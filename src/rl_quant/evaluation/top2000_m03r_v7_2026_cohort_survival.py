"""Censoring-aware score-origin holding evidence for the 2026 retrospective.

The date-by-age telemetry used for descriptive exit charts is not a coherent
resampling unit: one cohort contributes to many dates and ages.  This module
instead retains every scored entry cohort's complete return-neutral unit
trajectory.  Partial discretionary sales are events, mandatory removals are
cause-typed censoring, and units still open at the final boundary are right
censored.  Bootstrap draws resample complete origin-date trajectories in
common circular blocks across settings.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from rl_quant.envs.hold30 import AGE_BIN_COUNT, TurnoverCause
from rl_quant.evaluation.top2000_m03r_v7_2026_execution_view import (
    Top2000M03RV72026EconomicExecutionView,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    Top2000M03RV72026RetrospectiveData,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_COHORT_BOOTSTRAP_SEED_SHA256,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.training.hold30_runtime import Hold30CanonicalTrace

TOP2000_M03R_V7_2026_COHORT_TRAJECTORY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-cohort-trajectories-v1"
)
TOP2000_M03R_V7_2026_COHORT_RMST_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-cohort-rmst60-v1"
)
TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES = 10_000
TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_BLOCK = 21
TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_SEED_SHA256 = (
    M03R_SEED17_TOP2000_2026_YTD_COHORT_BOOTSTRAP_SEED_SHA256
)

_FORCED_CENSOR_CAUSES = (
    TurnoverCause.MEMBERSHIP_FORCED,
    TurnoverCause.AVAILABILITY_FORCED,
    TurnoverCause.RISK_FORCED,
    TurnoverCause.TERMINAL,
)
_TRADE_ORDER = (*_FORCED_CENSOR_CAUSES[:-1], TurnoverCause.DISCRETIONARY)
_TOLERANCE = 3.0e-6


class Top2000M03RV72026CohortSurvivalError(ValueError):
    """A cohort path, censoring event, or bootstrap identity drifted."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort evidence is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}
        )
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _as_nonnegative_array(
    name: str,
    value: object,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != shape
        or not np.isfinite(array).all()
        or np.any(array < -_TOLERANCE)
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            f"{name} must be finite nonnegative with shape {shape}"
        )
    result = np.ascontiguousarray(array)
    result[result < 0.0] = 0.0
    return result


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026CohortTrajectoryReceipt:
    setting_id: str
    checkpoint_sha256: str
    checkpoint_fold_index: int
    chronology_receipt_sha256: str
    economic_execution_receipt_sha256: str
    score_dates_sha256: str
    entry_units_sha256: str
    discretionary_event_units_by_age_sha256: str
    forced_censor_units_by_cause_and_age_sha256: tuple[tuple[str, str], ...]
    terminal_censor_units_by_age_sha256: str
    origin_rows: int
    age_bins: int = AGE_BIN_COUNT
    weighting: str = "score-origin-return-neutral-entry-units"
    partial_sales_are_fractional_events: bool = True
    forced_removals_are_censoring: bool = True
    terminal_open_units_are_right_censored: bool = True
    complete_origin_trajectories: bool = True
    transition_major_vectorized_attribution: bool = True
    development_only: bool = True
    future_selected_universe: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_COHORT_TRAJECTORY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "chronology_receipt_sha256",
            "economic_execution_receipt_sha256",
            "score_dates_sha256",
            "entry_units_sha256",
            "discretionary_event_units_by_age_sha256",
            "terminal_censor_units_by_age_sha256",
        ):
            _require_digest(name, getattr(self, name))
        expected_causes = tuple(cause.value for cause in _FORCED_CENSOR_CAUSES)
        if (
            self.setting_id not in M03R_SEED17_TOP2000_SETTING_IDS
            or self.checkpoint_fold_index not in range(6)
            or self.origin_rows <= 30
            or self.age_bins != AGE_BIN_COUNT
            or tuple(name for name, _digest in self.forced_censor_units_by_cause_and_age_sha256)
            != expected_causes
            or self.weighting != "score-origin-return-neutral-entry-units"
            or not self.partial_sales_are_fractional_events
            or not self.forced_removals_are_censoring
            or not self.terminal_open_units_are_right_censored
            or not self.complete_origin_trajectories
            or not self.transition_major_vectorized_attribution
            or not self.development_only
            or not self.future_selected_universe
            or self.scientific_reporting_eligible
            or self.promotion_eligible
            or self.schema != TOP2000_M03R_V7_2026_COHORT_TRAJECTORY_SCHEMA
        ):
            raise Top2000M03RV72026CohortSurvivalError(
                "cohort receipt identity or censoring semantics drifted"
            )
        for cause, digest in self.forced_censor_units_by_cause_and_age_sha256:
            _require_digest(f"forced censor array {cause}", digest)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026CohortTrajectories:
    origin_dates: tuple[str, ...]
    entry_units: np.ndarray
    discretionary_event_units_by_age: np.ndarray
    forced_censor_units_by_cause_and_age: Mapping[str, np.ndarray]
    terminal_censor_units_by_age: np.ndarray
    receipt: Top2000M03RV72026CohortTrajectoryReceipt

    def __post_init__(self) -> None:
        origins = len(self.origin_dates)
        if (
            origins != self.receipt.origin_rows
            or len(set(self.origin_dates)) != origins
            or tuple(sorted(self.origin_dates)) != self.origin_dates
            or any(
                not isinstance(value, str)
                or len(value) != 10
                or not value.startswith("2026-")
                for value in self.origin_dates
            )
        ):
            raise Top2000M03RV72026CohortSurvivalError(
                "cohort origin-date axis drifted"
            )
        entry = _as_nonnegative_array("entry_units", self.entry_units, (origins,))
        events = _as_nonnegative_array(
            "discretionary_event_units_by_age",
            self.discretionary_event_units_by_age,
            (origins, AGE_BIN_COUNT),
        )
        terminal = _as_nonnegative_array(
            "terminal_censor_units_by_age",
            self.terminal_censor_units_by_age,
            (origins, AGE_BIN_COUNT),
        )
        expected_causes = tuple(cause.value for cause in _FORCED_CENSOR_CAUSES)
        if tuple(self.forced_censor_units_by_cause_and_age) != expected_causes:
            raise Top2000M03RV72026CohortSurvivalError(
                "forced-censor cause inventory drifted"
            )
        forced = {
            cause: _as_nonnegative_array(
                f"forced_censor_units_by_cause_and_age[{cause}]",
                self.forced_censor_units_by_cause_and_age[cause],
                (origins, AGE_BIN_COUNT),
            )
            for cause in expected_causes
        }
        removed = events + terminal
        for value in forced.values():
            removed = removed + value
        remaining = entry.copy()
        for age in range(AGE_BIN_COUNT):
            removed_at_age = removed[:, age]
            if np.any(removed_at_age > remaining + _TOLERANCE):
                raise Top2000M03RV72026CohortSurvivalError(
                    "cohort removal precedes or exceeds its remaining risk set"
                )
            remaining = np.maximum(remaining - removed_at_age, 0.0)
        if not np.allclose(
            removed.sum(axis=1),
            entry,
            rtol=2.0e-5,
            atol=_TOLERANCE,
        ):
            raise Top2000M03RV72026CohortSurvivalError(
                "entry units do not reconcile with events and censoring"
            )
        if np.any(np.count_nonzero(terminal > _TOLERANCE, axis=1) > 1):
            raise Top2000M03RV72026CohortSurvivalError(
                "one origin cannot have multiple administrative censor ages"
            )
        if (
            self.receipt.score_dates_sha256
            != _sha256(list(self.origin_dates))
            or self.receipt.entry_units_sha256
            != _array_sha256("entry_units", entry)
            or self.receipt.discretionary_event_units_by_age_sha256
            != _array_sha256("discretionary_event_units_by_age", events)
            or self.receipt.terminal_censor_units_by_age_sha256
            != _array_sha256("terminal_censor_units_by_age", terminal)
            or tuple(
                (cause, _array_sha256(f"forced_censor/{cause}", forced[cause]))
                for cause in expected_causes
            )
            != self.receipt.forced_censor_units_by_cause_and_age_sha256
        ):
            raise Top2000M03RV72026CohortSurvivalError(
                "cohort arrays do not match their receipt"
            )


def validate_top2000_m03r_v7_2026_cohort_trajectories(
    value: object,
) -> None:
    """Revalidate mutable NumPy arrays and their typed receipt at use time."""

    if type(value) is not Top2000M03RV72026CohortTrajectories:
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort trajectories must use the exact typed artifact"
        )
    assert isinstance(value, Top2000M03RV72026CohortTrajectories)
    if type(value.receipt) is not Top2000M03RV72026CohortTrajectoryReceipt:
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort trajectories require the exact typed receipt"
        )
    value.receipt.__post_init__()
    value.__post_init__()


def _fractional_removal(
    tagged: torch.Tensor,
    total: torch.Tensor,
    sold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe = torch.where(total > 0.0, total, torch.ones_like(total))
    fraction = torch.where(total > 0.0, sold / safe, torch.zeros_like(total))
    if bool((fraction < -_TOLERANCE).any()) or bool(
        (fraction > 1.0 + _TOLERANCE).any()
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort removal fraction lies outside [0, 1]"
        )
    tagged_sold = tagged * fraction.clamp(0.0, 1.0)
    return (
        (tagged - tagged_sold).clamp_min(0.0),
        (total - sold).clamp_min(0.0),
        tagged_sold,
    )


def build_top2000_m03r_v7_2026_cohort_trajectories(
    trace: Hold30CanonicalTrace,
    retrospective: Top2000M03RV72026RetrospectiveData,
    execution_view: Top2000M03RV72026EconomicExecutionView,
    *,
    setting_id: str,
    checkpoint_sha256: str,
    checkpoint_fold_index: int,
) -> Top2000M03RV72026CohortTrajectories:
    """Attribute every scored entry's complete event/censor trajectory."""

    _require_digest("checkpoint_sha256", checkpoint_sha256)
    if (
        type(retrospective) is not Top2000M03RV72026RetrospectiveData
        or type(execution_view) is not Top2000M03RV72026EconomicExecutionView
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort attribution requires exact retrospective and execution artifacts"
        )
    retrospective.__post_init__()
    execution_view.__post_init__()
    receipt = execution_view.receipt
    receipt.__post_init__()
    sequence = execution_view.sequence
    if (
        not isinstance(trace, Hold30CanonicalTrace)
        or len(trace.transitions) != receipt.executed_transition_rows
        or len(trace.boundary_states) != receipt.executed_state_rows
        or sequence.n_positions != receipt.executed_state_rows
        or sequence.batch_size != 1
        or checkpoint_fold_index != receipt.training_fold_index
        or setting_id not in M03R_SEED17_TOP2000_SETTING_IDS
        or receipt.chronology_receipt_sha256
        != retrospective.identity.receipt_sha256
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            "trace does not match the leakage-safe economic execution view"
        )
    start = receipt.local_score_transition_start
    stop = receipt.local_score_transition_stop_exclusive
    origin_dates = retrospective.score_return_dates
    if stop - start != len(origin_dates) or not 0 <= start < stop <= len(
        trace.transitions
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            "score origins do not match the trace-local score window"
        )
    origins = len(origin_dates)
    cash = sequence.cash_index
    reference = trace.transitions[start].retention_units_before_membership
    device = reference.device
    dtype = reference.dtype
    assets = sequence.num_assets
    tagged = torch.zeros((origins, assets), dtype=dtype, device=device)
    entry_tensor = torch.zeros(origins, dtype=dtype, device=device)
    events_tensor = torch.zeros(
        (origins, AGE_BIN_COUNT), dtype=dtype, device=device
    )
    forced_tensors = {
        cause.value: torch.zeros(
            (origins, AGE_BIN_COUNT), dtype=dtype, device=device
        )
        for cause in _FORCED_CENSOR_CAUSES
    }

    def by_origin_age(value: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        if (
            value.shape != (1, assets, AGE_BIN_COUNT)
            or value.device != device
            or value.dtype != dtype
        ):
            raise Top2000M03RV72026CohortSurvivalError(
                "trace cohort tensor geometry changed within the chronology"
            )
        by_asset_age = value.detach()[0].clone()
        by_asset_age[cash] = 0.0
        return by_asset_age.index_select(1, ages).transpose(0, 1).contiguous()

    for transition_index in range(start, len(trace.transitions)):
        transition = trace.transitions[transition_index]
        active_origins = min(max(transition_index - start, 0), origins)
        if active_origins:
            origin_indexes = torch.arange(
                active_origins, dtype=torch.long, device=device
            )
            ages = (
                transition_index - (start + origin_indexes)
            ).clamp_max(AGE_BIN_COUNT - 1)
            total = by_origin_age(
                transition.retention_units_before_membership,
                ages,
            )
            active_tagged = tagged[:active_origins]
            if bool((active_tagged - total > _TOLERANCE).any()):
                raise Top2000M03RV72026CohortSurvivalError(
                    "origin cohort exceeds the canonical retention ledger"
                )
            for cause in _TRADE_ORDER:
                sold = by_origin_age(
                    transition.accounting_by_cause[cause].sold_units_by_age,
                    ages,
                )
                active_tagged, total, removed = _fractional_removal(
                    active_tagged,
                    total,
                    sold,
                )
                removed_by_origin = removed.sum(dim=1)
                target = (
                    events_tensor
                    if cause is TurnoverCause.DISCRETIONARY
                    else forced_tensors[cause.value]
                )
                target[origin_indexes, ages] += removed_by_origin

            terminal_sold = by_origin_age(
                transition.accounting_by_cause[
                    TurnoverCause.TERMINAL
                ].sold_units_by_age,
                ages,
            )
            active_tagged, _total, removed = _fractional_removal(
                active_tagged,
                total,
                terminal_sold,
            )
            forced_tensors[TurnoverCause.TERMINAL.value][
                origin_indexes, ages
            ] += removed.sum(dim=1)
            tagged[:active_origins] = active_tagged

        if transition_index < stop:
            output_row = transition_index - start
            added = (
                transition.accounting_by_cause[TurnoverCause.DISCRETIONARY]
                .entry_units_added.detach()[0]
                .clone()
            )
            if added.shape != (assets,) or added.device != device or added.dtype != dtype:
                raise Top2000M03RV72026CohortSurvivalError(
                    "trace entry-unit geometry changed within the chronology"
                )
            added[cash] = 0.0
            tagged[output_row] = added
            entry_tensor[output_row] = added.sum()

    terminal_tensor = torch.zeros_like(events_tensor)
    origin_indexes = torch.arange(origins, dtype=torch.long, device=device)
    origin_transitions = start + origin_indexes
    censor_ages = (len(trace.transitions) - 1 - origin_transitions).clamp(
        min=0,
        max=AGE_BIN_COUNT - 1,
    )
    terminal_tensor[origin_indexes, censor_ages] = tagged.sum(dim=1)

    entry = entry_tensor.to(device="cpu", dtype=torch.float64).numpy().copy()
    events = events_tensor.to(device="cpu", dtype=torch.float64).numpy().copy()
    forced = {
        cause: value.to(device="cpu", dtype=torch.float64).numpy().copy()
        for cause, value in forced_tensors.items()
    }
    terminal = terminal_tensor.to(
        device="cpu", dtype=torch.float64
    ).numpy().copy()

    forced_hashes = tuple(
        (cause.value, _array_sha256(f"forced_censor/{cause.value}", forced[cause.value]))
        for cause in _FORCED_CENSOR_CAUSES
    )
    trajectory_receipt = Top2000M03RV72026CohortTrajectoryReceipt(
        setting_id=setting_id,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_fold_index=checkpoint_fold_index,
        chronology_receipt_sha256=retrospective.identity.receipt_sha256,
        economic_execution_receipt_sha256=receipt.receipt_sha256,
        score_dates_sha256=_sha256(list(origin_dates)),
        entry_units_sha256=_array_sha256("entry_units", entry),
        discretionary_event_units_by_age_sha256=_array_sha256(
            "discretionary_event_units_by_age", events
        ),
        forced_censor_units_by_cause_and_age_sha256=forced_hashes,
        terminal_censor_units_by_age_sha256=_array_sha256(
            "terminal_censor_units_by_age", terminal
        ),
        origin_rows=origins,
    )
    return Top2000M03RV72026CohortTrajectories(
        origin_dates=origin_dates,
        entry_units=entry,
        discretionary_event_units_by_age=events,
        forced_censor_units_by_cause_and_age=forced,
        terminal_censor_units_by_age=terminal,
        receipt=trajectory_receipt,
    )


def _origin_block_indices(*, origins: int, replicate: int) -> np.ndarray:
    seed = bytes.fromhex(TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_SEED_SHA256)
    digest = hashlib.sha256(
        seed
        + replicate.to_bytes(8, "big")
        + TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_BLOCK.to_bytes(2, "big")
        + origins.to_bytes(4, "big")
    ).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))
    starts = rng.integers(
        0,
        origins,
        size=math.ceil(origins / TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_BLOCK),
    )
    blocks = [
        (start + np.arange(TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_BLOCK))
        % origins
        for start in starts
    ]
    return np.concatenate(blocks)[:origins].astype(np.int64, copy=False)


def _rmst60_from_trajectories(
    entry: np.ndarray,
    events: np.ndarray,
    forced: np.ndarray,
    terminal: np.ndarray,
) -> np.ndarray:
    if entry.ndim != 1 or events.shape != forced.shape or events.shape != terminal.shape:
        raise Top2000M03RV72026CohortSurvivalError(
            "RMST trajectory arrays are misaligned"
        )
    draws = entry.shape[0]
    remaining = entry.copy()
    survival = np.ones(draws, dtype=np.float64)
    rmst = np.zeros(draws, dtype=np.float64)
    estimable = entry > 0.0
    for age in range(60):
        remaining_after_forced = remaining - forced[:, age]
        if np.any(remaining_after_forced < -_TOLERANCE):
            raise Top2000M03RV72026CohortSurvivalError(
                "forced censoring exceeds cohort risk set"
            )
        risk = np.maximum(remaining_after_forced, 0.0)
        event = events[:, age]
        if np.any(event > risk + _TOLERANCE):
            raise Top2000M03RV72026CohortSurvivalError(
                "discretionary events exceed post-forced risk set"
            )
        hazard = np.divide(event, risk, out=np.zeros_like(event), where=risk > 0.0)
        survival *= np.clip(1.0 - hazard, 0.0, 1.0)
        remaining = np.maximum(risk - event - terminal[:, age], 0.0)
        estimable &= (remaining > _TOLERANCE) | (survival <= _TOLERANCE)
        rmst += survival
    rmst[~estimable] = np.nan
    return rmst


def evaluate_top2000_m03r_v7_2026_cohort_rmst60(
    trajectories: Sequence[Top2000M03RV72026CohortTrajectories],
) -> dict[str, Any]:
    """Compute point RMST60 and coherent origin-block uncertainty for a panel."""

    rows = tuple(trajectories)
    for value in rows:
        validate_top2000_m03r_v7_2026_cohort_trajectories(value)
    if (
        len(rows) != len(M03R_SEED17_TOP2000_SETTING_IDS)
        or tuple(value.receipt.setting_id for value in rows)
        != M03R_SEED17_TOP2000_SETTING_IDS
        or len({value.receipt.checkpoint_fold_index for value in rows}) != 1
        or len({value.origin_dates for value in rows}) != 1
        or len(
            {
                value.receipt.chronology_receipt_sha256
                for value in rows
            }
        )
        != 1
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort RMST requires the exact ordered 12-setting single-fold panel"
        )
    origins = len(rows[0].origin_dates)
    indexes = np.stack(
        [
            _origin_block_indices(origins=origins, replicate=replicate)
            for replicate in range(TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES)
        ]
    )
    origin_block_schedule_sha256 = _array_sha256(
        "joint_origin_block_indexes",
        indexes.astype(">i8", copy=False),
    )
    counts = np.zeros(
        (TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES, origins),
        dtype=np.float64,
    )
    replicate_rows = np.repeat(np.arange(counts.shape[0]), origins)
    np.add.at(counts, (replicate_rows, indexes.reshape(-1)), 1.0)
    output_rows: list[dict[str, Any]] = []
    for value in rows:
        entry = np.asarray(value.entry_units, dtype=np.float64)
        events = np.asarray(value.discretionary_event_units_by_age, dtype=np.float64)
        forced = sum(
            (
                np.asarray(value.forced_censor_units_by_cause_and_age[cause.value])
                for cause in _FORCED_CENSOR_CAUSES
            ),
            start=np.zeros_like(events),
        )
        terminal = np.asarray(value.terminal_censor_units_by_age, dtype=np.float64)
        point = _rmst60_from_trajectories(
            np.asarray([entry.sum()]),
            events.sum(axis=0, keepdims=True),
            forced.sum(axis=0, keepdims=True),
            terminal.sum(axis=0, keepdims=True),
        )[0]
        draw_entry = counts @ entry
        draw_events = counts @ events
        draw_forced = counts @ forced
        draw_terminal = counts @ terminal
        draw_rmst = _rmst60_from_trajectories(
            draw_entry,
            draw_events,
            draw_forced,
            draw_terminal,
        )
        valid = draw_rmst[np.isfinite(draw_rmst)]
        if not np.isfinite(point) or valid.size != draw_rmst.size:
            uncertainty: dict[str, Any] = {
                "status": "unavailable",
                "reason": "no-score-origin-entry-units-in-one-or-more-bootstrap-draws",
                "valid_replicates": int(valid.size),
            }
            point_value: float | None = None if not np.isfinite(point) else float(point)
        else:
            uncertainty = {
                "status": "available",
                "reason": None,
                "valid_replicates": int(valid.size),
                "bootstrap_standard_error": float(valid.std(ddof=1)),
                "two_sided_95_percent_interval": [
                    float(np.quantile(valid, 0.025, method="inverted_cdf")),
                    float(np.quantile(valid, 0.975, method="inverted_cdf")),
                ],
            }
            point_value = float(point)
        output_rows.append(
            {
                "setting_id": value.receipt.setting_id,
                "status": uncertainty["status"],
                "reason": uncertainty["reason"],
                "rmst60_sessions": point_value,
                "uncertainty": uncertainty,
                "entry_units": float(entry.sum()),
                "trajectory_receipt_sha256": value.receipt.receipt_sha256,
            }
        )
    unsigned = {
        "schema": TOP2000_M03R_V7_2026_COHORT_RMST_SCHEMA,
        "checkpoint_fold_index": rows[0].receipt.checkpoint_fold_index,
        "origin_dates_sha256": _sha256(list(rows[0].origin_dates)),
        "resampling": "joint-complete-origin-trajectory-circular-block-by-entry-date",
        "forced_exit_treatment": "cause-typed-censoring-before-discretionary-event",
        "terminal_open_treatment": "administrative-right-censoring",
        "partial_sale_treatment": "fractional-discretionary-event-units",
        "block_length_origin_sessions": TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_BLOCK,
        "bootstrap_replicates": TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES,
        "bootstrap_seed_sha256": TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_SEED_SHA256,
        "origin_block_schedule_sha256": origin_block_schedule_sha256,
        "common_origin_draws_across_settings": True,
        "rows": output_rows,
        "development_only": True,
        "future_selected_universe": True,
        "scientific_reporting_eligible": False,
        "promotion_eligible": False,
    }
    return {**unsigned, "receipt_sha256": _sha256(unsigned)}


def validate_top2000_m03r_v7_2026_cohort_rmst60_receipt(
    value: object,
) -> None:
    """Validate a serialized panel RMST receipt before evaluator trust."""

    if not isinstance(value, dict):
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort RMST receipt must be a dict"
        )
    required = {
        "schema",
        "checkpoint_fold_index",
        "origin_dates_sha256",
        "resampling",
        "forced_exit_treatment",
        "terminal_open_treatment",
        "partial_sale_treatment",
        "block_length_origin_sessions",
        "bootstrap_replicates",
        "bootstrap_seed_sha256",
        "origin_block_schedule_sha256",
        "common_origin_draws_across_settings",
        "rows",
        "development_only",
        "future_selected_universe",
        "scientific_reporting_eligible",
        "promotion_eligible",
        "receipt_sha256",
    }
    if set(value) != required:
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort RMST receipt keys drifted"
        )
    if (
        value["schema"] != TOP2000_M03R_V7_2026_COHORT_RMST_SCHEMA
        or isinstance(value["checkpoint_fold_index"], bool)
        or not isinstance(value["checkpoint_fold_index"], int)
        or value["checkpoint_fold_index"] not in range(6)
        or value["resampling"]
        != "joint-complete-origin-trajectory-circular-block-by-entry-date"
        or value["forced_exit_treatment"]
        != "cause-typed-censoring-before-discretionary-event"
        or value["terminal_open_treatment"]
        != "administrative-right-censoring"
        or value["partial_sale_treatment"]
        != "fractional-discretionary-event-units"
        or value["block_length_origin_sessions"]
        != TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_BLOCK
        or value["bootstrap_replicates"]
        != TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES
        or value["bootstrap_seed_sha256"]
        != TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_SEED_SHA256
        or value["common_origin_draws_across_settings"] is not True
        or value["development_only"] is not True
        or value["future_selected_universe"] is not True
        or value["scientific_reporting_eligible"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort RMST semantics drifted"
        )
    for name in (
        "origin_dates_sha256",
        "bootstrap_seed_sha256",
        "origin_block_schedule_sha256",
    ):
        _require_digest(name, value[name])
    rows = value["rows"]
    if (
        not isinstance(rows, list)
        or tuple(row.get("setting_id") for row in rows)
        != M03R_SEED17_TOP2000_SETTING_IDS
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort RMST setting inventory drifted"
        )
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "setting_id",
            "status",
            "reason",
            "rmst60_sessions",
            "uncertainty",
            "entry_units",
            "trajectory_receipt_sha256",
        }:
            raise Top2000M03RV72026CohortSurvivalError(
                "cohort RMST row keys drifted"
            )
        _require_digest(
            "trajectory_receipt_sha256",
            row["trajectory_receipt_sha256"],
        )
        uncertainty = row["uncertainty"]
        if not isinstance(uncertainty, dict):
            raise Top2000M03RV72026CohortSurvivalError(
                "cohort RMST uncertainty must be a dict"
            )
        entry_units = row["entry_units"]
        if (
            isinstance(entry_units, bool)
            or not isinstance(entry_units, (int, float))
            or not math.isfinite(float(entry_units))
            or float(entry_units) < 0.0
        ):
            raise Top2000M03RV72026CohortSurvivalError(
                "cohort RMST entry units must be finite nonnegative"
            )
        if row["status"] == "available":
            rmst = row["rmst60_sessions"]
            standard_error = uncertainty.get("bootstrap_standard_error")
            interval = uncertainty.get("two_sided_95_percent_interval")
            if (
                row["reason"] is not None
                or set(uncertainty)
                != {
                    "status",
                    "reason",
                    "valid_replicates",
                    "bootstrap_standard_error",
                    "two_sided_95_percent_interval",
                }
                or uncertainty["status"] != "available"
                or uncertainty["reason"] is not None
                or uncertainty["valid_replicates"]
                != TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES
                or isinstance(uncertainty["valid_replicates"], bool)
                or isinstance(rmst, bool)
                or not isinstance(rmst, (int, float))
                or not math.isfinite(float(rmst))
                or not 0.0 <= float(rmst) <= 60.0
                or float(entry_units) <= 0.0
                or isinstance(standard_error, bool)
                or not isinstance(standard_error, (int, float))
                or not math.isfinite(float(standard_error))
                or float(standard_error) < 0.0
                or not isinstance(interval, list)
                or len(interval) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in interval
                )
                or not 0.0 <= float(interval[0]) <= float(interval[1]) <= 60.0
            ):
                raise Top2000M03RV72026CohortSurvivalError(
                    "available cohort RMST row drifted"
                )
        elif row["status"] == "unavailable":
            rmst = row["rmst60_sessions"]
            valid_replicates = uncertainty.get("valid_replicates")
            if (
                row["reason"]
                != "no-score-origin-entry-units-in-one-or-more-bootstrap-draws"
                or set(uncertainty)
                != {"status", "reason", "valid_replicates"}
                or uncertainty["status"] != "unavailable"
                or uncertainty["reason"] != row["reason"]
                or isinstance(valid_replicates, bool)
                or not isinstance(valid_replicates, int)
                or not 0 <= valid_replicates
                < TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES
                or (
                    rmst is not None
                    and (
                        isinstance(rmst, bool)
                        or not isinstance(rmst, (int, float))
                        or not math.isfinite(float(rmst))
                        or not 0.0 <= float(rmst) <= 60.0
                    )
                )
            ):
                raise Top2000M03RV72026CohortSurvivalError(
                    "unavailable cohort RMST row drifted"
                )
        else:
            raise Top2000M03RV72026CohortSurvivalError(
                "cohort RMST row status drifted"
            )
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if _require_digest("receipt_sha256", value["receipt_sha256"]) != _sha256(
        unsigned
    ):
        raise Top2000M03RV72026CohortSurvivalError(
            "cohort RMST receipt hash drifted"
        )


__all__ = [
    "TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_BLOCK",
    "TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_REPLICATES",
    "TOP2000_M03R_V7_2026_COHORT_BOOTSTRAP_SEED_SHA256",
    "TOP2000_M03R_V7_2026_COHORT_RMST_SCHEMA",
    "TOP2000_M03R_V7_2026_COHORT_TRAJECTORY_SCHEMA",
    "Top2000M03RV72026CohortSurvivalError",
    "Top2000M03RV72026CohortTrajectories",
    "Top2000M03RV72026CohortTrajectoryReceipt",
    "build_top2000_m03r_v7_2026_cohort_trajectories",
    "evaluate_top2000_m03r_v7_2026_cohort_rmst60",
    "validate_top2000_m03r_v7_2026_cohort_rmst60_receipt",
    "validate_top2000_m03r_v7_2026_cohort_trajectories",
]
