"""Typed v3 alpha/Sharpe evaluation for the Hold-30 mechanism screen.

The evaluator is intentionally promotion-closed until the manifest binds the
factor family, multiplicity procedure, and moving-block plan.  It computes
metrics from retained daily arrays; caller-provided pass booleans are never an
input.  Alpha-core and the A06 Sharpe-overlay are separate streams and cannot
share an endpoint receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from rl_quant.datasets.hold30_alpha import (
    Hold30AlphaDataBindingReceipt,
    Hold30AlphaEvaluationProvenance,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_HORIZONS as _PROTOCOL_HORIZONS,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_MECH8_IDS as _PROTOCOL_IDS,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_PROTOCOL_GENERATION as _PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_V3_CANONICAL_ID as _PROTOCOL_CANONICAL_ID,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_VALIDATION_COSTS_BPS as _PROTOCOL_COST_RUNGS,
)

HOLD30_ALPHA_GENERATION = _PROTOCOL_GENERATION
HOLD30_ALPHA_IDS = _PROTOCOL_IDS
HOLD30_ALPHA_CORE_ID = _PROTOCOL_CANONICAL_ID
HOLD30_A06_OVERLAY_ID = HOLD30_ALPHA_IDS[6]
HOLD30_ALPHA_COST_RUNGS = _PROTOCOL_COST_RUNGS
HOLD30_ALPHA_HORIZONS = _PROTOCOL_HORIZONS
HOLD30_ALPHA_HAC_LAGS = (10, 21, 30, 42)
HOLD30_ALPHA_FOLDS = 6
HOLD30_ALPHA_DAYS_PER_FOLD = 63
HOLD30_ALPHA_SEEDS = (17, 29, 43, 71, 101)
HOLD30_ALPHA_SCHEMA = "rl-quant.hold30.alpha-evaluation-v3"
HOLD30_ALPHA_INVENTORY_SCHEMA = "rl-quant.hold30.alpha-artifact-inventory-v1"
HOLD30_ALPHA_LOCKBOX_SCHEMA = "rl-quant.hold30.alpha-lockbox-consumption-v1"
HOLD30_ALPHA_STREAM_BY_ID = dict(
    zip(
        HOLD30_ALPHA_IDS,
        (
            "legacy_absolute",
            "persistent_absolute",
            "active_te",
            "alpha_core",
            "a04_no_uncertainty",
            "a05_no_te_floor",
            "a06_sharpe_overlay",
            "a07_direct_sharpe",
        ),
        strict=True,
    )
)

_DIGEST_CHARS = frozenset("0123456789abcdef")
_RNG_DOMAIN = b"rl-quant.hold30.alpha-moving-block-v1\x00"


class Hold30AlphaEvaluationError(ValueError):
    """Typed evaluation evidence is incomplete or scientifically inconsistent."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise Hold30AlphaEvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _array(value: Any, shape: tuple[int, ...], name: str, *, boolean: bool = False) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise Hold30AlphaEvaluationError(f"{name} must have shape {shape}")
    if boolean:
        if result.dtype != np.bool_:
            raise Hold30AlphaEvaluationError(f"{name} must be boolean")
    elif not np.issubdtype(result.dtype, np.floating) or not np.isfinite(result).all():
        raise Hold30AlphaEvaluationError(f"{name} must be finite floating-point")
    return np.ascontiguousarray(result)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _endpoint_tensor_sha256(value: np.ndarray) -> str:
    """Match ``hold30_endpoints._tensor_sha256`` for CPU float tensors."""

    array = np.ascontiguousarray(value)
    dtype_names = {np.dtype("float32"): "torch.float32", np.dtype("float64"): "torch.float64"}
    try:
        dtype_name = dtype_names[array.dtype]
    except KeyError as exc:
        raise Hold30AlphaEvaluationError("endpoint-bound returns must be float32 or float64") from exc
    digest = hashlib.sha256()
    digest.update(dtype_name.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_row_indices_sha256(value: Sequence[int]) -> str:
    """Bind exact source-axis row identities independently of display dates."""

    return _sha256(list(value))


def _factor_arrays_sha256(factors: Mapping[str, np.ndarray]) -> str:
    return _sha256({name: _array_sha256(np.asarray(factors[name])) for name in sorted(factors)})


def _cross_section_arrays_sha256(panel: Hold30AlphaFoldPanel) -> str:
    return _sha256(
        {
            "scores": {
                str(horizon): _array_sha256(np.asarray(panel.scores[horizon]))
                for horizon in HOLD30_ALPHA_HORIZONS
            },
            "future_excess_returns": {
                str(horizon): _array_sha256(np.asarray(panel.future_excess_returns[horizon]))
                for horizon in HOLD30_ALPHA_HORIZONS
            },
            "future_valid": {
                str(horizon): _array_sha256(np.asarray(panel.future_valid[horizon]))
                for horizon in HOLD30_ALPHA_HORIZONS
            },
            "uncertainty": _array_sha256(np.asarray(panel.uncertainty)),
            "alpha_pnl_by_age": _array_sha256(np.asarray(panel.alpha_pnl_by_age)),
        }
    )


@dataclass(frozen=True, slots=True)
class Hold30AlphaEvaluationPlan:
    """Manifest-bound choices; missing inferential choices block promotion."""

    factor_names: tuple[str, ...]
    bootstrap_seed_sha256: str | None = None
    bootstrap_replicates: int | None = None
    bootstrap_block_lengths: tuple[int, ...] | None = None
    interval_alpha: float | None = None
    factor_multiplicity_method: str | None = None
    factor_family_alpha: float | None = None
    annualization_sessions: int = 252
    hac_lags: tuple[int, ...] = HOLD30_ALPHA_HAC_LAGS
    protocol_generation: str = HOLD30_ALPHA_GENERATION

    def __post_init__(self) -> None:
        if self.protocol_generation != HOLD30_ALPHA_GENERATION:
            raise Hold30AlphaEvaluationError("v2 and other protocol generations are rejected")
        if not self.factor_names or len(set(self.factor_names)) != len(self.factor_names):
            raise Hold30AlphaEvaluationError("declared factor names must be nonempty and unique")
        if "PIT_CAP_MARKET_EXCESS" in self.factor_names:
            raise Hold30AlphaEvaluationError(
                "declared factors cannot shadow the explicit PIT market excess column"
            )
        if self.hac_lags != HOLD30_ALPHA_HAC_LAGS:
            raise Hold30AlphaEvaluationError("HAC lag sensitivity must be exactly (10,21,30,42)")
        if self.annualization_sessions != 252:
            raise Hold30AlphaEvaluationError("v3 annualization is fixed at 252 sessions")
        supplied_bootstrap = (
            self.bootstrap_seed_sha256,
            self.bootstrap_replicates,
            self.bootstrap_block_lengths,
            self.interval_alpha,
        )
        if any(value is not None for value in supplied_bootstrap) and any(
            value is None for value in supplied_bootstrap
        ):
            raise Hold30AlphaEvaluationError("moving-block choices must be supplied together")
        if self.bootstrap_seed_sha256 is not None:
            _require_digest("bootstrap_seed_sha256", self.bootstrap_seed_sha256)
            if self.bootstrap_replicates is None or self.bootstrap_replicates < 1_000:
                raise Hold30AlphaEvaluationError("bootstrap_replicates must be at least 1,000")
            if (
                self.bootstrap_block_lengths is None
                or not self.bootstrap_block_lengths
                or len(set(self.bootstrap_block_lengths)) != len(self.bootstrap_block_lengths)
                or any(not 1 <= value <= HOLD30_ALPHA_DAYS_PER_FOLD for value in self.bootstrap_block_lengths)
            ):
                raise Hold30AlphaEvaluationError("bootstrap block lengths are invalid")
            if self.interval_alpha is None or not 0.0 < self.interval_alpha < 0.5:
                raise Hold30AlphaEvaluationError("interval_alpha must lie in (0,0.5)")
        if (self.factor_multiplicity_method is None) != (self.factor_family_alpha is None):
            raise Hold30AlphaEvaluationError("factor multiplicity method and alpha must be supplied together")
        if self.factor_family_alpha is not None and not 0.0 < self.factor_family_alpha < 0.5:
            raise Hold30AlphaEvaluationError("factor_family_alpha must lie in (0,0.5)")

    @property
    def promotion_plan_complete(self) -> bool:
        return (
            self.bootstrap_seed_sha256 is not None
            and self.factor_multiplicity_method is not None
            and self.factor_family_alpha is not None
        )


@dataclass(frozen=True, slots=True)
class Hold30AlphaFoldPanel:
    protocol_generation: str
    setting_id: str
    stream_id: str
    fold_index: int
    dates: tuple[str, ...]
    source_row_indices: tuple[int, ...]
    policy_net_returns: Mapping[int, np.ndarray]
    c1_net_returns: Mapping[int, np.ndarray]
    pit_risk_free_returns: np.ndarray
    pit_market_total_returns: np.ndarray
    factor_returns: Mapping[str, np.ndarray]
    cash_index: int
    policy_weights: np.ndarray
    c1_weights: np.ndarray
    scores: Mapping[int, np.ndarray]
    future_excess_returns: Mapping[int, np.ndarray]
    future_valid: Mapping[int, np.ndarray]
    uncertainty: np.ndarray
    alpha_pnl_by_age: np.ndarray
    endpoint_receipt: Mapping[str, Any]
    evaluation_provenance: Hold30AlphaEvaluationProvenance
    data_binding_receipt: Hold30AlphaDataBindingReceipt
    risk_free_receipt_sha256: str
    factor_receipt_sha256: str
    cross_section_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.protocol_generation != HOLD30_ALPHA_GENERATION:
            raise Hold30AlphaEvaluationError("fold panel rejects superseded v2 generation")
        if self.setting_id not in HOLD30_ALPHA_IDS:
            raise Hold30AlphaEvaluationError("fold panel setting/stream identity is invalid")
        if self.stream_id != HOLD30_ALPHA_STREAM_BY_ID[self.setting_id]:
            raise Hold30AlphaEvaluationError("setting and stream identities cannot be relabeled")
        if not 0 <= self.fold_index < HOLD30_ALPHA_FOLDS:
            raise Hold30AlphaEvaluationError("fold_index must lie in [0,5]")
        if len(self.dates) != HOLD30_ALPHA_DAYS_PER_FOLD or len(set(self.dates)) != len(self.dates):
            raise Hold30AlphaEvaluationError("each fold needs 63 unique score dates")
        if (
            not isinstance(self.source_row_indices, tuple)
            or len(self.source_row_indices) != HOLD30_ALPHA_DAYS_PER_FOLD
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.source_row_indices
            )
        ):
            raise Hold30AlphaEvaluationError(
                "source_row_indices must be 63 nonnegative integer source rows"
            )
        if any(
            right <= left
            for left, right in zip(
                self.source_row_indices,
                self.source_row_indices[1:],
            )
        ):
            raise Hold30AlphaEvaluationError(
                "source_row_indices must be strictly increasing"
            )
        if set(self.policy_net_returns) != set(HOLD30_ALPHA_COST_RUNGS) or set(
            self.c1_net_returns
        ) != set(HOLD30_ALPHA_COST_RUNGS):
            raise Hold30AlphaEvaluationError("cost ladders must be exactly 10/20/40 bp")
        for cost in HOLD30_ALPHA_COST_RUNGS:
            _array(self.policy_net_returns[cost], (63,), f"policy returns {cost}")
            _array(self.c1_net_returns[cost], (63,), f"C1 returns {cost}")
        _array(self.pit_risk_free_returns, (63,), "PIT risk-free returns")
        _array(self.pit_market_total_returns, (63,), "PIT market total returns")
        if np.any(np.asarray(self.pit_market_total_returns) <= -1.0):
            raise Hold30AlphaEvaluationError("PIT market total returns must exceed -100%")
        if not isinstance(self.factor_returns, Mapping) or not self.factor_returns:
            raise Hold30AlphaEvaluationError("factor_returns must be a nonempty mapping")
        for name, value in self.factor_returns.items():
            if not isinstance(name, str) or not name:
                raise Hold30AlphaEvaluationError("factor names must be nonempty strings")
            _array(value, (63,), f"factor returns {name}")
        assets = np.asarray(self.policy_weights).shape[1]
        if assets < 2:
            raise Hold30AlphaEvaluationError("portfolio weights need CASH and risky assets")
        if (
            isinstance(self.cash_index, bool)
            or not isinstance(self.cash_index, int)
            or not 0 <= self.cash_index < assets
        ):
            raise Hold30AlphaEvaluationError("cash_index lies outside the asset axis")
        _array(self.policy_weights, (63, assets), "policy_weights")
        _array(self.c1_weights, (63, assets), "c1_weights")
        for name, weights in (
            ("policy_weights", self.policy_weights),
            ("c1_weights", self.c1_weights),
        ):
            if np.any(weights < -1e-10) or not np.allclose(
                np.sum(weights, axis=1), 1.0, atol=1e-8, rtol=1e-8
            ):
                raise Hold30AlphaEvaluationError(f"{name} must be a long-only unit simplex")
        if set(self.scores) != set(HOLD30_ALPHA_HORIZONS) or set(
            self.future_excess_returns
        ) != set(HOLD30_ALPHA_HORIZONS) or set(self.future_valid) != set(
            HOLD30_ALPHA_HORIZONS
        ):
            raise Hold30AlphaEvaluationError("cross-sectional horizons must be exactly 5/21/30/63")
        for horizon in HOLD30_ALPHA_HORIZONS:
            _array(self.scores[horizon], (63, assets), f"scores {horizon}")
            _array(
                self.future_excess_returns[horizon],
                (63, assets),
                f"future excess returns {horizon}",
            )
            _array(self.future_valid[horizon], (63, assets), f"future valid {horizon}", boolean=True)
            if bool(np.any(self.future_valid[horizon][:, self.cash_index])):
                raise Hold30AlphaEvaluationError("CASH cannot enter cross-sectional IC or deciles")
        uncertainty = _array(self.uncertainty, (63, assets), "uncertainty")
        if np.any(uncertainty < 0.0):
            raise Hold30AlphaEvaluationError("uncertainty must be nonnegative")
        _array(self.alpha_pnl_by_age, (63, 61), "alpha_pnl_by_age")
        if not isinstance(self.evaluation_provenance, Hold30AlphaEvaluationProvenance):
            raise Hold30AlphaEvaluationError("typed V3 evaluation provenance is required")
        if not isinstance(self.data_binding_receipt, Hold30AlphaDataBindingReceipt):
            raise Hold30AlphaEvaluationError("typed V3 data-binding receipt is required")
        provenance = self.evaluation_provenance
        binding = self.data_binding_receipt
        if tuple(self.factor_returns) != provenance.factor_names:
            raise Hold30AlphaEvaluationError("factor arrays differ from V3 provenance order")
        if binding.protocol_generation != HOLD30_ALPHA_GENERATION:
            raise Hold30AlphaEvaluationError("data binding is not V3")
        if binding.evaluation_provenance_id != provenance.receipt_id:
            raise Hold30AlphaEvaluationError("data binding names another evaluation provenance")
        if binding.source_axis_id != self.endpoint_receipt.get("source_axis_id"):
            raise Hold30AlphaEvaluationError("data binding names another source axis")
        if binding.evaluation_panel_id != self.endpoint_receipt.get("evaluation_panel_id"):
            raise Hold30AlphaEvaluationError("data binding names another evaluation panel")
        for name in (
            "risk_free_receipt_sha256",
            "factor_receipt_sha256",
            "cross_section_receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class Hold30AlphaAuxiliaryPaths:
    """Typed H3 seed, initialization, and C8 daily active-return paths."""

    protocol_generation: str
    seed_active_log_returns: np.ndarray
    seed_run_receipt_sha256s: tuple[tuple[str, ...], ...]
    initialization_active_log_returns: np.ndarray
    initialization_source_endpoint_sha256s: tuple[str, ...]
    c8_active_log_returns: np.ndarray
    c8_cross_fold_mapping_receipt_sha256: str
    c8_selection_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.protocol_generation != HOLD30_ALPHA_GENERATION:
            raise Hold30AlphaEvaluationError("auxiliary paths reject superseded v2 generation")
        _array(self.seed_active_log_returns, (5, 6, 63), "seed active log returns")
        _array(
            self.initialization_active_log_returns,
            (6, 63),
            "initialization active log returns",
        )
        _array(self.c8_active_log_returns, (64, 6, 63), "C8 active log returns")
        if len(self.seed_run_receipt_sha256s) != 5 or any(
            len(row) != 6 for row in self.seed_run_receipt_sha256s
        ):
            raise Hold30AlphaEvaluationError("seed source receipts must be exactly 5 x 6")
        if len(self.initialization_source_endpoint_sha256s) != 6:
            raise Hold30AlphaEvaluationError("initialization needs six endpoint source receipts")
        if len(self.c8_selection_receipt_sha256s) != 6:
            raise Hold30AlphaEvaluationError("C8 needs six fold selection receipts")
        values = [
            *[value for row in self.seed_run_receipt_sha256s for value in row],
            *self.initialization_source_endpoint_sha256s,
            self.c8_cross_fold_mapping_receipt_sha256,
            *self.c8_selection_receipt_sha256s,
        ]
        for value in values:
            _require_digest("auxiliary source receipt", value)
        if len({value for row in self.seed_run_receipt_sha256s for value in row}) != 30:
            raise Hold30AlphaEvaluationError("five-seed run receipts must be distinct")
        if len(set(self.c8_selection_receipt_sha256s)) != 6:
            raise Hold30AlphaEvaluationError("C8 fold selection receipts must be distinct")


def evaluate_hold30_alpha_auxiliary_paths(
    paths: Hold30AlphaAuxiliaryPaths,
    *,
    alpha_core_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute seed/init/C8 values; no caller-supplied scalar may substitute."""

    if alpha_core_receipt.get("setting_id") != HOLD30_ALPHA_CORE_ID:
        raise Hold30AlphaEvaluationError("auxiliary paths require the alpha-core receipt")
    endpoint_sources = tuple(alpha_core_receipt.get("endpoint_receipt_sha256s", ()))
    if endpoint_sources != paths.initialization_source_endpoint_sha256s:
        raise Hold30AlphaEvaluationError("initialization paths are not bound to alpha-core endpoints")
    seed_totals = np.sum(paths.seed_active_log_returns, axis=(1, 2))
    initialization_total = float(np.sum(paths.initialization_active_log_returns))
    c8_totals = np.sum(paths.c8_active_log_returns, axis=(1, 2))
    primary_active = alpha_core_receipt["cost_ladder"]["20"]["active"]
    candidate_total = float(np.log1p(primary_active["active_relative_wealth_return"]))
    payload: dict[str, Any] = {
        "schema": "rl-quant.hold30.alpha-auxiliary-paths-v1",
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "setting_id": HOLD30_ALPHA_CORE_ID,
        "seed_aggregate_active_log_returns": [float(value) for value in seed_totals],
        "positive_seed_count": int(np.sum(seed_totals > 0.0)),
        "initialization_aggregate_active_log_return": initialization_total,
        "C8_aggregate_active_log_returns_sorted": [
            float(value) for value in np.sort(c8_totals)
        ],
        "candidate_exceeds_61st_C8": candidate_total > float(np.sort(c8_totals)[60]),
        "array_sha256s": {
            "seed": _array_sha256(paths.seed_active_log_returns),
            "initialization": _array_sha256(paths.initialization_active_log_returns),
            "C8": _array_sha256(paths.c8_active_log_returns),
        },
        "source_receipts": {
            "seed_runs": [list(row) for row in paths.seed_run_receipt_sha256s],
            "initialization_endpoints": list(paths.initialization_source_endpoint_sha256s),
            "C8_cross_fold_mapping": paths.c8_cross_fold_mapping_receipt_sha256,
            "C8_fold_selections": list(paths.c8_selection_receipt_sha256s),
        },
        "live_recomputed": True,
        "promotion_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def _validate_endpoint_binding(panel: Hold30AlphaFoldPanel) -> None:
    receipt = panel.endpoint_receipt
    if not isinstance(receipt, Mapping):
        raise Hold30AlphaEvaluationError("endpoint receipt must be a mapping")
    if receipt.get("protocol_generation") != HOLD30_ALPHA_GENERATION:
        raise Hold30AlphaEvaluationError("endpoint receipt is not v3")
    if receipt.get("setting_id") != panel.setting_id or receipt.get("stream_id") != panel.stream_id:
        raise Hold30AlphaEvaluationError("endpoint receipt binds another return stream")
    if receipt.get("fold_index") != panel.fold_index:
        raise Hold30AlphaEvaluationError("endpoint receipt binds another fold")
    tensors = receipt.get("tensor_receipts")
    if not isinstance(tensors, Mapping) or set(tensors) != {"10", "20", "40"}:
        raise Hold30AlphaEvaluationError("endpoint tensor ladder is incomplete")
    for cost in HOLD30_ALPHA_COST_RUNGS:
        policy = np.asarray(panel.policy_net_returns[cost])
        c1 = np.asarray(panel.c1_net_returns[cost])
        if np.any(policy <= -1.0) or np.any(c1 <= -1.0):
            raise Hold30AlphaEvaluationError("daily net return reached -100%")
        active_log = np.log1p(policy) - np.log1p(c1)
        expected = {
            "policy_net_returns_sha256": _endpoint_tensor_sha256(policy),
            "C1_net_returns_sha256": _endpoint_tensor_sha256(c1),
            "active_log_returns_sha256": _endpoint_tensor_sha256(active_log),
        }
        if tensors[str(cost)] != expected:
            raise Hold30AlphaEvaluationError("daily arrays differ from endpoint tensor receipts")
    for name, value in (
        ("risk_free_receipt_sha256", panel.risk_free_receipt_sha256),
        ("factor_receipt_sha256", panel.factor_receipt_sha256),
        ("cross_section_receipt_sha256", panel.cross_section_receipt_sha256),
    ):
        if receipt.get(name) != value:
            raise Hold30AlphaEvaluationError(f"endpoint receipt does not bind {name}")
    exact_path_bindings = {
        "dates_sha256": _sha256(list(panel.dates)),
        "source_row_indices_sha256": _source_row_indices_sha256(
            panel.source_row_indices
        ),
        "policy_weights_sha256": _array_sha256(np.asarray(panel.policy_weights)),
        "C1_weights_sha256": _array_sha256(np.asarray(panel.c1_weights)),
    }
    for name, value in exact_path_bindings.items():
        if receipt.get(name) != value:
            raise Hold30AlphaEvaluationError(f"endpoint receipt does not bind exact {name}")
    if receipt.get("risk_free_returns_sha256") != _array_sha256(
        np.asarray(panel.pit_risk_free_returns)
    ):
        raise Hold30AlphaEvaluationError("endpoint receipt does not bind exact PIT risk-free returns")
    provenance = panel.evaluation_provenance
    binding = panel.data_binding_receipt
    if binding.cash_returns_sha256 != _endpoint_tensor_sha256(
        np.asarray(panel.pit_risk_free_returns)
    ):
        raise Hold30AlphaEvaluationError(
            "data binding cash_returns_sha256 does not bind exact PIT risk-free returns"
        )
    explicit_market_binding = {
        "market_benchmark_id": provenance.market_benchmark_id,
        "market_artifact_sha256": provenance.market_artifact_sha256,
        "market_total_returns_sha256": _array_sha256(
            np.asarray(panel.pit_market_total_returns)
        ),
        "evaluation_provenance_id": provenance.receipt_id,
        "data_binding_receipt_id": binding.receipt_id,
        "evaluation_panel_id": binding.evaluation_panel_id,
        "source_axis_id": binding.source_axis_id,
    }
    for name, value in explicit_market_binding.items():
        if receipt.get(name) != value:
            raise Hold30AlphaEvaluationError(f"endpoint receipt does not bind exact {name}")
    if panel.risk_free_receipt_sha256 != provenance.risk_free_artifact_sha256:
        raise Hold30AlphaEvaluationError("risk-free receipt differs from V3 provenance artifact")
    if panel.factor_receipt_sha256 != provenance.factor_artifact_sha256:
        raise Hold30AlphaEvaluationError("factor receipt differs from V3 provenance artifact")
    if receipt.get("factor_returns_sha256") != _factor_arrays_sha256(panel.factor_returns):
        raise Hold30AlphaEvaluationError("endpoint receipt does not bind exact factor arrays")
    if receipt.get("cross_section_inputs_sha256") != _cross_section_arrays_sha256(panel):
        raise Hold30AlphaEvaluationError("endpoint receipt does not bind exact cross-sectional arrays")
    claimed = _require_digest("endpoint receipt_sha256", receipt.get("receipt_sha256"))
    unsigned = dict(receipt)
    del unsigned["receipt_sha256"]
    if _sha256(unsigned) != claimed:
        raise Hold30AlphaEvaluationError("endpoint receipt self-hash mismatch")


def _sample_std(value: np.ndarray) -> float:
    return float(np.std(value, ddof=1)) if value.size > 1 else 0.0


def _max_drawdown(growth: np.ndarray) -> float:
    wealth = np.concatenate(([1.0], np.cumprod(growth)))
    peak = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / peak))


def _total_metrics(net: np.ndarray, risk_free: np.ndarray) -> dict[str, Any]:
    excess = net - risk_free
    std = _sample_std(excess)
    downside = float(np.sqrt(np.mean(np.square(np.minimum(excess, 0.0)))))
    wealth = float(np.prod(1.0 + net))
    annual_return = wealth ** (252.0 / net.size) - 1.0
    drawdown = _max_drawdown(1.0 + net)
    return {
        "total_net_return": wealth - 1.0,
        "annualized_net_return": annual_return,
        "annualized_volatility": _sample_std(net) * math.sqrt(252.0),
        "annualized_downside_deviation": downside * math.sqrt(252.0),
        "net_sharpe": math.sqrt(252.0) * float(np.mean(excess)) / std if std > 0.0 else None,
        "sortino": math.sqrt(252.0) * float(np.mean(excess)) / downside
        if downside > 0.0
        else None,
        "maximum_drawdown": drawdown,
        "calmar": annual_return / drawdown if drawdown > 0.0 else None,
    }


def _active_metrics(
    policy: np.ndarray,
    benchmark: np.ndarray,
    policy_weights: np.ndarray,
    benchmark_weights: np.ndarray,
) -> dict[str, Any]:
    active_log = np.log1p(policy) - np.log1p(benchmark)
    std = _sample_std(active_log)
    policy_std = _sample_std(policy)
    benchmark_std = _sample_std(benchmark)
    correlation = None
    if policy_std > 0.0 and benchmark_std > 0.0:
        correlation = float(np.corrcoef(policy, benchmark)[0, 1])
    return {
        "sum_active_log_return": float(np.sum(active_log)),
        "mean_active_log_return_daily": float(np.mean(active_log)),
        "tracking_error_annualized": std * math.sqrt(252.0),
        "information_ratio_annualized": math.sqrt(252.0)
        * float(np.mean(active_log))
        / std
        if std > 0.0
        else None,
        "active_relative_wealth_return": float(np.exp(np.sum(active_log)) - 1.0),
        "active_relative_wealth_max_drawdown": _max_drawdown(np.exp(active_log)),
        "daily_active_hit_rate": float(np.mean(active_log > 0.0)),
        "mean_active_share": float(np.mean(0.5 * np.sum(np.abs(policy_weights - benchmark_weights), axis=1))),
        "daily_return_correlation_to_C1": correlation,
    }


def _newey_west(x: np.ndarray, residual: np.ndarray, lag: int) -> np.ndarray:
    n = x.shape[0]
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for offset in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - offset / (lag + 1.0)
        gamma = scores[offset:].T @ scores[:-offset]
        meat += weight * (gamma + gamma.T)
    inverse = np.linalg.pinv(x.T @ x)
    return inverse @ meat @ inverse


def _regression(y: np.ndarray, factors: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    x = np.column_stack((np.ones(y.size), factors))
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coefficients
    residual_std = _sample_std(residual)
    hac: dict[str, Any] = {}
    for lag in HOLD30_ALPHA_HAC_LAGS:
        covariance = _newey_west(x, residual, lag)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        t_values = np.divide(
            coefficients,
            standard_errors,
            out=np.full_like(coefficients, np.nan),
            where=standard_errors > 0.0,
        )
        hac[str(lag)] = {
            "alpha_se_daily": float(standard_errors[0]),
            "alpha_t": float(t_values[0]) if math.isfinite(float(t_values[0])) else None,
            "alpha_two_sided_normal_p": 2.0 * (1.0 - NormalDist().cdf(abs(float(t_values[0]))))
            if math.isfinite(float(t_values[0]))
            else None,
        }
    payload = {
        "alpha_daily": float(coefficients[0]),
        "alpha_annualized_arithmetic": float(coefficients[0] * 252.0),
        "loadings": {name: float(value) for name, value in zip(names, coefficients[1:])},
        "residual_volatility_annualized": residual_std * math.sqrt(252.0),
        "residual_sharpe": math.sqrt(252.0) * float(coefficients[0]) / residual_std
        if residual_std > 0.0
        else None,
        "hac": hac,
    }
    if len(names) == 1:
        payload["beta"] = float(coefficients[1])
    return payload


def _rank_average(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    result = np.empty(value.size, dtype=np.float64)
    cursor = 0
    while cursor < value.size:
        end = cursor + 1
        while end < value.size and value[order[end]] == value[order[cursor]]:
            end += 1
        result[order[cursor:end]] = 0.5 * (cursor + end - 1) + 1.0
        cursor = end
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or _sample_std(left) == 0.0 or _sample_std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _cross_sectional(panels: Sequence[Hold30AlphaFoldPanel]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for horizon in HOLD30_ALPHA_HORIZONS:
        pearson: list[float] = []
        rank_ic: list[float] = []
        decile_sum = np.zeros(10, dtype=np.float64)
        decile_count = np.zeros(10, dtype=np.int64)
        bucket_ic: list[list[float]] = [[] for _ in range(5)]
        bucket_return_sum = np.zeros(5, dtype=np.float64)
        bucket_return_count = np.zeros(5, dtype=np.int64)
        for panel in panels:
            for day in range(63):
                mask = panel.future_valid[horizon][day]
                score = panel.scores[horizon][day, mask]
                outcome = panel.future_excess_returns[horizon][day, mask]
                uncertainty = panel.uncertainty[day, mask]
                if score.size < 10:
                    continue
                value = _correlation(score, outcome)
                rank = _correlation(_rank_average(score), _rank_average(outcome))
                if value is not None:
                    pearson.append(value)
                if rank is not None:
                    rank_ic.append(rank)
                ordering = np.argsort(score, kind="mergesort")
                for decile, indices in enumerate(np.array_split(ordering, 10)):
                    if indices.size:
                        decile_sum[decile] += float(np.sum(outcome[indices]))
                        decile_count[decile] += indices.size
                uncertainty_order = np.argsort(uncertainty, kind="mergesort")
                for bucket, indices in enumerate(np.array_split(uncertainty_order, 5)):
                    ic = _correlation(score[indices], outcome[indices])
                    if ic is not None:
                        bucket_ic[bucket].append(ic)
                    bucket_return_sum[bucket] += float(np.sum(outcome[indices]))
                    bucket_return_count[bucket] += indices.size
        means = np.divide(
            decile_sum,
            decile_count,
            out=np.full(10, np.nan),
            where=decile_count > 0,
        )
        output[str(horizon)] = {
            "pearson_ic_mean": float(np.mean(pearson)) if pearson else None,
            "pearson_ic_daily_observations": len(pearson),
            "rank_ic_mean": float(np.mean(rank_ic)) if rank_ic else None,
            "rank_ic_daily_observations": len(rank_ic),
            "score_decile_mean_excess_return": [
                float(value) if math.isfinite(float(value)) else None for value in means
            ],
            "top_minus_bottom_decile": float(means[-1] - means[0])
            if math.isfinite(float(means[-1])) and math.isfinite(float(means[0]))
            else None,
            "uncertainty_bucket_rank_ic": [
                float(np.mean(values)) if values else None for values in bucket_ic
            ],
            "uncertainty_bucket_mean_excess_return": [
                float(bucket_return_sum[index] / bucket_return_count[index])
                if bucket_return_count[index] > 0
                else None
                for index in range(5)
            ],
        }
    age = np.sum(np.stack([panel.alpha_pnl_by_age for panel in panels]), axis=(0, 1))
    output["alpha_decay_by_age"] = {
        "ages": list(range(61)),
        "pnl": [float(value) for value in age],
        "cumulative_pnl": [float(value) for value in np.cumsum(age)],
    }
    return output


def _block_indices(seed: bytes, replicate: int, fold: int, block_length: int) -> np.ndarray:
    pieces: list[int] = []
    block = 0
    choices = 63 - block_length + 1
    while len(pieces) < 63:
        material = b"".join(
            (
                _RNG_DOMAIN,
                seed,
                replicate.to_bytes(4, "big"),
                fold.to_bytes(2, "big"),
                block_length.to_bytes(2, "big"),
                block.to_bytes(2, "big"),
            )
        )
        start = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % choices
        pieces.extend(range(start, start + block_length))
        block += 1
    return np.asarray(pieces[:63], dtype=np.int64)


def _bootstrap_alpha_intervals(
    y_folds: np.ndarray,
    market_excess_folds: np.ndarray,
    factor_folds: np.ndarray,
    plan: Hold30AlphaEvaluationPlan,
) -> dict[str, Any] | None:
    if plan.bootstrap_seed_sha256 is None:
        return None
    assert plan.bootstrap_replicates is not None
    assert plan.bootstrap_block_lengths is not None
    assert plan.interval_alpha is not None
    seed = bytes.fromhex(plan.bootstrap_seed_sha256)
    output: dict[str, Any] = {}
    for block_length in plan.bootstrap_block_lengths:
        market_alpha = np.empty(plan.bootstrap_replicates, dtype=np.float64)
        factor_alpha = np.empty(plan.bootstrap_replicates, dtype=np.float64)
        for replicate in range(plan.bootstrap_replicates):
            ys: list[np.ndarray] = []
            markets: list[np.ndarray] = []
            xs: list[np.ndarray] = []
            for fold in range(6):
                index = _block_indices(seed, replicate, fold, block_length)
                ys.append(y_folds[fold, index])
                markets.append(market_excess_folds[fold, index])
                xs.append(factor_folds[fold, index])
            y = np.concatenate(ys)
            market = np.concatenate(markets)
            x = np.concatenate(xs)
            market_alpha[replicate] = np.linalg.lstsq(
                np.column_stack((np.ones(y.size), market)), y, rcond=None
            )[0][0]
            factor_alpha[replicate] = np.linalg.lstsq(
                np.column_stack((np.ones(y.size), market, x)), y, rcond=None
            )[0][0]
        output[str(block_length)] = {
            "market_alpha_daily_interval": [
                float(np.quantile(market_alpha, plan.interval_alpha, method="inverted_cdf")),
                float(np.quantile(market_alpha, 1.0 - plan.interval_alpha, method="inverted_cdf")),
            ],
            "multifactor_alpha_daily_interval": [
                float(np.quantile(factor_alpha, plan.interval_alpha, method="inverted_cdf")),
                float(np.quantile(factor_alpha, 1.0 - plan.interval_alpha, method="inverted_cdf")),
            ],
        }
    return output


def evaluate_hold30_alpha_stream(
    panels: Sequence[Hold30AlphaFoldPanel],
    *,
    plan: Hold30AlphaEvaluationPlan,
) -> dict[str, Any]:
    """Recompute one exact six-fold stream from endpoint-bound daily arrays."""

    ordered = tuple(sorted(panels, key=lambda value: value.fold_index))
    if len(ordered) != 6 or tuple(value.fold_index for value in ordered) != tuple(range(6)):
        raise Hold30AlphaEvaluationError("stream evaluation requires exact folds 0..5")
    if len({value.stream_id for value in ordered}) != 1 or len(
        {value.setting_id for value in ordered}
    ) != 1:
        raise Hold30AlphaEvaluationError("one stream cannot mix setting identities")
    if any(tuple(panel.factor_returns) != plan.factor_names for panel in ordered):
        raise Hold30AlphaEvaluationError("factor columns differ from the manifest-bound declaration")
    if any(panel.evaluation_provenance.factor_names != plan.factor_names for panel in ordered):
        raise Hold30AlphaEvaluationError("factor provenance differs from the manifest-bound declaration")
    factor_conventions = ordered[0].evaluation_provenance.factor_return_conventions
    if any(
        panel.evaluation_provenance.factor_return_conventions != factor_conventions
        for panel in ordered
    ):
        raise Hold30AlphaEvaluationError("factor return conventions differ across folds")
    all_dates = [date for panel in ordered for date in panel.dates]
    if len(set(all_dates)) != 378:
        raise Hold30AlphaEvaluationError("outer fold score dates must be disjoint")
    for name, values in (
        ("endpoint", [panel.endpoint_receipt["receipt_sha256"] for panel in ordered]),
        ("risk-free", [panel.risk_free_receipt_sha256 for panel in ordered]),
        ("factor", [panel.factor_receipt_sha256 for panel in ordered]),
        ("cross-section", [panel.cross_section_receipt_sha256 for panel in ordered]),
        ("data-binding", [panel.data_binding_receipt.receipt_id for panel in ordered]),
    ):
        if len(set(values)) != 6:
            raise Hold30AlphaEvaluationError(f"{name} fold receipts must be distinct")
    for panel in ordered:
        _validate_endpoint_binding(panel)

    risk_free = np.concatenate([panel.pit_risk_free_returns for panel in ordered])
    weights = np.concatenate([panel.policy_weights for panel in ordered])
    c1_weights = np.concatenate([panel.c1_weights for panel in ordered])
    cost_ladder: dict[str, Any] = {}
    for cost in HOLD30_ALPHA_COST_RUNGS:
        policy = np.concatenate([panel.policy_net_returns[cost] for panel in ordered])
        c1 = np.concatenate([panel.c1_net_returns[cost] for panel in ordered])
        cost_ladder[str(cost)] = {
            "policy_total": _total_metrics(policy, risk_free),
            "C1_total": _total_metrics(c1, risk_free),
            "active": _active_metrics(policy, c1, weights, c1_weights),
            "daily_array_sha256": {
                "policy": _array_sha256(policy),
                "C1": _array_sha256(c1),
                "active_log": _array_sha256(np.log1p(policy) - np.log1p(c1)),
            },
        }

    y_folds = np.stack(
        [panel.policy_net_returns[20] - panel.pit_risk_free_returns for panel in ordered]
    )
    transformed_factor_folds: list[np.ndarray] = []
    for panel in ordered:
        columns: list[np.ndarray] = []
        for name, convention in zip(plan.factor_names, factor_conventions, strict=True):
            values = np.asarray(panel.factor_returns[name])
            if convention == "total-return":
                values = values - np.asarray(panel.pit_risk_free_returns)
            elif convention not in {"zero-investment", "excess-over-risk-free"}:
                raise Hold30AlphaEvaluationError("unsupported factor return convention")
            columns.append(values)
        transformed_factor_folds.append(np.column_stack(columns))
    factor_folds = np.stack(transformed_factor_folds)
    market_excess_folds = np.stack(
        [panel.pit_market_total_returns - panel.pit_risk_free_returns for panel in ordered]
    )
    y = y_folds.reshape(-1)
    factors = factor_folds.reshape(-1, len(plan.factor_names))
    market_excess = market_excess_folds.reshape(-1)
    regression = {
        "market_only": _regression(y, market_excess[:, None], ["PIT_CAP_MARKET_EXCESS"]),
        "declared_multifactor": _regression(
            y,
            np.column_stack((market_excess, factors)),
            ("PIT_CAP_MARKET_EXCESS", *plan.factor_names),
        ),
        "moving_block_intervals": _bootstrap_alpha_intervals(
            y_folds,
            market_excess_folds,
            factor_folds,
            plan,
        ),
    }
    payload: dict[str, Any] = {
        "schema": HOLD30_ALPHA_SCHEMA,
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "setting_id": ordered[0].setting_id,
        "stream_id": ordered[0].stream_id,
        "folds": list(range(6)),
        "sessions": 378,
        "cost_ladder": cost_ladder,
        "regression": regression,
        "factor_return_conventions": {
            name: convention
            for name, convention in zip(plan.factor_names, factor_conventions, strict=True)
        },
        "cross_sectional": _cross_sectional(ordered),
        "endpoint_receipt_sha256s": [panel.endpoint_receipt["receipt_sha256"] for panel in ordered],
        "risk_free_receipt_sha256s": [panel.risk_free_receipt_sha256 for panel in ordered],
        "market_artifacts": [
            {
                "market_benchmark_id": panel.evaluation_provenance.market_benchmark_id,
                "market_artifact_sha256": panel.evaluation_provenance.market_artifact_sha256,
                "evaluation_provenance_id": panel.evaluation_provenance.receipt_id,
                "data_binding_receipt_id": panel.data_binding_receipt.receipt_id,
            }
            for panel in ordered
        ],
        "factor_receipt_sha256s": [panel.factor_receipt_sha256 for panel in ordered],
        "cross_section_receipt_sha256s": [panel.cross_section_receipt_sha256 for panel in ordered],
        "live_recomputed": True,
        "promotion_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def verify_hold30_c6_ownership(receipt: Mapping[str, Any]) -> None:
    """Require one 64-replay C6 receipt for every stable setting and fold."""

    required = {"schema", "protocol_generation", "rows", "receipt_sha256"}
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise Hold30AlphaEvaluationError("C6 ownership receipt is partial")
    if (
        receipt["schema"] != "rl-quant.hold30.alpha-c6-ownership-v1"
        or receipt["protocol_generation"] != HOLD30_ALPHA_GENERATION
    ):
        raise Hold30AlphaEvaluationError("C6 ownership is not v3")
    rows = receipt["rows"]
    if not isinstance(rows, list) or len(rows) != 48:
        raise Hold30AlphaEvaluationError("C6 requires exactly 8 x 6 ownership rows")
    expected = {(setting, fold) for setting in HOLD30_ALPHA_IDS for fold in range(6)}
    observed: set[tuple[str, int]] = set()
    hashes: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "setting_id",
            "fold_index",
            "replicates",
            "outer_score_rows",
            "other_rows_fixed",
            "canonical_five_seed_intent_receipt_sha256",
            "permutation_receipt_sha256",
        }:
            raise Hold30AlphaEvaluationError("C6 ownership row is malformed")
        key = (row["setting_id"], row["fold_index"])
        if key in observed:
            raise Hold30AlphaEvaluationError("C6 ownership contains a duplicate row")
        observed.add(key)
        if row["replicates"] != 64 or row["outer_score_rows"] != 63 or row["other_rows_fixed"] is not True:
            raise Hold30AlphaEvaluationError("C6 must permute only 63 outer rows in 64 replays")
        for name in (
            "canonical_five_seed_intent_receipt_sha256",
            "permutation_receipt_sha256",
        ):
            value = _require_digest(name, row[name])
            if value in hashes:
                raise Hold30AlphaEvaluationError("C6 ownership receipts cannot be reused")
            hashes.add(value)
    if observed != expected:
        raise Hold30AlphaEvaluationError("C6 ownership is incomplete or contains unknown settings")
    claimed = _require_digest("C6 receipt", receipt["receipt_sha256"])
    unsigned = dict(receipt)
    del unsigned["receipt_sha256"]
    if _sha256(unsigned) != claimed:
        raise Hold30AlphaEvaluationError("C6 ownership self-hash mismatch")


def verify_hold30_alpha_terminal_inventory(receipts: Sequence[Mapping[str, Any]]) -> None:
    """Reject missing, duplicate, selectively retried, or non-v3 trial rows."""

    expected = {
        (setting, fold, seed)
        for setting in HOLD30_ALPHA_IDS
        for fold in range(6)
        for seed in HOLD30_ALPHA_SEEDS
    }
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        raise Hold30AlphaEvaluationError("terminal receipts must be a sequence")
    observed: set[tuple[str, int, int]] = set()
    run_hashes: set[str] = set()
    for receipt in receipts:
        required = {
            "schema",
            "protocol_generation",
            "setting_id",
            "fold_index",
            "seed",
            "terminal_status",
            "selective_retry",
            "run_receipt_sha256",
            "artifact_graph_sha256",
            "receipt_sha256",
        }
        if not isinstance(receipt, Mapping) or set(receipt) != required:
            raise Hold30AlphaEvaluationError("terminal trial receipt is partial or unknown")
        if (
            receipt["schema"] != "rl-quant.hold30.alpha-terminal-trial-v1"
            or receipt["protocol_generation"] != HOLD30_ALPHA_GENERATION
            or receipt["terminal_status"] != "success"
            or receipt["selective_retry"] is not False
        ):
            raise Hold30AlphaEvaluationError("terminal trial is not an unselected v3 success")
        key = (receipt["setting_id"], receipt["fold_index"], receipt["seed"])
        if key in observed:
            raise Hold30AlphaEvaluationError("terminal trial identity is duplicated")
        observed.add(key)
        for name in ("run_receipt_sha256", "artifact_graph_sha256"):
            value = _require_digest(name, receipt[name])
            if value in run_hashes:
                raise Hold30AlphaEvaluationError("terminal trial artifacts cannot be reused")
            run_hashes.add(value)
        claimed = _require_digest("terminal receipt", receipt["receipt_sha256"])
        unsigned = dict(receipt)
        del unsigned["receipt_sha256"]
        if _sha256(unsigned) != claimed:
            raise Hold30AlphaEvaluationError("terminal trial self-hash mismatch")
    if observed != expected:
        raise Hold30AlphaEvaluationError("terminal inventory must be exactly 8 x 6 x 5")


def evaluate_hold30_matched_controls(
    *,
    alpha_core_active_log_returns: np.ndarray,
    control_active_log_returns: np.ndarray,
    target_profile: Mapping[str, float],
    control_profiles: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Recompute same-turnover/exposure/holding-duration control diagnostics."""

    target = _array(
        alpha_core_active_log_returns,
        (378,),
        "alpha-core active log returns",
    )
    controls = _array(
        control_active_log_returns,
        (64, 378),
        "matched-control active log returns",
    )
    tolerances = {
        "turnover": (0.05, "relative"),
        "risky_exposure": (0.01, "absolute"),
        "median_sale_age": (3.0, "absolute"),
        "survival_30": (0.05, "absolute"),
    }
    if set(target_profile) != set(tolerances) or set(control_profiles) != set(tolerances):
        raise Hold30AlphaEvaluationError("matched-control profiles must cover turnover/exposure/age/S30")
    match_pass = np.ones(64, dtype=np.bool_)
    profile_diagnostics: dict[str, Any] = {}
    for name, (tolerance, mode) in tolerances.items():
        target_value = float(target_profile[name])
        values = _array(control_profiles[name], (64,), f"control profile {name}")
        gap = np.abs(values - target_value)
        allowed = tolerance if mode == "absolute" else tolerance * abs(target_value)
        passed = gap <= allowed
        match_pass &= passed
        profile_diagnostics[name] = {
            "target": target_value,
            "tolerance": tolerance,
            "mode": mode,
            "maximum_gap": float(np.max(gap)),
            "matched_count": int(np.sum(passed)),
        }
    if not bool(np.all(match_pass)):
        raise Hold30AlphaEvaluationError("one or more declared matched controls violate profile tolerances")
    target_total = float(np.sum(target))
    control_totals = np.sum(controls, axis=1)
    target_ir = math.sqrt(252.0) * float(np.mean(target)) / _sample_std(target)
    control_irs = np.asarray(
        [math.sqrt(252.0) * float(np.mean(row)) / _sample_std(row) for row in controls]
    )
    return {
        "control_count": 64,
        "profile_diagnostics": profile_diagnostics,
        "target_active_log_total": target_total,
        "control_active_log_total_sorted": [float(value) for value in np.sort(control_totals)],
        "target_information_ratio": target_ir,
        "control_information_ratio_sorted": [float(value) for value in np.sort(control_irs)],
        "target_exceeds_61st_of_64_by_active_log_total": target_total
        > float(np.sort(control_totals)[60]),
        "target_exceeds_61st_of_64_by_information_ratio": target_ir
        > float(np.sort(control_irs)[60]),
        "return_arrays_sha256": {
            "target": _array_sha256(target),
            "controls": _array_sha256(controls),
        },
    }


def build_hold30_alpha_artifact_inventory(
    entries: Mapping[str, tuple[str, str] | tuple[str, str, str]],
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    paths: set[str] = set()
    for logical_id in sorted(entries):
        row = entries[logical_id]
        if len(row) == 2:
            path, digest = row
            payload_digest = None
        elif len(row) == 3:
            path, digest, payload_digest = row
            payload_digest = _require_digest(f"{logical_id} payload", payload_digest)
        else:
            raise Hold30AlphaEvaluationError("inventory entries must have two or three fields")
        relative = Path(path)
        if not logical_id or relative.is_absolute() or ".." in relative.parts:
            raise Hold30AlphaEvaluationError("inventory IDs/paths must be safe and relative")
        normalized = relative.as_posix()
        if normalized in paths:
            raise Hold30AlphaEvaluationError("inventory paths must be unique")
        paths.add(normalized)
        rows.append(
            {
                "logical_id": logical_id,
                "path": normalized,
                "sha256": _require_digest(logical_id, digest),
                "payload_sha256": payload_digest,
            }
        )
    payload: dict[str, Any] = {
        "schema": HOLD30_ALPHA_INVENTORY_SCHEMA,
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "entries": rows,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise Hold30AlphaEvaluationError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def verify_hold30_alpha_artifact_inventory(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    expected_json_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    required_manifest_sha256s: Sequence[str] = (),
) -> dict[str, dict[str, str | None]]:
    """Verify file bytes, then parse/compare JSON independent of formatting."""

    required = {"schema", "protocol_generation", "entries", "receipt_sha256"}
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise Hold30AlphaEvaluationError("artifact inventory is partial")
    if (
        receipt["schema"] != HOLD30_ALPHA_INVENTORY_SCHEMA
        or receipt["protocol_generation"] != HOLD30_ALPHA_GENERATION
    ):
        raise Hold30AlphaEvaluationError("artifact inventory rejects v2 generation")
    unsigned = dict(receipt)
    claimed = _require_digest("inventory receipt", unsigned.pop("receipt_sha256"))
    if _sha256(unsigned) != claimed:
        raise Hold30AlphaEvaluationError("artifact inventory self-hash mismatch")
    base = Path(root).resolve()
    if not base.is_dir():
        raise Hold30AlphaEvaluationError("artifact root is absent")
    live: dict[str, dict[str, str | None]] = {}
    used_paths: set[str] = set()
    for row in receipt["entries"]:
        if not isinstance(row, Mapping) or set(row) != {
            "logical_id",
            "path",
            "sha256",
            "payload_sha256",
        }:
            raise Hold30AlphaEvaluationError("inventory row is malformed")
        logical_id, relative_text = row["logical_id"], row["path"]
        if logical_id in live or relative_text in used_paths:
            raise Hold30AlphaEvaluationError("inventory IDs and paths must be unique")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise Hold30AlphaEvaluationError("inventory path escapes root")
        path = base / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(base):
            raise Hold30AlphaEvaluationError(f"inventory artifact {logical_id} is absent or unsafe")
        data = path.read_bytes()
        observed = hashlib.sha256(data).hexdigest()
        if observed != _require_digest(logical_id, row["sha256"]):
            raise Hold30AlphaEvaluationError(f"live byte hash mismatch for {logical_id}")
        payload_digest = row["payload_sha256"]
        if payload_digest is not None:
            payload_digest = _require_digest(f"{logical_id} payload", payload_digest)
            try:
                parsed_payload = json.loads(data, object_pairs_hook=_reject_duplicate_keys)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Hold30AlphaEvaluationError(
                    f"{logical_id} declares a payload digest but is not valid JSON"
                ) from exc
            if isinstance(parsed_payload, Mapping) and parsed_payload.get(
                "receipt_sha256"
            ) == payload_digest:
                unsigned_payload = dict(parsed_payload)
                del unsigned_payload["receipt_sha256"]
                observed_payload = _sha256(unsigned_payload)
            else:
                observed_payload = _sha256(parsed_payload)
            if observed_payload != payload_digest:
                raise Hold30AlphaEvaluationError(f"payload hash mismatch for {logical_id}")
        live[logical_id] = {"byte_sha256": observed, "payload_sha256": payload_digest}
        used_paths.add(relative_text)
        if expected_json_payloads is not None and logical_id in expected_json_payloads:
            try:
                parsed = json.loads(data, object_pairs_hook=_reject_duplicate_keys)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Hold30AlphaEvaluationError(f"{logical_id} is not valid JSON") from exc
            if _canonical_json(parsed) != _canonical_json(expected_json_payloads[logical_id]):
                raise Hold30AlphaEvaluationError(f"parsed JSON payload differs for {logical_id}")
    available = {
        digest
        for row in live.values()
        for digest in (row["byte_sha256"], row["payload_sha256"])
        if digest is not None
    }
    missing = sorted({_require_digest("manifest binding", value) for value in required_manifest_sha256s} - available)
    if missing:
        raise Hold30AlphaEvaluationError("manifest binding artifacts are absent: " + ", ".join(missing))
    return live


def manifest_binding_sha256s(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect every SHA-256 binding recursively; 40-char Git SHAs are not misclassified."""

    if manifest.get("protocol_generation") != HOLD30_ALPHA_GENERATION:
        raise Hold30AlphaEvaluationError("manifest rejects superseded v2 generation")
    values: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            candidate = value[7:] if key == "container_image_digest" and value.startswith("sha256:") else value
            if key == "manifest_sha256":
                return
            if key.endswith("sha256") or key == "container_image_digest":
                values.append(_require_digest(key, candidate))

    visit(manifest)
    return tuple(dict.fromkeys(values))


def publish_hold30_alpha_lockbox_marker(
    path: Path,
    *,
    verified_evaluation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish the one-shot consumed marker after evaluation passes."""

    required = {
        "schema",
        "protocol_generation",
        "manifest_sha256",
        "lockbox_id_sha256",
        "all_required_live_artifacts_verified",
        "evaluation_complete",
        "scientific_qualification",
        "promotion_authorized",
        "launch_authorized",
        "receipt_sha256",
    }
    if not isinstance(verified_evaluation_receipt, Mapping) or set(
        verified_evaluation_receipt
    ) != required:
        raise Hold30AlphaEvaluationError("lockbox publication requires the exact final receipt")
    if (
        verified_evaluation_receipt["schema"]
        != "rl-quant.hold30.alpha-final-evaluation-v1"
        or verified_evaluation_receipt["protocol_generation"] != HOLD30_ALPHA_GENERATION
        or verified_evaluation_receipt["all_required_live_artifacts_verified"] is not True
        or verified_evaluation_receipt["evaluation_complete"] is not True
        or verified_evaluation_receipt["launch_authorized"] is not False
        or not isinstance(verified_evaluation_receipt["scientific_qualification"], bool)
        or not isinstance(verified_evaluation_receipt["promotion_authorized"], bool)
    ):
        raise Hold30AlphaEvaluationError("final evaluation receipt is incomplete or invalid")
    evaluation_unsigned = dict(verified_evaluation_receipt)
    evaluation_claimed = _require_digest(
        "final evaluation receipt", evaluation_unsigned.pop("receipt_sha256")
    )
    if _sha256(evaluation_unsigned) != evaluation_claimed:
        raise Hold30AlphaEvaluationError("final evaluation receipt self-hash mismatch")
    payload: dict[str, Any] = {
        "schema": HOLD30_ALPHA_LOCKBOX_SCHEMA,
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "manifest_sha256": _require_digest(
            "manifest", verified_evaluation_receipt["manifest_sha256"]
        ),
        "lockbox_id_sha256": _require_digest(
            "lockbox", verified_evaluation_receipt["lockbox_id_sha256"]
        ),
        "reveal_receipt_sha256": evaluation_claimed,
        "consumed": True,
        "reuse_for_selection_permitted": False,
        "historical_2026_S0_S7_consumed": True,
        "historical_2026_evidence_used": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    encoded = _canonical_json(payload) + b"\n"
    marker = Path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if marker.read_bytes() != encoded:
            raise Hold30AlphaEvaluationError("lockbox already has a different consumption marker")
        return payload
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(marker.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload


def build_hold30_alpha_mech8_summary(
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind all eight V3 mechanisms and recompute the predeclared contrasts."""

    if not isinstance(receipts, Mapping) or set(receipts) != set(HOLD30_ALPHA_IDS):
        raise Hold30AlphaEvaluationError("mechanism summary requires the exact eight V3 settings")
    endpoint_sources: list[str] = []
    per_setting: dict[str, Any] = {}
    for setting_id in HOLD30_ALPHA_IDS:
        receipt = receipts[setting_id]
        expected_stream = HOLD30_ALPHA_STREAM_BY_ID[setting_id]
        if (
            receipt.get("schema") != HOLD30_ALPHA_SCHEMA
            or receipt.get("protocol_generation") != HOLD30_ALPHA_GENERATION
            or receipt.get("setting_id") != setting_id
            or receipt.get("stream_id") != expected_stream
            or receipt.get("promotion_authorized") is not False
        ):
            raise Hold30AlphaEvaluationError("mechanism stream receipt identity is invalid")
        claimed = _require_digest("mechanism receipt", receipt.get("receipt_sha256"))
        unsigned = dict(receipt)
        del unsigned["receipt_sha256"]
        if _sha256(unsigned) != claimed:
            raise Hold30AlphaEvaluationError("mechanism stream receipt self-hash mismatch")
        sources = receipt.get("endpoint_receipt_sha256s")
        if not isinstance(sources, list) or len(sources) != HOLD30_ALPHA_FOLDS:
            raise Hold30AlphaEvaluationError("each mechanism needs six endpoint receipts")
        endpoint_sources.extend(sources)
        primary = receipt["cost_ladder"]["20"]
        per_setting[setting_id] = {
            "stream_id": expected_stream,
            "receipt_sha256": claimed,
            "total_net_return": primary["policy_total"]["total_net_return"],
            "net_sharpe": primary["policy_total"]["net_sharpe"],
            "maximum_drawdown": primary["policy_total"]["maximum_drawdown"],
            "active_relative_wealth_return": primary["active"][
                "active_relative_wealth_return"
            ],
            "information_ratio_annualized": primary["active"][
                "information_ratio_annualized"
            ],
            "tracking_error_annualized": primary["active"][
                "tracking_error_annualized"
            ],
            "market_beta": receipt["regression"]["market_only"]["beta"],
        }
    if len(set(endpoint_sources)) != len(HOLD30_ALPHA_IDS) * HOLD30_ALPHA_FOLDS:
        raise Hold30AlphaEvaluationError("the eight mechanisms cannot reuse endpoint receipts")

    def contrast(left: str, right: str) -> dict[str, Any]:
        fields = (
            "total_net_return",
            "net_sharpe",
            "maximum_drawdown",
            "active_relative_wealth_return",
            "information_ratio_annualized",
            "tracking_error_annualized",
            "market_beta",
        )
        result: dict[str, Any] = {"left": left, "right": right}
        for field in fields:
            left_value = per_setting[left][field]
            right_value = per_setting[right][field]
            result[f"delta_{field}"] = (
                float(left_value - right_value)
                if left_value is not None and right_value is not None
                else None
            )
        return result

    m00, m01, m02, m03, a04, a05, a06, a07 = HOLD30_ALPHA_IDS
    contrasts = {
        "m01_minus_m00_persistence": contrast(m01, m00),
        "m02_minus_m01_active_objective": contrast(m02, m01),
        "m03_minus_m02_alpha_heads": contrast(m03, m02),
        "m03_minus_a04_uncertainty": contrast(m03, a04),
        "m03_minus_a05_te_floor": contrast(m03, a05),
        "a06_minus_m03_sharpe_overlay": contrast(a06, m03),
        "a07_minus_m03_direct_sharpe": contrast(a07, m03),
    }
    payload: dict[str, Any] = {
        "schema": "rl-quant.hold30.alpha-mech8-summary-v1",
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "primary_cost_bps": 20,
        "setting_order": list(HOLD30_ALPHA_IDS),
        "promotion_candidate": HOLD30_ALPHA_CORE_ID,
        "per_setting": per_setting,
        "contrasts": contrasts,
        "promotion_authorized": False,
        "launch_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def build_hold30_alpha_tranche(
    *,
    mech8_summary: Mapping[str, Any],
    alpha_core: Mapping[str, Any],
    a06_overlay: Mapping[str, Any],
    matched_controls: Mapping[str, Any],
    plan: Hold30AlphaEvaluationPlan,
) -> dict[str, Any]:
    """Build a default-closed tranche from two independently audited streams."""

    if (
        mech8_summary.get("schema") != "rl-quant.hold30.alpha-mech8-summary-v1"
        or mech8_summary.get("protocol_generation") != HOLD30_ALPHA_GENERATION
        or mech8_summary.get("promotion_candidate") != HOLD30_ALPHA_CORE_ID
        or mech8_summary.get("promotion_authorized") is not False
    ):
        raise Hold30AlphaEvaluationError("exact eight-mechanism summary is required")
    summary_unsigned = dict(mech8_summary)
    summary_claimed = _require_digest("mech8 summary", summary_unsigned.pop("receipt_sha256"))
    if _sha256(summary_unsigned) != summary_claimed:
        raise Hold30AlphaEvaluationError("mech8 summary self-hash mismatch")
    for receipt, setting, stream in (
        (alpha_core, HOLD30_ALPHA_CORE_ID, "alpha_core"),
        (a06_overlay, HOLD30_A06_OVERLAY_ID, "a06_sharpe_overlay"),
    ):
        if (
            receipt.get("protocol_generation") != HOLD30_ALPHA_GENERATION
            or receipt.get("setting_id") != setting
            or receipt.get("stream_id") != stream
            or receipt.get("promotion_authorized") is not False
        ):
            raise Hold30AlphaEvaluationError("alpha-core/A06 stream receipt identity is invalid")
        if mech8_summary.get("per_setting", {}).get(setting, {}).get(
            "receipt_sha256"
        ) != receipt.get("receipt_sha256"):
            raise Hold30AlphaEvaluationError("mech8 summary does not bind core/A06 receipts")
    if alpha_core["endpoint_receipt_sha256s"] == a06_overlay["endpoint_receipt_sha256s"]:
        raise Hold30AlphaEvaluationError("alpha core and A06 overlay cannot reuse endpoint receipts")
    primary = alpha_core["cost_ladder"]["20"]
    active = primary["active"]
    policy_sharpe = primary["policy_total"]["net_sharpe"]
    c1_sharpe = primary["C1_total"]["net_sharpe"]
    market_beta = alpha_core["regression"]["market_only"]["beta"]
    point_gates = {
        "pooled_information_ratio_gt_0.5": active["information_ratio_annualized"] is not None
        and active["information_ratio_annualized"] > 0.5,
        "tracking_error_in_2pct_6pct": 0.02 <= active["tracking_error_annualized"] <= 0.06,
        "market_beta_in_0.9_1.1": 0.9 <= market_beta <= 1.1,
        "net_sharpe_noninferiority_minus_0.10": policy_sharpe is not None
        and c1_sharpe is not None
        and policy_sharpe - c1_sharpe >= -0.10,
        "matched_control_active_log_total_exceeds_61st_of_64": matched_controls.get(
            "target_exceeds_61st_of_64_by_active_log_total"
        )
        is True,
    }
    blockers: list[str] = []
    if not plan.promotion_plan_complete:
        if plan.bootstrap_seed_sha256 is None:
            blockers.append("moving_block_interval_plan_not_frozen")
        if plan.factor_multiplicity_method is None:
            blockers.append("factor_alpha_multiplicity_procedure_not_frozen")
    # The user froze the point gates, but not which factor-alpha hypotheses form
    # the confirmatory family nor the adjusted lower-bound decision rule.  Even
    # with a populated method name, a separate manifest-bound hypothesis list
    # is required before promotion can become true.
    blockers.append("confirmatory_factor_alpha_hypothesis_family_not_frozen")
    if not all(point_gates.values()):
        blockers.append("one_or_more_alpha_core_point_gates_failed")
    payload: dict[str, Any] = {
        "schema": "rl-quant.hold30.alpha-tranche-v1",
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "candidate": HOLD30_ALPHA_CORE_ID,
        "overlay": HOLD30_A06_OVERLAY_ID,
        "alpha_core_receipt_sha256": alpha_core["receipt_sha256"],
        "a06_overlay_receipt_sha256": a06_overlay["receipt_sha256"],
        "mech8_summary_receipt_sha256": summary_claimed,
        "matched_controls": dict(matched_controls),
        "point_gates": point_gates,
        "promotion_blockers": blockers,
        "scientific_qualification": False,
        "promotion_authorized": False,
        "launch_authorized": False,
        "lockbox_consumed_by_this_function": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


__all__ = [
    "HOLD30_A06_OVERLAY_ID",
    "HOLD30_ALPHA_CORE_ID",
    "HOLD30_ALPHA_GENERATION",
    "HOLD30_ALPHA_HAC_LAGS",
    "HOLD30_ALPHA_HORIZONS",
    "HOLD30_ALPHA_IDS",
    "HOLD30_ALPHA_STREAM_BY_ID",
    "Hold30AlphaAuxiliaryPaths",
    "Hold30AlphaEvaluationError",
    "Hold30AlphaEvaluationPlan",
    "Hold30AlphaFoldPanel",
    "build_hold30_alpha_artifact_inventory",
    "build_hold30_alpha_mech8_summary",
    "build_hold30_alpha_tranche",
    "evaluate_hold30_alpha_auxiliary_paths",
    "evaluate_hold30_alpha_stream",
    "evaluate_hold30_matched_controls",
    "manifest_binding_sha256s",
    "publish_hold30_alpha_lockbox_marker",
    "verify_hold30_alpha_artifact_inventory",
    "verify_hold30_alpha_terminal_inventory",
    "verify_hold30_c6_ownership",
]
