"""Immutable TOP2000 M03R-v7 data adapter for the 2026 retrospective.

This module is intentionally separate from the pre-2026 training adapter.
It opens the raw 2026 partition namespace only after training completion,
reuses the existing causal raw-bar-to-daily aggregation, and materializes one
small content-bound cache that every setting can share.  The resulting
chronology contains exactly 252 pre-2026 context states followed by every
available 2026 state through 2026-06-23.  It is executed once without a reset;
only transitions whose *return date* is in 2026 are scored.

The current TOP2000 universe was selected during 2026 and has no point-in-time
membership history.  Every public type therefore fails closed unless it stays
future-selected, development-only, nonreportable, and nonpromotable.  This is
retrospective mechanism evidence, never a lockbox or canonical PIT result.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from rl_quant.datasets.provenance import declared_universe_actions
from rl_quant.envs.hold30 import CohortLedger
from rl_quant.training import hold30_top2000_development as training_data
from rl_quant.training.hold30_runtime import Hold30Sequence

TOP2000_M03R_V7_2026_RETROSPECTIVE_CACHE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-2026-retrospective-cache-v1"
)
TOP2000_M03R_V7_2026_RETROSPECTIVE_IDENTITY_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-2026-retrospective-data-v1"
)
TOP2000_M03R_V7_2026_RETROSPECTIVE_SOURCE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-2026-retrospective-source-v1"
)
TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK = (
    "I acknowledge the 2026 TOP2000 retrospective is future-selected, "
    "development-only, nonreportable, and nonpromotable"
)
TOP2000_M03R_V7_2026_START = dt.date(2026, 1, 1)
TOP2000_M03R_V7_2026_CUTOFF = dt.date(2026, 6, 23)
TOP2000_M03R_V7_2026_UNIVERSE_SELECTION_DATE = dt.date(2026, 6, 12)
TOP2000_M03R_V7_2026_CONTEXT_STATES = 252
TOP2000_M03R_V7_2026_MAX_TOTAL_STATES = 378
TOP2000_M03R_V7_2026_COST_RATE = 0.002
TOP2000_M03R_V7_2026_MAX_STOCK_WEIGHT = 0.01


class Top2000M03RV72026RetrospectiveDataError(ValueError):
    """The 2026 retrospective data boundary is absent or inconsistent."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV72026RetrospectiveDataError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _parse_dates(values: Sequence[str], *, name: str) -> tuple[dt.date, ...]:
    try:
        parsed = tuple(dt.date.fromisoformat(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026RetrospectiveDataError(
            f"{name} contains an invalid ISO exchange date"
        ) from exc
    if not parsed or any(left >= right for left, right in pairwise(parsed)):
        raise Top2000M03RV72026RetrospectiveDataError(
            f"{name} must be nonempty, unique, and strictly increasing"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026RetrospectiveSourceEvidence:
    """Full-content raw-source and completed-training evidence.

    ``test_partition_inventory_sha256`` binds the ``EvaluationPlan.test``
    rows.  Those rows are produced by the legacy package's evaluation-plan
    builder with full Parquet-content signatures, not its cheaper search-plan
    footer signatures.
    """

    base_dataset_identity: str
    search_identity: str
    lockbox_partition_names_hash: str
    test_identity: str
    test_partition_inventory_sha256: str
    manifest_sha256: str
    universe_sha256: str
    training_completion_receipt_sha256: str
    evaluation_contract_sha256: str
    raw_first_exchange_date: str
    raw_last_exchange_date: str
    universe_selection_date: str = (
        TOP2000_M03R_V7_2026_UNIVERSE_SELECTION_DATE.isoformat()
    )
    bar_seconds: int = training_data.TOP2000_HOLD30_SOURCE_BAR_SECONDS
    development_only: bool = True
    dataset_reportable: bool = False
    future_selected_universe: bool = True
    point_in_time_membership: bool = False
    static_universe: bool = True
    promotion_eligible: bool = False
    scientific_reporting_eligible: bool = False
    training_complete_before_2026_access: bool = True
    schema: str = TOP2000_M03R_V7_2026_RETROSPECTIVE_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "base_dataset_identity",
            "search_identity",
            "lockbox_partition_names_hash",
            "test_identity",
            "test_partition_inventory_sha256",
            "manifest_sha256",
            "universe_sha256",
            "training_completion_receipt_sha256",
            "evaluation_contract_sha256",
        ):
            _require_digest(name, getattr(self, name))
        try:
            raw_first = dt.date.fromisoformat(self.raw_first_exchange_date)
            raw_last = dt.date.fromisoformat(self.raw_last_exchange_date)
            selected = dt.date.fromisoformat(self.universe_selection_date)
        except (TypeError, ValueError) as exc:
            raise Top2000M03RV72026RetrospectiveDataError(
                "source evidence dates must be valid ISO dates"
            ) from exc
        if (
            self.schema != TOP2000_M03R_V7_2026_RETROSPECTIVE_SOURCE_SCHEMA
            or self.bar_seconds != training_data.TOP2000_HOLD30_SOURCE_BAR_SECONDS
            or raw_first >= TOP2000_M03R_V7_2026_START
            or raw_last != TOP2000_M03R_V7_2026_CUTOFF
            or selected != TOP2000_M03R_V7_2026_UNIVERSE_SELECTION_DATE
            or not self.development_only
            or self.dataset_reportable
            or not self.future_selected_universe
            or self.point_in_time_membership
            or not self.static_universe
            or self.promotion_eligible
            or self.scientific_reporting_eligible
            or not self.training_complete_before_2026_access
        ):
            raise Top2000M03RV72026RetrospectiveDataError(
                "2026 source evidence must remain completed-training, "
                "future-selected, development-only, nonreportable, and nonpromotable"
            )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026RetrospectiveIdentity:
    """Content identity and immutable interpretation of one chronology."""

    source_evidence_sha256: str
    pre2026_cache_sha256: str
    pre2026_cache_identity: str
    search_identity: str
    action_hash: str
    action_ids_sha256: str
    exchange_dates_sha256: str
    context_dates_sha256: str
    score_return_dates_sha256: str
    daily_ohlcv_sha256: str
    availability_sha256: str
    score_availability_sha256: str
    asset_returns_sha256: str
    benchmark_weights_sha256: str
    benchmark_gross_returns_sha256: str
    benchmark_total_turnover_sha256: str
    benchmark_net_returns_sha256: str
    benchmark_trace_sha256: str
    first_context_date: str
    last_context_date: str
    first_score_return_date: str
    last_score_return_date: str
    state_rows: int
    transition_rows: int
    context_state_rows: int
    score_transition_start: int
    score_transition_stop_exclusive: int
    score_transition_rows: int
    action_count: int
    one_way_cost_rate: float = TOP2000_M03R_V7_2026_COST_RATE
    max_stock_weight: float = TOP2000_M03R_V7_2026_MAX_STOCK_WEIGHT
    benchmark_id: str = training_data.TOP2000_HOLD30_BENCHMARK_ID
    chronology_rule: str = (
        "warm-252-pre2026-states-then-one-continuous-2026-runtime-no-reset"
    )
    score_date_rule: str = "score-transition-by-return-date-in-2026"
    development_only: bool = True
    future_selected_universe: bool = True
    point_in_time_membership: bool = False
    retrospective_only: bool = True
    outer_lockbox_claim_authorized: bool = False
    promotion_eligible: bool = False
    scientific_reporting_eligible: bool = False
    single_continuous_chronology: bool = True
    state_reset_count_within_2026: int = 0
    schema: str = TOP2000_M03R_V7_2026_RETROSPECTIVE_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_evidence_sha256",
            "pre2026_cache_sha256",
            "pre2026_cache_identity",
            "search_identity",
            "action_hash",
            "action_ids_sha256",
            "exchange_dates_sha256",
            "context_dates_sha256",
            "score_return_dates_sha256",
            "daily_ohlcv_sha256",
            "availability_sha256",
            "score_availability_sha256",
            "asset_returns_sha256",
            "benchmark_weights_sha256",
            "benchmark_gross_returns_sha256",
            "benchmark_total_turnover_sha256",
            "benchmark_net_returns_sha256",
            "benchmark_trace_sha256",
        ):
            _require_digest(name, getattr(self, name))
        try:
            first_context = dt.date.fromisoformat(self.first_context_date)
            last_context = dt.date.fromisoformat(self.last_context_date)
            first_score = dt.date.fromisoformat(self.first_score_return_date)
            last_score = dt.date.fromisoformat(self.last_score_return_date)
        except (TypeError, ValueError) as exc:
            raise Top2000M03RV72026RetrospectiveDataError(
                "retrospective identity dates must be valid ISO dates"
            ) from exc
        if (
            self.schema != TOP2000_M03R_V7_2026_RETROSPECTIVE_IDENTITY_SCHEMA
            or self.benchmark_id != training_data.TOP2000_HOLD30_BENCHMARK_ID
            or self.one_way_cost_rate != TOP2000_M03R_V7_2026_COST_RATE
            or self.max_stock_weight != TOP2000_M03R_V7_2026_MAX_STOCK_WEIGHT
            or self.context_state_rows != TOP2000_M03R_V7_2026_CONTEXT_STATES
            or self.state_rows != self.transition_rows + 1
            or not 2 <= self.state_rows <= TOP2000_M03R_V7_2026_MAX_TOTAL_STATES
            or self.score_transition_start != self.context_state_rows - 1
            or self.score_transition_stop_exclusive != self.transition_rows
            or self.score_transition_rows
            != self.score_transition_stop_exclusive - self.score_transition_start
            or self.score_transition_rows <= 0
            or self.action_count < 2
            or not first_context < last_context < TOP2000_M03R_V7_2026_START
            or not TOP2000_M03R_V7_2026_START <= first_score <= last_score
            or last_score != TOP2000_M03R_V7_2026_CUTOFF
            or not self.development_only
            or not self.future_selected_universe
            or self.point_in_time_membership
            or not self.retrospective_only
            or self.outer_lockbox_claim_authorized
            or self.promotion_eligible
            or self.scientific_reporting_eligible
            or not self.single_continuous_chronology
            or self.state_reset_count_within_2026 != 0
        ):
            raise Top2000M03RV72026RetrospectiveDataError(
                "2026 identity geometry or nonreportable semantics drifted"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())

    @property
    def axis_id(self) -> str:
        return f"top2000-m03r-v7-2026-retrospective:{self.receipt_sha256}"


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026RetrospectiveData:
    """One raw-input sequence with a warm-up prefix and exact score slice."""

    sequence: Hold30Sequence
    benchmark: training_data.Top2000MonthlyEqualWeightBuyAndDriftTrace
    source_evidence: Top2000M03RV72026RetrospectiveSourceEvidence
    identity: Top2000M03RV72026RetrospectiveIdentity
    exchange_dates: tuple[str, ...]
    action_ids: tuple[str, ...]
    score_return_dates: tuple[str, ...]
    cache_file_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.cache_file_sha256 is not None:
            _require_digest("cache_file_sha256", self.cache_file_sha256)
        if self.identity.source_evidence_sha256 != self.source_evidence.receipt_sha256:
            raise Top2000M03RV72026RetrospectiveDataError(
                "source evidence does not match the chronology identity"
            )
        if self.sequence.axis_id != self.identity.axis_id:
            raise Top2000M03RV72026RetrospectiveDataError(
                "runtime sequence does not match the chronology identity"
            )
        if (
            self.sequence.batch_size != 1
            or self.sequence.n_positions != self.identity.state_rows
            or self.sequence.num_assets != self.identity.action_count
            or len(self.exchange_dates) != self.identity.state_rows
            or len(self.action_ids) != self.identity.action_count
            or self.action_ids[0] != "CASH"
            or len(self.score_return_dates) != self.identity.score_transition_rows
        ):
            raise Top2000M03RV72026RetrospectiveDataError(
                "retrospective axes do not match the bound sequence"
            )
        expected_score_dates = self.exchange_dates[
            self.identity.context_state_rows :
        ]
        if self.score_return_dates != expected_score_dates:
            raise Top2000M03RV72026RetrospectiveDataError(
                "score dates must be the 2026 return-date suffix"
            )

    @property
    def score_transition_slice(self) -> slice:
        return slice(
            self.identity.score_transition_start,
            self.identity.score_transition_stop_exclusive,
        )


def _validate_daily_axes(
    daily_ohlcv: torch.Tensor,
    availability: torch.Tensor,
    exchange_dates: tuple[str, ...],
    action_ids: tuple[str, ...],
) -> None:
    if (
        not isinstance(daily_ohlcv, torch.Tensor)
        or daily_ohlcv.ndim != 3
        or daily_ohlcv.shape[-1] != 5
        or not daily_ohlcv.is_floating_point()
        or daily_ohlcv.device.type != "cpu"
        or not bool(torch.isfinite(daily_ohlcv).all())
    ):
        raise Top2000M03RV72026RetrospectiveDataError(
            "daily OHLCV must be finite floating CPU [date,action,5]"
        )
    if (
        not isinstance(availability, torch.Tensor)
        or availability.dtype != torch.bool
        or availability.device.type != "cpu"
        or availability.shape != daily_ohlcv.shape[:2]
    ):
        raise Top2000M03RV72026RetrospectiveDataError(
            "availability must be bool CPU [date,action]"
        )
    if (
        len(exchange_dates) != daily_ohlcv.shape[0]
        or len(action_ids) != daily_ohlcv.shape[1]
        or len(set(action_ids)) != len(action_ids)
        or not action_ids
        or action_ids[0] != "CASH"
    ):
        raise Top2000M03RV72026RetrospectiveDataError(
            "daily tensor axes do not match unique CASH-first identities"
        )
    _parse_dates(exchange_dates, name="exchange_dates")


def _selected_retrospective_data(
    *,
    daily_ohlcv: torch.Tensor,
    availability: torch.Tensor,
    exchange_dates: tuple[str, ...],
    action_ids: tuple[str, ...],
    source_evidence: Top2000M03RV72026RetrospectiveSourceEvidence,
    pre2026_cache_sha256: str,
    pre2026_cache_identity: str,
    action_hash: str,
    output_device: str | torch.device,
    cache_file_sha256: str | None = None,
) -> Top2000M03RV72026RetrospectiveData:
    """Build and revalidate the selected 252-state-context chronology."""

    _validate_daily_axes(daily_ohlcv, availability, exchange_dates, action_ids)
    for name, value in (
        ("pre2026_cache_sha256", pre2026_cache_sha256),
        ("pre2026_cache_identity", pre2026_cache_identity),
        ("action_hash", action_hash),
    ):
        _require_digest(name, value)
    parsed = _parse_dates(exchange_dates, name="selected exchange dates")
    first_2026 = next(
        (index for index, value in enumerate(parsed) if value >= TOP2000_M03R_V7_2026_START),
        None,
    )
    if (
        first_2026 != TOP2000_M03R_V7_2026_CONTEXT_STATES
        or any(value >= TOP2000_M03R_V7_2026_START for value in parsed[:first_2026])
        or any(
            not TOP2000_M03R_V7_2026_START
            <= value
            <= TOP2000_M03R_V7_2026_CUTOFF
            for value in parsed[first_2026:]
        )
        or parsed[-1] != TOP2000_M03R_V7_2026_CUTOFF
        or len(parsed) > TOP2000_M03R_V7_2026_MAX_TOTAL_STATES
    ):
        raise Top2000M03RV72026RetrospectiveDataError(
            "selected chronology must contain exactly 252 pre-2026 states and "
            "one complete 2026 suffix through 2026-06-23"
        )

    # Derive and bind every economic array on CPU exactly once.  Moving these
    # immutable results to an inference device is a byte-preserving transfer;
    # recomputing C1 independently on CPU and CUDA would make one source cache
    # acquire device-dependent receipt hashes.
    decision_state_cpu, asset_returns_cpu, masks_cpu = (
        training_data._daily_ohlcv_sequence_tensors(
            daily_ohlcv,
            availability,
            exchange_dates,
            output_device="cpu",
        )
    )
    risk_caps_cpu = torch.zeros_like(masks_cpu, dtype=asset_returns_cpu.dtype)
    risk_caps_cpu[..., training_data.TOP2000_HOLD30_CASH_INDEX] = 1.0
    risk_caps_cpu[..., 1:] = (
        masks_cpu[..., 1:].to(asset_returns_cpu.dtype)
        * TOP2000_M03R_V7_2026_MAX_STOCK_WEIGHT
    )
    risk_gross_cpu = asset_returns_cpu.new_ones((len(exchange_dates), 1))
    benchmark_cpu = training_data._monthly_equal_weight_buy_and_drift(
        asset_returns_cpu[:, 0],
        masks_cpu[:, 0],
        risk_caps_cpu[:, 0],
        risk_gross_cpu[:, 0],
        exchange_dates,
        cost_rate=TOP2000_M03R_V7_2026_COST_RATE,
        cash_index=training_data.TOP2000_HOLD30_CASH_INDEX,
        max_stock_weight=TOP2000_M03R_V7_2026_MAX_STOCK_WEIGHT,
    )
    risky_benchmark = benchmark_cpu.weights[:, 1:]
    if (
        bool((risky_benchmark > TOP2000_M03R_V7_2026_MAX_STOCK_WEIGHT + 5e-6).any())
        or bool((benchmark_cpu.weights[~masks_cpu[:, 0]] > 5e-6).any())
    ):
        raise Top2000M03RV72026RetrospectiveDataError(
            "C1 is not feasible on the selected causal availability axis"
        )

    device = torch.device(output_device)
    decision_state = decision_state_cpu.to(device)
    asset_returns = asset_returns_cpu.to(device)
    masks = masks_cpu.to(device)
    risk_caps = risk_caps_cpu.to(device)
    risk_gross = risk_gross_cpu.to(device)
    benchmark = training_data.Top2000MonthlyEqualWeightBuyAndDriftTrace(
        weights=benchmark_cpu.weights.to(device),
        gross_returns=benchmark_cpu.gross_returns.to(device),
        availability_forced_one_way_turnover=(
            benchmark_cpu.availability_forced_one_way_turnover.to(device)
        ),
        monthly_rebalance_one_way_turnover=(
            benchmark_cpu.monthly_rebalance_one_way_turnover.to(device)
        ),
        risk_forced_one_way_turnover=(
            benchmark_cpu.risk_forced_one_way_turnover.to(device)
        ),
        total_one_way_turnover=benchmark_cpu.total_one_way_turnover.to(device),
        costs=benchmark_cpu.costs.to(device),
        net_returns=benchmark_cpu.net_returns.to(device),
    )

    score_start = TOP2000_M03R_V7_2026_CONTEXT_STATES - 1
    score_stop = len(exchange_dates) - 1
    score_dates = exchange_dates[TOP2000_M03R_V7_2026_CONTEXT_STATES :]
    identity = Top2000M03RV72026RetrospectiveIdentity(
        source_evidence_sha256=source_evidence.receipt_sha256,
        pre2026_cache_sha256=pre2026_cache_sha256,
        pre2026_cache_identity=pre2026_cache_identity,
        search_identity=source_evidence.search_identity,
        action_hash=action_hash,
        action_ids_sha256=_canonical_sha256(list(action_ids)),
        exchange_dates_sha256=_canonical_sha256(list(exchange_dates)),
        context_dates_sha256=_canonical_sha256(
            list(exchange_dates[:TOP2000_M03R_V7_2026_CONTEXT_STATES])
        ),
        score_return_dates_sha256=_canonical_sha256(list(score_dates)),
        daily_ohlcv_sha256=_tensor_sha256(daily_ohlcv),
        availability_sha256=_tensor_sha256(availability),
        score_availability_sha256=_tensor_sha256(
            availability[TOP2000_M03R_V7_2026_CONTEXT_STATES :]
        ),
        asset_returns_sha256=_tensor_sha256(asset_returns_cpu),
        benchmark_weights_sha256=_tensor_sha256(benchmark_cpu.weights),
        benchmark_gross_returns_sha256=_tensor_sha256(benchmark_cpu.gross_returns),
        benchmark_total_turnover_sha256=_tensor_sha256(
            benchmark_cpu.total_one_way_turnover
        ),
        benchmark_net_returns_sha256=_tensor_sha256(benchmark_cpu.net_returns),
        benchmark_trace_sha256=benchmark_cpu.trace_sha256,
        first_context_date=exchange_dates[0],
        last_context_date=exchange_dates[
            TOP2000_M03R_V7_2026_CONTEXT_STATES - 1
        ],
        first_score_return_date=score_dates[0],
        last_score_return_date=score_dates[-1],
        state_rows=len(exchange_dates),
        transition_rows=len(exchange_dates) - 1,
        context_state_rows=TOP2000_M03R_V7_2026_CONTEXT_STATES,
        score_transition_start=score_start,
        score_transition_stop_exclusive=score_stop,
        score_transition_rows=score_stop - score_start,
        action_count=len(action_ids),
    )
    initial_weights = benchmark.weights[0].unsqueeze(0)
    sequence = Hold30Sequence(
        decision_state=decision_state,
        asset_returns=asset_returns,
        decision_available=masks.clone(),
        fill_membership=masks.clone(),
        fill_availability=masks.clone(),
        benchmark_weights=benchmark.weights.unsqueeze(1),
        risk_asset_caps=risk_caps,
        risk_gross_max=risk_gross,
        benchmark_net_returns=benchmark.net_returns.unsqueeze(1),
        initial_ledger=CohortLedger.from_staggered_endowment(
            initial_weights,
            cash_index=training_data.TOP2000_HOLD30_CASH_INDEX,
            youngest_age=0,
            oldest_age=29,
            track_initial_units=False,
        ),
        cost_rate=TOP2000_M03R_V7_2026_COST_RATE,
        initial_equity=asset_returns.new_ones((1,)),
        track_entry_units=torch.ones(
            len(exchange_dates) - 1,
            dtype=torch.bool,
            device=asset_returns.device,
        ),
        axis_id=identity.axis_id,
    )
    return Top2000M03RV72026RetrospectiveData(
        sequence=sequence,
        benchmark=benchmark,
        source_evidence=source_evidence,
        identity=identity,
        exchange_dates=exchange_dates,
        action_ids=action_ids,
        score_return_dates=score_dates,
        cache_file_sha256=cache_file_sha256,
    )


def compose_top2000_m03r_v7_2026_retrospective_data(
    pre2026_cache: training_data.Top2000VerifiedDevelopmentCache,
    *,
    retrospective_daily_ohlcv: torch.Tensor,
    retrospective_availability: torch.Tensor,
    retrospective_exchange_dates: Sequence[str],
    retrospective_action_ids: Sequence[str],
    source_evidence: Top2000M03RV72026RetrospectiveSourceEvidence,
    output_device: str | torch.device = "cpu",
    acknowledgement: str,
) -> Top2000M03RV72026RetrospectiveData:
    """Merge exact cache/raw overlaps and retain one bounded causal chronology."""

    if acknowledgement != TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK:
        raise Top2000M03RV72026RetrospectiveDataError(
            "2026 retrospective composition requires the exact development acknowledgement"
        )
    if not isinstance(pre2026_cache, training_data.Top2000VerifiedDevelopmentCache):
        raise Top2000M03RV72026RetrospectiveDataError(
            "composition requires the one-time verified pre-2026 cache"
        )
    pre2026_cache.validate_unmodified()
    raw_dates = tuple(retrospective_exchange_dates)
    raw_actions = tuple(retrospective_action_ids)
    _validate_daily_axes(
        retrospective_daily_ohlcv,
        retrospective_availability,
        raw_dates,
        raw_actions,
    )
    parsed_raw = _parse_dates(raw_dates, name="retrospective exchange dates")
    if (
        raw_actions != pre2026_cache.action_ids
        or pre2026_cache.search_identity != source_evidence.search_identity
        or parsed_raw[0] >= TOP2000_M03R_V7_2026_START
        or parsed_raw[-1] != TOP2000_M03R_V7_2026_CUTOFF
        or any(value > TOP2000_M03R_V7_2026_CUTOFF for value in parsed_raw)
    ):
        raise Top2000M03RV72026RetrospectiveDataError(
            "raw retrospective actions, source identity, bridge, or cutoff do not match"
        )

    by_date: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        date_value: (
            pre2026_cache.daily_ohlcv[index],
            pre2026_cache.availability[index],
        )
        for index, date_value in enumerate(pre2026_cache.exchange_dates)
    }
    for index, date_value in enumerate(raw_dates):
        incoming = (
            retrospective_daily_ohlcv[index],
            retrospective_availability[index],
        )
        previous = by_date.get(date_value)
        if previous is not None and (
            previous[0].dtype != incoming[0].dtype
            or not torch.equal(previous[0], incoming[0])
            or not torch.equal(previous[1], incoming[1])
        ):
            raise Top2000M03RV72026RetrospectiveDataError(
                f"pre-2026 cache and raw retrospective disagree on overlap {date_value}"
            )
        by_date[date_value] = incoming

    ordered_dates = tuple(sorted(by_date))
    parsed = _parse_dates(ordered_dates, name="combined exchange dates")
    first_2026 = next(
        (index for index, value in enumerate(parsed) if value >= TOP2000_M03R_V7_2026_START),
        None,
    )
    if first_2026 is None or first_2026 < TOP2000_M03R_V7_2026_CONTEXT_STATES:
        raise Top2000M03RV72026RetrospectiveDataError(
            "at least 252 pre-2026 exchange states are required"
        )
    start = first_2026 - TOP2000_M03R_V7_2026_CONTEXT_STATES
    selected_dates = ordered_dates[start:]
    if len(selected_dates) > TOP2000_M03R_V7_2026_MAX_TOTAL_STATES:
        raise Top2000M03RV72026RetrospectiveDataError(
            "252-state context plus the complete 2026 suffix exceeds 378 states"
        )
    daily = torch.stack([by_date[value][0] for value in selected_dates]).contiguous()
    availability = torch.stack(
        [by_date[value][1] for value in selected_dates]
    ).contiguous()
    return _selected_retrospective_data(
        daily_ohlcv=daily,
        availability=availability,
        exchange_dates=selected_dates,
        action_ids=raw_actions,
        source_evidence=source_evidence,
        pre2026_cache_sha256=pre2026_cache.cache_sha256,
        pre2026_cache_identity=pre2026_cache.cache_identity,
        action_hash=pre2026_cache.action_hash,
        output_device=output_device,
    )


def _read_json_object(path: Path, *, name: str) -> tuple[dict[str, Any], str]:
    try:
        payload_bytes = path.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV72026RetrospectiveDataError(
            f"{name} is absent or invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise Top2000M03RV72026RetrospectiveDataError(
            f"{name} must be a JSON object"
        )
    return payload, hashlib.sha256(payload_bytes).hexdigest()


def materialize_top2000_m03r_v7_2026_retrospective_cache(
    dataset_root: str | Path,
    pre2026_cache_path: str | Path,
    output_path: str | Path,
    *,
    expected_pre2026_cache_sha256: str,
    expected_base_dataset_identity: str,
    expected_search_identity: str,
    expected_lockbox_partition_names_hash: str,
    training_completion_receipt_sha256: str,
    evaluation_contract_sha256: str,
    acknowledgement: str,
) -> dict[str, Any]:
    """Aggregate the raw namespace once and publish one immutable small cache."""

    if acknowledgement != TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK:
        raise Top2000M03RV72026RetrospectiveDataError(
            "2026 cache materialization requires the exact development acknowledgement"
        )
    for name, value in (
        ("expected_pre2026_cache_sha256", expected_pre2026_cache_sha256),
        ("expected_base_dataset_identity", expected_base_dataset_identity),
        ("expected_search_identity", expected_search_identity),
        (
            "expected_lockbox_partition_names_hash",
            expected_lockbox_partition_names_hash,
        ),
        ("training_completion_receipt_sha256", training_completion_receipt_sha256),
        ("evaluation_contract_sha256", evaluation_contract_sha256),
    ):
        _require_digest(name, value)
    root = Path(dataset_root)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable 2026 retrospective cache {destination}"
        )

    manifest, manifest_sha256 = _read_json_object(
        root / "manifest.json", name="TOP2000 manifest"
    )
    universe, universe_sha256 = _read_json_object(
        root / "universe.json", name="TOP2000 universe"
    )
    reportability_errors = manifest.get("reportability_errors")
    if (
        manifest.get("dataset_reportable") is not False
        or manifest.get("membership_mode") != "static"
        or manifest.get("universe_selection_date")
        != TOP2000_M03R_V7_2026_UNIVERSE_SELECTION_DATE.isoformat()
        or str(manifest.get("built_at_utc", ""))[:10]
        != TOP2000_M03R_V7_2026_CUTOFF.isoformat()
        or not isinstance(reportability_errors, list)
        or not reportability_errors
    ):
        raise Top2000M03RV72026RetrospectiveDataError(
            "TOP2000 manifest must disclose its static future-selected, "
            "nonreportable 2026 identity"
        )

    # Lazy import keeps the generic statistical package free of workflow-side
    # import cycles while deliberately reusing the one qualified raw-to-daily
    # aggregation path for this compatibility dataset.
    from rl_quant.workflows import top2000_ppo

    plan = top2000_ppo.build_evaluation_plan(
        root,
        bar_seconds=training_data.TOP2000_HOLD30_SOURCE_BAR_SECONDS,
    )
    for name, actual, expected in (
        ("base_dataset_identity", plan.base_dataset_identity, expected_base_dataset_identity),
        ("search_identity", plan.search_identity, expected_search_identity),
        (
            "lockbox_partition_names_hash",
            plan.lockbox_partition_names_hash,
            expected_lockbox_partition_names_hash,
        ),
    ):
        if actual != expected:
            raise Top2000M03RV72026RetrospectiveDataError(
                f"2026 EvaluationPlan {name} does not match frozen pre-2026 evidence"
            )
    if plan.development_only is not True or plan.label != top2000_ppo.DEVELOPMENT_LABEL:
        raise Top2000M03RV72026RetrospectiveDataError(
            "2026 EvaluationPlan must remain development-only"
        )

    pre2026 = training_data.load_verified_top2000_hold30_development_cache(
        pre2026_cache_path,
        expected_cache_sha256=expected_pre2026_cache_sha256,
        acknowledgement=training_data.DEVELOPMENT_ACK,
    )
    actions = tuple(declared_universe_actions(root))
    if actions != pre2026.action_ids or tuple(universe.get("actions", ())) != actions:
        raise Top2000M03RV72026RetrospectiveDataError(
            "raw, universe-manifest, and pre-2026 action identities differ"
        )
    if (
        universe.get("cash_index") != 0
        or universe.get("action_count") != len(actions)
    ):
        raise Top2000M03RV72026RetrospectiveDataError(
            "TOP2000 universe axis metadata is inconsistent"
        )

    market, raw_dates = top2000_ppo.load_market_data(
        root,
        plan.test,
        bar_seconds=plan.bar_seconds,
        device="cpu",
        date_end=TOP2000_M03R_V7_2026_CUTOFF,
    )
    if set(market.features) != {"daily_ohlcv"} or market.batch_size != 1:
        raise Top2000M03RV72026RetrospectiveDataError(
            "raw aggregation must return one bars-only chronology"
        )
    daily = market.features["daily_ohlcv"][0].contiguous()
    available = market.availability[0].contiguous()
    partition_inventory = tuple(asdict(value) for value in plan.test)
    source = Top2000M03RV72026RetrospectiveSourceEvidence(
        base_dataset_identity=plan.base_dataset_identity,
        search_identity=plan.search_identity,
        lockbox_partition_names_hash=plan.lockbox_partition_names_hash,
        test_identity=plan.test_identity,
        test_partition_inventory_sha256=_canonical_sha256(partition_inventory),
        manifest_sha256=manifest_sha256,
        universe_sha256=universe_sha256,
        training_completion_receipt_sha256=training_completion_receipt_sha256,
        evaluation_contract_sha256=evaluation_contract_sha256,
        raw_first_exchange_date=raw_dates[0],
        raw_last_exchange_date=raw_dates[-1],
    )
    built = compose_top2000_m03r_v7_2026_retrospective_data(
        pre2026,
        retrospective_daily_ohlcv=daily,
        retrospective_availability=available,
        retrospective_exchange_dates=raw_dates,
        retrospective_action_ids=actions,
        source_evidence=source,
        output_device="cpu",
        acknowledgement=acknowledgement,
    )
    payload = {
        "schema": TOP2000_M03R_V7_2026_RETROSPECTIVE_CACHE_SCHEMA,
        "source_evidence": asdict(built.source_evidence),
        "identity": built.identity.canonical_payload(),
        "exchange_dates": built.exchange_dates,
        "action_ids": built.action_ids,
        "daily_ohlcv": built.sequence.decision_state[:, 0].to(device="cpu"),
        "availability": built.sequence.decision_available[:, 0].to(device="cpu"),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    cache_sha256 = _file_sha256(destination)
    return {
        "cache_path": str(destination),
        "cache_sha256": cache_sha256,
        "data_receipt_sha256": built.identity.receipt_sha256,
        "source_evidence_sha256": built.source_evidence.receipt_sha256,
        "exchange_date_range": [built.exchange_dates[0], built.exchange_dates[-1]],
        "score_return_date_range": [
            built.score_return_dates[0],
            built.score_return_dates[-1],
        ],
        "state_rows": built.identity.state_rows,
        "score_transition_rows": built.identity.score_transition_rows,
        "action_count": built.identity.action_count,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }


def load_top2000_m03r_v7_2026_retrospective_cache(
    cache_path: str | Path,
    *,
    expected_cache_sha256: str,
    output_device: str | torch.device = "cpu",
    acknowledgement: str,
) -> Top2000M03RV72026RetrospectiveData:
    """Verify once, rebuild every economic array, and return the chronology."""

    if acknowledgement != TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK:
        raise Top2000M03RV72026RetrospectiveDataError(
            "2026 cache load requires the exact development acknowledgement"
        )
    _require_digest("expected_cache_sha256", expected_cache_sha256)
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"2026 retrospective cache is absent: {path}")
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_cache_sha256:
        raise Top2000M03RV72026RetrospectiveDataError(
            "2026 retrospective cache SHA-256 mismatch"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema",
        "source_evidence",
        "identity",
        "exchange_dates",
        "action_ids",
        "daily_ohlcv",
        "availability",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise Top2000M03RV72026RetrospectiveDataError(
            "2026 retrospective cache schema is incomplete or has unknown fields"
        )
    if payload["schema"] != TOP2000_M03R_V7_2026_RETROSPECTIVE_CACHE_SCHEMA:
        raise Top2000M03RV72026RetrospectiveDataError(
            "2026 retrospective cache schema identity drifted"
        )
    try:
        source = Top2000M03RV72026RetrospectiveSourceEvidence(
            **dict(payload["source_evidence"])
        )
        expected_identity = Top2000M03RV72026RetrospectiveIdentity(
            **dict(payload["identity"])
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, Top2000M03RV72026RetrospectiveDataError):
            raise
        raise Top2000M03RV72026RetrospectiveDataError(
            "2026 retrospective cache receipt payload is invalid"
        ) from exc
    rebuilt = _selected_retrospective_data(
        daily_ohlcv=payload["daily_ohlcv"],
        availability=payload["availability"],
        exchange_dates=tuple(payload["exchange_dates"]),
        action_ids=tuple(payload["action_ids"]),
        source_evidence=source,
        pre2026_cache_sha256=expected_identity.pre2026_cache_sha256,
        pre2026_cache_identity=expected_identity.pre2026_cache_identity,
        action_hash=expected_identity.action_hash,
        output_device=output_device,
        cache_file_sha256=actual_sha256,
    )
    if rebuilt.identity.canonical_payload() != expected_identity.canonical_payload():
        raise Top2000M03RV72026RetrospectiveDataError(
            "2026 retrospective cache arrays do not reproduce its identity"
        )
    return rebuilt


__all__ = [
    "TOP2000_M03R_V7_2026_CONTEXT_STATES",
    "TOP2000_M03R_V7_2026_COST_RATE",
    "TOP2000_M03R_V7_2026_CUTOFF",
    "TOP2000_M03R_V7_2026_MAX_STOCK_WEIGHT",
    "TOP2000_M03R_V7_2026_MAX_TOTAL_STATES",
    "TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK",
    "TOP2000_M03R_V7_2026_RETROSPECTIVE_CACHE_SCHEMA",
    "TOP2000_M03R_V7_2026_RETROSPECTIVE_IDENTITY_SCHEMA",
    "TOP2000_M03R_V7_2026_RETROSPECTIVE_SOURCE_SCHEMA",
    "TOP2000_M03R_V7_2026_START",
    "TOP2000_M03R_V7_2026_UNIVERSE_SELECTION_DATE",
    "Top2000M03RV72026RetrospectiveData",
    "Top2000M03RV72026RetrospectiveDataError",
    "Top2000M03RV72026RetrospectiveIdentity",
    "Top2000M03RV72026RetrospectiveSourceEvidence",
    "compose_top2000_m03r_v7_2026_retrospective_data",
    "load_top2000_m03r_v7_2026_retrospective_cache",
    "materialize_top2000_m03r_v7_2026_retrospective_cache",
]
