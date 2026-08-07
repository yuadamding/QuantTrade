"""Bounded TOP2000-cache adapter for Hold-30 development qualification.

This module deliberately does *not* turn the legacy future-selected TOP2000
cache into a point-in-time Active-300 or promotion dataset.  It only adapts an
explicit, content-addressed, pre-2026 cache slice into the tensor contract used
by :class:`rl_quant.training.hold30_runtime.Hold30Sequence`, with a transparent
monthly equal-weight rebalance-and-drift development benchmark.

The cache stores daily OHLCV rows aggregated from the approved five-minute-bar
source.  The adapter binds that distinction in its receipt: it does not claim
that the resulting decision tensor contains intraday five-minute tokens.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from rl_quant.datasets.raw_window import BAR_FIELDS
from rl_quant.envs.hold30 import CohortLedger, reconcile_cash_simplex_roundoff
from rl_quant.training.hold30_runtime import Hold30Sequence

TOP2000_HOLD30_LEGACY_DEVELOPMENT_ADAPTER_SCHEMA = (
    "rl-quant.top2000-hold30-development-cache-adapter-v1"
)
TOP2000_HOLD30_DEVELOPMENT_ADAPTER_SCHEMA = (
    "rl-quant.top2000-hold30-development-cache-adapter-v2"
)
TOP2000_HOLD30_SOURCE_BAR_SECONDS = 300
TOP2000_HOLD30_SOURCE_BAR_IDENTITY = "5-minute-bars"
TOP2000_HOLD30_OBSERVATION_REPRESENTATION = "daily-ohlcv-aggregated-from-5-minute-bars"
TOP2000_HOLD30_UNIVERSE_IDENTITY = "future-selected-top2000-development-only"
TOP2000_HOLD30_LEGACY_BENCHMARK_ID = (
    "C1-monthly-point-in-time-equal-weight-rebalance-and-drift-development-v1"
)
TOP2000_HOLD30_BENCHMARK_ID = (
    "C1-monthly-point-in-time-equal-weight-rebalance-drift-and-risk-repair-"
    "development-v2"
)
TOP2000_HOLD30_MAX_STATE_ROWS = 378
TOP2000_HOLD30_CASH_INDEX = 0
TOP2000_HOLD30_TRAINING_COST_RATE = 0.002
TOP2000_HOLD30_MAX_STOCK_WEIGHT = 0.01
TOP2000_HOLD30_DEFAULT_MAX_STOCK_WEIGHT = 1.0
TOP2000_HOLD30_BENCHMARK_INITIALIZATION_RULE = (
    "equal-weight-risky-endowment-at-first-state-then-hard-risk-repair-"
    "to-cash-no-startup-cost"
)
TOP2000_HOLD30_BENCHMARK_RISK_REPAIR_RULE = (
    "after-drift-availability-and-any-monthly-target;"
    "clip-risky-names-to-min-fill-cap-and-bound-max-stock-weight;"
    "scale-to-fill-gross-ceiling;release-to-cash;"
    "charge-post-startup-one-way-turnover"
)
DEVELOPMENT_ACK = "I acknowledge TOP2000 results are development-only"

_TOP2000_CACHE_SCHEMA_VERSION = 1
_TOP2000_FEATURE_CACHE_VERSION = 1
_TOP2000_LOCKBOX_START = dt.date(2026, 1, 1)


class Top2000Hold30DevelopmentError(ValueError):
    """The cache cannot support the bounded development-only adapter."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000Hold30DevelopmentError(
            f"{name} must be a lowercase SHA-256 digest"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_axis_sha256(values: tuple[str, ...]) -> str:
    """Reproduce the legacy cache builder's newline-terminated axis digest."""

    encoded = (
        json.dumps(
            list(values),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_verified_daily_cache_payload(
    cache_path: str | Path,
    *,
    expected_cache_sha256: str,
) -> dict[str, Any]:
    """Own the minimal immutable-cache contract needed by this adapter.

    The legacy cache was authored by a workflow module, but consuming it is a
    training-layer responsibility.  Keeping the reader here avoids coupling a
    reusable adapter to a CLI/workflow boundary while preserving the original
    on-disk schema and content checks.
    """

    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Required immutable daily-bars cache is absent: {path}.")
    actual_hash = _file_sha256(path)
    if actual_hash != expected_cache_sha256:
        raise Top2000Hold30DevelopmentError("Daily-bars cache SHA-256 mismatch")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "feature_cache_version",
        "development_only",
        "bars_only",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
        "cache_identity",
        "actions",
        "action_hash",
        "exchange_dates",
        "date_hash",
        "daily_ohlcv",
        "availability",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise Top2000Hold30DevelopmentError("Daily-bars cache schema is incomplete")
    if (
        payload["schema_version"] != _TOP2000_CACHE_SCHEMA_VERSION
        or payload["feature_cache_version"] != _TOP2000_FEATURE_CACHE_VERSION
        or payload["development_only"] is not True
        or payload["bars_only"] is not True
    ):
        raise Top2000Hold30DevelopmentError(
            "Daily-bars cache schema/labels are incompatible"
        )

    dates = tuple(payload["exchange_dates"])
    actions = tuple(payload["actions"])
    try:
        parsed_dates = tuple(dt.date.fromisoformat(value) for value in dates)
    except (TypeError, ValueError) as exc:
        raise Top2000Hold30DevelopmentError(
            "Daily-bars cache exchange dates are invalid"
        ) from exc
    if any(value >= _TOP2000_LOCKBOX_START for value in parsed_dates):
        raise Top2000Hold30DevelopmentError(
            "Development cache illegally contains a 2026-or-later exchange date"
        )
    if _cache_axis_sha256(dates) != payload["date_hash"]:
        raise Top2000Hold30DevelopmentError(
            "Daily-bars cache date identity mismatch"
        )
    if _cache_axis_sha256(actions) != payload["action_hash"]:
        raise Top2000Hold30DevelopmentError(
            "Daily-bars cache action identity mismatch"
        )

    daily = payload["daily_ohlcv"]
    availability = payload["availability"]
    if (
        not isinstance(daily, torch.Tensor)
        or daily.shape != (len(dates), len(actions), len(BAR_FIELDS))
    ):
        raise Top2000Hold30DevelopmentError(
            "Daily-bars cache tensor shape does not match date/action identities"
        )
    if (
        not isinstance(availability, torch.Tensor)
        or availability.shape != daily.shape[:2]
        or availability.dtype != torch.bool
    ):
        raise Top2000Hold30DevelopmentError(
            "Daily-bars cache availability schema is invalid"
        )
    payload["daily_ohlcv"] = daily
    payload["availability"] = availability
    payload["exchange_dates"] = dates
    payload["actions"] = actions
    payload["cache_sha256"] = actual_hash
    return payload


def _daily_ohlcv_sequence_tensors(
    daily_ohlcv: torch.Tensor,
    availability: torch.Tensor,
    exchange_dates: tuple[str, ...],
    *,
    output_device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build position-major Hold30 tensors from causal daily OHLCV rows."""

    if daily_ohlcv.ndim != 3 or daily_ohlcv.shape[-1] != len(BAR_FIELDS):
        raise Top2000Hold30DevelopmentError(
            "daily_ohlcv must have shape [date, action, 5]"
        )
    dates, actions, _features = daily_ohlcv.shape
    if dates < 2 or actions < 2:
        raise Top2000Hold30DevelopmentError(
            "Need at least two exchange dates and CASH plus one risky action"
        )
    if availability.shape != (dates, actions) or availability.dtype != torch.bool:
        raise Top2000Hold30DevelopmentError(
            "availability must be bool [date, action]"
        )
    try:
        parsed = tuple(dt.date.fromisoformat(value) for value in exchange_dates)
    except ValueError as exc:
        raise Top2000Hold30DevelopmentError("exchange_dates are invalid") from exc
    if len(parsed) != dates or any(left >= right for left, right in pairwise(parsed)):
        raise Top2000Hold30DevelopmentError(
            "exchange_dates must be unique and strictly increasing"
        )
    if not bool(availability[:, TOP2000_HOLD30_CASH_INDEX].all()):
        raise Top2000Hold30DevelopmentError(
            "Synthetic CASH must be available on every date"
        )
    if not daily_ohlcv.is_floating_point() or not bool(
        torch.isfinite(daily_ohlcv).all()
    ):
        raise Top2000Hold30DevelopmentError(
            "daily_ohlcv must be finite floating point"
        )
    risky_available = availability[:, 1:]
    risky_bars = daily_ohlcv[:, 1:]
    if bool((risky_bars[..., :4][risky_available] <= 0).any()):
        raise Top2000Hold30DevelopmentError(
            "Available risky OHLC prices must be positive"
        )
    if bool((risky_bars[..., 4][risky_available] < 0).any()):
        raise Top2000Hold30DevelopmentError(
            "Available risky volume must be nonnegative"
        )

    close = daily_ohlcv[..., 3]
    pair_valid = availability[:-1] & availability[1:]
    safe_previous = torch.where(pair_valid, close[:-1], torch.ones_like(close[:-1]))
    simple_returns = torch.where(
        pair_valid,
        close[1:] / safe_previous - 1.0,
        torch.zeros_like(close[:-1]),
    )
    simple_returns[:, TOP2000_HOLD30_CASH_INDEX] = 0.0
    if not bool(torch.isfinite(simple_returns).all()) or bool(
        (simple_returns <= -1.0).any()
    ):
        raise Top2000Hold30DevelopmentError(
            "Daily simple returns must be finite and greater than -1"
        )

    device = torch.device(output_device)
    # Hold30Sequence is position-major with a singleton chronology batch.
    return (
        daily_ohlcv.to(device).unsqueeze(1).contiguous(),
        simple_returns.to(device).unsqueeze(1).contiguous(),
        availability.to(device).unsqueeze(1).contiguous(),
    )


@dataclass(frozen=True, slots=True)
class Top2000VerifiedDevelopmentCache:
    """One SHA-verified CPU cache that can serve many bounded slices.

    Full-file SHA verification and ``torch.load`` happen only in
    :func:`load_verified_top2000_hold30_development_cache`.  Slice builders use
    this resident object and perform only constant-time tensor mutation checks
    before indexing it.
    """

    daily_ohlcv: torch.Tensor
    availability: torch.Tensor
    exchange_dates: tuple[str, ...]
    action_ids: tuple[str, ...]
    cache_sha256: str
    cache_identity: str
    search_identity: str
    action_hash: str
    bar_seconds: int
    acknowledgement: str
    development_only: bool
    bars_only: bool
    _daily_ohlcv_version: int = field(init=False, repr=False)
    _availability_version: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "cache_sha256",
            "cache_identity",
            "search_identity",
            "action_hash",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.acknowledgement != DEVELOPMENT_ACK
            or not self.development_only
            or not self.bars_only
            or self.bar_seconds != TOP2000_HOLD30_SOURCE_BAR_SECONDS
        ):
            raise Top2000Hold30DevelopmentError(
                "verified cache must retain the acknowledged 300-second "
                "(five-minute) development identity"
            )
        if (
            not isinstance(self.daily_ohlcv, torch.Tensor)
            or self.daily_ohlcv.ndim != 3
            or self.daily_ohlcv.shape[-1] != 5
            or self.daily_ohlcv.device.type != "cpu"
            or not self.daily_ohlcv.is_floating_point()
            or not bool(torch.isfinite(self.daily_ohlcv).all())
        ):
            raise Top2000Hold30DevelopmentError(
                "verified daily_ohlcv must be finite floating CPU [date,action,5]"
            )
        if (
            not isinstance(self.availability, torch.Tensor)
            or self.availability.dtype != torch.bool
            or self.availability.device.type != "cpu"
            or self.availability.shape != self.daily_ohlcv.shape[:2]
        ):
            raise Top2000Hold30DevelopmentError(
                "verified availability must be bool CPU [date,action]"
            )
        dates, actions = self.daily_ohlcv.shape[:2]
        if (
            len(self.exchange_dates) != dates
            or len(self.action_ids) != actions
            or actions < 2
            or self.action_ids[TOP2000_HOLD30_CASH_INDEX] != "CASH"
        ):
            raise Top2000Hold30DevelopmentError(
                "verified date/action axes do not match cache tensors"
            )
        if not bool(self.availability[:, TOP2000_HOLD30_CASH_INDEX].all()):
            raise Top2000Hold30DevelopmentError("CASH must always be available")
        object.__setattr__(self, "_daily_ohlcv_version", self.daily_ohlcv._version)
        object.__setattr__(self, "_availability_version", self.availability._version)

    def validate_unmodified(self) -> None:
        """Fail if resident tensors changed after their one-time verification."""

        if (
            self.daily_ohlcv._version != self._daily_ohlcv_version
            or self.availability._version != self._availability_version
        ):
            raise Top2000Hold30DevelopmentError(
                "verified cache tensors changed after load; reload the immutable cache"
            )


def load_verified_top2000_hold30_development_cache(
    cache_path: str | Path,
    *,
    expected_cache_sha256: str,
    acknowledgement: str,
) -> Top2000VerifiedDevelopmentCache:
    """SHA-check and deserialize the real cache exactly once on CPU."""

    if acknowledgement != DEVELOPMENT_ACK:
        raise Top2000Hold30DevelopmentError(
            f"TOP2000 adaptation requires acknowledgement {DEVELOPMENT_ACK!r}"
        )
    _require_digest("expected_cache_sha256", expected_cache_sha256)
    payload = _load_verified_daily_cache_payload(
        cache_path,
        expected_cache_sha256=expected_cache_sha256,
    )
    bar_seconds = payload.get("bar_seconds")
    development_only = payload.get("development_only")
    bars_only = payload.get("bars_only")
    if not isinstance(bar_seconds, int):
        raise Top2000Hold30DevelopmentError(
            "cache must explicitly bind an integer bar_seconds identity"
        )
    if not isinstance(development_only, bool) or not isinstance(bars_only, bool):
        raise Top2000Hold30DevelopmentError(
            "cache must explicitly bind boolean development and bars-only labels"
        )
    return Top2000VerifiedDevelopmentCache(
        daily_ohlcv=payload["daily_ohlcv"],
        availability=payload["availability"],
        exchange_dates=tuple(payload["exchange_dates"]),
        action_ids=tuple(payload["actions"]),
        cache_sha256=payload["cache_sha256"],
        cache_identity=payload["cache_identity"],
        search_identity=payload["search_identity"],
        action_hash=payload["action_hash"],
        bar_seconds=bar_seconds,
        acknowledgement=acknowledgement,
        development_only=development_only,
        bars_only=bars_only,
    )


@dataclass(frozen=True, slots=True)
class Top2000MonthlyEqualWeightBuyAndDriftTrace:
    """Transparent, fill-feasible C1 benchmark rebalanced monthly.

    The first book is an uncharged equal-weight endowment over risky names
    available at the first state, repaired into the policy's hard cap/gross
    envelope without a startup charge.  Thereafter it earns the next
    close-to-close return, performs availability- and risk-forced sales into
    CASH, and re-establishes equal weight at the first state of each new
    calendar month over names that were visible at the preceding decision and
    remain tradable at the fill.  The final fill book is then repaired through
    the same deterministic risk projection as the policy.  Every post-startup
    forced or rebalance trade pays the same one-way linear cost as the policy.
    """

    weights: torch.Tensor
    gross_returns: torch.Tensor
    availability_forced_one_way_turnover: torch.Tensor
    monthly_rebalance_one_way_turnover: torch.Tensor
    risk_forced_one_way_turnover: torch.Tensor
    total_one_way_turnover: torch.Tensor
    costs: torch.Tensor
    net_returns: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.weights, torch.Tensor) or self.weights.ndim != 2:
            raise Top2000Hold30DevelopmentError(
                "benchmark weights must have shape [position, asset]"
            )
        positions, _assets = self.weights.shape
        expected = (positions - 1,)
        for name in (
            "gross_returns",
            "availability_forced_one_way_turnover",
            "monthly_rebalance_one_way_turnover",
            "risk_forced_one_way_turnover",
            "total_one_way_turnover",
            "costs",
            "net_returns",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected
                or value.dtype != self.weights.dtype
                or value.device != self.weights.device
                or not bool(torch.isfinite(value).all())
            ):
                raise Top2000Hold30DevelopmentError(
                    f"benchmark {name} must be finite and align with transitions"
                )
        if bool((self.weights < 0).any()) or not bool(
            torch.allclose(
                self.weights.sum(-1),
                torch.ones(
                    positions, dtype=self.weights.dtype, device=self.weights.device
                ),
                atol=1e-6,
                rtol=1e-6,
            )
        ):
            raise Top2000Hold30DevelopmentError(
                "benchmark weights must be nonnegative simplexes"
            )
        if (
            bool((self.availability_forced_one_way_turnover < 0).any())
            or bool((self.monthly_rebalance_one_way_turnover < 0).any())
            or bool((self.risk_forced_one_way_turnover < 0).any())
            or bool((self.total_one_way_turnover < 0).any())
            or bool((self.costs < 0).any())
        ):
            raise Top2000Hold30DevelopmentError(
                "benchmark turnover and cost cannot be negative"
            )
        if not bool(
            torch.allclose(
                self.total_one_way_turnover,
                self.availability_forced_one_way_turnover
                + self.monthly_rebalance_one_way_turnover
                + self.risk_forced_one_way_turnover,
            )
        ):
            raise Top2000Hold30DevelopmentError(
                "benchmark total turnover must reconcile by cause"
            )
        if not bool(torch.allclose(self.net_returns, self.gross_returns - self.costs)):
            raise Top2000Hold30DevelopmentError(
                "benchmark net return must equal gross return minus trading cost"
            )

    @property
    def trace_sha256(self) -> str:
        """Bind every economic and cause-typed benchmark array."""

        return _canonical_sha256(
            {
                name: _tensor_sha256(getattr(self, name))
                for name in (
                    "weights",
                    "gross_returns",
                    "availability_forced_one_way_turnover",
                    "monthly_rebalance_one_way_turnover",
                    "risk_forced_one_way_turnover",
                    "total_one_way_turnover",
                    "costs",
                    "net_returns",
                )
            }
        )


@dataclass(frozen=True, slots=True)
class Top2000Hold30DevelopmentIdentity:
    """Content identity and explicit scientific limitations for one slice."""

    cache_sha256: str
    cache_identity: str
    search_identity: str
    action_hash: str
    date_slice_sha256: str
    benchmark_weights_sha256: str
    benchmark_net_returns_sha256: str
    benchmark_trace_sha256: str
    state_start_index: int
    state_stop_index_exclusive: int
    first_exchange_date: str
    last_exchange_date: str
    state_rows: int
    transition_rows: int
    action_count: int
    one_way_cost_rate: float
    max_stock_weight: float
    source_bar_seconds: int = TOP2000_HOLD30_SOURCE_BAR_SECONDS
    source_bar_identity: str = TOP2000_HOLD30_SOURCE_BAR_IDENTITY
    observation_representation: str = TOP2000_HOLD30_OBSERVATION_REPRESENTATION
    universe_identity: str = TOP2000_HOLD30_UNIVERSE_IDENTITY
    benchmark_id: str = TOP2000_HOLD30_BENCHMARK_ID
    benchmark_initialization: str = TOP2000_HOLD30_BENCHMARK_INITIALIZATION_RULE
    policy_initial_ledger_rule: str = (
        "common-c1-endowment-staggered-untracked-ages-0-through-29"
    )
    benchmark_rebalance_rule: str = (
        "first-state-of-calendar-month-equal-weight-over-prior-decision-visible-and-"
        "fill-available-risky;"
        "availability-forced-sales-to-cash"
    )
    benchmark_risk_repair_rule: str = TOP2000_HOLD30_BENCHMARK_RISK_REPAIR_RULE
    availability_semantics: str = (
        "legacy-cache-availability-used-for-membership-and-tradability"
    )
    development_only: bool = True
    future_selected_universe: bool = True
    outer_evaluation_authorized: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_HOLD30_DEVELOPMENT_ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "cache_sha256",
            "cache_identity",
            "search_identity",
            "action_hash",
            "date_slice_sha256",
            "benchmark_weights_sha256",
            "benchmark_net_returns_sha256",
            "benchmark_trace_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.schema != TOP2000_HOLD30_DEVELOPMENT_ADAPTER_SCHEMA
            or self.source_bar_seconds != TOP2000_HOLD30_SOURCE_BAR_SECONDS
            or self.source_bar_identity != TOP2000_HOLD30_SOURCE_BAR_IDENTITY
            or self.observation_representation
            != TOP2000_HOLD30_OBSERVATION_REPRESENTATION
            or self.universe_identity != TOP2000_HOLD30_UNIVERSE_IDENTITY
            or self.benchmark_id != TOP2000_HOLD30_BENCHMARK_ID
            or self.benchmark_initialization
            != TOP2000_HOLD30_BENCHMARK_INITIALIZATION_RULE
            or self.benchmark_risk_repair_rule
            != TOP2000_HOLD30_BENCHMARK_RISK_REPAIR_RULE
        ):
            raise Top2000Hold30DevelopmentError(
                "TOP2000 development adapter identity drifted"
            )
        if (
            not self.development_only
            or not self.future_selected_universe
            or self.outer_evaluation_authorized
            or self.promotion_eligible
        ):
            raise Top2000Hold30DevelopmentError(
                "TOP2000 adapter must remain development-only and nonpromotable"
            )
        if (
            isinstance(self.state_start_index, bool)
            or not isinstance(self.state_start_index, int)
            or self.state_start_index < 0
            or isinstance(self.state_stop_index_exclusive, bool)
            or not isinstance(self.state_stop_index_exclusive, int)
            or self.state_stop_index_exclusive <= self.state_start_index
            or self.state_rows
            != self.state_stop_index_exclusive - self.state_start_index
            or self.transition_rows != self.state_rows - 1
            or not 2 <= self.state_rows <= TOP2000_HOLD30_MAX_STATE_ROWS
            or self.action_count < 2
            or self.one_way_cost_rate != TOP2000_HOLD30_TRAINING_COST_RATE
            or isinstance(self.max_stock_weight, bool)
            or not isinstance(self.max_stock_weight, (int, float))
            or not 0.0 < float(self.max_stock_weight) <= 1.0
        ):
            raise Top2000Hold30DevelopmentError(
                "TOP2000 slice geometry or cost identity is invalid"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())

    @property
    def axis_id(self) -> str:
        return f"top2000-hold30-development:{self.receipt_sha256}"


@dataclass(frozen=True, slots=True)
class Top2000Hold30DevelopmentSequence:
    """A runtime-compatible sequence plus its nonpromotable cache receipt."""

    sequence: Hold30Sequence
    benchmark: Top2000MonthlyEqualWeightBuyAndDriftTrace
    identity: Top2000Hold30DevelopmentIdentity
    exchange_dates: tuple[str, ...]
    action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.sequence.axis_id != self.identity.axis_id:
            raise Top2000Hold30DevelopmentError(
                "sequence axis does not match adapter receipt"
            )
        if self.sequence.n_positions != self.identity.state_rows:
            raise Top2000Hold30DevelopmentError(
                "sequence length does not match adapter receipt"
            )
        if self.sequence.num_assets != self.identity.action_count:
            raise Top2000Hold30DevelopmentError(
                "asset count does not match adapter receipt"
            )
        if self.sequence.batch_size != 1:
            raise Top2000Hold30DevelopmentError("adapter must emit one chronology")
        if len(self.exchange_dates) != self.identity.state_rows:
            raise Top2000Hold30DevelopmentError("exchange-date slice length drifted")
        if (
            len(self.action_ids) != self.identity.action_count
            or self.action_ids[0] != "CASH"
        ):
            raise Top2000Hold30DevelopmentError("action axis must start with CASH")


def _equal_weight_target(
    availability: torch.Tensor,
    *,
    cash_index: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    target = torch.zeros(
        availability.shape,
        dtype=dtype,
        device=availability.device,
    )
    risky = availability.clone()
    risky[cash_index] = False
    count = int(risky.sum().item())
    if count:
        target[risky] = 1.0 / count
    else:
        target[cash_index] = 1.0
    return target


def _benchmark_risk_project(
    weights: torch.Tensor,
    asset_caps: torch.Tensor,
    gross_max: torch.Tensor,
    *,
    cash_index: int,
    max_stock_weight: float,
) -> torch.Tensor:
    """Apply the caller-bound cap/gross envelope without hidden globals."""

    risky = torch.ones_like(weights, dtype=torch.bool)
    risky[:, cash_index] = False
    cap = torch.where(
        risky,
        torch.minimum(
            asset_caps.clamp_min(0.0),
            weights.new_tensor(max_stock_weight),
        ),
        torch.zeros_like(asset_caps),
    )
    held = torch.where(risky, weights.clamp_min(0.0), torch.zeros_like(weights))
    held = torch.minimum(held, cap)
    hard_gross = torch.minimum(
        torch.ones_like(gross_max),
        torch.minimum(gross_max.clamp_min(0.0), cap.sum(-1)),
    )
    gross = held.sum(-1)
    scale = torch.where(
        gross > hard_gross,
        hard_gross / gross.clamp_min(1.0e-18),
        torch.ones_like(gross),
    )
    held = held * scale.unsqueeze(-1)
    target = held.clone()
    target[:, cash_index] = 1.0 - held.sum(-1)
    return reconcile_cash_simplex_roundoff(
        target,
        cash_index=cash_index,
        risky_gross_limit=hard_gross,
    )


def _monthly_equal_weight_buy_and_drift(
    asset_returns: torch.Tensor,
    availability: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    exchange_dates: tuple[str, ...],
    *,
    cost_rate: float,
    cash_index: int,
    max_stock_weight: float = TOP2000_HOLD30_DEFAULT_MAX_STOCK_WEIGHT,
) -> Top2000MonthlyEqualWeightBuyAndDriftTrace:
    transitions, assets = asset_returns.shape
    if availability.shape != (transitions + 1, assets):
        raise Top2000Hold30DevelopmentError(
            "benchmark availability must have one row beyond returns"
        )
    if (
        risk_asset_caps.shape != (transitions + 1, assets)
        or risk_asset_caps.dtype != asset_returns.dtype
        or risk_asset_caps.device != asset_returns.device
        or not bool(torch.isfinite(risk_asset_caps).all())
        or bool((risk_asset_caps < 0).any())
    ):
        raise Top2000Hold30DevelopmentError(
            "benchmark risk caps must be finite, nonnegative, and align with states"
        )
    if (
        risk_gross_max.shape != (transitions + 1,)
        or risk_gross_max.dtype != asset_returns.dtype
        or risk_gross_max.device != asset_returns.device
        or not bool(torch.isfinite(risk_gross_max).all())
        or bool(((risk_gross_max < 0) | (risk_gross_max > 1)).any())
    ):
        raise Top2000Hold30DevelopmentError(
            "benchmark gross ceilings must be finite [state] values in [0,1]"
        )
    if len(exchange_dates) != transitions + 1:
        raise Top2000Hold30DevelopmentError(
            "benchmark exchange dates must align with states"
        )
    if (
        isinstance(max_stock_weight, bool)
        or not isinstance(max_stock_weight, (int, float))
        or not 0.0 < float(max_stock_weight) <= 1.0
    ):
        raise Top2000Hold30DevelopmentError(
            "benchmark max_stock_weight must be a finite fraction in (0,1]"
        )
    max_stock_weight = float(max_stock_weight)
    parsed_dates = tuple(dt.date.fromisoformat(value) for value in exchange_dates)
    weights = asset_returns.new_zeros((transitions + 1, assets))
    initial_target = _equal_weight_target(
        availability[0], cash_index=cash_index, dtype=asset_returns.dtype
    )
    weights[0] = _benchmark_risk_project(
        initial_target.unsqueeze(0),
        risk_asset_caps[0].unsqueeze(0),
        risk_gross_max[0].reshape(1),
        cash_index=cash_index,
        max_stock_weight=max_stock_weight,
    ).squeeze(0)
    if bool((weights[0][~availability[0]] != 0).any()):
        raise Top2000Hold30DevelopmentError(
            "risk-repaired benchmark endowment violated initial availability"
        )

    gross = asset_returns.new_zeros(transitions)
    availability_turnover = asset_returns.new_zeros(transitions)
    rebalance_turnover = asset_returns.new_zeros(transitions)
    risk_turnover = asset_returns.new_zeros(transitions)
    costs = asset_returns.new_zeros(transitions)
    for index in range(transitions):
        gross[index] = (weights[index] * asset_returns[index]).sum()
        growth = 1.0 + gross[index]
        if not bool(torch.isfinite(growth)) or float(growth) <= 0.0:
            raise Top2000Hold30DevelopmentError(
                "benchmark growth must remain finite and positive"
            )
        drifted = weights[index] * (1.0 + asset_returns[index]) / growth
        repaired = drifted.clone()
        unavailable = ~availability[index + 1]
        unavailable[cash_index] = False
        released = repaired.masked_select(unavailable).sum()
        repaired[unavailable] = 0.0
        repaired[cash_index] += released
        availability_turnover[index] = 0.5 * (repaired - drifted).abs().sum()
        if (
            parsed_dates[index].year,
            parsed_dates[index].month,
        ) != (
            parsed_dates[index + 1].year,
            parsed_dates[index + 1].month,
        ):
            # The benchmark order is decided at ``index`` and fills at
            # ``index + 1``, exactly like a policy order.  A name that first
            # appears at the fill was not decision-visible and cannot enter
            # either book until a later decision.  Intersecting the two masks
            # is execution feasibility, not future-information leakage.
            causal_fill_availability = availability[index] & availability[index + 1]
            target = _equal_weight_target(
                causal_fill_availability,
                cash_index=cash_index,
                dtype=asset_returns.dtype,
            )
            rebalance_turnover[index] = 0.5 * (target - repaired).abs().sum()
            repaired = target
        risk_repaired = _benchmark_risk_project(
            repaired.unsqueeze(0),
            risk_asset_caps[index + 1].unsqueeze(0),
            risk_gross_max[index + 1].reshape(1),
            cash_index=cash_index,
            max_stock_weight=max_stock_weight,
        ).squeeze(0)
        risk_turnover[index] = 0.5 * (risk_repaired - repaired).abs().sum()
        repaired = risk_repaired
        costs[index] = cost_rate * (
            availability_turnover[index]
            + rebalance_turnover[index]
            + risk_turnover[index]
        )
        weights[index + 1] = repaired
    net = gross - costs
    return Top2000MonthlyEqualWeightBuyAndDriftTrace(
        weights=weights,
        gross_returns=gross,
        availability_forced_one_way_turnover=availability_turnover,
        monthly_rebalance_one_way_turnover=rebalance_turnover,
        risk_forced_one_way_turnover=risk_turnover,
        total_one_way_turnover=(
            availability_turnover + rebalance_turnover + risk_turnover
        ),
        costs=costs,
        net_returns=net,
    )


def build_top2000_hold30_development_sequence_from_loaded_cache(
    cache: Top2000VerifiedDevelopmentCache,
    *,
    state_start_index: int,
    state_stop_index_exclusive: int,
    max_state_rows: int = TOP2000_HOLD30_MAX_STATE_ROWS,
    max_stock_weight: float = TOP2000_HOLD30_DEFAULT_MAX_STOCK_WEIGHT,
    output_device: str | torch.device = "cpu",
) -> Top2000Hold30DevelopmentSequence:
    """Build one causal sequence without rereading or rehashing the cache file."""

    if not isinstance(cache, Top2000VerifiedDevelopmentCache):
        raise Top2000Hold30DevelopmentError(
            "slice builder requires a one-time verified TOP2000 cache"
        )
    cache.validate_unmodified()
    if (
        isinstance(max_state_rows, bool)
        or not isinstance(max_state_rows, int)
        or not 2 <= max_state_rows <= TOP2000_HOLD30_MAX_STATE_ROWS
    ):
        raise Top2000Hold30DevelopmentError(
            f"max_state_rows must lie in [2,{TOP2000_HOLD30_MAX_STATE_ROWS}]"
        )
    if (
        isinstance(max_stock_weight, bool)
        or not isinstance(max_stock_weight, (int, float))
        or not 0.0 < float(max_stock_weight) <= 1.0
    ):
        raise Top2000Hold30DevelopmentError(
            "max_stock_weight must be a finite fraction in (0,1]"
        )
    max_stock_weight = float(max_stock_weight)
    if (
        isinstance(state_start_index, bool)
        or not isinstance(state_start_index, int)
        or state_start_index < 0
        or isinstance(state_stop_index_exclusive, bool)
        or not isinstance(state_stop_index_exclusive, int)
        or state_stop_index_exclusive <= state_start_index
        or state_stop_index_exclusive - state_start_index > max_state_rows
    ):
        raise Top2000Hold30DevelopmentError(
            "requested state slice must be positive, explicit, and bounded"
        )
    dates = cache.exchange_dates
    actions = cache.action_ids
    if state_stop_index_exclusive > len(dates):
        raise Top2000Hold30DevelopmentError(
            "requested state slice extends beyond the verified cache"
        )

    selected_dates = dates[state_start_index:state_stop_index_exclusive]
    daily = cache.daily_ohlcv[state_start_index:state_stop_index_exclusive]
    available = cache.availability[state_start_index:state_stop_index_exclusive]
    decision_state, asset_returns, masks = _daily_ohlcv_sequence_tensors(
        daily,
        available,
        selected_dates,
        output_device=output_device,
    )
    risk_caps = torch.zeros_like(masks, dtype=asset_returns.dtype)
    risk_caps[..., TOP2000_HOLD30_CASH_INDEX] = 1.0
    risk_caps[..., 1:] = (
        masks[..., 1:].to(asset_returns.dtype) * max_stock_weight
    )
    risk_gross = asset_returns.new_ones((len(selected_dates), 1))
    benchmark = _monthly_equal_weight_buy_and_drift(
        asset_returns[:, 0],
        masks[:, 0],
        risk_caps[:, 0],
        risk_gross[:, 0],
        selected_dates,
        cost_rate=TOP2000_HOLD30_TRAINING_COST_RATE,
        cash_index=TOP2000_HOLD30_CASH_INDEX,
        max_stock_weight=max_stock_weight,
    )

    # Fail before any model or GPU work if the benchmark cannot serve as the
    # exact factor-neutral execution anchor under the policy's causal fill
    # constraints.  Position zero has no preceding order; every later row must
    # additionally have been visible at the prior decision.
    causal_fill_mask = masks[:, 0].clone()
    causal_fill_mask[1:] &= masks[:-1, 0]
    causal_fill_mask[:, TOP2000_HOLD30_CASH_INDEX] = True
    risky = torch.ones_like(causal_fill_mask, dtype=torch.bool)
    risky[:, TOP2000_HOLD30_CASH_INDEX] = False
    effective_caps = torch.where(
        risky & causal_fill_mask,
        torch.minimum(
            risk_caps[:, 0],
            risk_caps.new_tensor(max_stock_weight),
        ),
        torch.zeros_like(risk_caps[:, 0]),
    )
    effective_caps[:, TOP2000_HOLD30_CASH_INDEX] = 1.0
    benchmark_gross = torch.where(
        risky,
        benchmark.weights,
        torch.zeros_like(benchmark.weights),
    ).sum(-1)
    feasibility_tolerance = 5.0e-6
    infeasible = (
        (benchmark.weights < -feasibility_tolerance)
        | (benchmark.weights - effective_caps > feasibility_tolerance)
    )
    gross_infeasible = (
        benchmark_gross - risk_gross[:, 0] > feasibility_tolerance
    )
    if bool(infeasible.any()) or bool(gross_infeasible.any()):
        first_state = int(
            torch.nonzero(
                infeasible.any(-1) | gross_infeasible,
                as_tuple=False,
            )[0, 0].item()
        )
        first_asset = (
            int(torch.nonzero(infeasible[first_state], as_tuple=False)[0, 0].item())
            if bool(infeasible[first_state].any())
            else TOP2000_HOLD30_CASH_INDEX
        )
        raise Top2000Hold30DevelopmentError(
            "fill-time benchmark feasibility preflight failed at "
            f"state={first_state} date={selected_dates[first_state]} "
            f"asset={actions[first_asset]} "
            f"weight={float(benchmark.weights[first_state, first_asset]):g} "
            f"cap={float(effective_caps[first_state, first_asset]):g} "
            f"gross={float(benchmark_gross[first_state]):g} "
            f"gross_max={float(risk_gross[first_state, 0]):g}"
        )

    identity = Top2000Hold30DevelopmentIdentity(
        cache_sha256=cache.cache_sha256,
        cache_identity=cache.cache_identity,
        search_identity=cache.search_identity,
        action_hash=cache.action_hash,
        date_slice_sha256=_canonical_sha256({"exchange_dates": list(selected_dates)}),
        benchmark_weights_sha256=_tensor_sha256(benchmark.weights),
        benchmark_net_returns_sha256=_tensor_sha256(benchmark.net_returns),
        benchmark_trace_sha256=benchmark.trace_sha256,
        state_start_index=state_start_index,
        state_stop_index_exclusive=state_stop_index_exclusive,
        first_exchange_date=selected_dates[0],
        last_exchange_date=selected_dates[-1],
        state_rows=len(selected_dates),
        transition_rows=len(selected_dates) - 1,
        action_count=len(actions),
        one_way_cost_rate=TOP2000_HOLD30_TRAINING_COST_RATE,
        max_stock_weight=max_stock_weight,
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
            cash_index=TOP2000_HOLD30_CASH_INDEX,
            youngest_age=0,
            oldest_age=29,
            track_initial_units=False,
        ),
        cost_rate=TOP2000_HOLD30_TRAINING_COST_RATE,
        initial_equity=asset_returns.new_ones((1,)),
        track_entry_units=torch.ones(
            len(selected_dates) - 1,
            dtype=torch.bool,
            device=asset_returns.device,
        ),
        axis_id=identity.axis_id,
    )
    return Top2000Hold30DevelopmentSequence(
        sequence=sequence,
        benchmark=benchmark,
        identity=identity,
        exchange_dates=selected_dates,
        action_ids=actions,
    )


def build_top2000_hold30_development_sequence(
    cache_path: str | Path,
    *,
    expected_cache_sha256: str,
    state_start_index: int,
    state_stop_index_exclusive: int,
    max_state_rows: int = TOP2000_HOLD30_MAX_STATE_ROWS,
    max_stock_weight: float = TOP2000_HOLD30_DEFAULT_MAX_STOCK_WEIGHT,
    output_device: str | torch.device = "cpu",
    acknowledgement: str,
) -> Top2000Hold30DevelopmentSequence:
    """One-shot path wrapper around the reusable verified-cache API.

    Long-running workers should call
    :func:`load_verified_top2000_hold30_development_cache` once and then use
    :func:`build_top2000_hold30_development_sequence_from_loaded_cache` for
    every chronology.  No synthetic or generated market fallback exists.
    """

    cache = load_verified_top2000_hold30_development_cache(
        cache_path,
        expected_cache_sha256=expected_cache_sha256,
        acknowledgement=acknowledgement,
    )
    return build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=state_start_index,
        state_stop_index_exclusive=state_stop_index_exclusive,
        max_state_rows=max_state_rows,
        max_stock_weight=max_stock_weight,
        output_device=output_device,
    )


__all__ = [
    "DEVELOPMENT_ACK",
    "TOP2000_HOLD30_BENCHMARK_ID",
    "TOP2000_HOLD30_BENCHMARK_INITIALIZATION_RULE",
    "TOP2000_HOLD30_BENCHMARK_RISK_REPAIR_RULE",
    "TOP2000_HOLD30_DEFAULT_MAX_STOCK_WEIGHT",
    "TOP2000_HOLD30_DEVELOPMENT_ADAPTER_SCHEMA",
    "TOP2000_HOLD30_LEGACY_BENCHMARK_ID",
    "TOP2000_HOLD30_LEGACY_DEVELOPMENT_ADAPTER_SCHEMA",
    "TOP2000_HOLD30_MAX_STATE_ROWS",
    "TOP2000_HOLD30_MAX_STOCK_WEIGHT",
    "TOP2000_HOLD30_OBSERVATION_REPRESENTATION",
    "TOP2000_HOLD30_SOURCE_BAR_IDENTITY",
    "TOP2000_HOLD30_SOURCE_BAR_SECONDS",
    "TOP2000_HOLD30_TRAINING_COST_RATE",
    "TOP2000_HOLD30_UNIVERSE_IDENTITY",
    "Top2000Hold30DevelopmentError",
    "Top2000Hold30DevelopmentIdentity",
    "Top2000Hold30DevelopmentSequence",
    "Top2000MonthlyEqualWeightBuyAndDriftTrace",
    "Top2000VerifiedDevelopmentCache",
    "build_top2000_hold30_development_sequence",
    "build_top2000_hold30_development_sequence_from_loaded_cache",
    "load_verified_top2000_hold30_development_cache",
]
