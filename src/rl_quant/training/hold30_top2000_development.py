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
from rl_quant.envs.hold30 import CohortLedger
from rl_quant.training.hold30_runtime import Hold30Sequence

TOP2000_HOLD30_DEVELOPMENT_ADAPTER_SCHEMA = (
    "rl-quant.top2000-hold30-development-cache-adapter-v1"
)
TOP2000_HOLD30_SOURCE_BAR_SECONDS = 300
TOP2000_HOLD30_SOURCE_BAR_IDENTITY = "5-minute-bars"
TOP2000_HOLD30_OBSERVATION_REPRESENTATION = "daily-ohlcv-aggregated-from-5-minute-bars"
TOP2000_HOLD30_UNIVERSE_IDENTITY = "future-selected-top2000-development-only"
TOP2000_HOLD30_BENCHMARK_ID = (
    "C1-monthly-point-in-time-equal-weight-rebalance-and-drift-development-v1"
)
TOP2000_HOLD30_MAX_STATE_ROWS = 378
TOP2000_HOLD30_CASH_INDEX = 0
TOP2000_HOLD30_TRAINING_COST_RATE = 0.002
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
    """Transparent C1 benchmark rebalanced at the first state of each month.

    The first book is an uncharged equal-weight endowment over risky names
    available at the first state.  Thereafter it earns the next close-to-close
    return, performs availability-forced sales into CASH, and re-establishes
    equal weight over the point-in-time available risky set at the first state
    of each new calendar month.  Both forced and rebalance turnover pay the
    same one-way linear cost as the policy.
    """

    weights: torch.Tensor
    gross_returns: torch.Tensor
    availability_forced_one_way_turnover: torch.Tensor
    monthly_rebalance_one_way_turnover: torch.Tensor
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
                + self.monthly_rebalance_one_way_turnover,
            )
        ):
            raise Top2000Hold30DevelopmentError(
                "benchmark total turnover must reconcile by cause"
            )
        if not bool(torch.allclose(self.net_returns, self.gross_returns - self.costs)):
            raise Top2000Hold30DevelopmentError(
                "benchmark net return must equal gross return minus forced-sale cost"
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
    state_start_index: int
    state_stop_index_exclusive: int
    first_exchange_date: str
    last_exchange_date: str
    state_rows: int
    transition_rows: int
    action_count: int
    one_way_cost_rate: float
    source_bar_seconds: int = TOP2000_HOLD30_SOURCE_BAR_SECONDS
    source_bar_identity: str = TOP2000_HOLD30_SOURCE_BAR_IDENTITY
    observation_representation: str = TOP2000_HOLD30_OBSERVATION_REPRESENTATION
    universe_identity: str = TOP2000_HOLD30_UNIVERSE_IDENTITY
    benchmark_id: str = TOP2000_HOLD30_BENCHMARK_ID
    benchmark_initialization: str = (
        "equal-weight-risky-endowment-at-first-state-no-startup-cost"
    )
    policy_initial_ledger_rule: str = (
        "common-c1-endowment-staggered-untracked-ages-0-through-29"
    )
    benchmark_rebalance_rule: str = (
        "first-state-of-calendar-month-equal-weight-over-point-in-time-available-risky;"
        "availability-forced-sales-to-cash"
    )
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


def _monthly_equal_weight_buy_and_drift(
    asset_returns: torch.Tensor,
    availability: torch.Tensor,
    exchange_dates: tuple[str, ...],
    *,
    cost_rate: float,
    cash_index: int,
) -> Top2000MonthlyEqualWeightBuyAndDriftTrace:
    transitions, assets = asset_returns.shape
    if availability.shape != (transitions + 1, assets):
        raise Top2000Hold30DevelopmentError(
            "benchmark availability must have one row beyond returns"
        )
    if len(exchange_dates) != transitions + 1:
        raise Top2000Hold30DevelopmentError(
            "benchmark exchange dates must align with states"
        )
    parsed_dates = tuple(dt.date.fromisoformat(value) for value in exchange_dates)
    weights = asset_returns.new_zeros((transitions + 1, assets))
    weights[0] = _equal_weight_target(
        availability[0], cash_index=cash_index, dtype=asset_returns.dtype
    )

    gross = asset_returns.new_zeros(transitions)
    availability_turnover = asset_returns.new_zeros(transitions)
    rebalance_turnover = asset_returns.new_zeros(transitions)
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
            target = _equal_weight_target(
                availability[index + 1],
                cash_index=cash_index,
                dtype=asset_returns.dtype,
            )
            rebalance_turnover[index] = 0.5 * (target - repaired).abs().sum()
            repaired = target
        costs[index] = cost_rate * (
            availability_turnover[index] + rebalance_turnover[index]
        )
        weights[index + 1] = repaired
    net = gross - costs
    return Top2000MonthlyEqualWeightBuyAndDriftTrace(
        weights=weights,
        gross_returns=gross,
        availability_forced_one_way_turnover=availability_turnover,
        monthly_rebalance_one_way_turnover=rebalance_turnover,
        total_one_way_turnover=availability_turnover + rebalance_turnover,
        costs=costs,
        net_returns=net,
    )


def build_top2000_hold30_development_sequence_from_loaded_cache(
    cache: Top2000VerifiedDevelopmentCache,
    *,
    state_start_index: int,
    state_stop_index_exclusive: int,
    max_state_rows: int = TOP2000_HOLD30_MAX_STATE_ROWS,
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
    benchmark = _monthly_equal_weight_buy_and_drift(
        asset_returns[:, 0],
        masks[:, 0],
        selected_dates,
        cost_rate=TOP2000_HOLD30_TRAINING_COST_RATE,
        cash_index=TOP2000_HOLD30_CASH_INDEX,
    )

    identity = Top2000Hold30DevelopmentIdentity(
        cache_sha256=cache.cache_sha256,
        cache_identity=cache.cache_identity,
        search_identity=cache.search_identity,
        action_hash=cache.action_hash,
        date_slice_sha256=_canonical_sha256({"exchange_dates": list(selected_dates)}),
        benchmark_weights_sha256=_tensor_sha256(benchmark.weights),
        benchmark_net_returns_sha256=_tensor_sha256(benchmark.net_returns),
        state_start_index=state_start_index,
        state_stop_index_exclusive=state_stop_index_exclusive,
        first_exchange_date=selected_dates[0],
        last_exchange_date=selected_dates[-1],
        state_rows=len(selected_dates),
        transition_rows=len(selected_dates) - 1,
        action_count=len(actions),
        one_way_cost_rate=TOP2000_HOLD30_TRAINING_COST_RATE,
    )

    risk_caps = torch.zeros_like(masks, dtype=asset_returns.dtype)
    risk_caps[..., TOP2000_HOLD30_CASH_INDEX] = 1.0
    risk_caps[..., 1:] = masks[..., 1:].to(asset_returns.dtype) * 0.01
    risk_gross = asset_returns.new_ones((len(selected_dates), 1))
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
        output_device=output_device,
    )


__all__ = [
    "DEVELOPMENT_ACK",
    "TOP2000_HOLD30_BENCHMARK_ID",
    "TOP2000_HOLD30_DEVELOPMENT_ADAPTER_SCHEMA",
    "TOP2000_HOLD30_MAX_STATE_ROWS",
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
