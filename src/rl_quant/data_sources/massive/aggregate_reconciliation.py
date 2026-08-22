"""Reconstruct condition-qualified five-minute bars from replayed trades."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import math
from typing import Sequence

from rl_quant.data_sources.massive.session_calendar import MassiveExchangeSession
from rl_quant.data_sources.massive.trade_replay import (
    MassiveTradeEventV1,
    MassiveTradeReplayResult,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_RECONSTRUCTED_BAR_SCHEMA = "rl-quant.massive-reconstructed-bar-v1"
MASSIVE_AGGREGATE_RECONCILIATION_SCHEMA = (
    "rl-quant.massive-aggregate-reconciliation-v1"
)


class MassiveAggregateReconciliationError(ValueError):
    """Trade-derived bars and vendor aggregates do not reconcile."""


@dataclass(frozen=True, slots=True)
class MassiveFiveMinuteBar:
    security_id: str
    session_date: str
    interval_index: int
    interval_start_ns: int
    interval_end_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    transaction_count: int
    source_trade_replay_receipt_sha256: str
    schema: str = MASSIVE_RECONSTRUCTED_BAR_SCHEMA

    def validate(self) -> None:
        if self.schema != MASSIVE_RECONSTRUCTED_BAR_SCHEMA:
            raise MassiveAggregateReconciliationError("bar schema drifted")
        if not self.security_id or not self.session_date:
            raise MassiveAggregateReconciliationError("bar identity is absent")
        if (
            isinstance(self.interval_index, bool)
            or not isinstance(self.interval_index, int)
            or self.interval_index < 0
            or self.interval_end_ns <= self.interval_start_ns
        ):
            raise MassiveAggregateReconciliationError("bar interval is invalid")
        prices = (self.open, self.high, self.low, self.close, self.vwap)
        if any(not math.isfinite(value) or value <= 0 for value in prices):
            raise MassiveAggregateReconciliationError("bar price is invalid")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise MassiveAggregateReconciliationError("bar OHLC relationship is invalid")
        if self.volume <= 0 or self.transaction_count <= 0:
            raise MassiveAggregateReconciliationError("bar activity is empty")
        if (
            len(self.source_trade_replay_receipt_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_trade_replay_receipt_sha256
            )
        ):
            raise MassiveAggregateReconciliationError("bar source receipt is invalid")


@dataclass(frozen=True, slots=True)
class MassiveAggregateReconciliation:
    reconstructed_bar_receipt_sha256: str
    vendor_aggregate_receipt_sha256: str
    compared_interval_count: int
    missing_reconstructed_intervals: tuple[int, ...]
    missing_vendor_intervals: tuple[int, ...]
    mismatched_intervals: tuple[int, ...]
    exact_support: bool
    all_values_match: bool
    receipt_sha256: str
    schema: str = MASSIVE_AGGREGATE_RECONCILIATION_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_AGGREGATE_RECONCILIATION_SCHEMA:
            raise MassiveAggregateReconciliationError(
                "aggregate reconciliation schema drifted"
            )
        for name in (
            "reconstructed_bar_receipt_sha256",
            "vendor_aggregate_receipt_sha256",
            "receipt_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise MassiveAggregateReconciliationError(f"{name} is invalid")
        if self.compared_interval_count < 0:
            raise MassiveAggregateReconciliationError("comparison count is negative")
        for inventory in (
            self.missing_reconstructed_intervals,
            self.missing_vendor_intervals,
            self.mismatched_intervals,
        ):
            if inventory != tuple(sorted(set(inventory))):
                raise MassiveAggregateReconciliationError(
                    "reconciliation inventory is not canonical"
                )
        if self.exact_support != (
            not self.missing_reconstructed_intervals
            and not self.missing_vendor_intervals
        ):
            raise MassiveAggregateReconciliationError("support flag drifted")
        if self.all_values_match != (
            self.exact_support and not self.mismatched_intervals
        ):
            raise MassiveAggregateReconciliationError("value parity flag drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveAggregateReconciliationError(
                "aggregate reconciliation receipt differs"
            )


def reconstruct_massive_five_minute_bars(
    replay: MassiveTradeReplayResult,
    *,
    session: MassiveExchangeSession,
) -> tuple[MassiveFiveMinuteBar, ...]:
    """Reconstruct only observed regular-session intervals; never fill gaps."""

    replay.validate()
    session.validate()
    if replay.session_date != session.session_date:
        raise MassiveAggregateReconciliationError("replay and session dates differ")
    by_interval: dict[int, list[MassiveTradeEventV1]] = {}
    for event in replay.active_events:
        if not event.regular_session or not session.is_regular(event.participant_timestamp_ns):
            continue
        index = session.five_minute_interval(event.participant_timestamp_ns)
        by_interval.setdefault(index, []).append(event)
    bars: list[MassiveFiveMinuteBar] = []
    for index in sorted(by_interval):
        events = sorted(
            by_interval[index],
            key=lambda event: (
                event.participant_timestamp_ns,
                event.sip_timestamp_ns,
                event.sequence_number,
                event.event_key,
            ),
        )
        price_events = [event for event in events if event.price_forming]
        volume_events = [event for event in events if event.volume_forming]
        if not price_events or not volume_events:
            continue
        volume = sum(Decimal(event.decimal_size) for event in volume_events)
        dollar = sum(
            Decimal(str(event.price)) * Decimal(event.decimal_size)
            for event in volume_events
        )
        prices = [event.price for event in price_events]
        bar = MassiveFiveMinuteBar(
            security_id=replay.security_id,
            session_date=replay.session_date,
            interval_index=index,
            interval_start_ns=session.regular_open_ns + index * 300_000_000_000,
            interval_end_ns=session.regular_open_ns + (index + 1) * 300_000_000_000,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=float(volume),
            vwap=float(dollar / volume),
            transaction_count=len(volume_events),
            source_trade_replay_receipt_sha256=replay.receipt_sha256,
        )
        bar.validate()
        bars.append(bar)
    return tuple(bars)


def reconcile_massive_aggregate_bars(
    reconstructed: Sequence[MassiveFiveMinuteBar],
    vendor: Sequence[MassiveFiveMinuteBar],
    *,
    vendor_aggregate_receipt_sha256: str,
    absolute_tolerance: float = 1e-10,
) -> MassiveAggregateReconciliation:
    """Compare exact interval support and frozen numerical fields."""

    for row in (*reconstructed, *vendor):
        row.validate()
    left = {row.interval_index: row for row in reconstructed}
    right = {row.interval_index: row for row in vendor}
    if len(left) != len(reconstructed) or len(right) != len(vendor):
        raise MassiveAggregateReconciliationError("duplicate aggregate interval")
    left_keys = set(left)
    right_keys = set(right)
    common = tuple(sorted(left_keys & right_keys))
    mismatched: list[int] = []
    for index in common:
        first = left[index]
        second = right[index]
        scalar_pairs = (
            (first.open, second.open),
            (first.high, second.high),
            (first.low, second.low),
            (first.close, second.close),
            (first.volume, second.volume),
            (first.vwap, second.vwap),
        )
        if (
            first.security_id != second.security_id
            or first.session_date != second.session_date
            or first.interval_start_ns != second.interval_start_ns
            or first.interval_end_ns != second.interval_end_ns
            or first.transaction_count != second.transaction_count
            or any(abs(left_value - right_value) > absolute_tolerance for left_value, right_value in scalar_pairs)
        ):
            mismatched.append(index)
    reconstructed_receipt = semantic_sha256([asdict(row) for row in reconstructed])
    missing_reconstructed = tuple(sorted(right_keys - left_keys))
    missing_vendor = tuple(sorted(left_keys - right_keys))
    body = {
        "schema": MASSIVE_AGGREGATE_RECONCILIATION_SCHEMA,
        "reconstructed_bar_receipt_sha256": reconstructed_receipt,
        "vendor_aggregate_receipt_sha256": vendor_aggregate_receipt_sha256,
        "compared_interval_count": len(common),
        "missing_reconstructed_intervals": missing_reconstructed,
        "missing_vendor_intervals": missing_vendor,
        "mismatched_intervals": tuple(mismatched),
        "exact_support": not missing_reconstructed and not missing_vendor,
        "all_values_match": not missing_reconstructed
        and not missing_vendor
        and not mismatched,
    }
    result = MassiveAggregateReconciliation(
        reconstructed_bar_receipt_sha256=reconstructed_receipt,
        vendor_aggregate_receipt_sha256=vendor_aggregate_receipt_sha256,
        compared_interval_count=len(common),
        missing_reconstructed_intervals=missing_reconstructed,
        missing_vendor_intervals=missing_vendor,
        mismatched_intervals=tuple(mismatched),
        exact_support=not missing_reconstructed and not missing_vendor,
        all_values_match=not missing_reconstructed
        and not missing_vendor
        and not mismatched,
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_AGGREGATE_RECONCILIATION_SCHEMA",
    "MASSIVE_RECONSTRUCTED_BAR_SCHEMA",
    "MassiveAggregateReconciliation",
    "MassiveAggregateReconciliationError",
    "MassiveFiveMinuteBar",
    "reconcile_massive_aggregate_bars",
    "reconstruct_massive_five_minute_bars",
]
