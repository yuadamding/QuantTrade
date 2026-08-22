"""Reconstruct condition-qualified five-minute bars from replayed trades."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import math
from typing import Sequence

from rl_quant.data_sources.massive.session_calendar import MassiveExchangeSession
from rl_quant.data_sources.massive.trade_replay import (
    MassiveTradeEventV2,
    MassiveTradeReplayResult,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_RECONSTRUCTED_BAR_SCHEMA = "rl-quant.massive-reconstructed-bar-v2"
MASSIVE_VENDOR_AGGREGATE_BAR_SCHEMA = "rl-quant.massive-vendor-aggregate-bar-v1"
MASSIVE_AGGREGATE_RECONCILIATION_SPEC_SCHEMA = (
    "rl-quant.massive-aggregate-reconciliation-spec-v1"
)
MASSIVE_AGGREGATE_RECONCILIATION_SCHEMA = (
    "rl-quant.massive-aggregate-reconciliation-v2"
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
        if self.high < self.low:
            raise MassiveAggregateReconciliationError("bar OHLC relationship is invalid")
        if (
            not math.isfinite(self.volume)
            or self.volume <= 0
            or isinstance(self.transaction_count, bool)
            or not isinstance(self.transaction_count, int)
            or self.transaction_count <= 0
        ):
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
class MassiveVendorAggregateBar:
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
    vendor_source_object_receipt_sha256: str
    vendor_schema_receipt_sha256: str
    adjusted: bool
    timespan: str
    multiplier: int
    schema: str = MASSIVE_VENDOR_AGGREGATE_BAR_SCHEMA

    def validate(self) -> None:
        if self.schema != MASSIVE_VENDOR_AGGREGATE_BAR_SCHEMA:
            raise MassiveAggregateReconciliationError("vendor bar schema drifted")
        synthetic = MassiveFiveMinuteBar(
            security_id=self.security_id,
            session_date=self.session_date,
            interval_index=self.interval_index,
            interval_start_ns=self.interval_start_ns,
            interval_end_ns=self.interval_end_ns,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            vwap=self.vwap,
            transaction_count=self.transaction_count,
            source_trade_replay_receipt_sha256="0" * 64,
        )
        synthetic.validate()
        for name in (
            "vendor_source_object_receipt_sha256",
            "vendor_schema_receipt_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise MassiveAggregateReconciliationError(
                    f"{name} is not a SHA-256 digest"
                )
        if not isinstance(self.adjusted, bool):
            raise MassiveAggregateReconciliationError(
                "vendor adjusted flag must be Boolean"
            )
        if self.adjusted:
            raise MassiveAggregateReconciliationError(
                "trade replay must reconcile against unadjusted vendor bars"
            )
        if self.timespan != "minute" or self.multiplier != 5:
            raise MassiveAggregateReconciliationError(
                "vendor aggregate is not a five-minute bar"
            )


@dataclass(frozen=True, slots=True)
class MassiveAggregateReconciliationSpec:
    price_absolute_tolerance: str
    volume_absolute_tolerance: str
    vwap_absolute_tolerance: str
    transaction_count_exact: bool
    receipt_sha256: str
    schema: str = MASSIVE_AGGREGATE_RECONCILIATION_SPEC_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def tolerances(self) -> tuple[Decimal, Decimal, Decimal]:
        values: list[Decimal] = []
        for name in (
            "price_absolute_tolerance",
            "volume_absolute_tolerance",
            "vwap_absolute_tolerance",
        ):
            try:
                value = Decimal(getattr(self, name))
            except InvalidOperation as exc:
                raise MassiveAggregateReconciliationError(
                    f"{name} is not decimal"
                ) from exc
            if not value.is_finite() or value < 0:
                raise MassiveAggregateReconciliationError(
                    f"{name} must be finite and nonnegative"
                )
            values.append(value)
        return values[0], values[1], values[2]

    def validate(self) -> None:
        if self.schema != MASSIVE_AGGREGATE_RECONCILIATION_SPEC_SCHEMA:
            raise MassiveAggregateReconciliationError(
                "aggregate reconciliation specification drifted"
            )
        self.tolerances()
        if self.transaction_count_exact is not True:
            raise MassiveAggregateReconciliationError(
                "transaction-count reconciliation must remain exact"
            )
        if len(self.receipt_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.receipt_sha256
        ):
            raise MassiveAggregateReconciliationError(
                "reconciliation specification receipt is invalid"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveAggregateReconciliationError(
                "reconciliation specification receipt differs"
            )

    @classmethod
    def build(
        cls,
        *,
        price_absolute_tolerance: str = "1e-10",
        volume_absolute_tolerance: str = "1e-10",
        vwap_absolute_tolerance: str = "1e-10",
    ) -> MassiveAggregateReconciliationSpec:
        body = {
            "schema": MASSIVE_AGGREGATE_RECONCILIATION_SPEC_SCHEMA,
            "price_absolute_tolerance": price_absolute_tolerance,
            "volume_absolute_tolerance": volume_absolute_tolerance,
            "vwap_absolute_tolerance": vwap_absolute_tolerance,
            "transaction_count_exact": True,
        }
        value = cls(
            price_absolute_tolerance=price_absolute_tolerance,
            volume_absolute_tolerance=volume_absolute_tolerance,
            vwap_absolute_tolerance=vwap_absolute_tolerance,
            transaction_count_exact=True,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveAggregateReconciliation:
    security_id: str
    session_date: str
    reconstructed_bar_receipt_sha256: str
    vendor_aggregate_receipt_sha256: str
    vendor_source_object_receipt_sha256: str
    reconciliation_spec_receipt_sha256: str
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
            "vendor_source_object_receipt_sha256",
            "reconciliation_spec_receipt_sha256",
            "receipt_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise MassiveAggregateReconciliationError(f"{name} is invalid")
        if not self.security_id or not self.session_date:
            raise MassiveAggregateReconciliationError("comparison identity is absent")
        if self.compared_interval_count <= 0:
            raise MassiveAggregateReconciliationError("comparison count is not positive")
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
    by_interval: dict[int, list[MassiveTradeEventV2]] = {}
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
        open_close_events = [event for event in events if event.updates_open_close]
        high_low_events = [event for event in events if event.updates_high_low]
        volume_events = [event for event in events if event.updates_volume]
        if not open_close_events or not high_low_events or not volume_events:
            continue
        volume = sum(Decimal(event.decimal_size) for event in volume_events)
        dollar = sum(
            Decimal(str(event.price)) * Decimal(event.decimal_size)
            for event in volume_events
        )
        open_close_prices = [event.price for event in open_close_events]
        high_low_prices = [event.price for event in high_low_events]
        bar = MassiveFiveMinuteBar(
            security_id=replay.security_id,
            session_date=replay.session_date,
            interval_index=index,
            interval_start_ns=session.regular_open_ns + index * 300_000_000_000,
            interval_end_ns=session.regular_open_ns + (index + 1) * 300_000_000_000,
            open=open_close_prices[0],
            high=max(high_low_prices),
            low=min(high_low_prices),
            close=open_close_prices[-1],
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
    vendor: Sequence[MassiveVendorAggregateBar],
    *,
    specification: MassiveAggregateReconciliationSpec,
) -> MassiveAggregateReconciliation:
    """Compare exact interval support and frozen numerical fields."""

    if not reconstructed or not vendor:
        raise MassiveAggregateReconciliationError(
            "aggregate reconciliation requires nonempty inputs"
        )
    specification.validate()
    for reconstructed_row in reconstructed:
        reconstructed_row.validate()
    for vendor_row in vendor:
        vendor_row.validate()
    left_identities = {(row.security_id, row.session_date) for row in reconstructed}
    right_identities = {(row.security_id, row.session_date) for row in vendor}
    if len(left_identities) != 1 or len(right_identities) != 1:
        raise MassiveAggregateReconciliationError(
            "each aggregate side must contain one security-session"
        )
    if left_identities != right_identities:
        raise MassiveAggregateReconciliationError(
            "vendor and reconstructed identities differ"
        )
    source_replays = {
        row.source_trade_replay_receipt_sha256 for row in reconstructed
    }
    vendor_sources = {row.vendor_source_object_receipt_sha256 for row in vendor}
    vendor_schemas = {row.vendor_schema_receipt_sha256 for row in vendor}
    if len(source_replays) != 1 or len(vendor_sources) != 1 or len(vendor_schemas) != 1:
        raise MassiveAggregateReconciliationError(
            "aggregate provenance is not source-homogeneous"
        )
    left = {row.interval_index: row for row in reconstructed}
    right = {row.interval_index: row for row in vendor}
    if len(left) != len(reconstructed) or len(right) != len(vendor):
        raise MassiveAggregateReconciliationError("duplicate aggregate interval")
    left_keys = set(left)
    right_keys = set(right)
    common = tuple(sorted(left_keys & right_keys))
    if not common:
        raise MassiveAggregateReconciliationError(
            "aggregate reconciliation has no common intervals"
        )
    price_tolerance, volume_tolerance, vwap_tolerance = specification.tolerances()
    mismatched: list[int] = []
    for index in common:
        first = left[index]
        second = right[index]
        price_pairs = (
            (first.open, second.open),
            (first.high, second.high),
            (first.low, second.low),
            (first.close, second.close),
        )
        if (
            first.interval_start_ns != second.interval_start_ns
            or first.interval_end_ns != second.interval_end_ns
            or first.transaction_count != second.transaction_count
            or any(
                abs(Decimal(str(left_value)) - Decimal(str(right_value)))
                > price_tolerance
                for left_value, right_value in price_pairs
            )
            or abs(Decimal(str(first.volume)) - Decimal(str(second.volume)))
            > volume_tolerance
            or abs(Decimal(str(first.vwap)) - Decimal(str(second.vwap)))
            > vwap_tolerance
        ):
            mismatched.append(index)
    reconstructed_receipt = semantic_sha256([asdict(row) for row in reconstructed])
    vendor_receipt = semantic_sha256([asdict(row) for row in vendor])
    security_id, session_date = next(iter(left_identities))
    missing_reconstructed = tuple(sorted(right_keys - left_keys))
    missing_vendor = tuple(sorted(left_keys - right_keys))
    body = {
        "schema": MASSIVE_AGGREGATE_RECONCILIATION_SCHEMA,
        "security_id": security_id,
        "session_date": session_date,
        "reconstructed_bar_receipt_sha256": reconstructed_receipt,
        "vendor_aggregate_receipt_sha256": vendor_receipt,
        "vendor_source_object_receipt_sha256": next(iter(vendor_sources)),
        "reconciliation_spec_receipt_sha256": specification.receipt_sha256,
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
        security_id=security_id,
        session_date=session_date,
        reconstructed_bar_receipt_sha256=reconstructed_receipt,
        vendor_aggregate_receipt_sha256=vendor_receipt,
        vendor_source_object_receipt_sha256=next(iter(vendor_sources)),
        reconciliation_spec_receipt_sha256=specification.receipt_sha256,
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
    "MASSIVE_AGGREGATE_RECONCILIATION_SPEC_SCHEMA",
    "MASSIVE_AGGREGATE_RECONCILIATION_SCHEMA",
    "MASSIVE_RECONSTRUCTED_BAR_SCHEMA",
    "MASSIVE_VENDOR_AGGREGATE_BAR_SCHEMA",
    "MassiveAggregateReconciliation",
    "MassiveAggregateReconciliationError",
    "MassiveAggregateReconciliationSpec",
    "MassiveFiveMinuteBar",
    "MassiveVendorAggregateBar",
    "reconcile_massive_aggregate_bars",
    "reconstruct_massive_five_minute_bars",
]
