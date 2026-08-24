"""PIT identity-routed partitions for finalized V0 trade-source rows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
)
from rl_quant.data_sources.massive.trade_extraction import MassiveExtractedTradeRow
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256


MASSIVE_FINALIZED_FEATURE_DOMAIN_SPEC_V0_SCHEMA = (
    "rl-quant.massive-finalized-feature-domain-spec-v0"
)
MASSIVE_DAILY_TRADE_SECURITY_PARTITION_V0_SCHEMA = (
    "rl-quant.massive-daily-trade-security-partition-v0"
)
MASSIVE_DAILY_TRADE_PARTITION_MANIFEST_V0_SCHEMA = (
    "rl-quant.massive-daily-trade-partition-manifest-v0"
)
MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256 = semantic_sha256(
    {
        "identity": "PIT-permanent-security-at-participant-time",
        "event_domain": "participant-timestamp-in-[regular-open,regular-close)",
        "all_source_rows": "route-exactly-once-before-domain-filter",
        "correction_order": "sip-sequence-exchange-trf-trade-source-row",
        "late_regular_reports": "eligible-after-finalized-correction-replay",
        "after_hours": "retained-in-source-partition-but-excluded-from-feature-domain",
        "partition_receipt": "semantic-canonical-row-and-source-provenance-inventory",
    }
)
MASSIVE_DAILY_TRADE_PARTITION_SOURCE_SHA256 = file_sha256(Path(__file__))


class MassiveDailyTradePartitionError(ValueError):
    """Whole-file rows cannot be reconciled into PIT security partitions."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveDailyTradePartitionError(f"{name} must be a lowercase SHA-256")
    return value


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveDailyTradePartitionError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True, slots=True)
class MassiveFinalizedFeatureDomainSpecV0:
    feature_domain_id: str
    economic_event_clock: str
    regular_session_interval: str
    closing_endpoint_included: bool
    after_hours_features_authorized: bool
    correction_order_clock: str
    finalized_post_close_corrections_applied: bool
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_FEATURE_DOMAIN_SPEC_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_FEATURE_DOMAIN_SPEC_V0_SCHEMA:
            raise MassiveDailyTradePartitionError("feature-domain schema drifted")
        expected = {
            "feature_domain_id": "massive-finalized-regular-session-trades-v0",
            "economic_event_clock": "participant-timestamp",
            "regular_session_interval": "[regular-open,regular-close)",
            "correction_order_clock": "sip-timestamp-sequence",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise MassiveDailyTradePartitionError(f"{name} drifted")
        if self.closing_endpoint_included is not False:
            raise MassiveDailyTradePartitionError("regular close must be exclusive")
        if self.after_hours_features_authorized is not False:
            raise MassiveDailyTradePartitionError("after-hours features are forbidden")
        if self.finalized_post_close_corrections_applied is not True:
            raise MassiveDailyTradePartitionError("post-close corrections must apply")
        for name in (
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveDailyTradePartitionError("feature-domain receipt differs")


def build_massive_finalized_feature_domain_spec_v0(
    *,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
) -> MassiveFinalizedFeatureDomainSpecV0:
    condition_authority.validate()
    correction_authority.validate()
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_FEATURE_DOMAIN_SPEC_V0_SCHEMA,
        "feature_domain_id": "massive-finalized-regular-session-trades-v0",
        "economic_event_clock": "participant-timestamp",
        "regular_session_interval": "[regular-open,regular-close)",
        "closing_endpoint_included": False,
        "after_hours_features_authorized": False,
        "correction_order_clock": "sip-timestamp-sequence",
        "finalized_post_close_corrections_applied": True,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
    }
    result = MassiveFinalizedFeatureDomainSpecV0(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveDailyTradeSecurityPartitionV0:
    security_id: str
    source_tickers: tuple[str, ...]
    source_row_count: int
    premarket_row_count: int
    regular_session_input_row_count: int
    after_hours_row_count: int
    active_regular_session_row_count: int
    cancelled_event_count: int
    all_row_inventory_sha256: str
    active_regular_row_inventory_sha256: str
    partition_receipt_sha256: str
    schema: str = MASSIVE_DAILY_TRADE_SECURITY_PARTITION_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        if self.schema != MASSIVE_DAILY_TRADE_SECURITY_PARTITION_V0_SCHEMA:
            raise MassiveDailyTradePartitionError("security partition schema drifted")
        if not self.security_id:
            raise MassiveDailyTradePartitionError(
                "security partition identity is absent"
            )
        if not self.source_tickers or self.source_tickers != tuple(
            sorted(set(self.source_tickers))
        ):
            raise MassiveDailyTradePartitionError("partition tickers are not canonical")
        for name in (
            "source_row_count",
            "premarket_row_count",
            "regular_session_input_row_count",
            "after_hours_row_count",
            "active_regular_session_row_count",
            "cancelled_event_count",
        ):
            _count(name, getattr(self, name))
        if self.source_row_count != (
            self.premarket_row_count
            + self.regular_session_input_row_count
            + self.after_hours_row_count
        ):
            raise MassiveDailyTradePartitionError(
                "partition row domains do not reconcile"
            )
        if self.active_regular_session_row_count > self.regular_session_input_row_count:
            raise MassiveDailyTradePartitionError("active regular rows exceed inputs")
        for name in (
            "all_row_inventory_sha256",
            "active_regular_row_inventory_sha256",
            "partition_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        expected = semantic_sha256(
            {
                key: value
                for key, value in asdict(self).items()
                if key != "partition_receipt_sha256"
            }
        )
        if self.partition_receipt_sha256 != expected:
            raise MassiveDailyTradePartitionError("security partition receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveDailyTradePartitionManifestV0:
    source_session_date: str
    source_file_scan_receipt_sha256: str
    identity_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    feature_domain_spec_receipt_sha256: str
    partition_spec_receipt_sha256: str
    partition_source_sha256: str
    security_partitions: tuple[MassiveDailyTradeSecurityPartitionV0, ...]
    global_row_count: int
    partitioned_row_count: int
    rejected_row_count: int
    global_row_inventory_sha256: str
    global_partition_inventory_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_DAILY_TRADE_PARTITION_MANIFEST_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_DAILY_TRADE_PARTITION_MANIFEST_V0_SCHEMA:
            raise MassiveDailyTradePartitionError("partition manifest schema drifted")
        if not self.source_session_date:
            raise MassiveDailyTradePartitionError("partition session is absent")
        for name in (
            "source_file_scan_receipt_sha256",
            "identity_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "feature_domain_spec_receipt_sha256",
            "partition_spec_receipt_sha256",
            "partition_source_sha256",
            "global_row_inventory_sha256",
            "global_partition_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.partition_spec_receipt_sha256
            != MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256
        ):
            raise MassiveDailyTradePartitionError("partition specification drifted")
        if self.partition_source_sha256 != MASSIVE_DAILY_TRADE_PARTITION_SOURCE_SHA256:
            raise MassiveDailyTradePartitionError("partition implementation drifted")
        for name in (
            "global_row_count",
            "partitioned_row_count",
            "rejected_row_count",
        ):
            _count(name, getattr(self, name))
        if self.global_row_count <= 0 or self.global_row_count != (
            self.partitioned_row_count + self.rejected_row_count
        ):
            raise MassiveDailyTradePartitionError(
                "global partition rows do not reconcile"
            )
        if (
            self.rejected_row_count != 0
            or self.partitioned_row_count != self.global_row_count
        ):
            raise MassiveDailyTradePartitionError(
                "every source row must be partitioned"
            )
        keys = tuple(row.security_id for row in self.security_partitions)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveDailyTradePartitionError(
                "security partitions are not sorted and unique"
            )
        for row in self.security_partitions:
            row.validate()
        if (
            sum(row.source_row_count for row in self.security_partitions)
            != self.global_row_count
        ):
            raise MassiveDailyTradePartitionError(
                "partition counts differ from global count"
            )
        expected_inventory = semantic_sha256(
            tuple(row.partition_receipt_sha256 for row in self.security_partitions)
        )
        if self.global_partition_inventory_sha256 != expected_inventory:
            raise MassiveDailyTradePartitionError("partition inventory differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveDailyTradePartitionError("partition manifest receipt differs")


def _resolve_security_id(
    authority: PITSecurityUniverseAuthority,
    *,
    ticker: str,
    participant_timestamp_ns: int,
) -> str:
    observed_at_ms = participant_timestamp_ns // 1_000_000
    matches = tuple(
        row
        for row in authority.ticker_history
        if row.ticker == ticker
        and row.valid_from_ms <= observed_at_ms
        and (row.valid_to_ms is None or observed_at_ms < row.valid_to_ms)
        and row.available_at_ms <= observed_at_ms
    )
    if len(matches) != 1:
        raise MassiveDailyTradePartitionError(
            f"ticker {ticker} does not resolve uniquely at participant time"
        )
    return matches[0].security_id


def _event_key(row: MassiveExtractedTradeRow) -> tuple[str, int, int, str]:
    record = row.canonical_record
    return (
        record.ticker,
        record.exchange_id,
        -1 if record.trf_id is None else record.trf_id,
        record.trade_id,
    )


def build_massive_daily_trade_partition_manifest_v0(
    *,
    rows: Sequence[MassiveExtractedTradeRow],
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
) -> MassiveDailyTradePartitionManifestV0:
    """Resolve every scanned row and replay finalized corrections by security."""

    scan_evidence.validate()
    identity_authority.validate()
    condition_authority.validate()
    correction_authority.validate()
    feature_domain_spec.validate()
    if (
        feature_domain_spec.condition_authority_receipt_sha256
        != condition_authority.receipt_sha256
        or feature_domain_spec.correction_authority_receipt_sha256
        != correction_authority.receipt_sha256
        or scan_evidence.correction_authority_receipt_sha256
        != correction_authority.receipt_sha256
    ):
        raise MassiveDailyTradePartitionError("feature authorities differ from scan")
    source_rows = tuple(sorted(rows, key=lambda row: row.source_row_number))
    if len(source_rows) != scan_evidence.source_row_count:
        raise MassiveDailyTradePartitionError("partition input count differs from scan")
    for row in source_rows:
        row.validate()
    canonical_inventory = semantic_sha256(
        tuple(row.canonical_record.receipt_sha256 for row in source_rows)
    )
    provenance_inventory = semantic_sha256(
        tuple(
            (
                row.source_row_number,
                row.raw_row_sha256,
                row.canonical_record.receipt_sha256,
            )
            for row in source_rows
        )
    )
    if (
        canonical_inventory != scan_evidence.all_row_canonical_inventory_sha256
        or provenance_inventory != scan_evidence.all_row_provenance_inventory_sha256
    ):
        raise MassiveDailyTradePartitionError(
            "partition rows differ from whole-file scan"
        )

    by_security: dict[str, list[MassiveExtractedTradeRow]] = defaultdict(list)
    for row in source_rows:
        condition_authority.resolve(row.canonical_record.conditions)
        security_id = _resolve_security_id(
            identity_authority,
            ticker=row.canonical_record.ticker,
            participant_timestamp_ns=row.canonical_record.participant_timestamp_ns,
        )
        by_security[security_id].append(row)

    partitions: list[MassiveDailyTradeSecurityPartitionV0] = []
    for security_id, security_rows in sorted(by_security.items()):
        ordered = tuple(
            sorted(
                security_rows,
                key=lambda row: (
                    row.canonical_record.sip_timestamp_ns,
                    row.canonical_record.sequence_number,
                    row.canonical_record.exchange_id,
                    -1
                    if row.canonical_record.trf_id is None
                    else row.canonical_record.trf_id,
                    row.canonical_record.trade_id,
                    row.source_row_number,
                ),
            )
        )
        active: dict[tuple[str, int, int, str], MassiveExtractedTradeRow] = {}
        cancelled: set[tuple[str, int, int, str]] = set()
        for row in ordered:
            record = row.canonical_record
            kind = correction_authority.resolve(
                0 if record.correction_code is None else record.correction_code
            )
            key = _event_key(row)
            if kind in {"new-trade", "late-report"}:
                existing = active.get(key)
                if (
                    existing is not None
                    and existing.canonical_record.receipt_sha256
                    != record.receipt_sha256
                ):
                    raise MassiveDailyTradePartitionError(
                        "conflicting duplicate finalized trade"
                    )
                active[key] = row
                cancelled.discard(key)
            elif kind == "replacement":
                if key not in active:
                    raise MassiveDailyTradePartitionError(
                        "replacement lacks predecessor"
                    )
                active[key] = row
                cancelled.discard(key)
            elif kind == "cancellation":
                if key not in active:
                    raise MassiveDailyTradePartitionError(
                        "cancellation lacks predecessor"
                    )
                del active[key]
                cancelled.add(key)
        regular_inputs = tuple(
            row
            for row in ordered
            if scan_evidence.regular_open_ns
            <= row.canonical_record.participant_timestamp_ns
            < scan_evidence.regular_close_ns
        )
        active_regular = tuple(
            sorted(
                (
                    row
                    for row in active.values()
                    if scan_evidence.regular_open_ns
                    <= row.canonical_record.participant_timestamp_ns
                    < scan_evidence.regular_close_ns
                ),
                key=lambda row: _event_key(row),
            )
        )
        premarket = sum(
            row.canonical_record.participant_timestamp_ns
            < scan_evidence.regular_open_ns
            for row in ordered
        )
        after_hours = sum(
            row.canonical_record.participant_timestamp_ns
            >= scan_evidence.regular_close_ns
            for row in ordered
        )
        partition_body: dict[str, object] = {
            "schema": MASSIVE_DAILY_TRADE_SECURITY_PARTITION_V0_SCHEMA,
            "security_id": security_id,
            "source_tickers": tuple(
                sorted({row.canonical_record.ticker for row in ordered})
            ),
            "source_row_count": len(ordered),
            "premarket_row_count": premarket,
            "regular_session_input_row_count": len(regular_inputs),
            "after_hours_row_count": after_hours,
            "active_regular_session_row_count": len(active_regular),
            "cancelled_event_count": len(cancelled),
            "all_row_inventory_sha256": semantic_sha256(
                tuple(
                    (
                        row.source_row_number,
                        row.raw_row_sha256,
                        row.canonical_record.receipt_sha256,
                    )
                    for row in sorted(
                        ordered, key=lambda value: value.source_row_number
                    )
                )
            ),
            "active_regular_row_inventory_sha256": semantic_sha256(
                tuple(row.canonical_record.receipt_sha256 for row in active_regular)
            ),
        }
        partition = MassiveDailyTradeSecurityPartitionV0(
            **partition_body,  # type: ignore[arg-type]
            partition_receipt_sha256=semantic_sha256(partition_body),
        )
        partition.validate()
        partitions.append(partition)
    partition_rows = tuple(partitions)
    body: dict[str, object] = {
        "schema": MASSIVE_DAILY_TRADE_PARTITION_MANIFEST_V0_SCHEMA,
        "source_session_date": scan_evidence.source_session_date,
        "source_file_scan_receipt_sha256": scan_evidence.receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "feature_domain_spec_receipt_sha256": feature_domain_spec.receipt_sha256,
        "partition_spec_receipt_sha256": MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
        "partition_source_sha256": MASSIVE_DAILY_TRADE_PARTITION_SOURCE_SHA256,
        "security_partitions": partition_rows,
        "global_row_count": len(source_rows),
        "partitioned_row_count": len(source_rows),
        "rejected_row_count": 0,
        "global_row_inventory_sha256": provenance_inventory,
        "global_partition_inventory_sha256": semantic_sha256(
            tuple(row.partition_receipt_sha256 for row in partition_rows)
        ),
    }
    provisional = MassiveDailyTradePartitionManifestV0(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveDailyTradePartitionManifestV0(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


def build_massive_daily_trade_partition_manifest_from_security_partitions_v0(
    *,
    scan_evidence: MassiveDailyTradeFileScanEvidenceV0,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    security_partitions: Sequence[MassiveDailyTradeSecurityPartitionV0],
) -> MassiveDailyTradePartitionManifestV0:
    """Build the V0 manifest from independently replayed security partitions."""

    scan_evidence.validate()
    identity_authority.validate()
    condition_authority.validate()
    correction_authority.validate()
    feature_domain_spec.validate()
    if (
        feature_domain_spec.condition_authority_receipt_sha256
        != condition_authority.receipt_sha256
        or feature_domain_spec.correction_authority_receipt_sha256
        != correction_authority.receipt_sha256
        or scan_evidence.correction_authority_receipt_sha256
        != correction_authority.receipt_sha256
    ):
        raise MassiveDailyTradePartitionError(
            "security-partition authorities differ from scan"
        )
    partitions = tuple(sorted(security_partitions, key=lambda row: row.security_id))
    if not partitions:
        raise MassiveDailyTradePartitionError("security partitions are absent")
    for partition in partitions:
        partition.validate()
    partitioned = sum(row.source_row_count for row in partitions)
    if partitioned != scan_evidence.source_row_count:
        raise MassiveDailyTradePartitionError(
            "security partition rows differ from whole-file scan"
        )
    body: dict[str, object] = {
        "schema": MASSIVE_DAILY_TRADE_PARTITION_MANIFEST_V0_SCHEMA,
        "source_session_date": scan_evidence.source_session_date,
        "source_file_scan_receipt_sha256": scan_evidence.receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "feature_domain_spec_receipt_sha256": feature_domain_spec.receipt_sha256,
        "partition_spec_receipt_sha256": MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
        "partition_source_sha256": MASSIVE_DAILY_TRADE_PARTITION_SOURCE_SHA256,
        "security_partitions": partitions,
        "global_row_count": scan_evidence.source_row_count,
        "partitioned_row_count": partitioned,
        "rejected_row_count": 0,
        "global_row_inventory_sha256": (
            scan_evidence.all_row_provenance_inventory_sha256
        ),
        "global_partition_inventory_sha256": semantic_sha256(
            tuple(row.partition_receipt_sha256 for row in partitions)
        ),
    }
    provisional = MassiveDailyTradePartitionManifestV0(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveDailyTradePartitionManifestV0(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_DAILY_TRADE_PARTITION_MANIFEST_V0_SCHEMA",
    "MASSIVE_DAILY_TRADE_PARTITION_SOURCE_SHA256",
    "MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256",
    "MASSIVE_DAILY_TRADE_SECURITY_PARTITION_V0_SCHEMA",
    "MASSIVE_FINALIZED_FEATURE_DOMAIN_SPEC_V0_SCHEMA",
    "MassiveDailyTradePartitionError",
    "MassiveDailyTradePartitionManifestV0",
    "MassiveDailyTradeSecurityPartitionV0",
    "MassiveFinalizedFeatureDomainSpecV0",
    "build_massive_daily_trade_partition_manifest_v0",
    "build_massive_daily_trade_partition_manifest_from_security_partitions_v0",
    "build_massive_finalized_feature_domain_spec_v0",
]
