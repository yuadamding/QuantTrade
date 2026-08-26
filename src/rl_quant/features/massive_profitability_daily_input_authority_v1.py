"""Source-derived daily market inputs for the bounded Massive P0 lane.

This authority is the common authenticated input boundary for feature marks,
tape summaries, entry/exit fills, and target marks.  Every supplied V0 bar and
tape row is independently reconstructed from a whole-file-scanned,
semantically validated persisted partition.  Missing supported securities are
represented by an explicit zero-plus-false-mask row rather than disappearing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
import math
from pathlib import Path
from statistics import median
from typing import TypeVar, cast

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_daily_scan import (
    MassiveDailyTradeFileScanEvidenceV0,
    scan_massive_daily_trade_file_v0,
)
from rl_quant.data_sources.massive.finalized_listing import (
    coverage_session_from_massive_trade_key,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveDailyTradePartitionManifestV0,
)
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    load_massive_persisted_security_rows_v2,
    validate_massive_persisted_partitions_semantically_v2,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.data_sources.massive.trade_extraction import MassiveExtractedTradeRow
from rl_quant.features.massive_daily_bars_v0 import (
    MASSIVE_DAILY_BARS_V0_FIELDS,
    MassiveDailyBarsArtifactV0,
    MassiveDailyBarsRowV0,
    validate_massive_daily_bars_v0,
)
from rl_quant.features.massive_daily_tape_v0 import (
    MASSIVE_DAILY_TAPE_V0_FIELDS,
    MassiveDailyTapeArtifactV0,
    MassiveDailyTapeRowV0,
    validate_massive_daily_tape_v0,
)
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MassiveProfitabilityArchiveFreezeV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    MassiveProfitabilitySecuritySupportV2,
    massive_profitability_identity_semantic_receipt_v2,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    derive_massive_profitability_tape_populations_v2,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveProfitabilityProductionAcquisitionV2,
    validate_massive_profitability_production_acquisition_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-profitability-daily-input-authority-v1"
)
MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "scope": "archive-feature-base-through-latest-H63-endpoint",
        "source": "fixed-runtime-authenticated-Massive-object-GET",
        "scan": "whole-file-rescan",
        "partitions": "semantic-validation-and-independent-correction-replay",
        "bars": "independently-rederived-exact-V0-equality",
        "tape": "independently-rederived-exact-V0-equality",
        "signed_flow_population": (
            "terminal-active-regular-session-volume-forming-trades"
        ),
        "correction_population": "regular-session-economic-event-linked",
        "missing": "zero-plus-independent-false-mask",
        "performance_authorization": False,
    }
)

MASSIVE_PROFITABILITY_DAILY_INPUT_V1_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_DAILY_INPUT_V1_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_DAILY_INPUT_V1_LOCKBOX_ACCESS_AUTHORIZED = False


class MassiveProfitabilityDailyInputAuthorityV1Error(ValueError):
    """Daily input evidence differs from its authenticated source chain."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(name: str, value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            f"{name} must be finite"
        )
    return float(value)


def _nearest_rank(values: tuple[Decimal, ...], quantile: float) -> Decimal:
    ordered = tuple(sorted(values))
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _independent_bar_row(
    *,
    security_id: str,
    active_rows: Sequence[MassiveExtractedTradeRow],
    condition_authority: MassiveConditionAuthority,
) -> MassiveDailyBarsRowV0:
    ordered = tuple(
        sorted(
            active_rows,
            key=lambda row: (
                row.canonical_record.participant_timestamp_ns,
                row.canonical_record.sip_timestamp_ns,
                row.canonical_record.sequence_number,
                row.source_row_number,
            ),
        )
    )
    if not ordered:
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            "daily bar reconstruction received an empty active partition"
        )
    flags = tuple(
        condition_authority.resolve(row.canonical_record.conditions) for row in ordered
    )
    open_close = tuple(
        row for row, eligible in zip(ordered, flags, strict=True) if eligible[0]
    )
    high_low = tuple(
        row for row, eligible in zip(ordered, flags, strict=True) if eligible[1]
    )
    volume = tuple(
        row for row, eligible in zip(ordered, flags, strict=True) if eligible[2]
    )
    open_close_prices = tuple(
        Decimal(row.canonical_record.price_decimal) for row in open_close
    )
    high_low_prices = tuple(
        Decimal(row.canonical_record.price_decimal) for row in high_low
    )
    opening = open_close_prices[0] if open_close_prices else Decimal(0)
    closing = open_close_prices[-1] if open_close_prices else Decimal(0)
    high = max(high_low_prices) if high_low_prices else Decimal(0)
    low = min(high_low_prices) if high_low_prices else Decimal(0)
    shares = sum(
        (Decimal(row.canonical_record.size_decimal) for row in volume), Decimal(0)
    )
    dollars = sum(
        (
            Decimal(row.canonical_record.price_decimal)
            * Decimal(row.canonical_record.size_decimal)
            for row in volume
        ),
        Decimal(0),
    )
    combined = bool(open_close_prices and high_low_prices)
    high_low_range = (
        Decimal(0) if not combined or closing == 0 else (high - low) / closing
    )
    close_location = (
        Decimal(0)
        if not combined
        else Decimal("0.5")
        if high == low
        else (closing - low) / (high - low)
    )
    body: dict[str, object] = {
        "security_id": security_id,
        "values": tuple(
            float(value)
            for value in (
                opening,
                high,
                low,
                closing,
                shares,
                dollars,
                high_low_range,
                close_location,
            )
        ),
        "valid": (
            bool(open_close_prices),
            bool(high_low_prices),
            bool(high_low_prices),
            bool(open_close_prices),
            bool(volume),
            bool(volume),
            combined and closing != 0,
            combined,
        ),
        "source_active_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
    }
    result = MassiveDailyBarsRowV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _independent_tape_row(
    *,
    security_id: str,
    active_rows: Sequence[MassiveExtractedTradeRow],
    corrections: tuple[dict[str, object], ...],
    condition_authority: MassiveConditionAuthority,
) -> MassiveDailyTapeRowV0:
    ordered = tuple(
        sorted(
            active_rows,
            key=lambda row: (
                row.canonical_record.participant_timestamp_ns,
                row.canonical_record.sip_timestamp_ns,
                row.canonical_record.sequence_number,
                row.source_row_number,
            ),
        )
    )
    if not ordered:
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            "daily tape reconstruction received an empty active partition"
        )
    for row in ordered:
        condition_authority.resolve(row.canonical_record.conditions)
    prices = tuple(Decimal(row.canonical_record.price_decimal) for row in ordered)
    sizes = tuple(Decimal(row.canonical_record.size_decimal) for row in ordered)
    dollars = tuple(
        price * size for price, size in zip(prices, sizes, strict=True)
    )
    total = sum(dollars, Decimal(0))
    if total <= 0:
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            "daily tape dollar volume is nonpositive"
        )
    signs: list[int] = []
    prior_sign = 0
    for index, price in enumerate(prices):
        sign = (
            0
            if index == 0
            else 1
            if price > prices[index - 1]
            else -1
            if price < prices[index - 1]
            else prior_sign
        )
        signs.append(sign)
        if sign:
            prior_sign = sign
    signed = sum(
        (Decimal(sign) * value for sign, value in zip(signs, dollars, strict=True)),
        Decimal(0),
    )
    venue_dollars: defaultdict[int, Decimal] = defaultdict(Decimal)
    for row, value in zip(ordered, dollars, strict=True):
        venue_dollars[row.canonical_record.exchange_id] += value
    venue_shares = tuple(float(value / total) for value in venue_dollars.values())
    tapes = {
        tape: sum(
            value
            for row, value in zip(ordered, dollars, strict=True)
            if row.canonical_record.tape_id == tape
        )
        / total
        for tape in (1, 2, 3)
    }
    response = Decimal(0) if signed == 0 else (prices[-1] - prices[0]) / abs(signed)
    values = (
        math.log1p(len(ordered)),
        float(median(sizes)),
        float(_nearest_rank(sizes, 0.90)),
        float(sum(value for value in dollars if value >= Decimal("100000")) / total),
        float(signed),
        float(abs(signed) / total),
        float(response),
        float(
            sum(
                value
                for row, value in zip(ordered, dollars, strict=True)
                if row.canonical_record.trf_id is not None
            )
            / total
        ),
        -sum(value * math.log(value) for value in venue_shares if value > 0),
        max(venue_shares),
        float(tapes[1]),
        float(tapes[2]),
        float(tapes[3]),
        sum(bool(row.canonical_record.conditions) for row in ordered) / len(ordered),
        len(corrections) / len(ordered),
    )
    body: dict[str, object] = {
        "security_id": security_id,
        "values": values,
        "valid": (True,) * len(values),
        "source_active_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
        "source_correction_inventory_sha256": semantic_sha256(corrections),
    }
    result = MassiveDailyTapeRowV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDailySecurityInputV1:
    source_session_date: str
    security_id: str
    bars_values: tuple[float, ...]
    bars_valid: tuple[bool, ...]
    tape_values: tuple[float, ...]
    tape_valid: tuple[bool, ...]
    signed_dollar_flow: float
    same_population_dollar_volume: float
    absolute_signed_flow_imbalance: float
    same_population_valid: bool
    regular_session_event_count: int
    replacement_event_count: int
    cancellation_event_count: int
    late_report_event_count: int
    daily_bar_row_receipt_sha256: str | None
    daily_tape_row_receipt_sha256: str | None
    tape_population_row_receipt_sha256: str | None
    persisted_partition_receipt_sha256: str | None
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.source_session_date or not self.security_id:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily security input identity differs"
            )
        for values, masks, fields in (
            (self.bars_values, self.bars_valid, MASSIVE_DAILY_BARS_V0_FIELDS),
            (self.tape_values, self.tape_valid, MASSIVE_DAILY_TAPE_V0_FIELDS),
        ):
            if (
                len(values) != len(fields)
                or len(masks) != len(fields)
                or any(not isinstance(mask, bool) for mask in masks)
                or any(not math.isfinite(float(value)) for value in values)
                or any(value != 0.0 for value, valid in zip(values, masks) if not valid)
            ):
                raise MassiveProfitabilityDailyInputAuthorityV1Error(
                    "daily security values or masks differ"
                )
        if not isinstance(self.same_population_valid, bool):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "same-population validity is not Boolean"
            )
        signed = _finite("daily signed flow", self.signed_dollar_flow)
        volume = _finite(
            "daily same-population volume", self.same_population_dollar_volume
        )
        imbalance = _finite(
            "daily absolute signed-flow imbalance",
            self.absolute_signed_flow_imbalance,
        )
        if self.same_population_valid:
            if volume <= 0.0 or abs(signed) > volume + 1e-9 or not 0 <= imbalance <= 1:
                raise MassiveProfitabilityDailyInputAuthorityV1Error(
                    "same-population daily values differ"
                )
        elif any(value != 0.0 for value in (signed, volume, imbalance)):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "invalid same-population values are not zero placeholders"
            )
        counts = (
            self.regular_session_event_count,
            self.replacement_event_count,
            self.cancellation_event_count,
            self.late_report_event_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ) or sum(counts[1:]) > counts[0]:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily correction population differs"
            )
        observed = self.daily_bar_row_receipt_sha256 is not None
        if observed != (self.daily_tape_row_receipt_sha256 is not None):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily bar and tape observation support differs"
            )
        for value in (
            self.daily_bar_row_receipt_sha256,
            self.daily_tape_row_receipt_sha256,
            self.tape_population_row_receipt_sha256,
            self.persisted_partition_receipt_sha256,
        ):
            if value is not None:
                _digest("daily optional source receipt", value)
        if self.same_population_valid != (
            self.tape_population_row_receipt_sha256 is not None
        ):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "same-population receipt presence differs"
            )
        _digest("daily security row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily security row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDailyInputSessionV1:
    source_session_date: str
    regular_open_at_ms: int
    regular_close_at_ms: int
    vendor_last_modified_at_ms: int
    authenticated_download_receipt_sha256: str
    whole_file_scan_receipt_sha256: str
    semantic_partition_manifest_receipt_sha256: str
    persisted_partition_manifest_receipt_sha256: str
    daily_bars_artifact_receipt_sha256: str
    daily_tape_artifact_receipt_sha256: str
    supported_security_row_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not self.source_session_date
            or self.regular_open_at_ms < 0
            or self.regular_close_at_ms <= self.regular_open_at_ms
            or not self.regular_close_at_ms <= self.vendor_last_modified_at_ms
        ):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily input session chronology differs"
            )
        for name in (
            "authenticated_download_receipt_sha256",
            "whole_file_scan_receipt_sha256",
            "semantic_partition_manifest_receipt_sha256",
            "persisted_partition_manifest_receipt_sha256",
            "daily_bars_artifact_receipt_sha256",
            "daily_tape_artifact_receipt_sha256",
            "supported_security_row_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily input session receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDailyInputAuthorityV1:
    coverage_start_session_date: str
    coverage_end_session_date: str
    data_freeze_at_ms: int
    supported_security_ids: tuple[str, ...]
    sessions: tuple[MassiveProfitabilityDailyInputSessionV1, ...]
    rows: tuple[MassiveProfitabilityDailySecurityInputV1, ...]
    archive_freeze_semantic_receipt_sha256: str | None
    security_support_semantic_receipt_sha256: str | None
    session_authority_receipt_sha256: str
    normalized_identity_semantic_receipt_sha256: str
    condition_authority_receipt_sha256: str
    correction_authority_receipt_sha256: str
    event_domain_spec_receipt_sha256: str
    session_inventory_sha256: str
    row_inventory_sha256: str
    source_transport_qualified: bool
    daily_input_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    acquisition_audit_receipt_sha256: str
    audit_receipt_sha256: str
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "acquisition_audit_receipt_sha256",
                "audit_receipt_sha256",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256
            or self.coverage_end_session_date < self.coverage_start_session_date
            or self.data_freeze_at_ms < 0
            or not self.supported_security_ids
            or self.supported_security_ids
            != tuple(sorted(set(self.supported_security_ids)))
        ):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily input authority identity or support differs"
            )
        dates = tuple(row.source_session_date for row in self.sessions)
        if (
            not dates
            or dates != tuple(sorted(set(dates)))
            or dates[0] != self.coverage_start_session_date
            or dates[-1] != self.coverage_end_session_date
        ):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily input session inventory differs"
            )
        for session_row in self.sessions:
            session_row.validate()
            if session_row.vendor_last_modified_at_ms > self.data_freeze_at_ms:
                raise MassiveProfitabilityDailyInputAuthorityV1Error(
                    "daily source became available after the archive freeze"
                )
        keys = tuple((row.source_session_date, row.security_id) for row in self.rows)
        expected = tuple(
            (session_date, security_id)
            for session_date in dates
            for security_id in self.supported_security_ids
        )
        if keys != expected:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily inputs do not form the exact supported rectangle"
            )
        for security_row in self.rows:
            security_row.validate()
        if self.session_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.sessions)
        ) or self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily input authority inventory differs"
            )
        if (
            not isinstance(self.source_transport_qualified, bool)
            or not isinstance(self.daily_input_data_qualified, bool)
            or self.daily_input_data_qualified and not self.source_transport_qualified
            or any(
                (
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily input qualification or authorization differs"
            )
        for value in (
            self.archive_freeze_semantic_receipt_sha256,
            self.security_support_semantic_receipt_sha256,
        ):
            if value is not None:
                _digest("daily frozen component", value)
        for name in (
            "session_authority_receipt_sha256",
            "normalized_identity_semantic_receipt_sha256",
            "condition_authority_receipt_sha256",
            "correction_authority_receipt_sha256",
            "event_domain_spec_receipt_sha256",
            "session_inventory_sha256",
            "row_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "acquisition_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily input semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "acquisition_audit_receipt_sha256": (
                    self.acquisition_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily input audit receipt differs"
            )

    def row(self, *, session_date: str, security_id: str) -> MassiveProfitabilityDailySecurityInputV1:
        for value in self.rows:
            if value.source_session_date == session_date and value.security_id == security_id:
                return value
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            "requested daily input row is outside the frozen rectangle"
        )


_T = TypeVar("_T")


def _unique_by_date(values: Sequence[_T], *, field: str, name: str) -> dict[str, _T]:
    result: dict[str, _T] = {}
    for value in values:
        session_date = cast(str, getattr(value, field))
        if session_date in result:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                f"{name} duplicates source session {session_date}"
            )
        result[session_date] = value
    return result


def _session_dates_between(
    *,
    session_authority: MassiveSessionAuthority,
    start: str,
    end: str,
) -> tuple[str, ...]:
    dates = tuple(
        row.session_date
        for row in session_authority.sessions
        if start <= row.session_date <= end
    )
    if not dates or dates[0] != start or dates[-1] != end:
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            "daily input scope is not exhausted by the session authority"
        )
    return dates


def _build_massive_profitability_daily_input_authority_v1(
    *,
    source_root: str | Path,
    persisted_root: str | Path,
    daily_bars_root: str | Path,
    daily_tape_root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    required_session_dates: Sequence[str],
    supported_security_ids: Sequence[str],
    data_freeze_at_ms: int,
    scan_evidence: Sequence[MassiveDailyTradeFileScanEvidenceV0],
    semantic_partition_manifests: Sequence[MassiveDailyTradePartitionManifestV0],
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    daily_tape: Sequence[MassiveDailyTapeArtifactV0],
    archive_freeze_semantic_receipt_sha256: str | None,
    security_support_semantic_receipt_sha256: str | None,
    require_fixed_runtime: bool,
) -> MassiveProfitabilityDailyInputAuthorityV1:
    session_authority.validate()
    identity_authority.validate()
    condition_authority.validate()
    correction_authority.validate()
    validate_massive_profitability_production_acquisition_v2(
        root=source_root,
        acquisition=acquisition,
        require_fixed_runtime=require_fixed_runtime,
    )
    required_dates = tuple(required_session_dates)
    support = tuple(sorted(set(supported_security_ids)))
    if (
        not required_dates
        or required_dates != tuple(sorted(set(required_dates)))
        or not support
        or not set(support)
        <= {row.security_id for row in identity_authority.security_master}
    ):
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            "daily input requested scope or support differs"
        )
    authority_dates = _session_dates_between(
        session_authority=session_authority,
        start=required_dates[0],
        end=required_dates[-1],
    )
    if authority_dates != required_dates:
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            "daily input sessions are not one consecutive XNYS interval"
        )
    downloads = {
        coverage_session_from_massive_trade_key(row.source_object_key): row
        for row in acquisition.authenticated_downloads
    }
    if len(downloads) != len(acquisition.authenticated_downloads):
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            "authenticated daily downloads duplicate a source session"
        )
    scans = _unique_by_date(
        scan_evidence, field="source_session_date", name="whole-file scan"
    )
    semantic_manifests = _unique_by_date(
        semantic_partition_manifests,
        field="source_session_date",
        name="semantic partition manifest",
    )
    persisted_manifests = _unique_by_date(
        persisted_partition_manifests,
        field="source_session_date",
        name="persisted partition manifest",
    )
    bars = _unique_by_date(daily_bars, field="source_session_date", name="daily bars")
    tapes = _unique_by_date(daily_tape, field="source_session_date", name="daily tape")
    for name, mapping in (
        ("authenticated downloads", downloads),
        ("whole-file scans", scans),
        ("semantic partition manifests", semantic_manifests),
        ("persisted partition manifests", persisted_manifests),
        ("daily bars", bars),
        ("daily tape", tapes),
    ):
        if not set(required_dates) <= set(mapping):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                f"{name} do not cover the complete requested interval"
            )
    listing_by_receipt = {
        row.acquisition_evidence.receipt_sha256: row
        for row in acquisition.captured_listings
    }
    sessions: list[MassiveProfitabilityDailyInputSessionV1] = []
    result_rows: list[MassiveProfitabilityDailySecurityInputV1] = []
    common_event_domain: str | None = None
    for source_date in required_dates:
        download = downloads[source_date]
        scan = scans[source_date]
        semantic_manifest = semantic_manifests[source_date]
        persisted_manifest = persisted_manifests[source_date]
        bar_artifact = bars[source_date]
        tape_artifact = tapes[source_date]
        session = session_authority.resolve(exchange="XNYS", session_date=source_date)
        captured = listing_by_receipt.get(
            download.listing_acquisition_receipt_sha256
        )
        if captured is None:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily download lacks its captured listing"
            )
        listing_entry = captured.committed_listing.resolve(
            source_object_key=download.source_object_key
        )
        close_ms = session.regular_close_ns // 1_000_000
        if not close_ms <= listing_entry.vendor_last_modified_at_ms <= data_freeze_at_ms:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily source availability lies outside session close and archive freeze"
            )
        rescanned_rows, rescanned = scan_massive_daily_trade_file_v0(
            root=source_root,
            loaded_source=download.loaded_source,
            session_authority=session_authority,
            session=session,
            correction_authority=correction_authority,
        )
        if rescanned != scan or len(rescanned_rows) != scan.source_row_count:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily whole-file scan differs from the authenticated download"
            )
        if (
            scan.loaded_source_receipt_sha256 != download.loaded_source.receipt_sha256
            or semantic_manifest.source_file_scan_receipt_sha256
            != scan.receipt_sha256
            or semantic_manifest.identity_authority_receipt_sha256
            != identity_authority.receipt_sha256
            or semantic_manifest.condition_authority_receipt_sha256
            != condition_authority.receipt_sha256
            or semantic_manifest.correction_authority_receipt_sha256
            != correction_authority.receipt_sha256
        ):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily semantic partition authorities differ"
            )
        event_domain = semantic_manifest.feature_domain_spec_receipt_sha256
        if common_event_domain is None:
            common_event_domain = event_domain
        elif common_event_domain != event_domain:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily sessions use different event-domain authorities"
            )
        validate_massive_persisted_partitions_semantically_v2(
            root=persisted_root,
            manifest=persisted_manifest,
            scan_evidence=scan,
            semantic_partition_manifest=semantic_manifest,
            identity_authority=identity_authority,
            correction_authority=correction_authority,
        )
        validate_massive_daily_bars_v0(root=daily_bars_root, artifact=bar_artifact)
        validate_massive_daily_tape_v0(root=daily_tape_root, artifact=tape_artifact)
        if (
            bar_artifact.persisted_partition_manifest_receipt_sha256
            != persisted_manifest.receipt_sha256
            or tape_artifact.persisted_partition_manifest_receipt_sha256
            != persisted_manifest.receipt_sha256
            or bar_artifact.condition_authority_receipt_sha256
            != condition_authority.receipt_sha256
            or tape_artifact.condition_authority_receipt_sha256
            != condition_authority.receipt_sha256
        ):
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily bar or tape artifact is detached from qualified partitions"
            )
        rederived_bars: list[MassiveDailyBarsRowV0] = []
        rederived_tapes: list[MassiveDailyTapeRowV0] = []
        partition_by_security = {
            row.security_id: row for row in persisted_manifest.partitions
        }
        for partition in persisted_manifest.partitions:
            _, active, corrections = load_massive_persisted_security_rows_v2(
                root=persisted_root, partition=partition
            )
            if not active:
                continue
            rederived_bars.append(
                _independent_bar_row(
                    security_id=partition.security_id,
                    active_rows=active,
                    condition_authority=condition_authority,
                )
            )
            rederived_tapes.append(
                _independent_tape_row(
                    security_id=partition.security_id,
                    active_rows=active,
                    corrections=corrections,
                    condition_authority=condition_authority,
                )
            )
        ordered_bars = tuple(sorted(rederived_bars, key=lambda row: row.security_id))
        ordered_tapes = tuple(sorted(rederived_tapes, key=lambda row: row.security_id))
        if ordered_bars != bar_artifact.rows:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily bars differ from authenticated partition rederivation"
            )
        if ordered_tapes != tape_artifact.rows:
            raise MassiveProfitabilityDailyInputAuthorityV1Error(
                "daily tape differs from authenticated partition rederivation"
            )
        population_rows = derive_massive_profitability_tape_populations_v2(
            persisted_root=persisted_root,
            manifest=persisted_manifest,
            session_authority=session_authority,
            condition_authority=condition_authority,
        )
        bar_by_security = {row.security_id: row for row in ordered_bars}
        tape_by_security = {row.security_id: row for row in ordered_tapes}
        population_by_security = {row.security_id: row for row in population_rows}
        session_security_rows: list[MassiveProfitabilityDailySecurityInputV1] = []
        for security_id in support:
            bar = bar_by_security.get(security_id)
            tape = tape_by_security.get(security_id)
            population = population_by_security.get(security_id)
            if (bar is None) != (tape is None):
                raise MassiveProfitabilityDailyInputAuthorityV1Error(
                    "daily bar and tape reconstructed support differs"
                )
            partition = partition_by_security.get(security_id)
            body: dict[str, object] = {
                "source_session_date": source_date,
                "security_id": security_id,
                "bars_values": (
                    (0.0,) * len(MASSIVE_DAILY_BARS_V0_FIELDS)
                    if bar is None
                    else bar.values
                ),
                "bars_valid": (
                    (False,) * len(MASSIVE_DAILY_BARS_V0_FIELDS)
                    if bar is None
                    else bar.valid
                ),
                "tape_values": (
                    (0.0,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS)
                    if tape is None
                    else tape.values
                ),
                "tape_valid": (
                    (False,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS)
                    if tape is None
                    else tape.valid
                ),
                "signed_dollar_flow": (
                    0.0 if population is None else population.signed_dollar_flow
                ),
                "same_population_dollar_volume": (
                    0.0 if population is None else population.dollar_volume
                ),
                "absolute_signed_flow_imbalance": (
                    0.0
                    if population is None
                    else population.absolute_signed_flow_imbalance
                ),
                "same_population_valid": population is not None,
                "regular_session_event_count": (
                    0
                    if population is None
                    else population.regular_session_event_count
                ),
                "replacement_event_count": (
                    0 if population is None else population.replacement_event_count
                ),
                "cancellation_event_count": (
                    0 if population is None else population.cancellation_event_count
                ),
                "late_report_event_count": (
                    0 if population is None else population.late_report_event_count
                ),
                "daily_bar_row_receipt_sha256": (
                    None if bar is None else bar.receipt_sha256
                ),
                "daily_tape_row_receipt_sha256": (
                    None if tape is None else tape.receipt_sha256
                ),
                "tape_population_row_receipt_sha256": (
                    None if population is None else population.receipt_sha256
                ),
                "persisted_partition_receipt_sha256": (
                    None if partition is None else partition.receipt_sha256
                ),
            }
            security_result = MassiveProfitabilityDailySecurityInputV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
            security_result.validate()
            session_security_rows.append(security_result)
            result_rows.append(security_result)
        session_inventory = semantic_sha256(
            tuple(row.receipt_sha256 for row in session_security_rows)
        )
        session_body: dict[str, object] = {
            "source_session_date": source_date,
            "regular_open_at_ms": session.regular_open_ns // 1_000_000,
            "regular_close_at_ms": close_ms,
            "vendor_last_modified_at_ms": listing_entry.vendor_last_modified_at_ms,
            "authenticated_download_receipt_sha256": download.receipt_sha256,
            "whole_file_scan_receipt_sha256": scan.receipt_sha256,
            "semantic_partition_manifest_receipt_sha256": (
                semantic_manifest.receipt_sha256
            ),
            "persisted_partition_manifest_receipt_sha256": (
                persisted_manifest.receipt_sha256
            ),
            "daily_bars_artifact_receipt_sha256": bar_artifact.receipt_sha256,
            "daily_tape_artifact_receipt_sha256": tape_artifact.receipt_sha256,
            "supported_security_row_inventory_sha256": session_inventory,
        }
        result_session = MassiveProfitabilityDailyInputSessionV1(
            source_session_date=source_date,
            regular_open_at_ms=session.regular_open_ns // 1_000_000,
            regular_close_at_ms=close_ms,
            vendor_last_modified_at_ms=listing_entry.vendor_last_modified_at_ms,
            authenticated_download_receipt_sha256=download.receipt_sha256,
            whole_file_scan_receipt_sha256=scan.receipt_sha256,
            semantic_partition_manifest_receipt_sha256=(
                semantic_manifest.receipt_sha256
            ),
            persisted_partition_manifest_receipt_sha256=(
                persisted_manifest.receipt_sha256
            ),
            daily_bars_artifact_receipt_sha256=bar_artifact.receipt_sha256,
            daily_tape_artifact_receipt_sha256=tape_artifact.receipt_sha256,
            supported_security_row_inventory_sha256=session_inventory,
            receipt_sha256=semantic_sha256(session_body),
        )
        result_session.validate()
        sessions.append(result_session)
    assert common_event_domain is not None
    source_transport = require_fixed_runtime and acquisition.fixed_runtime_captured
    semantic_body: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
        "coverage_start_session_date": required_dates[0],
        "coverage_end_session_date": required_dates[-1],
        "data_freeze_at_ms": data_freeze_at_ms,
        "supported_security_ids": support,
        "sessions": tuple(asdict(row) for row in sessions),
        "rows": tuple(asdict(row) for row in result_rows),
        "archive_freeze_semantic_receipt_sha256": (
            archive_freeze_semantic_receipt_sha256
        ),
        "security_support_semantic_receipt_sha256": (
            security_support_semantic_receipt_sha256
        ),
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "normalized_identity_semantic_receipt_sha256": (
            massive_profitability_identity_semantic_receipt_v2(identity_authority)
        ),
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "correction_authority_receipt_sha256": correction_authority.receipt_sha256,
        "event_domain_spec_receipt_sha256": common_event_domain,
        "session_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in sessions)
        ),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in result_rows)
        ),
        "source_transport_qualified": source_transport,
        "daily_input_data_qualified": source_transport,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256
        ),
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic_body)
    acquisition_audit = acquisition.receipt_sha256
    authority_result = MassiveProfitabilityDailyInputAuthorityV1(
        coverage_start_session_date=required_dates[0],
        coverage_end_session_date=required_dates[-1],
        data_freeze_at_ms=data_freeze_at_ms,
        supported_security_ids=support,
        sessions=tuple(sessions),
        rows=tuple(result_rows),
        archive_freeze_semantic_receipt_sha256=archive_freeze_semantic_receipt_sha256,
        security_support_semantic_receipt_sha256=(
            security_support_semantic_receipt_sha256
        ),
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        normalized_identity_semantic_receipt_sha256=semantic_body[
            "normalized_identity_semantic_receipt_sha256"
        ],  # type: ignore[arg-type]
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        correction_authority_receipt_sha256=correction_authority.receipt_sha256,
        event_domain_spec_receipt_sha256=common_event_domain,
        session_inventory_sha256=semantic_body["session_inventory_sha256"],  # type: ignore[arg-type]
        row_inventory_sha256=semantic_body["row_inventory_sha256"],  # type: ignore[arg-type]
        source_transport_qualified=source_transport,
        daily_input_data_qualified=source_transport,
        protocol_receipt_sha256=MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        specification_sha256=MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SOURCE_SHA256,
        semantic_receipt_sha256=semantic_receipt,
        acquisition_audit_receipt_sha256=acquisition_audit,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "acquisition_audit_receipt_sha256": acquisition_audit,
            }
        ),
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    authority_result.validate()
    return authority_result


def build_massive_profitability_daily_input_authority_v1(
    *,
    source_root: str | Path,
    persisted_root: str | Path,
    daily_bars_root: str | Path,
    daily_tape_root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    security_support: MassiveProfitabilitySecuritySupportV2,
    scan_evidence: Sequence[MassiveDailyTradeFileScanEvidenceV0],
    semantic_partition_manifests: Sequence[MassiveDailyTradePartitionManifestV0],
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    daily_tape: Sequence[MassiveDailyTapeArtifactV0],
) -> MassiveProfitabilityDailyInputAuthorityV1:
    """Build the fixed-runtime archive-wide daily-input authority."""

    archive_freeze.validate()
    security_support.validate()
    dates = _session_dates_between(
        session_authority=session_authority,
        start=archive_freeze.earliest_feature_base_session_date,
        end=archive_freeze.latest_h63_endpoint_session_date,
    )
    if (
        security_support.normalized_identity_semantic_receipt_sha256
        != massive_profitability_identity_semantic_receipt_v2(identity_authority)
    ):
        raise MassiveProfitabilityDailyInputAuthorityV1Error(
            "daily input identity differs from frozen experiment support"
        )
    return _build_massive_profitability_daily_input_authority_v1(
        source_root=source_root,
        persisted_root=persisted_root,
        daily_bars_root=daily_bars_root,
        daily_tape_root=daily_tape_root,
        session_authority=session_authority,
        identity_authority=identity_authority,
        condition_authority=condition_authority,
        correction_authority=correction_authority,
        acquisition=acquisition,
        required_session_dates=dates,
        supported_security_ids=security_support.all_supported_security_ids,
        data_freeze_at_ms=archive_freeze.data_freeze_at_ms,
        scan_evidence=scan_evidence,
        semantic_partition_manifests=semantic_partition_manifests,
        persisted_partition_manifests=persisted_partition_manifests,
        daily_bars=daily_bars,
        daily_tape=daily_tape,
        archive_freeze_semantic_receipt_sha256=(
            archive_freeze.semantic_receipt_sha256
        ),
        security_support_semantic_receipt_sha256=(
            security_support.semantic_receipt_sha256
        ),
        require_fixed_runtime=True,
    )


def build_massive_profitability_daily_input_authority_for_test_v1(
    *,
    source_root: str | Path,
    persisted_root: str | Path,
    daily_bars_root: str | Path,
    daily_tape_root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    required_session_dates: Sequence[str],
    supported_security_ids: Sequence[str],
    data_freeze_at_ms: int,
    scan_evidence: Sequence[MassiveDailyTradeFileScanEvidenceV0],
    semantic_partition_manifests: Sequence[MassiveDailyTradePartitionManifestV0],
    persisted_partition_manifests: Sequence[MassivePersistedPartitionManifestV1],
    daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    daily_tape: Sequence[MassiveDailyTapeArtifactV0],
) -> MassiveProfitabilityDailyInputAuthorityV1:
    """Exercise exact source reconstruction without authorizing real data."""

    return _build_massive_profitability_daily_input_authority_v1(
        source_root=source_root,
        persisted_root=persisted_root,
        daily_bars_root=daily_bars_root,
        daily_tape_root=daily_tape_root,
        session_authority=session_authority,
        identity_authority=identity_authority,
        condition_authority=condition_authority,
        correction_authority=correction_authority,
        acquisition=acquisition,
        required_session_dates=required_session_dates,
        supported_security_ids=supported_security_ids,
        data_freeze_at_ms=data_freeze_at_ms,
        scan_evidence=scan_evidence,
        semantic_partition_manifests=semantic_partition_manifests,
        persisted_partition_manifests=persisted_partition_manifests,
        daily_bars=daily_bars,
        daily_tape=daily_tape,
        archive_freeze_semantic_receipt_sha256=None,
        security_support_semantic_receipt_sha256=None,
        require_fixed_runtime=False,
    )


__all__ = [
    "MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256",
    "MASSIVE_PROFITABILITY_DAILY_INPUT_V1_LOCKBOX_ACCESS_AUTHORIZED",
    "MASSIVE_PROFITABILITY_DAILY_INPUT_V1_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_DAILY_INPUT_V1_PROFITABILITY_REPORTING_AUTHORIZED",
    "MassiveProfitabilityDailyInputAuthorityV1",
    "MassiveProfitabilityDailyInputAuthorityV1Error",
    "MassiveProfitabilityDailyInputSessionV1",
    "MassiveProfitabilityDailySecurityInputV1",
    "build_massive_profitability_daily_input_authority_for_test_v1",
    "build_massive_profitability_daily_input_authority_v1",
]
