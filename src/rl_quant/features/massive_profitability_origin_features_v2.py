"""Bounded P0 origin features backed by holding-safe feature accounting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from statistics import fmean, pstdev

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    load_massive_persisted_security_rows_v2,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_daily_tape_v0 import MASSIVE_DAILY_TAPE_V0_FIELDS
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    reject_massive_profitability_legacy_generation_v2,
)
from rl_quant.features.massive_profitability_feature_accounting_v1 import (
    MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA,
    MassiveProfitabilityFeatureAccountingV1,
    MassiveProfitabilityFeatureEconomicValueRowV1,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MassiveProfitabilityDecisionOriginV1,
)
from rl_quant.features.massive_session_panel_v1 import (
    MASSIVE_SESSION_PANEL_V1_SCHEMA,
    MassiveSessionPanelRowV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

BARS_MIN_V2_FIELDS = (
    "economic_total_return_1",
    "economic_total_return_5",
    "economic_total_return_21",
    "economic_total_return_63",
    "reversal_1",
    "reversal_5",
    "trend_21_minus_5",
    "trend_63_minus_21",
    "realized_volatility_5",
    "realized_volatility_21",
    "downside_volatility_21",
    "high_low_range",
    "close_location",
    "log_dollar_volume",
    "dollar_volume_surprise_21",
    "amihud_21",
    "listing_age_sessions",
    "listing_age_left_censored",
    "valid_history_fraction_63",
)
TAPE_MIN_V2_FIELDS = (
    "log_trade_count",
    "median_trade_size",
    "p90_trade_size",
    "large_trade_dollar_fraction",
    "signed_dollar_flow_fraction",
    "absolute_signed_flow_imbalance",
    "trf_off_exchange_dollar_fraction",
    "venue_entropy",
    "largest_venue_share",
    "tape_a_fraction",
    "tape_b_fraction",
    "tape_c_fraction",
    "replacement_event_fraction",
    "cancellation_event_fraction",
    "late_report_event_fraction",
)
MASSIVE_PROFITABILITY_TAPE_POPULATION_V2_SCHEMA = (
    "rl-quant.massive-profitability-tape-population-v2"
)
MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA = (
    "rl-quant.massive-profitability-origin-features-v2"
)
MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "support": "decision-time-PIT-members",
        "history": "exact-64-XNYS-sessions-ending-at-source",
        "source_staleness_sessions": 2,
        "bars_fields": BARS_MIN_V2_FIELDS,
        "tape_fields": TAPE_MIN_V2_FIELDS,
        "signed_flow_population": (
            "terminal-active-regular-session-volume-forming-trades"
        ),
        "correction_population": (
            "events-linked-to-regular-session-economic-trades-including-after-close-corrections"
        ),
        "missing": "zero-plus-independent-false-mask",
        "economic_input": MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA,
        "legacy_economic_index": "prohibited",
        "corporate_action_predictors": "prohibited",
        "performance_authorization": False,
    }
)

MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_LOCKBOX_ACCESS_AUTHORIZED = False


class MassiveProfitabilityOriginFeaturesV2Error(ValueError):
    """A bounded origin feature cross-section differs from P0."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityOriginFeaturesV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(name: str, value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise MassiveProfitabilityOriginFeaturesV2Error(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTapePopulationRowV2:
    source_session_date: str
    security_id: str
    signed_dollar_flow: float
    dollar_volume: float
    absolute_signed_flow_imbalance: float
    regular_session_event_count: int
    replacement_event_count: int
    cancellation_event_count: int
    late_report_event_count: int
    population_receipt_sha256: str
    source_data_qualified: bool
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_TAPE_POPULATION_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_TAPE_POPULATION_V2_SCHEMA
            or not self.source_session_date
            or not self.security_id
            or not isinstance(self.source_data_qualified, bool)
        ):
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "tape population identity differs"
            )
        signed = _finite("signed dollar flow", self.signed_dollar_flow)
        volume = _finite("same-population dollar volume", self.dollar_volume)
        imbalance = _finite(
            "same-population absolute imbalance", self.absolute_signed_flow_imbalance
        )
        if volume <= 0.0 or abs(signed) > volume + 1e-9 or not 0.0 <= imbalance <= 1.0:
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "same-population signed-flow values differ"
            )
        counts = (
            self.regular_session_event_count,
            self.replacement_event_count,
            self.cancellation_event_count,
            self.late_report_event_count,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            )
            or sum(counts[1:]) > counts[0]
        ):
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "regular-session-linked correction counts differ"
            )
        _digest("tape population", self.population_receipt_sha256)
        _digest("tape population row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "tape population row receipt differs"
            )

    @classmethod
    def build(
        cls,
        *,
        source_session_date: str,
        security_id: str,
        signed_dollar_flow: float,
        dollar_volume: float,
        regular_session_event_count: int,
        replacement_event_count: int,
        cancellation_event_count: int,
        late_report_event_count: int,
        population_receipt_sha256: str,
    ) -> MassiveProfitabilityTapePopulationRowV2:
        volume = float(dollar_volume)
        signed = float(signed_dollar_flow)
        body = {
            "source_session_date": source_session_date,
            "security_id": security_id,
            "signed_dollar_flow": signed,
            "dollar_volume": volume,
            "absolute_signed_flow_imbalance": abs(signed) / volume,
            "regular_session_event_count": regular_session_event_count,
            "replacement_event_count": replacement_event_count,
            "cancellation_event_count": cancellation_event_count,
            "late_report_event_count": late_report_event_count,
            "population_receipt_sha256": population_receipt_sha256,
            "source_data_qualified": False,
            "schema": MASSIVE_PROFITABILITY_TAPE_POPULATION_V2_SCHEMA,
        }
        row = cls(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        return row


def derive_massive_profitability_tape_populations_v2(
    *,
    persisted_root: str | Path,
    manifest: MassivePersistedPartitionManifestV1,
    session_authority: MassiveSessionAuthority,
    condition_authority: MassiveConditionAuthority,
) -> tuple[MassiveProfitabilityTapePopulationRowV2, ...]:
    """Rederive same-population flow and correction support from committed rows."""

    manifest.validate()
    session_authority.validate()
    condition_authority.validate()
    session = session_authority.resolve(
        exchange="XNYS", session_date=manifest.source_session_date
    )
    rows: list[MassiveProfitabilityTapePopulationRowV2] = []
    for partition in manifest.partitions:
        events, active, corrections = load_massive_persisted_security_rows_v2(
            root=persisted_root, partition=partition
        )
        volume_rows = tuple(
            sorted(
                (
                    row
                    for row in active
                    if condition_authority.resolve(row.canonical_record.conditions)[2]
                ),
                key=lambda row: (
                    row.canonical_record.participant_timestamp_ns,
                    row.canonical_record.sip_timestamp_ns,
                    row.canonical_record.sequence_number,
                    row.source_row_number,
                ),
            )
        )
        if not volume_rows:
            continue
        dollars = tuple(
            Decimal(row.canonical_record.price_decimal)
            * Decimal(row.canonical_record.size_decimal)
            for row in volume_rows
        )
        total = sum(dollars, Decimal(0))
        if total <= 0:
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "volume-forming tape population has nonpositive notional"
            )
        signs: list[int] = []
        prior_sign = 0
        prices = tuple(
            Decimal(row.canonical_record.price_decimal) for row in volume_rows
        )
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
        event_by_source = {row.source_row_number: row for row in events}
        regular_events = tuple(
            row
            for row in events
            if session.regular_open_ns
            <= row.canonical_record.participant_timestamp_ns
            < session.regular_close_ns
        )
        correction_counts = {
            "replacement": 0,
            "cancellation": 0,
            "late-report": 0,
        }
        for correction in corrections:
            source_row = correction["source_row_number"]
            linked = event_by_source.get(source_row)
            if linked is None:
                raise MassiveProfitabilityOriginFeaturesV2Error(
                    "correction is not linked to its committed event"
                )
            if not (
                session.regular_open_ns
                <= linked.canonical_record.participant_timestamp_ns
                < session.regular_close_ns
            ):
                continue
            kind = correction["correction_kind"]
            if kind not in correction_counts:
                raise MassiveProfitabilityOriginFeaturesV2Error(
                    "regular-session correction kind is unsupported"
                )
            correction_counts[kind] += 1
        body = {
            "source_session_date": manifest.source_session_date,
            "security_id": partition.security_id,
            "signed_dollar_flow": float(signed),
            "dollar_volume": float(total),
            "absolute_signed_flow_imbalance": float(abs(signed) / total),
            "regular_session_event_count": len(regular_events),
            "replacement_event_count": correction_counts["replacement"],
            "cancellation_event_count": correction_counts["cancellation"],
            "late_report_event_count": correction_counts["late-report"],
            "population_receipt_sha256": semantic_sha256(
                {
                    "partition_receipt_sha256": partition.receipt_sha256,
                    "condition_authority_receipt_sha256": (
                        condition_authority.receipt_sha256
                    ),
                    "volume_forming_trade_receipts": tuple(
                        row.receipt_sha256 for row in volume_rows
                    ),
                    "regular_event_receipts": tuple(
                        row.receipt_sha256 for row in regular_events
                    ),
                    "correction_inventory": corrections,
                }
            ),
            "source_data_qualified": True,
            "schema": MASSIVE_PROFITABILITY_TAPE_POPULATION_V2_SCHEMA,
        }
        result = MassiveProfitabilityTapePopulationRowV2(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        result.validate()
        rows.append(result)
    return tuple(sorted(rows, key=lambda row: row.security_id))


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityOriginFeatureRowV2:
    decision_session_date: str
    source_session_date: str
    security_id: str
    decision_membership_rank: int
    source_staleness_sessions: int
    source_listed: bool
    source_tradable: bool
    source_observed_regular_trade: bool
    source_halt_or_no_print: bool
    bars_values: tuple[float, ...]
    bars_valid: tuple[bool, ...]
    tape_values: tuple[float, ...]
    tape_valid: tuple[bool, ...]
    source_panel_row_receipt_sha256: str
    feature_accounting_security_inventory_sha256: str
    tape_population_row_receipt_sha256: str | None
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not self.decision_session_date
            or not self.source_session_date
            or not self.security_id
            or isinstance(self.decision_membership_rank, bool)
            or not isinstance(self.decision_membership_rank, int)
            or self.decision_membership_rank <= 0
            or self.source_staleness_sessions != 2
        ):
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "origin feature row identity differs"
            )
        for flag in (
            self.source_listed,
            self.source_tradable,
            self.source_observed_regular_trade,
            self.source_halt_or_no_print,
        ):
            if not isinstance(flag, bool):
                raise MassiveProfitabilityOriginFeaturesV2Error(
                    "origin feature source state is not Boolean"
                )
        for values, masks, fields in (
            (self.bars_values, self.bars_valid, BARS_MIN_V2_FIELDS),
            (self.tape_values, self.tape_valid, TAPE_MIN_V2_FIELDS),
        ):
            if (
                len(values) != len(fields)
                or len(masks) != len(fields)
                or any(not isinstance(value, bool) for value in masks)
                or any(not math.isfinite(float(value)) for value in values)
                or any(value != 0.0 for value, valid in zip(values, masks) if not valid)
            ):
                raise MassiveProfitabilityOriginFeaturesV2Error(
                    "origin feature values or masks differ"
                )
        for value in (
            self.source_panel_row_receipt_sha256,
            self.feature_accounting_security_inventory_sha256,
            self.receipt_sha256,
        ):
            _digest("origin feature row digest", value)
        if self.tape_population_row_receipt_sha256 is not None:
            _digest(
                "origin tape population row", self.tape_population_row_receipt_sha256
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "origin feature row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityOriginFeaturesV2:
    origin_receipt_sha256: str
    decision_session_date: str
    source_session_date: str
    feature_cutoff_at_ms: int
    source_staleness_sessions: int
    input_session_dates: tuple[str, ...]
    rows: tuple[MassiveProfitabilityOriginFeatureRowV2, ...]
    origin_membership_receipt_sha256: str
    feature_accounting_semantic_receipt_sha256: str
    session_panel_receipt_sha256: str
    source_input_inventory_sha256: str
    row_inventory_sha256: str
    input_schemas: tuple[str, ...]
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    tape_population_data_qualified: bool
    feature_accounting_data_qualified: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "audit_receipt_sha256"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA
            or self.source_staleness_sessions != 2
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SOURCE_SHA256
            or len(self.input_session_dates) != 64
            or self.input_session_dates != tuple(sorted(set(self.input_session_dates)))
            or self.input_session_dates[-1] != self.source_session_date
            or any(
                (
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
            or not isinstance(self.tape_population_data_qualified, bool)
            or not isinstance(self.feature_accounting_data_qualified, bool)
        ):
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "origin feature artifact identity or authorization differs"
            )
        for input_schema in self.input_schemas:
            reject_massive_profitability_legacy_generation_v2(input_schema)
        if self.input_schemas != tuple(sorted(set(self.input_schemas))) or set(
            self.input_schemas
        ) != {
            MASSIVE_SESSION_PANEL_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA,
        }:
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "origin feature input generations differ"
            )
        keys = tuple(row.security_id for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "origin feature cross-section is not canonical"
            )
        for row in self.rows:
            row.validate()
            if (
                row.decision_session_date != self.decision_session_date
                or row.source_session_date != self.source_session_date
            ):
                raise MassiveProfitabilityOriginFeaturesV2Error(
                    "origin feature row chronology differs"
                )
        for value in (
            self.origin_receipt_sha256,
            self.origin_membership_receipt_sha256,
            self.feature_accounting_semantic_receipt_sha256,
            self.session_panel_receipt_sha256,
            self.source_input_inventory_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("origin feature artifact digest", value)
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "origin feature semantic receipt differs"
            )


def _daily_value(
    row: MassiveSessionPanelRowV1, name: str, *, tape: bool = False
) -> tuple[float, bool]:
    fields = MASSIVE_DAILY_TAPE_V0_FIELDS if tape else MASSIVE_DAILY_BARS_V0_FIELDS
    index = fields.index(name)
    values = row.tape_values if tape else row.bars_values
    valid = row.tape_valid if tape else row.bars_valid
    return float(values[index]), valid[index]


def _economic_return(
    rows: dict[int, MassiveProfitabilityFeatureEconomicValueRowV1],
    end_offset: int,
    horizon: int,
) -> tuple[float, bool]:
    start = rows.get(end_offset - horizon)
    end = rows.get(end_offset)
    if (
        start is None
        or end is None
        or not start.valid
        or not end.valid
        or start.economic_value <= 0.0
    ):
        return 0.0, False
    return end.economic_value / start.economic_value - 1.0, True


def _listing_age(
    *,
    security_id: str,
    source_session_date: str,
    identity_authority: PITSecurityUniverseAuthority,
    session_authority: MassiveSessionAuthority,
) -> tuple[float, bool]:
    master = next(
        (
            row
            for row in identity_authority.security_master
            if row.security_id == security_id
        ),
        None,
    )
    if master is None:
        raise MassiveProfitabilityOriginFeaturesV2Error(
            "decision member is absent from identity authority"
        )
    sessions = tuple(session_authority.sessions)
    source = next(
        (
            index
            for index, row in enumerate(sessions)
            if row.session_date == source_session_date
        ),
        None,
    )
    if source is None:
        raise MassiveProfitabilityOriginFeaturesV2Error(
            "source is absent from session authority"
        )
    first_open_ms = sessions[0].regular_open_ns // 1_000_000
    if master.listing_at_ms < first_open_ms:
        return float(source + 1), True
    listing_index = next(
        (
            index
            for index, row in enumerate(sessions[: source + 1])
            if row.regular_close_ns // 1_000_000 >= master.listing_at_ms
        ),
        None,
    )
    if listing_index is None:
        return 0.0, False
    return float(source - listing_index + 1), False


def _bars_features(
    *,
    panel: dict[int, MassiveSessionPanelRowV1],
    economics: dict[int, MassiveProfitabilityFeatureEconomicValueRowV1],
    listing_age: float,
    listing_left_censored: bool,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    values = {name: 0.0 for name in BARS_MIN_V2_FIELDS}
    valid = {name: False for name in BARS_MIN_V2_FIELDS}
    returns: dict[int, float] = {}
    for horizon in (1, 5, 21, 63):
        result, is_valid = _economic_return(economics, 0, horizon)
        name = f"economic_total_return_{horizon}"
        if is_valid:
            values[name] = result
            valid[name] = True
            returns[horizon] = result
    for horizon in (1, 5):
        if horizon in returns:
            values[f"reversal_{horizon}"] = -returns[horizon]
            valid[f"reversal_{horizon}"] = True
    for name, end, horizon in (
        ("trend_21_minus_5", -5, 16),
        ("trend_63_minus_21", -21, 42),
    ):
        result, is_valid = _economic_return(economics, end, horizon)
        if is_valid:
            values[name] = result
            valid[name] = True
    daily_returns: dict[int, float] = {}
    for offset in range(-20, 1):
        result, is_valid = _economic_return(economics, offset, 1)
        if is_valid:
            daily_returns[offset] = result
    for window, minimum in ((5, 4), (21, 16)):
        selected = [
            value for offset, value in daily_returns.items() if offset >= -window + 1
        ]
        if len(selected) >= minimum:
            values[f"realized_volatility_{window}"] = pstdev(selected)
            valid[f"realized_volatility_{window}"] = True
        if window == 21 and len(selected) >= minimum:
            values["downside_volatility_21"] = math.sqrt(
                fmean(min(value, 0.0) ** 2 for value in selected)
            )
            valid["downside_volatility_21"] = True
    source = panel[0]
    for name in ("high_low_range", "close_location"):
        result, is_valid = _daily_value(source, name)
        if is_valid:
            values[name] = result
            valid[name] = True
    dollar, dollar_valid = _daily_value(source, "dollar_volume")
    if dollar_valid and dollar >= 0.0:
        values["log_dollar_volume"] = math.log1p(dollar)
        valid["log_dollar_volume"] = True
    prior_dollars: list[float] = []
    for offset in range(-21, 0):
        candidate, is_valid = _daily_value(panel[offset], "dollar_volume")
        if is_valid and candidate >= 0.0:
            prior_dollars.append(candidate)
    if dollar_valid and dollar >= 0.0 and len(prior_dollars) >= 16:
        values["dollar_volume_surprise_21"] = math.log1p(dollar) - math.log1p(
            fmean(prior_dollars)
        )
        valid["dollar_volume_surprise_21"] = True
    amihud: list[float] = []
    for offset, daily_return in daily_returns.items():
        candidate, is_valid = _daily_value(panel[offset], "dollar_volume")
        if is_valid and candidate > 0.0:
            amihud.append(abs(daily_return) / candidate)
    if len(amihud) >= 16:
        values["amihud_21"] = fmean(amihud)
        valid["amihud_21"] = True
    if source.listed:
        values["listing_age_sessions"] = listing_age
        valid["listing_age_sessions"] = True
        values["listing_age_left_censored"] = float(listing_left_censored)
        valid["listing_age_left_censored"] = True
    values["valid_history_fraction_63"] = (
        sum(economics[offset].valid for offset in range(-62, 1)) / 63.0
    )
    valid["valid_history_fraction_63"] = True
    return (
        tuple(values[name] for name in BARS_MIN_V2_FIELDS),
        tuple(valid[name] for name in BARS_MIN_V2_FIELDS),
    )


def _tape_features(
    *,
    panel_row: MassiveSessionPanelRowV1,
    population: MassiveProfitabilityTapePopulationRowV2 | None,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    values = {name: 0.0 for name in TAPE_MIN_V2_FIELDS}
    valid = {name: False for name in TAPE_MIN_V2_FIELDS}
    for name in (
        "log_trade_count",
        "median_trade_size",
        "p90_trade_size",
        "large_trade_dollar_fraction",
        "trf_off_exchange_dollar_fraction",
        "venue_entropy",
        "largest_venue_share",
        "tape_a_fraction",
        "tape_b_fraction",
        "tape_c_fraction",
    ):
        result, is_valid = _daily_value(panel_row, name, tape=True)
        if is_valid:
            values[name] = result
            valid[name] = True
    if population is not None:
        population.validate()
        values["signed_dollar_flow_fraction"] = (
            population.signed_dollar_flow / population.dollar_volume
        )
        valid["signed_dollar_flow_fraction"] = True
        values["absolute_signed_flow_imbalance"] = (
            population.absolute_signed_flow_imbalance
        )
        valid["absolute_signed_flow_imbalance"] = True
        if population.regular_session_event_count > 0:
            denominator = float(population.regular_session_event_count)
            for name, count in (
                ("replacement_event_fraction", population.replacement_event_count),
                ("cancellation_event_fraction", population.cancellation_event_count),
                ("late_report_event_fraction", population.late_report_event_count),
            ):
                values[name] = count / denominator
                valid[name] = True
    return (
        tuple(values[name] for name in TAPE_MIN_V2_FIELDS),
        tuple(valid[name] for name in TAPE_MIN_V2_FIELDS),
    )


def build_massive_profitability_origin_features_v2(
    *,
    origin: MassiveProfitabilityDecisionOriginV1,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    panel_rows: Sequence[MassiveSessionPanelRowV1],
    session_panel_receipt_sha256: str,
    feature_accounting: MassiveProfitabilityFeatureAccountingV1,
    tape_population_rows: Sequence[MassiveProfitabilityTapePopulationRowV2],
    input_schemas: Sequence[str] = (
        MASSIVE_SESSION_PANEL_V1_SCHEMA,
        MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA,
    ),
) -> MassiveProfitabilityOriginFeaturesV2:
    """Emit one decision-member cross-section from an exact 64-session rectangle."""

    origin.validate()
    session_authority.validate()
    identity_authority.validate()
    feature_accounting.validate()
    if origin.session_authority_receipt_sha256 != session_authority.receipt_sha256:
        raise MassiveProfitabilityOriginFeaturesV2Error(
            "origin and feature session authorities differ"
        )
    schemas = tuple(sorted(input_schemas))
    for schema in schemas:
        reject_massive_profitability_legacy_generation_v2(schema)
    if set(schemas) != {
        MASSIVE_SESSION_PANEL_V1_SCHEMA,
        MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_V1_SCHEMA,
    }:
        raise MassiveProfitabilityOriginFeaturesV2Error(
            "bounded features require only the V1 session panel and feature accounting"
        )
    if (
        feature_accounting.origin_receipt_sha256 != origin.receipt_sha256
        or feature_accounting.source_session_date != origin.source_session_date
        or feature_accounting.feature_cutoff_at_ms != origin.feature_cutoff_at_ms
    ):
        raise MassiveProfitabilityOriginFeaturesV2Error(
            "origin and feature accounting differ"
        )
    members = tuple(sorted(origin.decision_member_security_ids))
    rank_by_security = dict(
        zip(
            origin.decision_member_security_ids,
            origin.decision_member_universe_ranks,
            strict=True,
        )
    )
    panel_values = tuple(panel_rows)
    panel_map = {
        (row.source_session_date, row.security_id): row for row in panel_values
    }
    if len(panel_map) != len(panel_values):
        raise MassiveProfitabilityOriginFeaturesV2Error(
            "session panel contains duplicate date-security rows"
        )
    expected_panel = {
        (session_date, security_id)
        for session_date in feature_accounting.session_dates
        for security_id in members
    }
    if not expected_panel <= set(panel_map):
        raise MassiveProfitabilityOriginFeaturesV2Error(
            "session panel lacks the exact bounded decision-member rectangle"
        )
    for key in expected_panel:
        panel_map[key].validate()
        authority_session = session_authority.resolve(
            exchange="XNYS", session_date=key[0]
        )
        if (
            panel_map[key].regular_open_ns != authority_session.regular_open_ns
            or panel_map[key].regular_close_ns != authority_session.regular_close_ns
            or panel_map[key].regular_close_ns // 1_000_000
            > origin.feature_cutoff_at_ms
        ):
            raise MassiveProfitabilityOriginFeaturesV2Error(
                "feature input session coordinates or cutoff differ"
            )
    economic_map = {
        (row.source_session_offset, row.security_id): row
        for row in feature_accounting.rows
    }
    populations = {
        (row.source_session_date, row.security_id): row for row in tape_population_rows
    }
    if len(populations) != len(tuple(tape_population_rows)) or any(
        session_date != origin.source_session_date for session_date, _ in populations
    ):
        raise MassiveProfitabilityOriginFeaturesV2Error(
            "tape population support is not a unique source-date cross-section"
        )
    rows: list[MassiveProfitabilityOriginFeatureRowV2] = []
    source_inputs: list[str] = []
    for security_id in members:
        panels = {
            offset: panel_map[(session_date, security_id)]
            for offset, session_date in zip(
                range(-63, 1), feature_accounting.session_dates, strict=True
            )
        }
        economics = {
            offset: economic_map[(offset, security_id)] for offset in range(-63, 1)
        }
        age, censored = _listing_age(
            security_id=security_id,
            source_session_date=origin.source_session_date,
            identity_authority=identity_authority,
            session_authority=session_authority,
        )
        bars_values, bars_valid = _bars_features(
            panel=panels,
            economics=economics,
            listing_age=age,
            listing_left_censored=censored,
        )
        population = populations.get((origin.source_session_date, security_id))
        tape_values, tape_valid = _tape_features(
            panel_row=panels[0], population=population
        )
        accounting_inventory = semantic_sha256(
            tuple(economics[offset].receipt_sha256 for offset in range(-63, 1))
        )
        body = {
            "decision_session_date": origin.decision_session_date,
            "source_session_date": origin.source_session_date,
            "security_id": security_id,
            "decision_membership_rank": rank_by_security[security_id],
            "source_staleness_sessions": origin.source_staleness_sessions,
            "source_listed": panels[0].listed,
            "source_tradable": panels[0].tradable,
            "source_observed_regular_trade": panels[0].observed_regular_trade,
            "source_halt_or_no_print": panels[0].halt_or_no_print,
            "bars_values": bars_values,
            "bars_valid": bars_valid,
            "tape_values": tape_values,
            "tape_valid": tape_valid,
            "source_panel_row_receipt_sha256": panels[0].receipt_sha256,
            "feature_accounting_security_inventory_sha256": accounting_inventory,
            "tape_population_row_receipt_sha256": (
                None if population is None else population.receipt_sha256
            ),
        }
        row = MassiveProfitabilityOriginFeatureRowV2(
            **body, receipt_sha256=semantic_sha256(body)
        )
        row.validate()
        rows.append(row)
        source_inputs.extend(panel.receipt_sha256 for panel in panels.values())
        source_inputs.extend(
            economics[offset].receipt_sha256 for offset in range(-63, 1)
        )
        if population is not None:
            source_inputs.append(population.receipt_sha256)
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    source_inventory = semantic_sha256(tuple(sorted(source_inputs)))
    semantic = {
        "origin_receipt_sha256": origin.receipt_sha256,
        "decision_session_date": origin.decision_session_date,
        "source_session_date": origin.source_session_date,
        "feature_cutoff_at_ms": origin.feature_cutoff_at_ms,
        "source_staleness_sessions": origin.source_staleness_sessions,
        "input_session_dates": feature_accounting.session_dates,
        "rows": tuple(asdict(row) for row in rows),
        "origin_membership_receipt_sha256": origin.membership_group_semantic_receipt_sha256,
        "feature_accounting_semantic_receipt_sha256": feature_accounting.semantic_receipt_sha256,
        "session_panel_receipt_sha256": _digest(
            "session panel", session_panel_receipt_sha256
        ),
        "source_input_inventory_sha256": source_inventory,
        "row_inventory_sha256": row_inventory,
        "input_schemas": schemas,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SOURCE_SHA256,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "tape_population_data_qualified": bool(populations)
        and all(row.source_data_qualified for row in populations.values()),
        "feature_accounting_data_qualified": (
            feature_accounting.economic_values_data_qualified
        ),
        "schema": MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA,
    }
    semantic_receipt = semantic_sha256(semantic)
    runtime = dict(semantic)
    runtime["rows"] = tuple(rows)
    result = MassiveProfitabilityOriginFeaturesV2(
        **runtime,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "feature_accounting_audit_receipt_sha256": (
                    feature_accounting.audit_receipt_sha256
                ),
                "identity_authority_audit_receipt_sha256": (
                    identity_authority.receipt_sha256
                ),
            }
        ),
    )
    result.validate()
    return result


__all__ = [
    "BARS_MIN_V2_FIELDS",
    "MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SCHEMA",
    "MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V2_SPEC_SHA256",
    "MASSIVE_PROFITABILITY_TAPE_POPULATION_V2_SCHEMA",
    "TAPE_MIN_V2_FIELDS",
    "MassiveProfitabilityOriginFeatureRowV2",
    "MassiveProfitabilityOriginFeaturesV2",
    "MassiveProfitabilityOriginFeaturesV2Error",
    "MassiveProfitabilityTapePopulationRowV2",
    "build_massive_profitability_origin_features_v2",
    "derive_massive_profitability_tape_populations_v2",
]
