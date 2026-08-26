"""Source-derived bounded profitability feature cross-sections for Massive P0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
    MassiveProfitabilityAccountingFreezeV1,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
    MassiveProfitabilityDailyInputAuthorityV1,
    MassiveProfitabilityDailySecurityInputV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    reject_massive_profitability_legacy_generation_v2,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
    MassiveProfitabilityFeatureAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    BARS_MIN_V2_FIELDS,
    TAPE_MIN_V2_FIELDS,
    MassiveProfitabilityOriginFeatureRowV2,
    MassiveProfitabilityTapePopulationRowV2,
    _bars_features,
    _listing_age,
    _tape_features,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MassiveProfitabilityDecisionOriginV1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
    MassiveProfitabilityTerminalCoverageAuthorityV1,
)
from rl_quant.features.massive_session_panel_v1 import MassiveSessionPanelRowV1
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA = (
    "rl-quant.massive-profitability-origin-features-v3"
)
MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "origin": MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
        "daily_input": MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
        "accounting": MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
        "accounting_freeze": MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
        "terminal": MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
        "legacy_panel_rows": "prohibited-from-public-api",
        "history": "exact-source-minus-63-through-source",
        "output": "one-decision-member-cross-section",
        "corporate_action_predictors": False,
        "performance_authorization": False,
    }
)


class MassiveProfitabilityOriginFeaturesV3Error(ValueError):
    """A V3 feature cross-section is detached from frozen source authorities."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityOriginFeaturesV3Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _listed(
    *,
    security_id: str,
    at_ms: int,
    identity_authority: PITSecurityUniverseAuthority,
) -> bool:
    master = next(
        row
        for row in identity_authority.security_master
        if row.security_id == security_id
    )
    return master.listing_at_ms <= at_ms and (
        master.delisting_at_ms is None or at_ms < master.delisting_at_ms
    )


def _panel_row(
    *,
    source: MassiveProfitabilityDailySecurityInputV1,
    session_index: int,
    regular_open_ns: int,
    regular_close_ns: int,
    listed: bool,
) -> MassiveSessionPanelRowV1:
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    observed = (
        source.daily_bar_row_receipt_sha256 is not None
        and source.daily_tape_row_receipt_sha256 is not None
    )
    tradable = (
        listed
        and observed
        and source.bars_valid[close_index]
        and source.bars_valid[dollar_index]
    )
    body = {
        "source_session_index": session_index,
        "source_session_date": source.source_session_date,
        "regular_open_ns": regular_open_ns,
        "regular_close_ns": regular_close_ns,
        "security_id": source.security_id,
        "pit_member": True,
        "listed": listed,
        "tradable": tradable,
        "observed_regular_trade": observed,
        "halt_or_no_print": listed and not observed,
        "bars_values": source.bars_values,
        "bars_valid": source.bars_valid,
        "tape_values": source.tape_values,
        "tape_valid": source.tape_valid,
        "event_timeline_count": source.regular_session_event_count,
        "replacement_event_count": source.replacement_event_count,
        "cancellation_event_count": source.cancellation_event_count,
        "late_report_event_count": source.late_report_event_count,
        "daily_bars_row_receipt_sha256": source.daily_bar_row_receipt_sha256,
        "daily_tape_row_receipt_sha256": source.daily_tape_row_receipt_sha256,
        "event_counts_receipt_sha256": (
            source.tape_population_row_receipt_sha256
            if source.regular_session_event_count > 0
            else None
        ),
    }
    result = MassiveSessionPanelRowV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _population(
    *, source: MassiveProfitabilityDailySecurityInputV1, qualified: bool
) -> MassiveProfitabilityTapePopulationRowV2 | None:
    if not source.same_population_valid:
        return None
    assert source.tape_population_row_receipt_sha256 is not None
    body = {
        "source_session_date": source.source_session_date,
        "security_id": source.security_id,
        "signed_dollar_flow": source.signed_dollar_flow,
        "dollar_volume": source.same_population_dollar_volume,
        "absolute_signed_flow_imbalance": source.absolute_signed_flow_imbalance,
        "regular_session_event_count": source.regular_session_event_count,
        "replacement_event_count": source.replacement_event_count,
        "cancellation_event_count": source.cancellation_event_count,
        "late_report_event_count": source.late_report_event_count,
        "population_receipt_sha256": source.tape_population_row_receipt_sha256,
        "source_data_qualified": qualified,
        "schema": "rl-quant.massive-profitability-tape-population-v2",
    }
    result = MassiveProfitabilityTapePopulationRowV2(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityOriginFeaturesV3:
    origin_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    decision_session_date: str
    source_session_date: str
    feature_cutoff_at_ms: int
    maximum_economic_input_at_ms: int
    maximum_source_available_at_ms: int
    source_staleness_sessions: int
    input_session_dates: tuple[str, ...]
    rows: tuple[MassiveProfitabilityOriginFeatureRowV2, ...]
    daily_input_authority_semantic_receipt_sha256: str
    feature_accounting_authority_semantic_receipt_sha256: str
    accounting_freeze_semantic_receipt_sha256: str
    terminal_authority_semantic_receipt_sha256: str
    source_input_inventory_sha256: str
    row_inventory_sha256: str
    input_schemas: tuple[str, ...]
    source_inputs_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    development_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "audit_receipt_sha256"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SOURCE_SHA256
            or len(self.input_session_dates) != 64
            or self.input_session_dates != tuple(sorted(set(self.input_session_dates)))
            or self.input_session_dates[-1] != self.source_session_date
            or self.source_staleness_sessions != 2
            or self.maximum_economic_input_at_ms != self.feature_cutoff_at_ms
            or self.maximum_source_available_at_ms < self.feature_cutoff_at_ms
            or not isinstance(self.source_inputs_data_qualified, bool)
            or any(
                (
                    self.development_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityOriginFeaturesV3Error(
                "origin features V3 identity, cutoff, or authorization differs"
            )
        for schema in self.input_schemas:
            reject_massive_profitability_legacy_generation_v2(schema)
        if set(self.input_schemas) != {
            MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
            MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
            MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
            MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
            MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
        }:
            raise MassiveProfitabilityOriginFeaturesV3Error(
                "origin features V3 input generations differ"
            )
        keys = tuple(row.security_id for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveProfitabilityOriginFeaturesV3Error(
                "origin features V3 cross-section differs"
            )
        for row in self.rows:
            row.validate()
            if (
                row.decision_session_date != self.decision_session_date
                or row.source_session_date != self.source_session_date
            ):
                raise MassiveProfitabilityOriginFeaturesV3Error(
                    "origin features V3 row chronology differs"
                )
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveProfitabilityOriginFeaturesV3Error(
                "origin features V3 row inventory differs"
            )
        for value in (
            self.origin_receipt_sha256,
            self.origin_plan_semantic_receipt_sha256,
            self.daily_input_authority_semantic_receipt_sha256,
            self.feature_accounting_authority_semantic_receipt_sha256,
            self.accounting_freeze_semantic_receipt_sha256,
            self.terminal_authority_semantic_receipt_sha256,
            self.source_input_inventory_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("origin features V3", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityOriginFeaturesV3Error(
                "origin features V3 semantic receipt differs"
            )


def build_massive_profitability_origin_features_v3(
    *,
    origin: MassiveProfitabilityDecisionOriginV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    feature_accounting: MassiveProfitabilityFeatureAccountingAuthorityV2,
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
) -> MassiveProfitabilityOriginFeaturesV3:
    """Build one V3 cross-section without caller-supplied panel or economic rows."""

    origin.validate()
    origin_plan.validate()
    session_authority.validate()
    identity_authority.validate()
    daily_input_authority.validate()
    feature_accounting.validate()
    accounting_freeze.validate()
    terminal_authority.validate()
    if (
        origin.receipt_sha256
        not in {row.receipt_sha256 for row in origin_plan.origin_plan_v1.origins}
        or feature_accounting.origin_receipt_sha256 != origin.receipt_sha256
        or feature_accounting.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or feature_accounting.daily_input_authority_semantic_receipt_sha256
        != daily_input_authority.semantic_receipt_sha256
        or feature_accounting.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
        or accounting_freeze.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or accounting_freeze.archive_freeze_semantic_receipt_sha256
        != daily_input_authority.archive_freeze_semantic_receipt_sha256
        or accounting_freeze.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityOriginFeaturesV3Error(
            "origin feature authorities do not share one frozen experiment"
        )
    coverage_row = next(
        (
            row
            for row in accounting_freeze.coverage_rows
            if row.origin_receipt_sha256 == origin.receipt_sha256
        ),
        None,
    )
    if (
        coverage_row is None
        or coverage_row.economic_coverage_audit_receipt_sha256
        != feature_accounting.economic_archive_audit_receipt_sha256
    ):
        raise MassiveProfitabilityOriginFeaturesV3Error(
            "feature economic coverage is absent from the accounting freeze"
        )
    sessions = tuple(session_authority.sessions)
    by_date = {row.session_date: (index, row) for index, row in enumerate(sessions)}
    source = by_date[origin.source_session_date][0]
    dates = tuple(row.session_date for row in sessions[source - 63 : source + 1])
    if dates != feature_accounting.accounting.session_dates:
        raise MassiveProfitabilityOriginFeaturesV3Error(
            "feature accounting does not use the exact daily input rectangle"
        )
    economic_map = {
        (row.source_session_offset, row.security_id): row
        for row in feature_accounting.accounting.rows
    }
    ranks = dict(
        zip(
            origin.decision_member_security_ids,
            origin.decision_member_universe_ranks,
            strict=True,
        )
    )
    rows: list[MassiveProfitabilityOriginFeatureRowV2] = []
    source_receipts: list[str] = []
    for security_id in sorted(origin.decision_member_security_ids):
        panels: dict[int, MassiveSessionPanelRowV1] = {}
        for offset, session_date in zip(range(-63, 1), dates, strict=True):
            index, session = by_date[session_date]
            source_row = daily_input_authority.row(
                session_date=session_date, security_id=security_id
            )
            panels[offset] = _panel_row(
                source=source_row,
                session_index=index,
                regular_open_ns=session.regular_open_ns,
                regular_close_ns=session.regular_close_ns,
                listed=_listed(
                    security_id=security_id,
                    at_ms=session.regular_close_ns // 1_000_000,
                    identity_authority=identity_authority,
                ),
            )
            source_receipts.append(source_row.receipt_sha256)
        economics = {
            offset: economic_map[(offset, security_id)] for offset in range(-63, 1)
        }
        age, censored = _listing_age(
            security_id=security_id,
            source_session_date=origin.source_session_date,
            identity_authority=identity_authority,
            session_authority=session_authority,
        )
        bar_values, bar_masks = _bars_features(
            panel=panels,
            economics=economics,
            listing_age=age,
            listing_left_censored=censored,
        )
        source_daily = daily_input_authority.row(
            session_date=origin.source_session_date, security_id=security_id
        )
        population = _population(
            source=source_daily,
            qualified=daily_input_authority.daily_input_data_qualified,
        )
        tape_values, tape_masks = _tape_features(
            panel_row=panels[0], population=population
        )
        accounting_inventory = semantic_sha256(
            tuple(economics[offset].receipt_sha256 for offset in range(-63, 1))
        )
        body = {
            "decision_session_date": origin.decision_session_date,
            "source_session_date": origin.source_session_date,
            "security_id": security_id,
            "decision_membership_rank": ranks[security_id],
            "source_staleness_sessions": origin.source_staleness_sessions,
            "source_listed": panels[0].listed,
            "source_tradable": panels[0].tradable,
            "source_observed_regular_trade": panels[0].observed_regular_trade,
            "source_halt_or_no_print": panels[0].halt_or_no_print,
            "bars_values": bar_values,
            "bars_valid": bar_masks,
            "tape_values": tape_values,
            "tape_valid": tape_masks,
            "source_panel_row_receipt_sha256": source_daily.receipt_sha256,
            "feature_accounting_security_inventory_sha256": accounting_inventory,
            "tape_population_row_receipt_sha256": (
                None if population is None else population.receipt_sha256
            ),
        }
        feature_row = MassiveProfitabilityOriginFeatureRowV2(
            **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
        )
        feature_row.validate()
        rows.append(feature_row)
        source_receipts.extend(
            economics[offset].receipt_sha256 for offset in range(-63, 1)
        )
    source_sessions = {
        row.source_session_date: row for row in daily_input_authority.sessions
    }
    maximum_available = max(
        source_sessions[session_date].vendor_last_modified_at_ms
        for session_date in dates
    )
    if maximum_available > origin.decision_at_ms:
        raise MassiveProfitabilityOriginFeaturesV3Error(
            "feature source became available after the decision"
        )
    schemas = tuple(
        sorted(
            (
                MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
                MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
                MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
                MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
                MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
            )
        )
    )
    source_inventory = semantic_sha256(tuple(sorted(source_receipts)))
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    qualified = (
        daily_input_authority.daily_input_data_qualified
        and feature_accounting.economic_values_data_qualified
        and accounting_freeze.accounting_sources_frozen
        and accounting_freeze.capture_transport_qualified
        and terminal_authority.terminal_accounting_data_qualified
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA,
        "origin_receipt_sha256": origin.receipt_sha256,
        "origin_plan_semantic_receipt_sha256": origin_plan.semantic_receipt_sha256,
        "decision_session_date": origin.decision_session_date,
        "source_session_date": origin.source_session_date,
        "feature_cutoff_at_ms": origin.feature_cutoff_at_ms,
        "maximum_economic_input_at_ms": origin.feature_cutoff_at_ms,
        "maximum_source_available_at_ms": maximum_available,
        "source_staleness_sessions": origin.source_staleness_sessions,
        "input_session_dates": dates,
        "rows": tuple(asdict(row) for row in rows),
        "daily_input_authority_semantic_receipt_sha256": (
            daily_input_authority.semantic_receipt_sha256
        ),
        "feature_accounting_authority_semantic_receipt_sha256": (
            feature_accounting.semantic_receipt_sha256
        ),
        "accounting_freeze_semantic_receipt_sha256": (
            accounting_freeze.semantic_receipt_sha256
        ),
        "terminal_authority_semantic_receipt_sha256": (
            terminal_authority.semantic_receipt_sha256
        ),
        "source_input_inventory_sha256": source_inventory,
        "row_inventory_sha256": row_inventory,
        "input_schemas": schemas,
        "source_inputs_data_qualified": qualified,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SOURCE_SHA256,
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    receipt = semantic_sha256(semantic)
    runtime = dict(semantic)
    runtime.pop("rows")
    result = MassiveProfitabilityOriginFeaturesV3(
        **runtime,  # type: ignore[arg-type]
        rows=tuple(rows),
        semantic_receipt_sha256=receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": receipt,
                "daily_input_audit_receipt_sha256": (
                    daily_input_authority.audit_receipt_sha256
                ),
                "feature_accounting_audit_receipt_sha256": (
                    feature_accounting.audit_receipt_sha256
                ),
                "accounting_freeze_audit_receipt_sha256": (
                    accounting_freeze.audit_receipt_sha256
                ),
            }
        ),
    )
    result.validate()
    return result


__all__ = [
    "BARS_MIN_V2_FIELDS",
    "MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SCHEMA",
    "TAPE_MIN_V2_FIELDS",
    "MassiveProfitabilityOriginFeaturesV3",
    "MassiveProfitabilityOriginFeaturesV3Error",
    "build_massive_profitability_origin_features_v3",
]
