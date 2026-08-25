from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
    build_massive_session_authority,
)
from rl_quant.features.massive_monthly_rank_input_authority_v2 import (
    MassiveMonthlyRankInputAuthorityV2,
    build_massive_monthly_rank_input_authority_v2,
)
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_LOCKBOX_ACCESS_AUTHORIZED,
    MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PANEL_MATERIALIZATION_AUTHORIZED,
    MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PROFITABILITY_REPORTING_AUTHORIZED,
    MassiveProfitabilityArchiveFreezeV1,
    MassiveProfitabilityArchiveFreezeV1Error,
    MassiveProfitabilityArchiveMonthlyRankV1,
    MassiveProfitabilityArchiveSourceSessionV1,
    materialize_massive_profitability_archive_freeze_for_test_v1,
    materialize_massive_profitability_archive_freeze_v1,
    parse_massive_profitability_archive_freeze_v1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_LOCKBOX_ACCESS_AUTHORIZED,
    MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PANEL_MATERIALIZATION_AUTHORIZED,
    MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PROFITABILITY_REPORTING_AUTHORIZED,
    MASSIVE_PROFITABILITY_LEGACY_GENERATIONS_V2,
    MassiveProfitabilityExperimentCoverageV2Error,
    materialize_massive_profitability_experiment_coverage_for_test_v2,
    parse_massive_profitability_experiment_coverage_v2,
    reject_massive_profitability_legacy_generation_v2,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveMonthlyRankInputAuthorityV2 as OriginMonthlyRankInputAuthorityV2,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MASSIVE_PROFITABILITY_PHASE_PLAN_V1_LOCKBOX_ACCESS_AUTHORIZED,
    MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PANEL_MATERIALIZATION_AUTHORIZED,
    MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PROFITABILITY_REPORTING_AUTHORIZED,
    materialize_massive_profitability_phase_plan_v1,
    parse_massive_profitability_phase_plan_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

_EASTERN = ZoneInfo("America/New_York")
_ENTITLEMENT = "a" * 64
_SESSION_COUNT = 1_900


def _ms(day: str, value: time) -> int:
    return int(
        datetime.combine(date.fromisoformat(day), value, tzinfo=_EASTERN).timestamp()
        * 1_000
    )


def _sessions() -> MassiveSessionAuthority:
    days: list[str] = []
    current = date(2012, 1, 3)
    while len(days) < _SESSION_COUNT:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    source_receipt = semantic_sha256("archive-freeze-calendar")
    rows = tuple(
        MassiveExchangeSession(
            session_date=day,
            exchange="XNYS",
            regular_open_ns=_ms(day, time(9, 30)) * 1_000_000,
            regular_close_ns=_ms(day, time(16, 0)) * 1_000_000,
            scheduled_five_minute_intervals=78,
            special_session_reason=None,
            calendar_source_receipt_sha256=source_receipt,
        )
        for day in days
    )
    return build_massive_session_authority(
        rows, calendar_source_receipt_sha256=source_receipt
    )


def _inputs() -> tuple[
    MassiveSessionAuthority,
    tuple[MassiveProfitabilityArchiveSourceSessionV1, ...],
    tuple[MassiveProfitabilityArchiveMonthlyRankV1, ...],
    int,
]:
    sessions = _sessions()
    session_rows = sessions.sessions
    freeze_at = _ms(session_rows[-1].session_date, time(16, 0)) + 86_400_000
    sources = tuple(
        MassiveProfitabilityArchiveSourceSessionV1.build_for_test(
            source_session_date=row.session_date,
            vendor_last_modified_at_ms=_ms(row.session_date, time(18, 0)),
            authenticated_get_completed_at_ms=freeze_at - 1,
        )
        for row in session_rows
    )
    first_by_month: dict[str, int] = {}
    for index, row in enumerate(session_rows):
        first_by_month.setdefault(row.session_date[:7], index)
    lookback = (
        MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.ranking_lookback_sessions
    )
    lag = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.ranking_lag_sessions
    ranks = tuple(
        MassiveProfitabilityArchiveMonthlyRankV1.build_for_test(
            calendar_month=month,
            scheduled_rebalance_session_date=session_rows[index].session_date,
            activated_at_ms=_ms(session_rows[index].session_date, time(9, 30)),
            maximum_input_available_at_ms=_ms(
                session_rows[index - lag].session_date, time(18, 0)
            ),
        )
        for month, index in sorted(first_by_month.items())
        if index - lag - lookback + 1 >= 0
    )
    return sessions, sources, ranks, freeze_at


def _freeze(tmp_path: Path) -> MassiveProfitabilityArchiveFreezeV1:
    sessions, sources, ranks, freeze_at = _inputs()
    return materialize_massive_profitability_archive_freeze_for_test_v1(
        root=tmp_path,
        session_authority=sessions,
        source_rows=sources,
        rank_rows=ranks,
        data_freeze_at_ms=freeze_at,
        artifact_id="canary",
        committed_at_ms=freeze_at + 1,
        entitlement_receipt_sha256=_ENTITLEMENT,
    )


def test_archive_freeze_derives_fixed_candidate_phases_and_round_trips(
    tmp_path: Path,
) -> None:
    freeze = _freeze(tmp_path)
    protocol = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL

    assert freeze.calendar_geometry_complete is True
    assert freeze.source_transport_qualified is False
    assert freeze.rank_bar_data_qualified is False
    assert len(freeze.fixed_candidate_session_dates) >= 1_764
    assert (
        tuple(
            value
            for inventory in freeze.fixed_outer_test_session_inventories
            for value in inventory
        )
        == freeze.fixed_candidate_session_dates[-(4 * 126 + 252) : -252]
    )
    assert (
        freeze.fixed_lockbox_session_dates
        == freeze.fixed_candidate_session_dates[-protocol.historical_lockbox_sessions :]
    )
    assert all(
        len(inventory) == protocol.outer_fold_sessions
        for inventory in freeze.fixed_outer_test_session_inventories
    )
    assert not any(
        (
            freeze.panel_materialization_authorized,
            freeze.predictive_training_authorized,
            freeze.profitability_reporting_authorized,
            freeze.lockbox_access_authorized,
        )
    )

    reloaded = parse_massive_profitability_archive_freeze_v1(
        root=tmp_path, loaded_source=freeze.loaded_source
    )
    assert reloaded.semantic_receipt_sha256 == freeze.semantic_receipt_sha256
    assert reloaded.source_transport_qualified is False
    assert reloaded.rank_bar_data_qualified is False


def test_archive_freeze_prohibits_caller_dates_and_requires_complete_inputs(
    tmp_path: Path,
) -> None:
    signature = inspect.signature(materialize_massive_profitability_archive_freeze_v1)
    assert "first_candidate_decision_session_date" not in signature.parameters
    assert "last_candidate_decision_session_date" not in signature.parameters
    rank_signature = inspect.signature(build_massive_monthly_rank_input_authority_v2)
    assert "first_candidate_decision_session_date" not in rank_signature.parameters
    assert "last_candidate_decision_session_date" not in rank_signature.parameters

    sessions, sources, ranks, freeze_at = _inputs()
    with pytest.raises(
        MassiveProfitabilityArchiveFreezeV1Error,
        match="complete for every XNYS session",
    ):
        materialize_massive_profitability_archive_freeze_for_test_v1(
            root=tmp_path / "source-gap",
            session_authority=sessions,
            source_rows=sources[:500] + sources[501:],
            rank_rows=ranks,
            data_freeze_at_ms=freeze_at,
            artifact_id="source-gap",
            committed_at_ms=freeze_at + 1,
            entitlement_receipt_sha256=_ENTITLEMENT,
        )
    with pytest.raises(
        MassiveProfitabilityArchiveFreezeV1Error,
        match="cover every archive-supported month",
    ):
        materialize_massive_profitability_archive_freeze_for_test_v1(
            root=tmp_path / "rank-gap",
            session_authority=sessions,
            source_rows=sources,
            rank_rows=ranks[:-1],
            data_freeze_at_ms=freeze_at,
            artifact_id="rank-gap",
            committed_at_ms=freeze_at + 1,
            entitlement_receipt_sha256=_ENTITLEMENT,
        )


def test_phase_plan_freezes_candidate_date_folds_and_round_trips(
    tmp_path: Path,
) -> None:
    freeze = _freeze(tmp_path)
    phase = materialize_massive_profitability_phase_plan_v1(
        root=tmp_path,
        archive_freeze=freeze,
        artifact_id="canary",
        committed_at_ms=freeze.data_freeze_at_ms + 2,
        entitlement_receipt_sha256=_ENTITLEMENT,
    )
    protocol = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL

    assert phase.candidate_session_dates == freeze.fixed_candidate_session_dates
    assert phase.lockbox_session_dates == freeze.fixed_lockbox_session_dates
    assert tuple(fold.outer_test_session_dates for fold in phase.outer_folds) == (
        freeze.fixed_outer_test_session_inventories
    )
    assert len(phase.outer_folds) == protocol.outer_fold_count
    assert all(
        len(fold.fit_session_dates) >= protocol.minimum_initial_training_sessions
        and len(fold.inner_purge_session_dates) == protocol.inner_purge_sessions
        and len(fold.inner_validation_session_dates)
        == protocol.inner_validation_sessions
        and len(fold.outer_purge_session_dates)
        == protocol.target_overlap_purge_sessions
        for fold in phase.outer_folds
    )
    assert phase.archive_source_transport_qualified is False
    assert phase.archive_rank_bar_data_qualified is False
    assert not any(
        (
            phase.panel_materialization_authorized,
            phase.predictive_training_authorized,
            phase.profitability_reporting_authorized,
            phase.lockbox_access_authorized,
        )
    )
    reloaded = parse_massive_profitability_phase_plan_v1(
        root=tmp_path, loaded_source=phase.loaded_source
    )
    assert reloaded.semantic_receipt_sha256 == phase.semantic_receipt_sha256
    assert reloaded.archive_source_transport_qualified is False
    assert reloaded.archive_rank_bar_data_qualified is False


def test_monthly_rank_surface_is_v2_and_all_authorizations_remain_false() -> None:
    assert MassiveMonthlyRankInputAuthorityV2 is OriginMonthlyRankInputAuthorityV2
    assert not any(
        (
            MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PANEL_MATERIALIZATION_AUTHORIZED,
            MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PREDICTIVE_TRAINING_AUTHORIZED,
            MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_PROFITABILITY_REPORTING_AUTHORIZED,
            MASSIVE_PROFITABILITY_ARCHIVE_FREEZE_V1_LOCKBOX_ACCESS_AUTHORIZED,
            MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PANEL_MATERIALIZATION_AUTHORIZED,
            MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PREDICTIVE_TRAINING_AUTHORIZED,
            MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PROFITABILITY_REPORTING_AUTHORIZED,
            MASSIVE_PROFITABILITY_PHASE_PLAN_V1_LOCKBOX_ACCESS_AUTHORIZED,
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PANEL_MATERIALIZATION_AUTHORIZED,
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PREDICTIVE_TRAINING_AUTHORIZED,
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_PROFITABILITY_REPORTING_AUTHORIZED,
            MASSIVE_PROFITABILITY_EXPERIMENT_COVERAGE_V2_LOCKBOX_ACCESS_AUTHORIZED,
        )
    )


def test_v2_coverage_persists_fixed_dates_and_rejects_legacy_inputs(
    tmp_path: Path,
) -> None:
    freeze = _freeze(tmp_path)
    phase = materialize_massive_profitability_phase_plan_v1(
        root=tmp_path,
        archive_freeze=freeze,
        artifact_id="coverage-phase",
        committed_at_ms=freeze.data_freeze_at_ms + 2,
        entitlement_receipt_sha256=_ENTITLEMENT,
    )
    coverage = materialize_massive_profitability_experiment_coverage_for_test_v2(
        root=tmp_path,
        archive_freeze=freeze,
        phase_plan=phase,
        artifact_id="coverage",
        committed_at_ms=freeze.data_freeze_at_ms + 3,
        entitlement_receipt_sha256=_ENTITLEMENT,
    )

    assert coverage.candidate_session_dates == freeze.fixed_candidate_session_dates
    assert coverage.outer_test_session_inventories == (
        freeze.fixed_outer_test_session_inventories
    )
    assert coverage.lockbox_session_dates == freeze.fixed_lockbox_session_dates
    assert set(coverage.common_support_requirements) == {
        (session_date, 500, 400)
        for session_date in freeze.fixed_candidate_session_dates
    }
    assert coverage.data_gate_passed is False
    assert coverage.source_transport_qualified is False
    assert coverage.rank_bar_data_qualified is False
    assert not any(
        (
            coverage.panel_materialization_authorized,
            coverage.predictive_training_authorized,
            coverage.profitability_reporting_authorized,
            coverage.lockbox_access_authorized,
        )
    )
    reloaded = parse_massive_profitability_experiment_coverage_v2(
        root=tmp_path, loaded_source=coverage.loaded_source
    )
    assert reloaded.semantic_receipt_sha256 == coverage.semantic_receipt_sha256
    for schema in MASSIVE_PROFITABILITY_LEGACY_GENERATIONS_V2:
        with pytest.raises(
            MassiveProfitabilityExperimentCoverageV2Error,
            match="legacy profitability generation is prohibited",
        ):
            reject_massive_profitability_legacy_generation_v2(schema)


def test_archive_freeze_rejects_rewritten_semantic_row() -> None:
    _, sources, _, _ = _inputs()
    changed = replace(
        sources[0],
        vendor_last_modified_at_ms=sources[0].vendor_last_modified_at_ms + 1,
    )
    with pytest.raises(
        MassiveProfitabilityArchiveFreezeV1Error,
        match="source row receipt differs",
    ):
        changed.validate()
