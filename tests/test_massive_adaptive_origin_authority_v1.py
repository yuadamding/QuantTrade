from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from rl_quant.alpha.massive_universe_adapter import checked_pit_universe_rule
from rl_quant.alpha.pit_universe import (
    HistoricalMembershipRecord,
    SourcedSecurityMasterRecord,
)
from rl_quant.data_sources.massive.decision_clock import (
    build_massive_decision_clock_authority,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1,
    MassiveAdaptiveOriginAuthorityV1Error,
    build_massive_adaptive_origin_authority_v1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
)

_EASTERN = ZoneInfo("America/New_York")
_ASSETS = tuple(f"SEC-{index:02d}" for index in range(8))


def _ms(day: str, value: time) -> int:
    return int(
        datetime.combine(date.fromisoformat(day), value, tzinfo=_EASTERN).timestamp()
        * 1_000
    )


def _sessions():
    days = []
    current = date(2024, 1, 2)
    while len(days) < 70:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    source = semantic_sha256("adaptive-origin-calendar")
    rows = tuple(
        MassiveExchangeSession(
            session_date=day,
            exchange="XNYS",
            regular_open_ns=_ms(day, time(9, 30)) * 1_000_000,
            regular_close_ns=_ms(day, time(16)) * 1_000_000,
            scheduled_five_minute_intervals=78,
            special_session_reason=None,
            calendar_source_receipt_sha256=source,
        )
        for day in days
    )
    return build_massive_session_authority(
        rows, calendar_source_receipt_sha256=source
    )


def _membership(
    *, sessions, effective_index: int, available_index: int
) -> tuple[HistoricalMembershipRecord, ...]:
    effective = sessions.sessions[effective_index].regular_open_ns // 1_000_000
    available = sessions.sessions[available_index].regular_close_ns // 1_000_000
    rule = checked_pit_universe_rule(
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.action_universe_rule
    )
    group = semantic_sha256(("rank-group", effective))
    return tuple(
        HistoricalMembershipRecord(
            security_id=security_id,
            effective_at_ms=effective,
            available_at_ms=available,
            observation_end_ms=available,
            is_member=True,
            universe_rank=index + 1,
            eligibility_reason="top-k",
            rank_input_group_receipt_sha256=group,
            universe_rule_receipt_sha256=rule.receipt_sha256,
        )
        for index, security_id in enumerate(_ASSETS)
    )


def _identity(*, sessions, include_future: bool = False, wrong_rule: bool = False):
    listing = sessions.sessions[0].regular_open_ns // 1_000_000 - 1
    masters = tuple(
        SourcedSecurityMasterRecord(
            security_id=security_id,
            issuer_id=f"ISS-{security_id}",
            primary_exchange="XNYS",
            share_class="COMMON",
            security_type="common-stock",
            listing_at_ms=listing,
            delisting_at_ms=None,
            successor_security_id=None,
            corporate_action_chain_id=security_id,
            identity_source_receipt_sha256=semantic_sha256((security_id, "master")),
        )
        for security_id in _ASSETS
    )
    rule = checked_pit_universe_rule(
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.context_universe_rule
        if wrong_rule
        else MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.action_universe_rule
    )
    memberships = _membership(
        sessions=sessions, effective_index=63, available_index=62
    )
    if include_future:
        memberships += _membership(
            sessions=sessions, effective_index=64, available_index=63
        )
    return SimpleNamespace(
        rule=rule,
        security_master=masters,
        membership_events=memberships,
        validate=lambda: None,
    )


def _daily(*, sessions, missing_security: str | None = None, include_future: bool = False):
    selected_sessions = sessions.sessions[:65] if include_future else sessions.sessions[:64]
    session_rows = tuple(
        SimpleNamespace(
            source_session_date=session.session_date,
            vendor_last_modified_at_ms=session.regular_close_ns // 1_000_000,
            receipt_sha256=semantic_sha256((session.session_date, "session")),
        )
        for session in selected_sessions
    )
    rows = {}
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    for session_index, session in enumerate(selected_sessions):
        for asset_index, security_id in enumerate(_ASSETS):
            values = [1.0] * len(MASSIVE_DAILY_BARS_V0_FIELDS)
            values[close_index] = 50.0 + asset_index + 0.1 * session_index
            values[dollar_index] = 10_000_000.0 + 1_000.0 * asset_index
            valid = [True] * len(values)
            if missing_security == security_id and session_index == 20:
                values[close_index] = 0.0
                valid[close_index] = False
            rows[(session.session_date, security_id)] = SimpleNamespace(
                bars_values=tuple(values),
                bars_valid=tuple(valid),
                daily_bar_row_receipt_sha256=semantic_sha256(
                    (session.session_date, security_id, "bar")
                ),
                receipt_sha256=semantic_sha256(
                    (session.session_date, security_id, "daily-row")
                ),
            )
    return SimpleNamespace(
        session_authority_receipt_sha256=sessions.receipt_sha256,
        sessions=session_rows,
        semantic_receipt_sha256=semantic_sha256(
            ("daily", "future" if include_future else "base")
        ),
        validate=lambda: None,
        row=lambda *, session_date, security_id: rows[(session_date, security_id)],
    )


def _roots(monkeypatch, *, include_future_membership=False, include_future_daily=False, missing=None, wrong_rule=False):
    sessions = _sessions()
    clock = build_massive_decision_clock_authority(
        session_authority=sessions, session=sessions.sessions[63]
    )
    terminal_source = semantic_sha256("terminal-source")
    terminal = SimpleNamespace(
        rows=(),
        terminal_source_semantic_receipt_sha256=terminal_source,
        validate=lambda: None,
    )
    coverage = SimpleNamespace(
        loaded_source=object(),
        decision_at_ms=clock.decision_at_ns // 1_000_000,
        terminal_source_receipt_sha256=terminal_source,
        selected_events=(),
        validate=lambda: None,
    )
    monkeypatch.setattr(
        "rl_quant.features.massive_adaptive_origin_authority_v1.parse_massive_economic_origin_coverage_v8",
        lambda **_: coverage,
    )
    monkeypatch.setattr(
        "rl_quant.features.massive_adaptive_origin_authority_v1._ordered_events",
        lambda **_: (),
    )
    return {
        "economic_coverage_root": "/unused",
        "decision_clock": clock,
        "session_authority": sessions,
        "identity_authority": _identity(
            sessions=sessions,
            include_future=include_future_membership,
            wrong_rule=wrong_rule,
        ),
        "daily_input_authority": _daily(
            sessions=sessions,
            missing_security=missing,
            include_future=include_future_daily,
        ),
        "terminal_authority": terminal,
        "economic_coverage": coverage,
    }


def test_origin_derives_membership_and_six_source_time_exposures(monkeypatch) -> None:
    result = build_massive_adaptive_origin_authority_v1(**_roots(monkeypatch))

    assert result.security_ids == _ASSETS
    assert result.universe_ranks == tuple(range(1, 9))
    assert result.exposure_panel.exposure_names == MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1
    assert result.exposure_panel.qualified_asset_mask == (True,) * 8
    assert all(row.exposures[0] == 1.0 for row in result.rows)
    assert result.source_paths_replayed
    assert not result.predictive_training_authorized
    assert not result.profitability_reporting_authorized
    assert not result.lockbox_access_authorized
    assert not result.reinforcement_learning_authorized


def test_future_membership_and_daily_additions_do_not_change_origin(monkeypatch) -> None:
    baseline = build_massive_adaptive_origin_authority_v1(**_roots(monkeypatch))
    future = build_massive_adaptive_origin_authority_v1(
        **_roots(
            monkeypatch,
            include_future_membership=True,
            include_future_daily=True,
        )
    )

    assert future == baseline
    assert future.semantic_receipt_sha256 == baseline.semantic_receipt_sha256


def test_missing_history_is_zero_masked_without_changing_action_support(monkeypatch) -> None:
    result = build_massive_adaptive_origin_authority_v1(
        **_roots(monkeypatch, missing="SEC-07")
    )

    assert result.security_ids == _ASSETS
    assert result.exposure_panel.qualified_asset_mask == (True,) * 7 + (False,)
    assert result.rows[-1].exposures == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert result.rows[-1].regression_weight == 1.0


def test_context_universe_cannot_substitute_for_action_universe(monkeypatch) -> None:
    with pytest.raises(MassiveAdaptiveOriginAuthorityV1Error, match="roots"):
        build_massive_adaptive_origin_authority_v1(
            **_roots(monkeypatch, wrong_rule=True)
        )


def test_runtime_flag_mutation_breaks_origin_receipt(monkeypatch) -> None:
    result = build_massive_adaptive_origin_authority_v1(**_roots(monkeypatch))
    with pytest.raises(MassiveAdaptiveOriginAuthorityV1Error, match="authorizes"):
        replace(result, predictive_training_authorized=True).validate()
