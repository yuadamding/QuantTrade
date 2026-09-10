from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.alpha.massive_universe_adapter import checked_pit_universe_rule
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.decision_clock import (
    build_massive_decision_clock_authority,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1Error,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1,
    MassiveAdaptiveOriginAuthorityV1Error,
)
from rl_quant.features.massive_adaptive_real_source_bridge_v1 import (
    MassiveAdaptiveRealSourceBridgeV1Error,
    build_massive_adaptive_real_source_bridge_v1,
)
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    materialize_massive_profitability_accounting_freeze_for_test_v1,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    build_massive_profitability_origin_features_v3,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
)
from test_massive_profitability_source_derived_accounting_v2 import (
    _ENTITLEMENT,
    _coverage,
    _daily,
    _identity,
    _origin,
    _origin_plan,
    _sessions,
    _terminal,
)

_SOURCE_ASSET_IDS = tuple(f"SEC-{letter}" for letter in "ABCDEFG")


def _fixture_row(row, **changes):
    """Construct another explicitly synthetic native input row, not an output."""
    value = replace(row, **changes)
    value = replace(value, receipt_sha256=semantic_sha256(value.unsigned()))
    value.validate()
    return value


def _fixture_authority(authority, audit_field, **changes):
    value = replace(authority, **changes)
    receipt = semantic_sha256(value.semantic_unsigned())
    value = replace(
        value,
        semantic_receipt_sha256=receipt,
        audit_receipt_sha256=semantic_sha256(
            {"semantic_receipt_sha256": receipt, audit_field: getattr(value, audit_field)}
        ),
    )
    value.validate()
    return value


def _seven_asset_identity(identity):
    """Seven sourced synthetic identities leave one DOF beyond six exposures."""
    masters, tickers, listings, delistings, ranks = [], [], [], [], []
    for index, security_id in enumerate(_SOURCE_ASSET_IDS):
        ticker = f"FIX{index}"
        masters.append(replace(
            identity.security_master[0], security_id=security_id,
            issuer_id=f"ISS-{index}", corporate_action_chain_id=f"CHAIN-{index}",
            identity_source_receipt_sha256=semantic_sha256((security_id, "master")),
        ))
        tickers.append(replace(
            identity.ticker_history[0], security_id=security_id, ticker=ticker,
            source_receipt_sha256=semantic_sha256((security_id, "ticker")),
        ))
        listings.append(replace(
            identity.listing_events[0], security_id=security_id, ticker=ticker,
            event_id=f"LIST-{index}", source_receipt_sha256=semantic_sha256((security_id, "listing")),
        ))
        delistings.append(replace(
            identity.delisting_events[0], security_id=security_id, event_id=f"DELIST-{index}",
            source_receipt_sha256=semantic_sha256((security_id, "delisting")),
        ))
        ranks.append(replace(
            identity.rank_inputs[0], security_id=security_id, close_price=163.0 + index,
            source_receipt_sha256=semantic_sha256((security_id, "rank")),
        ))
    return PITSecurityUniverseAuthority.build(
        rule=identity.rule, security_master=tuple(masters), ticker_history=tuple(tickers),
        listing_events=tuple(listings), delisting_events=tuple(delistings), rank_inputs=tuple(ranks),
    )


def _seven_asset_sources(identity, origin, daily, terminal):
    # Author native typed source rows for every asset/session, then let the
    # production bridge derive features and exposures. Never patch its outputs
    # or factor validator, and preserve the fixture's unqualified source flags.
    rows, session_rows = [], []
    for daily_row, session_row in zip(daily.rows, daily.sessions, strict=True):
        group = []
        for index, security_id in enumerate(_SOURCE_ASSET_IDS):
            scale = 1.0 + index / 10.0
            bars = daily_row.bars_values
            values = (
                *(price * scale for price in bars[:4]),
                bars[4] * (index + 1), bars[5] * scale * (index + 1), *bars[6:],
            )
            group.append(_fixture_row(
                daily_row, security_id=security_id, bars_values=values,
                daily_bar_row_receipt_sha256=semantic_sha256((daily_row.source_session_date, security_id, "bar", values)),
                daily_tape_row_receipt_sha256=semantic_sha256((daily_row.source_session_date, security_id, "tape")),
                tape_population_row_receipt_sha256=semantic_sha256((daily_row.source_session_date, security_id, "population")),
                persisted_partition_receipt_sha256=semantic_sha256((daily_row.source_session_date, security_id, "partition")),
            ))
        rows.extend(group)
        session_rows.append(_fixture_row(
            session_row,
            supported_security_row_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in group)),
        ))
    daily = _fixture_authority(
        daily, "acquisition_audit_receipt_sha256", supported_security_ids=_SOURCE_ASSET_IDS,
        rows=tuple(rows), sessions=tuple(session_rows),
        row_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        session_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in session_rows)),
    )
    terminal_rows = tuple(
        _fixture_row(
            terminal.rows[0], security_id=event.security_id,
            listing_delisting_event_id=event.event_id,
            identity_delisting_receipt_sha256=event.source_receipt_sha256,
        )
        for event in identity.delisting_events
    )
    terminal = _fixture_authority(
        terminal, "terminal_source_audit_receipt_sha256", supported_security_ids=_SOURCE_ASSET_IDS,
        rows=terminal_rows, conservative_total_loss_count=len(terminal_rows),
        row_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in terminal_rows)),
    )
    origin = replace(
        origin, decision_member_security_ids=_SOURCE_ASSET_IDS,
        decision_member_universe_ranks=tuple(range(1, len(_SOURCE_ASSET_IDS) + 1)),
    )
    receipt = semantic_sha256(origin.semantic_unsigned())
    origin = replace(
        origin, receipt_sha256=receipt,
        audit_receipt_sha256=semantic_sha256({
            "semantic_receipt_sha256": receipt,
            "identity_authority_audit_receipt_sha256": origin.identity_authority_audit_receipt_sha256,
        }),
    )
    origin.validate()
    return origin, daily, terminal


def _with_rule(identity, rule, *, changed_rank_source=False):
    ranks = identity.rank_inputs
    if changed_rank_source:
        ranks = (
            replace(ranks[0], source_receipt_sha256=semantic_sha256("changed-rank")),
            *ranks[1:],
        )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=identity.security_master,
        ticker_history=identity.ticker_history,
        listing_events=identity.listing_events,
        delisting_events=identity.delisting_events,
        rank_inputs=ranks,
    )


def _inputs(root: Path, *, extra_context_member: bool = False, seven_assets: bool = False):
    # Typed synthetic source fixtures remain explicitly unqualified. The bridge
    # itself calls native accounting, feature and adaptive builders; no patching.
    sessions = _sessions()
    identity = _identity(sessions)
    if seven_assets:
        assert not extra_context_member
        identity = _seven_asset_identity(identity)
    if extra_context_member:
        master = replace(
            identity.security_master[0],
            security_id="SEC-B",
            issuer_id="ISS-B",
            corporate_action_chain_id="CHAIN-B",
        )
        ticker = replace(identity.ticker_history[0], security_id="SEC-B", ticker="BBB")
        listing = replace(
            identity.listing_events[0],
            event_id="LIST-B",
            security_id="SEC-B",
            ticker="BBB",
        )
        delisting = replace(
            identity.delisting_events[0], event_id="DELIST-B", security_id="SEC-B"
        )
        rank = replace(
            identity.rank_inputs[0],
            security_id="SEC-B",
            close_price=2.0,
            average_dollar_volume=1_000_000.0,
        )
        identity = PITSecurityUniverseAuthority.build(
            rule=identity.rule,
            security_master=(*identity.security_master, master),
            ticker_history=(*identity.ticker_history, ticker),
            listing_events=(*identity.listing_events, listing),
            delisting_events=(*identity.delisting_events, delisting),
            rank_inputs=(*identity.rank_inputs, rank),
        )
    origin = _origin(sessions)
    daily = _daily(sessions, identity)
    terminal = _terminal(sessions, identity)
    if seven_assets:
        origin, daily, terminal = _seven_asset_sources(identity, origin, daily, terminal)
    plan = _origin_plan(origin)
    clock = build_massive_decision_clock_authority(
        session_authority=sessions, session=sessions.sessions[65]
    )
    feature_coverage = _coverage(
        root=root,
        sessions=sessions,
        terminal=terminal,
        origin=origin,
        events=(),
        artifact_id="bridge-feature-economics",
    )
    # The fixture publisher needs only the timestamp; the object returned is a
    # genuine persisted+parsed MassiveEconomicOriginCoverageV8, not this stand-in.
    action_coverage = _coverage(
        root=root,
        sessions=sessions,
        terminal=terminal,
        origin=SimpleNamespace(decision_at_ms=clock.decision_at_ns // 1_000_000),
        events=(),
        artifact_id="bridge-action-economics",
    )
    freeze = materialize_massive_profitability_accounting_freeze_for_test_v1(
        root=root,
        archive_freeze_semantic_receipt_sha256=(
            daily.archive_freeze_semantic_receipt_sha256
        ),
        origin_plan=plan,
        terminal_authority=terminal,
        economic_coverages=(feature_coverage,),
        accounting_freeze_at_ms=clock.decision_at_ns // 1_000_000 + 1_000,
        entitlement_receipt_sha256=_ENTITLEMENT,
        artifact_id="bridge-accounting",
    )
    return {
        "economic_coverage_root": root,
        "feature_origin": origin,
        "feature_origin_plan": plan,
        "decision_clock": clock,
        "session_authority": sessions,
        "feature_identity_authority": identity,
        "context_identity_authority": _with_rule(
            identity,
            checked_pit_universe_rule(
                MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.context_universe_rule
            ),
        ),
        "action_identity_authority": _with_rule(
            identity,
            checked_pit_universe_rule(
                MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.action_universe_rule
            ),
        ),
        "daily_input_authority": daily,
        "feature_economic_coverage": feature_coverage,
        "action_economic_coverage": action_coverage,
        "accounting_freeze": freeze,
        "terminal_authority": terminal,
    }


def test_bridge_rebuilds_native_features_without_promoting_fixture(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, seven_assets=True)
    result = build_massive_adaptive_real_source_bridge_v1(**inputs)
    expected_features = build_massive_profitability_origin_features_v3(
        origin=inputs["feature_origin"],
        origin_plan=inputs["feature_origin_plan"],
        session_authority=inputs["session_authority"],
        identity_authority=inputs["feature_identity_authority"],
        daily_input_authority=inputs["daily_input_authority"],
        feature_accounting=result.feature_accounting,
        accounting_freeze=inputs["accounting_freeze"],
        terminal_authority=inputs["terminal_authority"],
    )
    assert result.features == expected_features
    assert result.features.source_staleness_sessions == 2
    assert len(result.features.input_session_dates) == 64
    assert len(result.features.rows[0].bars_values) == 19
    assert len(result.features.rows[0].tape_values) == 15
    assert (
        result.context_origin.security_ids
        == result.action_origin.security_ids
        == _SOURCE_ASSET_IDS
    )
    assert sum(result.action_origin.exposure_panel.qualified_asset_mask) > len(
        MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1
    )
    assert (
        result.decision_root.decision_at_ms
        == inputs["decision_clock"].decision_at_ns // 1_000_000
    )
    assert (
        result.decision_root.decision_at_ms
        != inputs["feature_origin"].decision_at_ms
    )
    assert not result.features.source_inputs_data_qualified
    assert not result.decision_root.source_data_qualified
    assert not result.training_ready
    assert not result.decision_root.reinforcement_learning_authorized


def test_bridge_rejects_p0_coverage_as_adaptive_action_coverage(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["action_economic_coverage"] = inputs["feature_economic_coverage"]
    with pytest.raises(MassiveAdaptiveOriginAuthorityV1Error, match="roots differ"):
        build_massive_adaptive_real_source_bridge_v1(**inputs)


def test_bridge_rejects_adaptive_coverage_as_p0_feature_coverage(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["feature_economic_coverage"] = inputs["action_economic_coverage"]
    with pytest.raises(MassiveAdaptiveRealSourceBridgeV1Error, match="P0 origin clock"):
        build_massive_adaptive_real_source_bridge_v1(**inputs)


def test_bridge_rejects_source_identity_substitution(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    context = inputs["context_identity_authority"]
    inputs["context_identity_authority"] = _with_rule(
        context, context.rule, changed_rank_source=True
    )
    with pytest.raises(
        MassiveAdaptiveRealSourceBridgeV1Error, match="source records differ"
    ):
        build_massive_adaptive_real_source_bridge_v1(**inputs)


def test_bridge_rejects_untyped_daily_rows(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["daily_input_authority"] = SimpleNamespace(daily_input_data_qualified=True)
    with pytest.raises(
        MassiveAdaptiveRealSourceBridgeV1Error, match="native source-authority type"
    ):
        build_massive_adaptive_real_source_bridge_v1(**inputs)


def test_bridge_does_not_expand_p0_features_to_context_members(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, extra_context_member=True)
    assert sum(
        row.is_member for row in inputs["context_identity_authority"].membership_events
    ) == 2
    assert sum(
        row.is_member for row in inputs["action_identity_authority"].membership_events
    ) == 1
    with pytest.raises(
        MassiveAdaptiveContextOriginAuthorityV1Error, match="exact PIT-1500"
    ):
        build_massive_adaptive_real_source_bridge_v1(**inputs)
