"""Dedicated public surface for acquired monthly PIT-500 rank authority V2.

The implementation remains frozen in ``massive_profitability_origin_v2``.
This module gives archive-freeze and panel code one narrow import boundary
without copying or silently changing the existing V2 semantics.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.finalized_listing import (
    coverage_session_from_massive_trade_key,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_daily_bars_v0 import MassiveDailyBarsArtifactV0
from rl_quant.features.massive_profitability_origin_v2 import (
    MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SCHEMA,
    MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
    MassiveMonthlyRankInputAuthorityV2,
    MassiveMonthlyRankInputGroupV2,
    MassiveProfitabilityOriginV2Error,
    MassiveProfitabilityProductionAcquisitionV2,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    build_massive_monthly_rank_input_authority_v2 as _build_rank_authority_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_BINDING_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SCHEMA,
        "implementation_source_sha256": (MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256),
        "archive_derived_surface_source_sha256": (
            MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SOURCE_SHA256
        ),
        "specification_sha256": MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
        "public_surface": "dedicated-immutable-import-boundary",
    }
)


def build_massive_monthly_rank_input_authority_v2(
    *,
    root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    daily_bars: Sequence[MassiveDailyBarsArtifactV0],
) -> MassiveMonthlyRankInputAuthorityV2:
    """Derive every archive-supported monthly rank; caller dates are prohibited."""

    session_authority.validate()
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    dates = tuple(row.session_date for row in sessions)
    positions = {value: index for index, value in enumerate(dates)}
    source_dates = tuple(
        sorted(
            coverage_session_from_massive_trade_key(row.source_object_key)
            for row in acquisition.authenticated_downloads
        )
    )
    if (
        not source_dates
        or len(source_dates) != len(set(source_dates))
        or source_dates[0] not in positions
        or source_dates[-1] not in positions
    ):
        raise MassiveProfitabilityOriginV2Error(
            "monthly rank acquisition source inventory differs"
        )
    source_start = positions[source_dates[0]]
    source_end = positions[source_dates[-1]]
    if source_dates != dates[source_start : source_end + 1]:
        raise MassiveProfitabilityOriginV2Error(
            "monthly rank acquisition is not complete for every XNYS session"
        )

    first_by_month: defaultdict[str, int] = defaultdict(lambda: len(sessions))
    for index, session in enumerate(sessions):
        first_by_month[session.session_date[:7]] = min(
            first_by_month[session.session_date[:7]], index
        )
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    supported = tuple(
        (month, index)
        for month, index in sorted(first_by_month.items())
        if source_start
        <= index - rule.ranking_lag_sessions - rule.ranking_lookback_sessions + 1
        and index <= source_end
    )
    if not supported:
        raise MassiveProfitabilityOriginV2Error(
            "monthly rank acquisition has no complete scheduled lookback"
        )
    result = _build_rank_authority_v2(
        root=root,
        session_authority=session_authority,
        identity_authority=identity_authority,
        acquisition=acquisition,
        daily_bars=daily_bars,
        first_candidate_decision_session_date=sessions[supported[0][1]].session_date,
        last_candidate_decision_session_date=sessions[source_end].session_date,
    )
    if tuple(group.calendar_month for group in result.groups) != tuple(
        month for month, _ in supported
    ):
        raise MassiveProfitabilityOriginV2Error(
            "derived monthly rank schedule differs from the acquired archive"
        )
    return result


__all__ = [
    "MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_BINDING_SHA256",
    "MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SCHEMA",
    "MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SOURCE_SHA256",
    "MassiveMonthlyRankInputAuthorityV2",
    "MassiveMonthlyRankInputGroupV2",
    "MassiveProfitabilityOriginV2Error",
    "build_massive_monthly_rank_input_authority_v2",
]
