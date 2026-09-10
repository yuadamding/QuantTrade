"""Rebuild native adaptive inputs from existing, typed source authorities.

This is a read-only integration adapter, not a new feature generation or a
source-qualification authority. It preserves the P0 Feature V3 clock, 64-session
history, accounting and two-session staleness, then invokes the native adaptive
context, action and decision builders. It cannot expand a P0 feature population
to a larger adaptive context population. Such a mismatch fails closed.

The caller still owns historical candidate-scope qualification, upstream archive
reconstruction, protected-date access and subsequent forecast/training gates.
No synthetic fixture, caller feature array or qualification override is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.decision_clock import MassiveDecisionClockAuthority
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
    build_massive_adaptive_context_origin_authority_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
    build_massive_adaptive_origin_authority_v1,
)
from rl_quant.features.massive_economic_coverage_v8 import (
    MassiveEconomicOriginCoverageV8,
)
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    MassiveProfitabilityAccountingFreezeV1,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_experiment_coverage_v2 import (
    massive_profitability_identity_semantic_receipt_v2,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    MassiveProfitabilityFeatureAccountingAuthorityV2,
    build_massive_profitability_feature_accounting_authority_v2,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
    build_massive_profitability_origin_features_v3,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MassiveProfitabilityDecisionOriginV1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MassiveProfitabilityTerminalCoverageAuthorityV1,
)


class MassiveAdaptiveRealSourceBridgeV1Error(ValueError):
    """Native feature and adaptive inputs do not share compatible source roots."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRealSourceBridgeV1:
    """Native builder outputs; this container grants no new authorization."""

    feature_accounting: MassiveProfitabilityFeatureAccountingAuthorityV2
    features: MassiveProfitabilityOriginFeaturesV3
    context_origin: MassiveAdaptiveContextOriginAuthorityV1
    action_origin: MassiveAdaptiveOriginAuthorityV1
    decision_root: MassiveAdaptiveDecisionRootV1

    @property
    def training_ready(self) -> bool:
        return False

    def validate(self) -> None:
        self.feature_accounting.validate()
        self.features.validate()
        self.context_origin.validate()
        self.action_origin.validate()
        self.decision_root.validate()
        if (
            self.features.feature_accounting_authority_semantic_receipt_sha256
            != self.feature_accounting.semantic_receipt_sha256
            or self.decision_root
            != build_massive_adaptive_decision_root_v1(
                context_origin=self.context_origin,
                action_origin=self.action_origin,
                features=self.features,
            )
        ):
            raise MassiveAdaptiveRealSourceBridgeV1Error(
                "native bridge output roots differ"
            )


def build_massive_adaptive_real_source_bridge_v1(
    *,
    economic_coverage_root: str | Path,
    feature_origin: MassiveProfitabilityDecisionOriginV1,
    feature_origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    decision_clock: MassiveDecisionClockAuthority,
    session_authority: MassiveSessionAuthority,
    feature_identity_authority: PITSecurityUniverseAuthority,
    context_identity_authority: PITSecurityUniverseAuthority,
    action_identity_authority: PITSecurityUniverseAuthority,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    feature_economic_coverage: MassiveEconomicOriginCoverageV8,
    action_economic_coverage: MassiveEconomicOriginCoverageV8,
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
) -> MassiveAdaptiveRealSourceBridgeV1:
    """Compose production builders without changing any upstream identity.

    Feature economic coverage binds the P0 origin; action economic coverage
    separately binds the adaptive close-plus-60-minute decision. Both are read
    back through their native persisted-source APIs. Distinct universe rules
    must use the same sourced identity/rank records, not substituted masters.
    Qualification remains exactly what the native builders derive; unqualified
    typed upstream evidence is usable for engineering but remains unqualified.
    """

    roots = (
        ("feature origin", feature_origin, MassiveProfitabilityDecisionOriginV1),
        (
            "feature origin plan", feature_origin_plan,
            MassiveProfitabilityDecisionOriginPlanV2,
        ),
        ("decision clock", decision_clock, MassiveDecisionClockAuthority),
        ("session authority", session_authority, MassiveSessionAuthority),
        ("feature identity", feature_identity_authority, PITSecurityUniverseAuthority),
        ("context identity", context_identity_authority, PITSecurityUniverseAuthority),
        ("action identity", action_identity_authority, PITSecurityUniverseAuthority),
        (
            "daily inputs", daily_input_authority,
            MassiveProfitabilityDailyInputAuthorityV1,
        ),
        ("feature economics", feature_economic_coverage, MassiveEconomicOriginCoverageV8),
        ("action economics", action_economic_coverage, MassiveEconomicOriginCoverageV8),
        ("accounting freeze", accounting_freeze, MassiveProfitabilityAccountingFreezeV1),
        (
            "terminal authority", terminal_authority,
            MassiveProfitabilityTerminalCoverageAuthorityV1,
        ),
    )
    for name, value, expected_type in roots:
        if type(value) is not expected_type:
            raise MassiveAdaptiveRealSourceBridgeV1Error(
                f"{name} must use its native source-authority type"
            )
        value.validate()
    if (
        feature_origin.decision_session_date != decision_clock.session_date
        or feature_origin.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or decision_clock.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or daily_input_authority.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or daily_input_authority.normalized_identity_semantic_receipt_sha256
        != massive_profitability_identity_semantic_receipt_v2(
            feature_identity_authority
        )
        or terminal_authority.normalized_identity_semantic_receipt_sha256
        != daily_input_authority.normalized_identity_semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRealSourceBridgeV1Error(
            "feature and adaptive session or daily identity roots differ"
        )
    if feature_economic_coverage.decision_at_ms != feature_origin.decision_at_ms:
        raise MassiveAdaptiveRealSourceBridgeV1Error(
            "feature economic coverage is not bound to the P0 origin clock"
        )
    for identity in (context_identity_authority, action_identity_authority):
        if any(
            getattr(identity, field) != getattr(feature_identity_authority, field)
            for field in (
                "security_master",
                "ticker_history",
                "listing_events",
                "delisting_events",
                "rank_inputs",
            )
        ):
            raise MassiveAdaptiveRealSourceBridgeV1Error(
                "feature, context and action identity source records differ"
            )
    feature_accounting = build_massive_profitability_feature_accounting_authority_v2(
        root=economic_coverage_root,
        origin=feature_origin,
        origin_plan=feature_origin_plan,
        session_authority=session_authority,
        identity_authority=feature_identity_authority,
        daily_input_authority=daily_input_authority,
        economic_coverage=feature_economic_coverage,
        terminal_authority=terminal_authority,
    )
    features = build_massive_profitability_origin_features_v3(
        origin=feature_origin,
        origin_plan=feature_origin_plan,
        session_authority=session_authority,
        identity_authority=feature_identity_authority,
        daily_input_authority=daily_input_authority,
        feature_accounting=feature_accounting,
        accounting_freeze=accounting_freeze,
        terminal_authority=terminal_authority,
    )
    context = build_massive_adaptive_context_origin_authority_v1(
        decision_clock=decision_clock,
        session_authority=session_authority,
        identity_authority=context_identity_authority,
        features=features,
    )
    action = build_massive_adaptive_origin_authority_v1(
        economic_coverage_root=economic_coverage_root,
        decision_clock=decision_clock,
        session_authority=session_authority,
        identity_authority=action_identity_authority,
        daily_input_authority=daily_input_authority,
        terminal_authority=terminal_authority,
        economic_coverage=action_economic_coverage,
    )
    result = MassiveAdaptiveRealSourceBridgeV1(
        feature_accounting=feature_accounting,
        features=features,
        context_origin=context,
        action_origin=action,
        decision_root=build_massive_adaptive_decision_root_v1(
            context_origin=context, action_origin=action, features=features
        ),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRealSourceBridgeV1",
    "MassiveAdaptiveRealSourceBridgeV1Error",
    "build_massive_adaptive_real_source_bridge_v1",
]
