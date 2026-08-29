"""Source-owned PIT-1500 context support for adaptive alpha decisions.

The P0 Feature V3 artifact proves the supplied feature rows, but it does not
prove that those rows are exactly the adaptive protocol's context universe.
This authority resolves the latest causal membership group under the frozen
PIT-1500 rule and requires exact equality with the feature cross-section.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.alpha.massive_universe_adapter import checked_pit_universe_rule
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.decision_clock import MassiveDecisionClockAuthority
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-context-origin-authority-v1"
)
MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "universe": "latest-complete-effective-and-available-pit1500-membership-group",
        "features": "exact-feature-v3-security-cross-section",
        "clock": "adaptive-close-plus-60-minute-decision",
        "future_survival": False,
        "generic_rows": "prohibited",
        "training_authorization": False,
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveContextOriginAuthorityV1Error(ValueError):
    """The context feature cross-section differs from its PIT authority."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveContextOriginAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveContextOriginAuthorityV1:
    decision_session_date: str
    decision_at_ms: int
    membership_effective_at_ms: int
    membership_available_at_ms: int
    source_session_date: str
    feature_cutoff_at_ms: int
    feature_input_session_dates: tuple[str, ...]
    security_ids: tuple[str, ...]
    universe_ranks: tuple[int, ...]
    decision_clock_receipt_sha256: str
    session_authority_receipt_sha256: str
    identity_authority_receipt_sha256: str
    context_universe_rule_receipt_sha256: str
    membership_group_inventory_sha256: str
    membership_row_inventory_sha256: str
    feature_semantic_receipt_sha256: str
    feature_row_inventory_sha256: str
    feature_source_input_inventory_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    source_paths_replayed: bool
    source_data_qualified: bool
    development_training_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SOURCE_SHA256
            or not self.decision_session_date
            or isinstance(self.decision_at_ms, bool)
            or not isinstance(self.decision_at_ms, int)
            or self.decision_at_ms <= 0
            or self.membership_effective_at_ms > self.decision_at_ms
            or self.membership_available_at_ms > self.decision_at_ms
            or self.feature_cutoff_at_ms > self.decision_at_ms
            or len(self.feature_input_session_dates) != 64
            or self.feature_input_session_dates
            != tuple(sorted(set(self.feature_input_session_dates)))
            or self.feature_input_session_dates[-1] != self.source_session_date
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or not self.security_ids
            or len(self.universe_ranks) != len(self.security_ids)
            or len(set(self.universe_ranks)) != len(self.universe_ranks)
            or any(rank <= 0 for rank in self.universe_ranks)
            or self.context_universe_rule_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.context_universe_rule.receipt_sha256
            or not self.source_paths_replayed
            or not isinstance(self.source_data_qualified, bool)
            or self.development_training_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveContextOriginAuthorityV1Error(
                "adaptive context origin identity or qualification differs"
            )
        for name in (
            "decision_clock_receipt_sha256",
            "session_authority_receipt_sha256",
            "identity_authority_receipt_sha256",
            "context_universe_rule_receipt_sha256",
            "membership_group_inventory_sha256",
            "membership_row_inventory_sha256",
            "feature_semantic_receipt_sha256",
            "feature_row_inventory_sha256",
            "feature_source_input_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveAdaptiveContextOriginAuthorityV1Error(
                "adaptive context origin receipt differs"
            )
        assert_no_adaptive_hold_semantics(asdict(self))


def build_massive_adaptive_context_origin_authority_v1(
    *,
    decision_clock: MassiveDecisionClockAuthority,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    features: MassiveProfitabilityOriginFeaturesV3,
) -> MassiveAdaptiveContextOriginAuthorityV1:
    """Resolve PIT-1500 support and bind it to one exact Feature V3 origin."""

    decision_clock.validate()
    session_authority.validate()
    identity_authority.validate()
    features.validate()
    context_rule = checked_pit_universe_rule(
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.context_universe_rule
    )
    decision_at_ms = decision_clock.decision_at_ns // 1_000_000
    if (
        identity_authority.rule.receipt_sha256 != context_rule.receipt_sha256
        or decision_clock.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or decision_clock.session_date != features.decision_session_date
        or features.feature_cutoff_at_ms > decision_at_ms
        or features.maximum_source_available_at_ms > decision_at_ms
    ):
        raise MassiveAdaptiveContextOriginAuthorityV1Error(
            "adaptive context clock, session, identity, or feature roots differ"
        )
    effective_times = tuple(
        sorted(
            {
                row.effective_at_ms
                for row in identity_authority.membership_events
                if row.effective_at_ms <= decision_at_ms
                and row.available_at_ms <= decision_at_ms
            }
        )
    )
    if not effective_times:
        raise MassiveAdaptiveContextOriginAuthorityV1Error(
            "no causal adaptive context membership exists"
        )
    effective = effective_times[-1]
    group = tuple(
        row
        for row in identity_authority.membership_events
        if row.effective_at_ms == effective
    )
    if not group or any(
        row.available_at_ms > decision_at_ms
        or row.observation_end_ms >= decision_at_ms
        for row in group
    ):
        raise MassiveAdaptiveContextOriginAuthorityV1Error(
            "adaptive context membership group is not causal"
        )
    members = tuple(
        sorted(
            (row for row in group if row.is_member),
            key=lambda row: row.security_id,
        )
    )
    security_ids = tuple(row.security_id for row in members)
    feature_ids = tuple(row.security_id for row in features.rows)
    if not security_ids or security_ids != feature_ids:
        raise MassiveAdaptiveContextOriginAuthorityV1Error(
            "Feature V3 rows are not the exact PIT-1500 context membership"
        )
    ranks = tuple(int(row.universe_rank or 0) for row in members)
    if any(rank <= 0 for rank in ranks):
        raise MassiveAdaptiveContextOriginAuthorityV1Error(
            "adaptive context membership rank is absent"
        )
    membership_rows = tuple(semantic_sha256(asdict(row)) for row in group)
    body = {
        "schema": MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SCHEMA,
        "decision_session_date": decision_clock.session_date,
        "decision_at_ms": decision_at_ms,
        "membership_effective_at_ms": effective,
        "membership_available_at_ms": max(row.available_at_ms for row in group),
        "source_session_date": features.source_session_date,
        "feature_cutoff_at_ms": features.feature_cutoff_at_ms,
        "feature_input_session_dates": features.input_session_dates,
        "security_ids": security_ids,
        "universe_ranks": ranks,
        "decision_clock_receipt_sha256": decision_clock.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "context_universe_rule_receipt_sha256": context_rule.receipt_sha256,
        "membership_group_inventory_sha256": semantic_sha256(
            tuple(row.security_id for row in group)
        ),
        "membership_row_inventory_sha256": semantic_sha256(membership_rows),
        "feature_semantic_receipt_sha256": features.semantic_receipt_sha256,
        "feature_row_inventory_sha256": features.row_inventory_sha256,
        "feature_source_input_inventory_sha256": features.source_input_inventory_sha256,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SOURCE_SHA256,
        "source_paths_replayed": True,
        "source_data_qualified": features.source_inputs_data_qualified,
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveAdaptiveContextOriginAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SCHEMA",
    "MassiveAdaptiveContextOriginAuthorityV1",
    "MassiveAdaptiveContextOriginAuthorityV1Error",
    "build_massive_adaptive_context_origin_authority_v1",
]
