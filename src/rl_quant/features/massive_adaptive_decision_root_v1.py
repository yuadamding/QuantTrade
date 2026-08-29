"""One reconciled context, action, and feature root per adaptive decision.

The context and action authorities are independently useful, but a training
path must not combine roots merely because they share an ISO date.  This
package-owned root binds their exact decision timestamp, clock, session
calendar, feature inventory, and nested universe relationship before any
model tensor or economic target is opened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-decision-root-v1"
)
MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "context": "exact-source-owned-pit1500-context-origin-v1",
        "action": "exact-source-owned-pit500-action-origin-v1",
        "features": "exact-feature-v3-cross-section",
        "clock": "one-identical-decision-timestamp-and-clock-receipt",
        "session": "one-identical-session-authority-receipt",
        "support": "action-security-ids-strict-subset-or-equal-context-security-ids",
        "history": "feature-input-inventory-bound-explicitly",
        "duration_prior": False,
        "downstream_authorization": False,
    }
)


class MassiveAdaptiveDecisionRootV1Error(ValueError):
    """Adaptive context, action, or feature roots cannot be reconciled."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveDecisionRootV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveDecisionRootV1:
    decision_session_date: str
    decision_at_ms: int
    feature_source_session_date: str
    feature_cutoff_at_ms: int
    feature_input_session_dates: tuple[str, ...]
    context_security_ids: tuple[str, ...]
    action_security_ids: tuple[str, ...]
    context_origin_receipt_sha256: str
    action_origin_receipt_sha256: str
    feature_semantic_receipt_sha256: str
    feature_audit_receipt_sha256: str
    decision_clock_receipt_sha256: str
    session_authority_receipt_sha256: str
    context_universe_rule_receipt_sha256: str
    action_universe_rule_receipt_sha256: str
    feature_row_inventory_sha256: str
    source_root_inventory_sha256: str
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
    schema: str = MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SOURCE_SHA256
            or not self.decision_session_date
            or isinstance(self.decision_at_ms, bool)
            or not isinstance(self.decision_at_ms, int)
            or self.decision_at_ms <= 0
            or self.feature_cutoff_at_ms > self.decision_at_ms
            or len(self.feature_input_session_dates) != 64
            or self.feature_input_session_dates
            != tuple(sorted(set(self.feature_input_session_dates)))
            or self.feature_input_session_dates[-1]
            != self.feature_source_session_date
            or self.context_security_ids
            != tuple(sorted(set(self.context_security_ids)))
            or self.action_security_ids
            != tuple(sorted(set(self.action_security_ids)))
            or not self.context_security_ids
            or not self.action_security_ids
            or not set(self.action_security_ids) <= set(self.context_security_ids)
            or not self.source_paths_replayed
            or not isinstance(self.source_data_qualified, bool)
            or self.development_training_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveDecisionRootV1Error(
                "adaptive decision-root identity or qualification differs"
            )
        expected_inventory = semantic_sha256(
            {
                "context_origin": self.context_origin_receipt_sha256,
                "action_origin": self.action_origin_receipt_sha256,
                "feature_semantic": self.feature_semantic_receipt_sha256,
                "feature_audit": self.feature_audit_receipt_sha256,
                "decision_clock": self.decision_clock_receipt_sha256,
                "session_authority": self.session_authority_receipt_sha256,
            }
        )
        if (
            self.source_root_inventory_sha256 != expected_inventory
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveDecisionRootV1Error(
                "adaptive decision-root receipt differs"
            )
        for name in (
            "context_origin_receipt_sha256",
            "action_origin_receipt_sha256",
            "feature_semantic_receipt_sha256",
            "feature_audit_receipt_sha256",
            "decision_clock_receipt_sha256",
            "session_authority_receipt_sha256",
            "context_universe_rule_receipt_sha256",
            "action_universe_rule_receipt_sha256",
            "feature_row_inventory_sha256",
            "source_root_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        assert_no_adaptive_hold_semantics(asdict(self))


def build_massive_adaptive_decision_root_v1(
    *,
    context_origin: MassiveAdaptiveContextOriginAuthorityV1,
    action_origin: MassiveAdaptiveOriginAuthorityV1,
    features: MassiveProfitabilityOriginFeaturesV3,
) -> MassiveAdaptiveDecisionRootV1:
    """Reconcile one feature cross-section with both universe authorities."""

    context_origin.validate()
    action_origin.validate()
    features.validate()
    feature_ids = tuple(row.security_id for row in features.rows)
    if (
        context_origin.decision_session_date
        != action_origin.decision_session_date
        or context_origin.decision_session_date
        != features.decision_session_date
        or context_origin.decision_at_ms != action_origin.decision_at_ms
        or context_origin.decision_clock_receipt_sha256
        != action_origin.decision_clock_receipt_sha256
        or context_origin.session_authority_receipt_sha256
        != action_origin.session_authority_receipt_sha256
        or context_origin.feature_semantic_receipt_sha256
        != features.semantic_receipt_sha256
        or context_origin.feature_row_inventory_sha256
        != features.row_inventory_sha256
        or context_origin.feature_source_input_inventory_sha256
        != features.source_input_inventory_sha256
        or context_origin.source_session_date != features.source_session_date
        or context_origin.feature_cutoff_at_ms != features.feature_cutoff_at_ms
        or context_origin.feature_input_session_dates
        != features.input_session_dates
        or context_origin.security_ids != feature_ids
        or action_origin.security_ids
        != tuple(sorted(action_origin.security_ids))
        or not set(action_origin.security_ids) <= set(context_origin.security_ids)
    ):
        raise MassiveAdaptiveDecisionRootV1Error(
            "adaptive context, action, feature, clock, or session roots differ"
        )
    inventory = semantic_sha256(
        {
            "context_origin": context_origin.semantic_receipt_sha256,
            "action_origin": action_origin.semantic_receipt_sha256,
            "feature_semantic": features.semantic_receipt_sha256,
            "feature_audit": features.audit_receipt_sha256,
            "decision_clock": context_origin.decision_clock_receipt_sha256,
            "session_authority": context_origin.session_authority_receipt_sha256,
        }
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SCHEMA,
        "decision_session_date": context_origin.decision_session_date,
        "decision_at_ms": context_origin.decision_at_ms,
        "feature_source_session_date": features.source_session_date,
        "feature_cutoff_at_ms": features.feature_cutoff_at_ms,
        "feature_input_session_dates": features.input_session_dates,
        "context_security_ids": context_origin.security_ids,
        "action_security_ids": action_origin.security_ids,
        "context_origin_receipt_sha256": context_origin.semantic_receipt_sha256,
        "action_origin_receipt_sha256": action_origin.semantic_receipt_sha256,
        "feature_semantic_receipt_sha256": features.semantic_receipt_sha256,
        "feature_audit_receipt_sha256": features.audit_receipt_sha256,
        "decision_clock_receipt_sha256": context_origin.decision_clock_receipt_sha256,
        "session_authority_receipt_sha256": context_origin.session_authority_receipt_sha256,
        "context_universe_rule_receipt_sha256": (
            context_origin.context_universe_rule_receipt_sha256
        ),
        "action_universe_rule_receipt_sha256": (
            action_origin.action_universe_rule_receipt_sha256
        ),
        "feature_row_inventory_sha256": features.row_inventory_sha256,
        "source_root_inventory_sha256": inventory,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SOURCE_SHA256,
        "source_paths_replayed": True,
        "source_data_qualified": (
            context_origin.source_data_qualified
            and features.source_inputs_data_qualified
        ),
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveAdaptiveDecisionRootV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SCHEMA",
    "MassiveAdaptiveDecisionRootV1",
    "MassiveAdaptiveDecisionRootV1Error",
    "build_massive_adaptive_decision_root_v1",
]
