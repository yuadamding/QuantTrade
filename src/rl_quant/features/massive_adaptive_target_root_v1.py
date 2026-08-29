"""One source-replayed economic target root per adaptive decision.

The source-target wrapper proves a local economic-path reconstruction.  This
root closes the experiment boundary by rebuilding that wrapper from the live
Massive authorities and reconciling every target-side receipt with the exact
dual-universe decision root used by the model.  Synthetic fixtures have a
separate nonqualifying binder; only the live-root builder may derive source
qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.decision_clock import MassiveDecisionClockAuthority
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import MassiveAdaptiveFillSourceV1
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_source_targets_v1 import (
    MassiveAdaptiveSourceTargetsV1,
    build_massive_adaptive_source_targets_v1,
)
from rl_quant.features.massive_economic_coverage_v8 import (
    MassiveEconomicOriginCoverageV8,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MassiveProfitabilityTerminalCoverageAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SCHEMA = "rl-quant.massive-adaptive-target-root-v1"
MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "decision": "exact-dual-universe-decision-root-v1",
        "target_replay": (
            "clock-calendar-action-identity-daily-fill-terminal-coverage-and-path"
        ),
        "qualification": "derived-only-from-live-source-authorities",
        "synthetic_binding": "always-unqualified",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveTargetRootV1Error(ValueError):
    """Adaptive target and decision roots do not belong to one experiment."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveTargetRootV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveTargetSourceRuntimeV1:
    """Live roots required to independently reconstruct one target artifact."""

    economic_coverage_root: str | Path
    decision_clock: MassiveDecisionClockAuthority
    session_authority: MassiveSessionAuthority
    action_identity_authority: PITSecurityUniverseAuthority
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1
    fill_source: MassiveAdaptiveFillSourceV1
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1
    economic_coverage: MassiveEconomicOriginCoverageV8
    action_origin: MassiveAdaptiveOriginAuthorityV1


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveTargetRootV1:
    decision_session_date: str
    decision_at_ms: int
    fill_session_date: str
    security_ids: tuple[str, ...]
    decision_root_receipt_sha256: str
    decision_source_inventory_sha256: str
    source_target_receipt_sha256: str
    action_origin_receipt_sha256: str
    decision_clock_receipt_sha256: str
    session_authority_receipt_sha256: str
    identity_authority_receipt_sha256: str
    daily_input_authority_receipt_sha256: str
    fill_source_receipt_sha256: str
    terminal_authority_receipt_sha256: str
    economic_coverage_receipt_sha256: str
    economic_path_inventory_sha256: str
    target_receipt_sha256: str
    experiment_source_receipt_sha256: str
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
    schema: str = MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SCHEMA
            or not self.decision_session_date
            or isinstance(self.decision_at_ms, bool)
            or not isinstance(self.decision_at_ms, int)
            or self.decision_at_ms <= 0
            or not self.fill_session_date
            or not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SOURCE_SHA256
            or not self.source_paths_replayed
            or not isinstance(self.source_data_qualified, bool)
            or self.development_training_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveTargetRootV1Error(
                "adaptive target-root identity or qualification differs"
            )
        expected_experiment = semantic_sha256(
            {
                "protocol": self.protocol_receipt_sha256,
                "decision_root": self.decision_root_receipt_sha256,
                "decision_source_inventory": self.decision_source_inventory_sha256,
                "action_origin": self.action_origin_receipt_sha256,
                "decision_clock": self.decision_clock_receipt_sha256,
                "session_authority": self.session_authority_receipt_sha256,
                "identity_authority": self.identity_authority_receipt_sha256,
                "daily_input_authority": self.daily_input_authority_receipt_sha256,
                "fill_source": self.fill_source_receipt_sha256,
                "terminal_authority": self.terminal_authority_receipt_sha256,
                "economic_coverage": self.economic_coverage_receipt_sha256,
            }
        )
        if self.experiment_source_receipt_sha256 != expected_experiment:
            raise MassiveAdaptiveTargetRootV1Error(
                "adaptive target experiment-source receipt differs"
            )
        for name in (
            "decision_root_receipt_sha256",
            "decision_source_inventory_sha256",
            "source_target_receipt_sha256",
            "action_origin_receipt_sha256",
            "decision_clock_receipt_sha256",
            "session_authority_receipt_sha256",
            "identity_authority_receipt_sha256",
            "daily_input_authority_receipt_sha256",
            "fill_source_receipt_sha256",
            "terminal_authority_receipt_sha256",
            "economic_coverage_receipt_sha256",
            "economic_path_inventory_sha256",
            "target_receipt_sha256",
            "experiment_source_receipt_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        assert_no_adaptive_hold_semantics(asdict(self))


def _bind_target_root(
    *,
    decision_root: MassiveAdaptiveDecisionRootV1,
    source_target: MassiveAdaptiveSourceTargetsV1,
    source_data_qualified: bool,
) -> MassiveAdaptiveTargetRootV1:
    decision_root.validate()
    source_target.validate()
    if (
        decision_root.decision_session_date != source_target.decision_session_date
        or decision_root.decision_at_ms != source_target.targets.decision_at_ms
        or decision_root.action_security_ids != source_target.security_ids
        or decision_root.action_origin_receipt_sha256
        != source_target.origin_authority_receipt_sha256
        or decision_root.decision_clock_receipt_sha256
        != source_target.decision_clock_receipt_sha256
        or decision_root.session_authority_receipt_sha256
        != source_target.session_authority_receipt_sha256
    ):
        raise MassiveAdaptiveTargetRootV1Error(
            "adaptive decision and target roots differ"
        )
    experiment = semantic_sha256(
        {
            "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
            "decision_root": decision_root.semantic_receipt_sha256,
            "decision_source_inventory": decision_root.source_root_inventory_sha256,
            "action_origin": source_target.origin_authority_receipt_sha256,
            "decision_clock": source_target.decision_clock_receipt_sha256,
            "session_authority": source_target.session_authority_receipt_sha256,
            "identity_authority": source_target.identity_authority_receipt_sha256,
            "daily_input_authority": (
                source_target.daily_input_authority_receipt_sha256
            ),
            "fill_source": source_target.fill_source_receipt_sha256,
            "terminal_authority": source_target.terminal_authority_receipt_sha256,
            "economic_coverage": source_target.economic_coverage_receipt_sha256,
        }
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SCHEMA,
        "decision_session_date": source_target.decision_session_date,
        "decision_at_ms": source_target.targets.decision_at_ms,
        "fill_session_date": source_target.fill_session_date,
        "security_ids": source_target.security_ids,
        "decision_root_receipt_sha256": decision_root.semantic_receipt_sha256,
        "decision_source_inventory_sha256": (
            decision_root.source_root_inventory_sha256
        ),
        "source_target_receipt_sha256": source_target.semantic_receipt_sha256,
        "action_origin_receipt_sha256": (
            source_target.origin_authority_receipt_sha256
        ),
        "decision_clock_receipt_sha256": (
            source_target.decision_clock_receipt_sha256
        ),
        "session_authority_receipt_sha256": (
            source_target.session_authority_receipt_sha256
        ),
        "identity_authority_receipt_sha256": (
            source_target.identity_authority_receipt_sha256
        ),
        "daily_input_authority_receipt_sha256": (
            source_target.daily_input_authority_receipt_sha256
        ),
        "fill_source_receipt_sha256": source_target.fill_source_receipt_sha256,
        "terminal_authority_receipt_sha256": (
            source_target.terminal_authority_receipt_sha256
        ),
        "economic_coverage_receipt_sha256": (
            source_target.economic_coverage_receipt_sha256
        ),
        "economic_path_inventory_sha256": (
            source_target.economic_path_inventory_sha256
        ),
        "target_receipt_sha256": source_target.target_receipt_sha256,
        "experiment_source_receipt_sha256": experiment,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SOURCE_SHA256,
        "source_paths_replayed": True,
        "source_data_qualified": source_data_qualified,
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveAdaptiveTargetRootV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def build_massive_adaptive_target_root_v1(
    *,
    decision_root: MassiveAdaptiveDecisionRootV1,
    source_target: MassiveAdaptiveSourceTargetsV1,
    source_runtime: MassiveAdaptiveTargetSourceRuntimeV1,
) -> MassiveAdaptiveTargetRootV1:
    """Reexecute the target path from live authorities and bind its root."""

    replayed = build_massive_adaptive_source_targets_v1(
        economic_coverage_root=source_runtime.economic_coverage_root,
        decision_clock=source_runtime.decision_clock,
        session_authority=source_runtime.session_authority,
        identity_authority=source_runtime.action_identity_authority,
        daily_input_authority=source_runtime.daily_input_authority,
        fill_source=source_runtime.fill_source,
        terminal_authority=source_runtime.terminal_authority,
        economic_coverage=source_runtime.economic_coverage,
        origin_authority=source_runtime.action_origin,
        built_at_ms=source_target.targets.built_at_ms,
    )
    if replayed != source_target:
        raise MassiveAdaptiveTargetRootV1Error(
            "adaptive source target does not replay from the live roots"
        )
    qualified = (
        decision_root.source_data_qualified
        and source_runtime.daily_input_authority.daily_input_data_qualified
        and source_runtime.fill_source.source_data_qualified
        and source_runtime.terminal_authority.terminal_accounting_data_qualified
        and source_runtime.economic_coverage.coverage_qualified
    )
    return _bind_target_root(
        decision_root=decision_root,
        source_target=replayed,
        source_data_qualified=qualified,
    )


def build_massive_adaptive_target_root_canary_v1(
    *,
    decision_root: MassiveAdaptiveDecisionRootV1,
    source_target: MassiveAdaptiveSourceTargetsV1,
) -> MassiveAdaptiveTargetRootV1:
    """Bind synthetic fixtures without ever deriving source qualification."""

    return _bind_target_root(
        decision_root=decision_root,
        source_target=source_target,
        source_data_qualified=False,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_TARGET_ROOT_V1_SCHEMA",
    "MassiveAdaptiveTargetRootV1",
    "MassiveAdaptiveTargetRootV1Error",
    "MassiveAdaptiveTargetSourceRuntimeV1",
    "build_massive_adaptive_target_root_canary_v1",
    "build_massive_adaptive_target_root_v1",
]
