"""Outer plan that freezes both PPO and its fit-selected comparator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_outer_forecast_archive_v1 import (
    MassiveAdaptiveOuterForecastArchiveV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v1 import (
    MassiveAdaptiveRLOuterPlanV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    MassiveAdaptiveRLFixedControlRegistryV1,
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)


MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SCHEMA = "rl-quant.massive-adaptive-rl-outer-plan-v2"
MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SPEC_SHA256 = semantic_sha256(
    {
        "ppo": "frozen-outer-plan-v1",
        "comparator": "fit-selected-static-control-frozen-before-outer-forecast",
        "substitution_after_outer_access": False,
        "profitability_reporting": False,
    }
)


class MassiveAdaptiveRLOuterPlanV2Error(ValueError):
    """The outer comparator was absent, substituted, or frozen too late."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterPlanV2Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterPlanV2:
    outer_plan_v1: MassiveAdaptiveRLOuterPlanV1
    fixed_control_registry_receipt_sha256: str
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_fixed_control_id: str
    selected_fixed_action_receipt_sha256: str
    comparator_frozen_at_ms: int
    outer_forecast_committed_at_ms: int
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SPEC_SHA256
    implementation_source_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SOURCE_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SCHEMA

    @property
    def fold_index(self) -> int:
        return self.outer_plan_v1.fold_index

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "outer_plan_v1_receipt_sha256": self.outer_plan_v1.semantic_receipt_sha256,
            "fixed_control_registry_receipt_sha256": (
                self.fixed_control_registry_receipt_sha256
            ),
            "fixed_control_fit_authority_receipt_sha256": (
                self.fixed_control_fit_authority_receipt_sha256
            ),
            "fixed_control_selection_authority_receipt_sha256": (
                self.fixed_control_selection_authority_receipt_sha256
            ),
            "selected_fixed_control_id": self.selected_fixed_control_id,
            "selected_fixed_action_receipt_sha256": (
                self.selected_fixed_action_receipt_sha256
            ),
            "comparator_frozen_at_ms": self.comparator_frozen_at_ms,
            "outer_forecast_committed_at_ms": self.outer_forecast_committed_at_ms,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.outer_plan_v1.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SCHEMA
            or not self.selected_fixed_control_id
            or self.comparator_frozen_at_ms < 0
            or self.outer_forecast_committed_at_ms < 0
            or self.comparator_frozen_at_ms >= self.outer_forecast_committed_at_ms
            or self.source_data_qualified
            != (
                self.outer_plan_v1.source_data_qualified
                and self.comparator_frozen_at_ms < self.outer_forecast_committed_at_ms
            )
            or self.outer_evaluation_authorized != self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterPlanV2Error(
                "comparator-bound adaptive RL outer plan differs"
            )
        for value in (
            self.fixed_control_registry_receipt_sha256,
            self.fixed_control_fit_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.selected_fixed_action_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("comparator-bound outer plan", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_outer_plan_v2(
    *,
    outer_plan_v1: MassiveAdaptiveRLOuterPlanV1,
    outer_forecast_archive: MassiveAdaptiveOuterForecastArchiveV1,
    fixed_control_registry: MassiveAdaptiveRLFixedControlRegistryV1,
    fixed_control_fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    fixed_control_selection_authority: (
        MassiveAdaptiveRLFixedControlSelectionAuthorityV1
    ),
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
) -> MassiveAdaptiveRLOuterPlanV2:
    """Freeze the selected static comparator before the outer forecast opens."""

    outer_plan_v1.validate()
    outer_forecast_archive.validate()
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
        registry=fixed_control_registry,
        fit_authority=fixed_control_fit_authority,
        selection_authority=fixed_control_selection_authority,
        chronology_authority=chronology_authority,
    )
    selection = fixed_control_selection_authority.runtime_selection
    if (
        selection is None
        or not fixed_control_selection_authority.runtime_selection_replayed
        or outer_plan_v1.fold_index != selection.fold_index
        or outer_plan_v1.outer_forecast_archive_receipt_sha256
        != outer_forecast_archive.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterPlanV2Error(
            "outer plan and selected static comparator differ"
        )
    frozen_at_ms = max(
        fixed_control_fit_authority.loaded_source.commit.committed_at_ms,
        fixed_control_selection_authority.loaded_source.commit.committed_at_ms,
    )
    outer_at_ms = outer_forecast_archive.loaded_source.commit.committed_at_ms
    if frozen_at_ms >= outer_at_ms:
        raise MassiveAdaptiveRLOuterPlanV2Error(
            "static comparator was not frozen before outer forecast access"
        )
    source_qualified = bool(
        outer_plan_v1.outer_evaluation_authorized
        and fixed_control_fit_authority.development_control_fit_authorized
        and fixed_control_selection_authority.development_control_selection_authorized
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SCHEMA,
        "outer_plan_v1": outer_plan_v1,
        "fixed_control_registry_receipt_sha256": (
            fixed_control_registry.semantic_receipt_sha256
        ),
        "fixed_control_fit_authority_receipt_sha256": (
            fixed_control_fit_authority.semantic_receipt_sha256
        ),
        "fixed_control_selection_authority_receipt_sha256": (
            fixed_control_selection_authority.semantic_receipt_sha256
        ),
        "selected_fixed_control_id": selection.selected_control_id,
        "selected_fixed_action_receipt_sha256": (
            selection.selected_action_receipt_sha256
        ),
        "comparator_frozen_at_ms": frozen_at_ms,
        "outer_forecast_committed_at_ms": outer_at_ms,
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V2_SOURCE_SHA256,
    }
    provisional = MassiveAdaptiveRLOuterPlanV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        outer_evaluation_authorized=source_qualified,
    )
    result = MassiveAdaptiveRLOuterPlanV2(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(
                provisional.semantic_unsigned()
            ),
            "outer_evaluation_authorized": source_qualified,
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLOuterPlanV2",
    "MassiveAdaptiveRLOuterPlanV2Error",
    "build_massive_adaptive_rl_outer_plan_v2",
]
