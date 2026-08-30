"""Outer plan with a hard pre-forecast access commitment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v1 import (
    MassiveAdaptiveOuterAccessCommitmentV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_forecast_archive_v1 import (
    MassiveAdaptiveRLOuterForecastArchiveV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_plan_v2 import (
    MassiveAdaptiveRLOuterPlanV2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SCHEMA = "rl-quant.massive-adaptive-rl-outer-plan-v3"
MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SPEC_SHA256 = semantic_sha256(
    {
        "ppo_and_comparator": "outer-plan-v2",
        "outer_access": "create-only-commitment-required-before-forecast",
        "timestamp_only_gate": False,
        "profitability_reporting": False,
        "lockbox": False,
    }
)


class MassiveAdaptiveRLOuterPlanV3Error(ValueError):
    """The outer plan bypassed its hard access commitment."""


def _digest(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterPlanV3Error(
            "outer plan V3 receipt must be a lowercase SHA-256"
        )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterPlanV3:
    outer_plan_v2: MassiveAdaptiveRLOuterPlanV2
    outer_access_commitment_receipt_sha256: str
    gated_outer_forecast_archive_receipt_sha256: str
    raw_outer_forecast_archive_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SPEC_SHA256
    implementation_source_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SOURCE_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SCHEMA

    @property
    def fold_index(self) -> int:
        return self.outer_plan_v2.fold_index

    @property
    def outer_plan_v1(self):
        return self.outer_plan_v2.outer_plan_v1

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "outer_plan_v2_receipt_sha256": self.outer_plan_v2.semantic_receipt_sha256,
            "outer_access_commitment_receipt_sha256": (
                self.outer_access_commitment_receipt_sha256
            ),
            "gated_outer_forecast_archive_receipt_sha256": (
                self.gated_outer_forecast_archive_receipt_sha256
            ),
            "raw_outer_forecast_archive_receipt_sha256": (
                self.raw_outer_forecast_archive_receipt_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.outer_plan_v2.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SCHEMA
            or self.source_data_qualified != self.outer_plan_v2.source_data_qualified
            or self.outer_evaluation_authorized != self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterPlanV3Error(
                "hard-gated adaptive RL outer plan differs"
            )
        for value in (
            self.outer_access_commitment_receipt_sha256,
            self.gated_outer_forecast_archive_receipt_sha256,
            self.raw_outer_forecast_archive_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest(value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_outer_plan_v3(
    *,
    outer_plan_v2: MassiveAdaptiveRLOuterPlanV2,
    outer_access_commitment: MassiveAdaptiveOuterAccessCommitmentV1,
    gated_outer_forecast_archive: MassiveAdaptiveRLOuterForecastArchiveV1,
) -> MassiveAdaptiveRLOuterPlanV3:
    """Bind the existing outer plan to the prerequisite-gated forecast path."""

    outer_plan_v2.validate()
    outer_access_commitment.validate()
    gated_outer_forecast_archive.validate()
    if (
        not outer_access_commitment.outer_forecast_access_authorized
        or not gated_outer_forecast_archive.outer_forecast_authorized
        or len(
            {
                outer_plan_v2.fold_index,
                outer_access_commitment.fold_index,
                gated_outer_forecast_archive.fold_index,
            }
        )
        != 1
        or outer_plan_v2.outer_plan_v1.outer_forecast_archive_receipt_sha256
        != gated_outer_forecast_archive.raw_outer_forecast_archive_receipt_sha256
        or outer_plan_v2.outer_plan_v1.outer_inference_plan_receipt_sha256
        != outer_access_commitment.outer_inference_plan_receipt_sha256
        or gated_outer_forecast_archive.outer_access_commitment_receipt_sha256
        != outer_access_commitment.semantic_receipt_sha256
        or outer_plan_v2.fixed_control_fit_authority_receipt_sha256
        != outer_access_commitment.fixed_control_fit_authority_receipt_sha256
        or outer_plan_v2.fixed_control_selection_authority_receipt_sha256
        != outer_access_commitment.fixed_control_selection_authority_receipt_sha256
        or outer_plan_v2.selected_fixed_action_receipt_sha256
        != outer_access_commitment.selected_fixed_action_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterPlanV3Error(
            "outer plan does not descend from the hard access commitment"
        )
    source_qualified = bool(
        outer_plan_v2.outer_evaluation_authorized
        and outer_access_commitment.outer_forecast_access_authorized
        and gated_outer_forecast_archive.outer_forecast_authorized
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SCHEMA,
        "outer_plan_v2": outer_plan_v2,
        "outer_access_commitment_receipt_sha256": (
            outer_access_commitment.semantic_receipt_sha256
        ),
        "gated_outer_forecast_archive_receipt_sha256": (
            gated_outer_forecast_archive.semantic_receipt_sha256
        ),
        "raw_outer_forecast_archive_receipt_sha256": (
            gated_outer_forecast_archive.raw_outer_forecast_archive_receipt_sha256
        ),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_OUTER_PLAN_V3_SOURCE_SHA256,
    }
    provisional = MassiveAdaptiveRLOuterPlanV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        outer_evaluation_authorized=source_qualified,
    )
    result = MassiveAdaptiveRLOuterPlanV3(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
            "outer_evaluation_authorized": source_qualified,
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLOuterPlanV3",
    "MassiveAdaptiveRLOuterPlanV3Error",
    "build_massive_adaptive_rl_outer_plan_v3",
]
