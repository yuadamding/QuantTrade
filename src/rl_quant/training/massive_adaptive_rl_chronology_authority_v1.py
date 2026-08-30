"""Disjoint PPO-fit, policy-selection, and outer chronologies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferencePlanV1,
)
from rl_quant.evaluation.massive_adaptive_outer_inference_plan_v1 import (
    MassiveAdaptiveOuterInferencePlanV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v2 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV2,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MassiveAdaptiveSplitPlanV1,
)


MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-chronology-authority-v1"
)
MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "roles": ("rl_fit", "rl_policy_selection", "outer_test"),
        "pairwise_disjoint": True,
        "chronological": "fit-before-selection-before-outer",
        "outer_opening": "split-dates-first-selected-inference-plan-bound-later",
        "policy_gradients": "rl-fit-only",
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLChronologyAuthorityV1Error(ValueError):
    """RL fitting, selection, and outer decision inventories overlap."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLChronologyAuthorityV1:
    fold_index: int
    training_forecast_authority_receipt_sha256: str
    validation_inference_plan_receipt_sha256: str
    split_plan_receipt_sha256: str
    outer_fold_receipt_sha256: str
    outer_inference_plan_receipt_sha256: str | None
    rl_fit_origin_dates: tuple[str, ...]
    rl_validation_origin_dates: tuple[str, ...]
    outer_origin_dates: tuple[str, ...]
    rl_fit_origin_inventory_sha256: str
    rl_validation_origin_inventory_sha256: str
    outer_origin_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_rl_training_authorized: bool
    development_policy_selection_authorized: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "development_rl_training_authorized",
                "development_policy_selection_authorized",
                "outer_evaluation_authorized",
            }
        }

    def validate(self) -> None:
        fit = set(self.rl_fit_origin_dates)
        validation = set(self.rl_validation_origin_dates)
        outer = set(self.outer_origin_dates)
        expected = self.source_data_qualified
        outer_bound = self.outer_inference_plan_receipt_sha256 is not None
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SCHEMA
            or self.fold_index < 0
            or not fit
            or not validation
            or not outer
            or fit & validation
            or fit & outer
            or validation & outer
            or self.rl_fit_origin_dates != tuple(sorted(set(self.rl_fit_origin_dates)))
            or self.rl_validation_origin_dates
            != tuple(sorted(set(self.rl_validation_origin_dates)))
            or self.outer_origin_dates != tuple(sorted(set(self.outer_origin_dates)))
            or self.rl_fit_origin_dates[-1] >= self.rl_validation_origin_dates[0]
            or self.rl_validation_origin_dates[-1] >= self.outer_origin_dates[0]
            or self.rl_fit_origin_inventory_sha256
            != semantic_sha256(self.rl_fit_origin_dates)
            or self.rl_validation_origin_inventory_sha256
            != semantic_sha256(self.rl_validation_origin_dates)
            or self.outer_origin_inventory_sha256
            != semantic_sha256(self.outer_origin_dates)
            or self.development_rl_training_authorized != expected
            or self.development_policy_selection_authorized != expected
            or self.outer_evaluation_authorized != (expected and outer_bound)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLChronologyAuthorityV1Error(
                "adaptive RL chronology authority differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_chronology_authority_v1(
    *,
    training_forecast_authority: (
        MassiveAdaptiveRLTrainingForecastAuthorityV1
        | MassiveAdaptiveRLTrainingForecastAuthorityV2
    ),
    validation_inference_plan: MassiveAdaptiveInferencePlanV1,
    split_plan: MassiveAdaptiveSplitPlanV1,
) -> MassiveAdaptiveRLChronologyAuthorityV1:
    """Freeze roles from the split without opening selected outer artifacts."""

    training_forecast_authority.validate()
    validation_inference_plan.validate()
    split_plan.validate()
    try:
        fold = split_plan.outer_folds[training_forecast_authority.outer_fold_index]
    except (IndexError, TypeError) as error:
        raise MassiveAdaptiveRLChronologyAuthorityV1Error(
            "adaptive RL chronology fold is absent"
        ) from error
    if (
        validation_inference_plan.fold_index
        != training_forecast_authority.outer_fold_index
        or validation_inference_plan.inference_role != "inner_validation"
        or validation_inference_plan.split_plan_receipt_sha256
        != split_plan.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLChronologyAuthorityV1Error(
            "adaptive RL chronology roles or folds differ"
        )
    fit_dates = training_forecast_authority.origin_session_dates
    validation_dates = tuple(
        row.decision_session_date for row in validation_inference_plan.rows
    )
    outer_dates = fold.outer_test_session_dates
    if validation_dates != fold.inner_validation_session_dates or not set(
        fit_dates
    ).issubset(fold.fit_session_dates):
        raise MassiveAdaptiveRLChronologyAuthorityV1Error(
            "adaptive RL fit or validation dates differ from the split fold"
        )
    source_qualified = bool(
        training_forecast_authority.reinforcement_learning_authorized
        and validation_inference_plan.source_data_qualified
        and split_plan.candidate_source_data_qualified
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SCHEMA,
        "fold_index": training_forecast_authority.outer_fold_index,
        "training_forecast_authority_receipt_sha256": (
            training_forecast_authority.semantic_receipt_sha256
        ),
        "validation_inference_plan_receipt_sha256": (
            validation_inference_plan.semantic_receipt_sha256
        ),
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "outer_fold_receipt_sha256": fold.receipt_sha256,
        "outer_inference_plan_receipt_sha256": None,
        "rl_fit_origin_dates": fit_dates,
        "rl_validation_origin_dates": validation_dates,
        "outer_origin_dates": outer_dates,
        "rl_fit_origin_inventory_sha256": semantic_sha256(fit_dates),
        "rl_validation_origin_inventory_sha256": semantic_sha256(validation_dates),
        "outer_origin_inventory_sha256": semantic_sha256(outer_dates),
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLChronologyAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        development_rl_training_authorized=source_qualified,
        development_policy_selection_authorized=source_qualified,
        outer_evaluation_authorized=False,
    )
    result.validate()
    return result


def bind_massive_adaptive_rl_outer_chronology_v1(
    *,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    outer_inference_plan: MassiveAdaptiveOuterInferencePlanV1,
) -> MassiveAdaptiveRLChronologyAuthorityV1:
    """Bind selected-checkpoint outer inference only after it is authorized."""

    chronology_authority.validate()
    outer_inference_plan.validate()
    if (
        chronology_authority.outer_inference_plan_receipt_sha256 is not None
        or chronology_authority.outer_evaluation_authorized
        or not chronology_authority.source_data_qualified
        or not outer_inference_plan.outer_inference_authorized
        or outer_inference_plan.fold_index != chronology_authority.fold_index
        or outer_inference_plan.split_plan_receipt_sha256
        != chronology_authority.split_plan_receipt_sha256
        or tuple(row.decision_session_date for row in outer_inference_plan.rows)
        != chronology_authority.outer_origin_dates
    ):
        raise MassiveAdaptiveRLChronologyAuthorityV1Error(
            "selected outer inference cannot bind the RL chronology"
        )
    provisional = replace(
        chronology_authority,
        outer_inference_plan_receipt_sha256=(
            outer_inference_plan.semantic_receipt_sha256
        ),
        semantic_receipt_sha256="0" * 64,
        outer_evaluation_authorized=True,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLChronologyAuthorityV1",
    "MassiveAdaptiveRLChronologyAuthorityV1Error",
    "bind_massive_adaptive_rl_outer_chronology_v1",
    "build_massive_adaptive_rl_chronology_authority_v1",
]
