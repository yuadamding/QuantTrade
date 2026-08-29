"""Legal training-checkpoint to target-free inference compatibility.

V1 forecast replay intentionally bound a checkpoint to its own training
window.  This authority separates immutable training provenance from an
independent inference provenance and admits only the registered transition
from a training checkpoint to the same fold's complete inner-validation
schedule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferencePlanV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256,
    MassiveAdaptiveAlphaModelSpecV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)


MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-forecast-eligibility-authority-v2"
)
MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "checkpoint_role": "training",
        "inference_role": "inner_validation",
        "fold": "identical",
        "training_and_inference_origins": "disjoint",
        "training_tensor_equals_inference_tensor": False,
        "training_roots_equal_inference_roots": False,
        "target_archive_required_for_inference": False,
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveForecastEligibilityAuthorityV2Error(ValueError):
    """A checkpoint cannot legally forecast the supplied inference schedule."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveForecastEligibilityAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveForecastEligibilityAuthorityV2:
    fold_index: int
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    training_tensor_receipt_sha256: str
    training_full_decision_root_inventory_sha256: str
    training_origin_decision_root_inventory_sha256: str
    training_window_plan_receipt_sha256: str
    inference_tensor_receipt_sha256: str
    inference_full_decision_root_inventory_sha256: str
    inference_origin_decision_root_inventory_sha256: str
    inference_plan_receipt_sha256: str
    inference_role: str
    split_plan_receipt_sha256: str
    model_spec_receipt_sha256: str
    model_source_sha256: str
    inference_origin_session_dates: tuple[str, ...]
    source_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    runtime_eligibility_replayed: bool
    development_forecast_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SCHEMA
            or self.inference_role != "inner_validation"
            or not self.inference_origin_session_dates
            or self.inference_origin_session_dates
            != tuple(sorted(set(self.inference_origin_session_dates)))
            or not isinstance(self.source_data_qualified, bool)
            or not self.runtime_eligibility_replayed
            or self.development_forecast_authorized != self.source_data_qualified
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SOURCE_SHA256
            or self.model_source_sha256 != MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveForecastEligibilityAuthorityV2Error(
                "adaptive forecast eligibility identity or authorization differs"
            )
        for value in (
            self.checkpoint_receipt_sha256,
            self.checkpoint_source_receipt_sha256,
            self.model_state_receipt_sha256,
            self.training_tensor_receipt_sha256,
            self.training_full_decision_root_inventory_sha256,
            self.training_origin_decision_root_inventory_sha256,
            self.training_window_plan_receipt_sha256,
            self.inference_tensor_receipt_sha256,
            self.inference_full_decision_root_inventory_sha256,
            self.inference_origin_decision_root_inventory_sha256,
            self.inference_plan_receipt_sha256,
            self.split_plan_receipt_sha256,
            self.model_spec_receipt_sha256,
            self.model_source_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive forecast eligibility", value)
        assert_no_adaptive_hold_semantics(asdict(self))


def build_massive_adaptive_forecast_eligibility_authority_v2(
    *,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    inference_tensor: MassiveAdaptiveDecisionTensorV1,
    inference_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    inference_plan: MassiveAdaptiveInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveForecastEligibilityAuthorityV2:
    """Reconcile disjoint training and inner-validation provenance."""

    checkpoint.validate()
    training_window_plan.validate()
    inference_tensor.validate()
    inference_plan.validate()
    model_spec.validate()
    if (
        checkpoint.runtime_state is None
        or not checkpoint.runtime_checkpoint_replayed
        or inference_tensor.runtime_tensor is None
        or not inference_tensor.runtime_source_replayed
    ):
        raise MassiveAdaptiveForecastEligibilityAuthorityV2Error(
            "adaptive checkpoint or inference tensor has not been replayed"
        )
    ordered_roots = tuple(
        sorted(inference_decision_roots, key=lambda row: row.decision_session_date)
    )
    for root in ordered_roots:
        root.validate()
    full_receipts = tuple(root.semantic_receipt_sha256 for root in ordered_roots)
    origin_receipts = tuple(
        row.decision_root_receipt_sha256 for row in inference_plan.rows
    )
    training_origins = {row.origin_session_date for row in training_window_plan.rows}
    inference_origins = tuple(row.decision_session_date for row in inference_plan.rows)
    if (
        training_window_plan.split_role != "training"
        or inference_plan.inference_role != "inner_validation"
        or training_window_plan.fold_index != inference_plan.fold_index
        or checkpoint.window_plan_receipt_sha256
        != training_window_plan.semantic_receipt_sha256
        or checkpoint.decision_tensor_receipt_sha256
        != training_window_plan.decision_tensor_receipt_sha256
        or checkpoint.full_decision_root_inventory_sha256
        != training_window_plan.full_decision_root_inventory_sha256
        or checkpoint.origin_decision_root_inventory_sha256
        != training_window_plan.origin_decision_root_inventory_sha256
        or checkpoint.split_plan_receipt_sha256
        != inference_plan.split_plan_receipt_sha256
        or training_window_plan.split_plan_receipt_sha256
        != inference_plan.split_plan_receipt_sha256
        or checkpoint.model_spec_receipt_sha256 != model_spec.receipt_sha256
        or inference_plan.model_spec_receipt_sha256 != model_spec.receipt_sha256
        or inference_plan.decision_tensor_receipt_sha256
        != inference_tensor.semantic_receipt_sha256
        or tuple(root.decision_session_date for root in ordered_roots)
        != inference_tensor.decision_session_dates
        or tuple(root.feature_semantic_receipt_sha256 for root in ordered_roots)
        != inference_tensor.feature_semantic_receipts
        or tuple(root.action_origin_receipt_sha256 for root in ordered_roots)
        != inference_tensor.action_origin_receipts
        or inference_plan.full_decision_root_inventory_sha256
        != semantic_sha256(full_receipts)
        or inference_plan.origin_decision_root_inventory_sha256
        != semantic_sha256(origin_receipts)
        or training_origins.intersection(inference_origins)
    ):
        raise MassiveAdaptiveForecastEligibilityAuthorityV2Error(
            "adaptive training and inference provenance is incompatible"
        )

    body = {
        "schema": MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SCHEMA,
        "fold_index": inference_plan.fold_index,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "checkpoint_source_receipt_sha256": (
            checkpoint.loaded_source.receipt.receipt_sha256
        ),
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "training_tensor_receipt_sha256": (checkpoint.decision_tensor_receipt_sha256),
        "training_full_decision_root_inventory_sha256": (
            checkpoint.full_decision_root_inventory_sha256
        ),
        "training_origin_decision_root_inventory_sha256": (
            checkpoint.origin_decision_root_inventory_sha256
        ),
        "training_window_plan_receipt_sha256": (
            training_window_plan.semantic_receipt_sha256
        ),
        "inference_tensor_receipt_sha256": (inference_tensor.semantic_receipt_sha256),
        "inference_full_decision_root_inventory_sha256": (
            inference_plan.full_decision_root_inventory_sha256
        ),
        "inference_origin_decision_root_inventory_sha256": (
            inference_plan.origin_decision_root_inventory_sha256
        ),
        "inference_plan_receipt_sha256": inference_plan.semantic_receipt_sha256,
        "inference_role": inference_plan.inference_role,
        "split_plan_receipt_sha256": inference_plan.split_plan_receipt_sha256,
        "model_spec_receipt_sha256": model_spec.receipt_sha256,
        "model_source_sha256": MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256,
        "inference_origin_session_dates": inference_origins,
        "source_data_qualified": (
            checkpoint.committed_development_training_authorized
            and inference_plan.source_data_qualified
            and inference_tensor.committed_source_data_qualified
            and all(root.source_data_qualified for root in ordered_roots)
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SOURCE_SHA256
        ),
        "runtime_eligibility_replayed": True,
        "development_forecast_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    body["development_forecast_authorized"] = body["source_data_qualified"]
    result = MassiveAdaptiveForecastEligibilityAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_FORECAST_ELIGIBILITY_AUTHORITY_V2_SCHEMA",
    "MassiveAdaptiveForecastEligibilityAuthorityV2",
    "MassiveAdaptiveForecastEligibilityAuthorityV2Error",
    "build_massive_adaptive_forecast_eligibility_authority_v2",
]
