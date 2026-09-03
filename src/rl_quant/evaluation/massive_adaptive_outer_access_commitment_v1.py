"""Create-only prerequisite for opening adaptive-RL outer forecasts.

The commitment is deliberately materialized before an RL outer forecast.  It
binds both competing policies and every nonmarket economic input, replacing a
caller-supplied timestamp comparison with a receipt dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_outer_inference_plan_v1 import (
    MassiveAdaptiveOuterInferencePlanV1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_frozen_rl_policy_v1 import (
    MassiveAdaptiveFrozenRLPolicyV1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
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
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV1,
)


MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-outer-access-commitment-v1"
)
MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_DATASET = (
    "massive-adaptive-outer-access-commitment-v1"
)
MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "prerequisite": "create-only-before-rl-outer-forecast",
        "policies": ("selected-ppo", "fit-selected-static"),
        "policy_selection_authority": "exact-v1-only",
        "caller_timestamp_authority": False,
        "outer_forecast_access": "receipt-gated",
        "profitability_reporting": False,
        "lockbox": False,
        "duration_semantics": False,
    }
)
MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SCHEMA,
        "payload": "canonical-commitment-metadata",
        "policy_selection_authority": "exact-v1-only",
        "generic_reload": "nonauthorizing",
        "promotion": "rebuild-from-frozen-policy-and-comparator-authorities",
    }
)


class MassiveAdaptiveOuterAccessCommitmentV1Error(ValueError):
    """Outer access was attempted without a complete prior commitment."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveOuterAccessCommitmentV1Error(
            "outer-access commitment ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveOuterAccessCommitmentV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOuterAccessCommitmentV1:
    fold_index: int
    outer_inference_plan_receipt_sha256: str
    outer_origin_inventory_sha256: str
    supervised_checkpoint_receipt_sha256: str
    supervised_model_state_receipt_sha256: str
    calibration_receipt_sha256: str
    rl_policy_selection_authority_receipt_sha256: str
    frozen_rl_policy_receipt_sha256: str
    frozen_rl_policy_model_state_receipt_sha256: str
    fixed_control_registry_receipt_sha256: str
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_fixed_control_id: str
    selected_fixed_action_receipt_sha256: str
    chronology_authority_receipt_sha256: str
    observation_specification_sha256: str
    action_specification_sha256: str
    reward_specification_sha256: str
    compiler_config_receipt_sha256: str
    benchmark_specification: str
    initial_book_specification: str
    primary_capital: float
    cost_ladder_basis_points: tuple[float, ...]
    maximum_fill_participation: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_commitment_replayed: bool
    outer_forecast_access_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "outer_inference_plan_receipt_sha256": (
                self.outer_inference_plan_receipt_sha256
            ),
            "outer_origin_inventory_sha256": self.outer_origin_inventory_sha256,
            "supervised_checkpoint_receipt_sha256": (
                self.supervised_checkpoint_receipt_sha256
            ),
            "supervised_model_state_receipt_sha256": (
                self.supervised_model_state_receipt_sha256
            ),
            "calibration_receipt_sha256": self.calibration_receipt_sha256,
            "rl_policy_selection_authority_receipt_sha256": (
                self.rl_policy_selection_authority_receipt_sha256
            ),
            "frozen_rl_policy_receipt_sha256": self.frozen_rl_policy_receipt_sha256,
            "frozen_rl_policy_model_state_receipt_sha256": (
                self.frozen_rl_policy_model_state_receipt_sha256
            ),
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
            "chronology_authority_receipt_sha256": (
                self.chronology_authority_receipt_sha256
            ),
            "observation_specification_sha256": self.observation_specification_sha256,
            "action_specification_sha256": self.action_specification_sha256,
            "reward_specification_sha256": self.reward_specification_sha256,
            "compiler_config_receipt_sha256": self.compiler_config_receipt_sha256,
            "benchmark_specification": self.benchmark_specification,
            "initial_book_specification": self.initial_book_specification,
            "primary_capital": self.primary_capital,
            "cost_ladder_basis_points": self.cost_ladder_basis_points,
            "maximum_fill_participation": self.maximum_fill_participation,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        expected = self.runtime_commitment_replayed and self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SCHEMA
            or self.fold_index < 0
            or not self.selected_fixed_control_id
            or self.benchmark_specification != "shared-buy-and-drift-book-v1"
            or self.initial_book_specification != "all-books-cash-v1"
            or self.primary_capital != 10_000_000.0
            or self.cost_ladder_basis_points != (10.0, 20.0, 40.0)
            or self.maximum_fill_participation != 0.02
            or self.outer_forecast_access_authorized != expected
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveOuterAccessCommitmentV1Error(
                "adaptive RL outer-access commitment differs"
            )
        for value in (
            self.outer_inference_plan_receipt_sha256,
            self.outer_origin_inventory_sha256,
            self.supervised_checkpoint_receipt_sha256,
            self.supervised_model_state_receipt_sha256,
            self.calibration_receipt_sha256,
            self.rl_policy_selection_authority_receipt_sha256,
            self.frozen_rl_policy_receipt_sha256,
            self.frozen_rl_policy_model_state_receipt_sha256,
            self.fixed_control_registry_receipt_sha256,
            self.fixed_control_fit_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.selected_fixed_action_receipt_sha256,
            self.chronology_authority_receipt_sha256,
            self.observation_specification_sha256,
            self.action_specification_sha256,
            self.reward_specification_sha256,
            self.compiler_config_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("outer-access commitment", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.rl_policy_selection_authority_receipt_sha256
        ):
            raise MassiveAdaptiveOuterAccessCommitmentV1Error(
                "outer-access commitment source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _metadata(
    *,
    outer_inference_plan: MassiveAdaptiveOuterInferencePlanV1,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    policy_selection_authority: MassiveAdaptiveRLPolicySelectionAuthorityV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV1,
    fixed_control_registry: MassiveAdaptiveRLFixedControlRegistryV1,
    fixed_control_fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    fixed_control_selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1,
) -> dict[str, object]:
    if type(policy_selection_authority) is not (
        MassiveAdaptiveRLPolicySelectionAuthorityV1
    ):
        raise MassiveAdaptiveOuterAccessCommitmentV1Error(
            "outer-access commitment V1 requires exact policy-selection authority V1"
        )
    for value in (
        outer_inference_plan,
        calibration,
        policy_selection_authority,
        frozen_policy,
        chronology_authority,
        compiler_config,
    ):
        value.validate()
    validate_massive_adaptive_rl_fixed_control_registry_coverage_v1(
        registry=fixed_control_registry,
        fit_authority=fixed_control_fit_authority,
        selection_authority=fixed_control_selection_authority,
        chronology_authority=chronology_authority,
    )
    policy_selection = policy_selection_authority.runtime_selection
    fixed_selection = fixed_control_selection_authority.runtime_selection
    dates = tuple(row.decision_session_date for row in outer_inference_plan.rows)
    if (
        policy_selection is None
        or fixed_selection is None
        or not policy_selection_authority.runtime_selection_replayed
        or not fixed_control_selection_authority.runtime_selection_replayed
        or len(
            {
                outer_inference_plan.fold_index,
                calibration.fold_index,
                policy_selection.fold_index,
                frozen_policy.fold_index,
                fixed_selection.fold_index,
                chronology_authority.fold_index,
            }
        )
        != 1
        or chronology_authority.outer_inference_plan_receipt_sha256
        != outer_inference_plan.semantic_receipt_sha256
        or chronology_authority.outer_origin_dates != dates
        or calibration.checkpoint_receipt_sha256
        != outer_inference_plan.selected_checkpoint_receipt_sha256
        or frozen_policy.policy_selection_authority_receipt_sha256
        != policy_selection_authority.semantic_receipt_sha256
        or frozen_policy.policy_selection_receipt_sha256
        != policy_selection.semantic_receipt_sha256
        or frozen_policy.selected_rl_checkpoint_receipt_sha256
        != policy_selection.selected_checkpoint_receipt_sha256
    ):
        raise MassiveAdaptiveOuterAccessCommitmentV1Error(
            "outer-access commitment components differ"
        )
    source_qualified = bool(
        outer_inference_plan.outer_inference_authorized
        and calibration.development_calibration_authorized
        and policy_selection_authority.outer_evaluation_authorized
        and frozen_policy.development_outer_policy_authorized
        and fixed_control_fit_authority.development_control_fit_authorized
        and fixed_control_selection_authority.development_control_selection_authorized
        and chronology_authority.outer_evaluation_authorized
    )
    return {
        "schema": MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SCHEMA,
        "fold_index": outer_inference_plan.fold_index,
        "outer_inference_plan_receipt_sha256": (
            outer_inference_plan.semantic_receipt_sha256
        ),
        "outer_origin_inventory_sha256": (
            chronology_authority.outer_origin_inventory_sha256
        ),
        "supervised_checkpoint_receipt_sha256": (
            outer_inference_plan.selected_checkpoint_receipt_sha256
        ),
        "supervised_model_state_receipt_sha256": calibration.model_state_receipt_sha256,
        "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
        "rl_policy_selection_authority_receipt_sha256": (
            policy_selection_authority.semantic_receipt_sha256
        ),
        "frozen_rl_policy_receipt_sha256": frozen_policy.semantic_receipt_sha256,
        "frozen_rl_policy_model_state_receipt_sha256": (
            frozen_policy.frozen_model_state_receipt_sha256
        ),
        "fixed_control_registry_receipt_sha256": (
            fixed_control_registry.semantic_receipt_sha256
        ),
        "fixed_control_fit_authority_receipt_sha256": (
            fixed_control_fit_authority.semantic_receipt_sha256
        ),
        "fixed_control_selection_authority_receipt_sha256": (
            fixed_control_selection_authority.semantic_receipt_sha256
        ),
        "selected_fixed_control_id": fixed_selection.selected_control_id,
        "selected_fixed_action_receipt_sha256": (
            fixed_selection.selected_action_receipt_sha256
        ),
        "chronology_authority_receipt_sha256": chronology_authority.semantic_receipt_sha256,
        "observation_specification_sha256": frozen_policy.observation_specification_sha256,
        "action_specification_sha256": frozen_policy.action_specification_sha256,
        "reward_specification_sha256": frozen_policy.reward_specification_sha256,
        "compiler_config_receipt_sha256": compiler_config.receipt_sha256,
        "benchmark_specification": "shared-buy-and-drift-book-v1",
        "initial_book_specification": "all-books-cash-v1",
        "primary_capital": 10_000_000.0,
        "cost_ladder_basis_points": (10.0, 20.0, 40.0),
        "maximum_fill_participation": 0.02,
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SOURCE_SHA256
        ),
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveOuterAccessCommitmentV1Error(
            "outer-access commitment is not canonical JSON"
        )
    result = dict(value)
    result["cost_ladder_basis_points"] = tuple(result["cost_ladder_basis_points"])
    return result


def parse_massive_adaptive_outer_access_commitment_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveOuterAccessCommitmentV1:
    """Load commitment metadata without granting outer-forecast access."""

    payload = _load_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveOuterAccessCommitmentV1(
        **payload,  # type: ignore[arg-type]
        loaded_source=loaded_source,
        runtime_commitment_replayed=False,
        outer_forecast_access_authorized=False,
    )
    result.validate()
    return result


def authorize_massive_adaptive_outer_access_commitment_v1(
    *,
    root: str | Path,
    commitment: MassiveAdaptiveOuterAccessCommitmentV1,
    outer_inference_plan: MassiveAdaptiveOuterInferencePlanV1,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    policy_selection_authority: MassiveAdaptiveRLPolicySelectionAuthorityV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV1,
    fixed_control_registry: MassiveAdaptiveRLFixedControlRegistryV1,
    fixed_control_fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    fixed_control_selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1,
) -> MassiveAdaptiveOuterAccessCommitmentV1:
    """Rebuild every frozen dependency before opening the outer forecast."""

    parsed = parse_massive_adaptive_outer_access_commitment_v1(
        root=root, loaded_source=commitment.loaded_source
    )
    metadata = _metadata(
        outer_inference_plan=outer_inference_plan,
        calibration=calibration,
        policy_selection_authority=policy_selection_authority,
        frozen_policy=frozen_policy,
        fixed_control_registry=fixed_control_registry,
        fixed_control_fit_authority=fixed_control_fit_authority,
        fixed_control_selection_authority=fixed_control_selection_authority,
        chronology_authority=chronology_authority,
        compiler_config=compiler_config,
    )
    expected = {**metadata, "semantic_receipt_sha256": semantic_sha256(metadata)}
    if (
        parsed.semantic_receipt_sha256 != commitment.semantic_receipt_sha256
        or _load_payload(root=root, loaded_source=commitment.loaded_source) != expected
    ):
        raise MassiveAdaptiveOuterAccessCommitmentV1Error(
            "outer-access commitment does not replay from frozen inputs"
        )
    result = replace(
        parsed,
        runtime_commitment_replayed=True,
        outer_forecast_access_authorized=parsed.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_outer_access_commitment_v1(
    *,
    root: str | Path,
    artifact_id: str,
    outer_inference_plan: MassiveAdaptiveOuterInferencePlanV1,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    policy_selection_authority: MassiveAdaptiveRLPolicySelectionAuthorityV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV1,
    fixed_control_registry: MassiveAdaptiveRLFixedControlRegistryV1,
    fixed_control_fit_authority: MassiveAdaptiveRLFixedControlFitAuthorityV1,
    fixed_control_selection_authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1,
    committed_at_ms: int,
) -> MassiveAdaptiveOuterAccessCommitmentV1:
    """Publish the hard prerequisite before any package-owned RL outer forecast."""

    identifier = _artifact_id(artifact_id)
    metadata = _metadata(
        outer_inference_plan=outer_inference_plan,
        calibration=calibration,
        policy_selection_authority=policy_selection_authority,
        frozen_policy=frozen_policy,
        fixed_control_registry=fixed_control_registry,
        fixed_control_fit_authority=fixed_control_fit_authority,
        fixed_control_selection_authority=fixed_control_selection_authority,
        chronology_authority=chronology_authority,
        compiler_config=compiler_config,
    )
    payload = {**metadata, "semantic_receipt_sha256": semantic_sha256(metadata)}
    relative = f"massive-adaptive/outer-access-commitment-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=(policy_selection_authority.semantic_receipt_sha256),
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-OUTER-ACCESS-COMMITMENT-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_outer_access_commitment_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_outer_access_commitment_v1(
        root=root,
        commitment=generic,
        outer_inference_plan=outer_inference_plan,
        calibration=calibration,
        policy_selection_authority=policy_selection_authority,
        frozen_policy=frozen_policy,
        fixed_control_registry=fixed_control_registry,
        fixed_control_fit_authority=fixed_control_fit_authority,
        fixed_control_selection_authority=fixed_control_selection_authority,
        chronology_authority=chronology_authority,
        compiler_config=compiler_config,
    )


__all__ = [
    "MassiveAdaptiveOuterAccessCommitmentV1",
    "MassiveAdaptiveOuterAccessCommitmentV1Error",
    "authorize_massive_adaptive_outer_access_commitment_v1",
    "materialize_massive_adaptive_outer_access_commitment_v1",
    "parse_massive_adaptive_outer_access_commitment_v1",
]
