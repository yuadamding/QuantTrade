"""Manifest-V4-bound inner-validation policy-selection computation.

V1 remains the compatibility implementation for the original eligibility-only
selector.  V2 implements Manifest V4 exactly: all registered candidates receive
one total ranking, eligible candidates form the normal selection pool, and the
highest-ranked candidate overall is selected diagnostically when that pool is
empty.  Its persisted authority is a computation-only witness: only Selection
V3 may authorize policy freezing or outer preparation.  All numeric comparisons
are exact finite IEEE-754 binary64 comparisons with canonical positive zero and
no tolerance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_authority_v1 import (
    MassiveAdaptiveRLFoldValidationAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptiveRLCheckpointV1
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
    build_massive_adaptive_rl_policy_candidate_v1,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_v1 import (
    MassiveAdaptiveRLFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256,
    MassiveAdaptiveRLExperimentManifestV4,
)

if TYPE_CHECKING:
    from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
        MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    )


MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-candidate-v2"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-selection-v2"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-selection-authority-v2"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_DATASET = (
    "massive-adaptive-rl-policy-selection-authority-v2"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256 = semantic_sha256(
    {
        "numeric_format": "ieee-754-binary64",
        "finite_values_required": True,
        "canonical_zero": "+0.0",
        "ranking_comparison": "exact-no-tolerance",
        "strict_positive": "value-greater-than-+0.0",
        "nonnegative": "value-greater-than-or-equal-to-+0.0",
        "maximum_drawdown": "value-less-than-or-equal-to-0.25",
        "cost_ladder": "low-greater-than-or-equal-primary-greater-than-or-equal-high",
    }
)
MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SPEC_SHA256 = semantic_sha256(
    {
        "source_evidence": "validated-policy-candidate-v1-and-exact-cost-traces",
        "manifest_selection_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
        ),
        "eligibility_criteria": MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1,
        "numerical_comparison_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
        ),
        "caller_metrics_authorizing": False,
        "profitability_reporting": False,
        "outer_access": False,
    }
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256 = semantic_sha256(
    {
        "manifest_selection_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
        ),
        "candidate_ranking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
        ),
        "candidate_tie_breaking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
        ),
        "numerical_comparison_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
        ),
        "normal_pool": "highest-ranked-eligible",
        "empty_eligible_pool": "highest-ranked-overall-diagnostic",
        "candidate_coverage": "exact-fold-fit-checkpoint-authority-inventory",
        "authority": "computation-only-witness",
        "policy_freezing": False,
        "outer_diagnostic_preparation": False,
        "profitability_reporting": False,
        "outer_access": False,
    }
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SCHEMA,
            "payload": "canonical-json-policy-selection-v2-and-candidates-v2",
            "authority": "computation-only-witness",
            "generic_reload": "nonauthorizing",
        }
    )
)


class MassiveAdaptiveRLPolicySelectionV2Error(ValueError):
    """Manifest V4 validation candidates or their selection differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "RL policy selection V2 artifact ID is not path safe"
        )
    return value


def _binary64(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            f"{name} must be finite IEEE-754 binary64"
        )
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            f"{name} must be finite IEEE-754 binary64"
        ) from error
    if not math.isfinite(result):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            f"{name} must be finite IEEE-754 binary64"
        )
    return 0.0 if result == 0.0 else result


def _canonical_binary64(value: object) -> bool:
    return bool(
        type(value) is float
        and math.isfinite(value)
        and not (value == 0.0 and math.copysign(1.0, value) < 0.0)
    )


def _validation_eligibility_failures_v1(
    *,
    primary_incremental_rl_log_wealth: float,
    ppo_minus_fc06_log_wealth: float,
    primary_strategy_active_log_wealth: float,
    low_cost_terminal_liquidation_adjusted_return: float,
    primary_cost_terminal_liquidation_adjusted_return: float,
    high_cost_terminal_liquidation_adjusted_return: float,
    maximum_drawdown: float,
) -> tuple[str, ...]:
    criteria = MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1
    failures: list[str] = []
    if primary_incremental_rl_log_wealth <= 0.0:
        failures.append(criteria[0])
    if ppo_minus_fc06_log_wealth <= 0.0:
        failures.append(criteria[1])
    if primary_strategy_active_log_wealth <= 0.0:
        failures.append(criteria[2])
    if high_cost_terminal_liquidation_adjusted_return < 0.0:
        failures.append(criteria[3])
    if not (
        low_cost_terminal_liquidation_adjusted_return
        >= primary_cost_terminal_liquidation_adjusted_return
        >= high_cost_terminal_liquidation_adjusted_return
    ):
        failures.append(criteria[4])
    if maximum_drawdown > 0.25:
        failures.append(criteria[5])
    return tuple(sorted(failures))


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicyCandidateV2:
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_index: int
    checkpoint_authority_receipt_sha256: str
    checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    update_index: int
    training_forecast_authority_receipt_sha256: str
    primary_trace_receipt_sha256: str
    low_cost_trace_receipt_sha256: str
    high_cost_trace_receipt_sha256: str
    decision_target_inventory_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_fc06_action_receipt_sha256: str
    fc06_validation_trace_receipt_sha256: str
    legacy_candidate_v1_receipt_sha256: str
    fc06_primary_incremental_log_wealth: float
    ppo_minus_fc06_log_wealth: float
    primary_incremental_rl_log_wealth: float
    primary_strategy_active_log_wealth: float
    low_cost_terminal_liquidation_adjusted_return: float
    primary_cost_terminal_liquidation_adjusted_return: float
    high_cost_terminal_liquidation_adjusted_return: float
    maximum_drawdown: float
    validation_eligibility_failures: tuple[str, ...]
    economically_eligible: bool
    source_data_qualified: bool
    validation_selection_specification_sha256: str
    candidate_ranking_specification_sha256: str
    candidate_tie_breaking_specification_sha256: str
    numerical_comparison_specification_sha256: str
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        metrics = (
            self.fc06_primary_incremental_log_wealth,
            self.ppo_minus_fc06_log_wealth,
            self.primary_incremental_rl_log_wealth,
            self.primary_strategy_active_log_wealth,
            self.low_cost_terminal_liquidation_adjusted_return,
            self.primary_cost_terminal_liquidation_adjusted_return,
            self.high_cost_terminal_liquidation_adjusted_return,
            self.maximum_drawdown,
        )
        expected_failures = _validation_eligibility_failures_v1(
            primary_incremental_rl_log_wealth=(self.primary_incremental_rl_log_wealth),
            ppo_minus_fc06_log_wealth=self.ppo_minus_fc06_log_wealth,
            primary_strategy_active_log_wealth=(
                self.primary_strategy_active_log_wealth
            ),
            low_cost_terminal_liquidation_adjusted_return=(
                self.low_cost_terminal_liquidation_adjusted_return
            ),
            primary_cost_terminal_liquidation_adjusted_return=(
                self.primary_cost_terminal_liquidation_adjusted_return
            ),
            high_cost_terminal_liquidation_adjusted_return=(
                self.high_cost_terminal_liquidation_adjusted_return
            ),
            maximum_drawdown=self.maximum_drawdown,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SCHEMA
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or isinstance(self.update_index, bool)
            or self.update_index < 0
            or not all(_canonical_binary64(value) for value in metrics)
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or self.ppo_minus_fc06_log_wealth
            != self.primary_incremental_rl_log_wealth
            - self.fc06_primary_incremental_log_wealth
            or self.validation_eligibility_failures != expected_failures
            or self.economically_eligible != (not expected_failures)
            or not isinstance(self.source_data_qualified, bool)
            or self.validation_selection_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
            or self.candidate_ranking_specification_sha256
            != MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
            or self.candidate_tie_breaking_specification_sha256
            != MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
            or self.numerical_comparison_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicySelectionV2Error(
                "adaptive RL policy candidate V2 differs"
            )
        for value in (
            self.manifest_v4_receipt_sha256,
            self.training_manifest_v3_receipt_sha256,
            self.checkpoint_authority_receipt_sha256,
            self.checkpoint_receipt_sha256,
            self.model_state_receipt_sha256,
            self.training_forecast_authority_receipt_sha256,
            self.primary_trace_receipt_sha256,
            self.low_cost_trace_receipt_sha256,
            self.high_cost_trace_receipt_sha256,
            self.decision_target_inventory_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.selected_fc06_action_receipt_sha256,
            self.fc06_validation_trace_receipt_sha256,
            self.legacy_candidate_v1_receipt_sha256,
            self.validation_selection_specification_sha256,
            self.candidate_ranking_specification_sha256,
            self.candidate_tie_breaking_specification_sha256,
            self.numerical_comparison_specification_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL policy candidate V2", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_policy_candidate_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    checkpoint_authority_receipt_sha256: str,
    checkpoint: MassiveAdaptiveRLCheckpointV1,
    primary_trace: MassiveAdaptiveRLPolicyTraceV1,
    low_cost_trace: MassiveAdaptiveRLPolicyTraceV1,
    high_cost_trace: MassiveAdaptiveRLPolicyTraceV1,
    fixed_control_selection_authority: (
        MassiveAdaptiveRLFixedControlSelectionAuthorityV1
    ),
    fixed_control_validation_trace: MassiveAdaptiveRLPolicyTraceV1,
) -> MassiveAdaptiveRLPolicyCandidateV2:
    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV4:
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy candidate V2 Manifest V4 differs"
        )
    manifest.validate()
    legacy = build_massive_adaptive_rl_policy_candidate_v1(
        checkpoint=checkpoint,
        primary_trace=primary_trace,
        low_cost_trace=low_cost_trace,
        high_cost_trace=high_cost_trace,
        fixed_control_selection_authority=fixed_control_selection_authority,
        fixed_control_validation_trace=fixed_control_validation_trace,
    )
    metrics = {
        "fc06_primary_incremental_log_wealth": _binary64(
            "FC06 primary incremental log wealth",
            legacy.best_fixed_control_incremental_log_wealth,
        ),
        "ppo_minus_fc06_log_wealth": _binary64(
            "PPO-minus-FC06 log wealth",
            legacy.ppo_minus_best_fixed_control_log_wealth,
        ),
        "primary_incremental_rl_log_wealth": _binary64(
            "primary incremental RL log wealth",
            legacy.primary_incremental_rl_log_wealth,
        ),
        "primary_strategy_active_log_wealth": _binary64(
            "primary strategy active log wealth",
            legacy.primary_strategy_active_log_wealth,
        ),
        "low_cost_terminal_liquidation_adjusted_return": _binary64(
            "low-cost terminal liquidation-adjusted return",
            low_cost_trace.terminal_liquidation_adjusted_return,
        ),
        "primary_cost_terminal_liquidation_adjusted_return": _binary64(
            "primary-cost terminal liquidation-adjusted return",
            primary_trace.terminal_liquidation_adjusted_return,
        ),
        "high_cost_terminal_liquidation_adjusted_return": _binary64(
            "high-cost terminal liquidation-adjusted return",
            high_cost_trace.terminal_liquidation_adjusted_return,
        ),
        "maximum_drawdown": _binary64(
            "maximum drawdown",
            legacy.maximum_drawdown,
        ),
    }
    failures = _validation_eligibility_failures_v1(
        primary_incremental_rl_log_wealth=metrics["primary_incremental_rl_log_wealth"],
        ppo_minus_fc06_log_wealth=metrics["ppo_minus_fc06_log_wealth"],
        primary_strategy_active_log_wealth=metrics[
            "primary_strategy_active_log_wealth"
        ],
        low_cost_terminal_liquidation_adjusted_return=metrics[
            "low_cost_terminal_liquidation_adjusted_return"
        ],
        primary_cost_terminal_liquidation_adjusted_return=metrics[
            "primary_cost_terminal_liquidation_adjusted_return"
        ],
        high_cost_terminal_liquidation_adjusted_return=metrics[
            "high_cost_terminal_liquidation_adjusted_return"
        ],
        maximum_drawdown=metrics["maximum_drawdown"],
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SCHEMA,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_index": legacy.fold_index,
        "checkpoint_authority_receipt_sha256": _digest(
            "candidate checkpoint authority receipt",
            checkpoint_authority_receipt_sha256,
        ),
        "checkpoint_receipt_sha256": legacy.checkpoint_receipt_sha256,
        "model_state_receipt_sha256": legacy.model_state_receipt_sha256,
        "update_index": legacy.update_index,
        "training_forecast_authority_receipt_sha256": (
            legacy.training_forecast_authority_receipt_sha256
        ),
        "primary_trace_receipt_sha256": legacy.primary_trace_receipt_sha256,
        "low_cost_trace_receipt_sha256": legacy.low_cost_trace_receipt_sha256,
        "high_cost_trace_receipt_sha256": legacy.high_cost_trace_receipt_sha256,
        "decision_target_inventory_sha256": (legacy.decision_target_inventory_sha256),
        "fixed_control_selection_authority_receipt_sha256": (
            legacy.fixed_control_selection_authority_receipt_sha256
        ),
        "selected_fc06_action_receipt_sha256": (
            legacy.selected_fixed_action_receipt_sha256
        ),
        "fc06_validation_trace_receipt_sha256": (
            legacy.fixed_control_validation_trace_receipt_sha256
        ),
        "legacy_candidate_v1_receipt_sha256": legacy.semantic_receipt_sha256,
        **metrics,
        "validation_eligibility_failures": failures,
        "economically_eligible": not failures,
        "source_data_qualified": legacy.source_data_qualified,
        "validation_selection_specification_sha256": (
            manifest.validation_selection_specification_sha256
        ),
        "candidate_ranking_specification_sha256": (
            manifest.candidate_ranking_specification_sha256
        ),
        "candidate_tie_breaking_specification_sha256": (
            manifest.candidate_tie_breaking_specification_sha256
        ),
        "numerical_comparison_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
        ),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLPolicyCandidateV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def validation_rank_key_v1(
    candidate: MassiveAdaptiveRLPolicyCandidateV2,
) -> tuple[float, float, float, float, float, int, str]:
    """Return the exact total ordering preregistered by Manifest V4."""

    if type(candidate) is not MassiveAdaptiveRLPolicyCandidateV2:
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL validation ranking candidate type differs"
        )
    candidate.validate()
    return (
        -candidate.primary_incremental_rl_log_wealth,
        -candidate.ppo_minus_fc06_log_wealth,
        -candidate.primary_strategy_active_log_wealth,
        -candidate.high_cost_terminal_liquidation_adjusted_return,
        candidate.maximum_drawdown,
        candidate.update_index,
        candidate.checkpoint_receipt_sha256,
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicySelectionV2:
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_fit_authority_receipt_sha256: str
    fold_index: int
    selected_checkpoint_authority_receipt_sha256: str
    selected_checkpoint_receipt_sha256: str
    selected_model_state_receipt_sha256: str
    selected_update_index: int
    selected_candidate_receipt_sha256: str
    selected_candidate_validation_eligible: bool
    validation_eligibility_failures: tuple[str, ...]
    selection_pool_kind: str
    expected_candidate_checkpoint_authority_receipts: tuple[str, ...]
    candidate_receipts: tuple[str, ...]
    candidate_inventory_sha256: str
    ranked_candidate_receipts: tuple[str, ...]
    ranked_candidate_inventory_sha256: str
    candidate_count: int
    training_forecast_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    source_data_qualified: bool
    positive_profitability_authorization_eligible: bool
    validation_selection_specification_sha256: str
    candidate_ranking_specification_sha256: str
    candidate_tie_breaking_specification_sha256: str
    numerical_comparison_specification_sha256: str
    no_eligible_candidate_policy: str
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SCHEMA
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or isinstance(self.selected_update_index, bool)
            or self.selected_update_index < 0
            or self.candidate_count != self.fold_index + 1
            or len(self.expected_candidate_checkpoint_authority_receipts)
            != self.candidate_count
            or len(set(self.expected_candidate_checkpoint_authority_receipts))
            != self.candidate_count
            or len(self.candidate_receipts) != self.candidate_count
            or len(set(self.candidate_receipts)) != self.candidate_count
            or len(self.ranked_candidate_receipts) != self.candidate_count
            or set(self.ranked_candidate_receipts) != set(self.candidate_receipts)
            or self.selected_candidate_receipt_sha256 not in self.candidate_receipts
            or self.selected_checkpoint_authority_receipt_sha256
            not in self.expected_candidate_checkpoint_authority_receipts
            or self.candidate_inventory_sha256
            != semantic_sha256(self.candidate_receipts)
            or self.ranked_candidate_inventory_sha256
            != semantic_sha256(self.ranked_candidate_receipts)
            or self.validation_eligibility_failures
            != tuple(sorted(set(self.validation_eligibility_failures)))
            or not set(self.validation_eligibility_failures).issubset(
                MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1
            )
            or self.selected_candidate_validation_eligible
            != (not self.validation_eligibility_failures)
            or self.selection_pool_kind not in {"eligible", "all-no-eligible"}
            or (self.selection_pool_kind == "eligible")
            != self.selected_candidate_validation_eligible
            or self.positive_profitability_authorization_eligible
            != bool(
                self.source_data_qualified
                and self.selected_candidate_validation_eligible
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.validation_selection_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
            or self.candidate_ranking_specification_sha256
            != MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
            or self.candidate_tie_breaking_specification_sha256
            != MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
            or self.numerical_comparison_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
            or self.no_eligible_candidate_policy
            != MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicySelectionV2Error(
                "adaptive RL policy selection V2 differs"
            )
        for value in (
            self.manifest_v4_receipt_sha256,
            self.training_manifest_v3_receipt_sha256,
            self.fold_fit_authority_receipt_sha256,
            self.selected_checkpoint_authority_receipt_sha256,
            self.selected_checkpoint_receipt_sha256,
            self.selected_model_state_receipt_sha256,
            self.selected_candidate_receipt_sha256,
            *self.expected_candidate_checkpoint_authority_receipts,
            *self.candidate_receipts,
            self.candidate_inventory_sha256,
            *self.ranked_candidate_receipts,
            self.ranked_candidate_inventory_sha256,
            self.training_forecast_authority_receipt_sha256,
            self.fixed_control_selection_authority_receipt_sha256,
            self.validation_selection_specification_sha256,
            self.candidate_ranking_specification_sha256,
            self.candidate_tie_breaking_specification_sha256,
            self.numerical_comparison_specification_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL policy selection V2", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _ordered_candidates_v2(
    *,
    expected_checkpoint_authority_receipts: tuple[str, ...],
    candidates: Sequence[MassiveAdaptiveRLPolicyCandidateV2],
) -> tuple[MassiveAdaptiveRLPolicyCandidateV2, ...]:
    rows = tuple(candidates)
    if any(type(row) is not MassiveAdaptiveRLPolicyCandidateV2 for row in rows):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy candidate V2 type differs"
        )
    for row in rows:
        row.validate()
    by_authority = {row.checkpoint_authority_receipt_sha256: row for row in rows}
    if (
        len(by_authority) != len(rows)
        or set(by_authority) != set(expected_checkpoint_authority_receipts)
        or len(set(expected_checkpoint_authority_receipts))
        != len(expected_checkpoint_authority_receipts)
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy candidate checkpoint coverage differs"
        )
    return tuple(
        by_authority[receipt] for receipt in expected_checkpoint_authority_receipts
    )


def select_massive_adaptive_rl_policy_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_fit_authority_receipt_sha256: str,
    expected_candidate_checkpoint_authority_receipts: Sequence[str],
    candidates: Sequence[MassiveAdaptiveRLPolicyCandidateV2],
) -> MassiveAdaptiveRLPolicySelectionV2:
    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV4:
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy selection V2 Manifest V4 differs"
        )
    manifest.validate()
    expected_receipts = tuple(
        _digest("candidate checkpoint authority receipt", value)
        for value in expected_candidate_checkpoint_authority_receipts
    )
    ordered = _ordered_candidates_v2(
        expected_checkpoint_authority_receipts=expected_receipts,
        candidates=candidates,
    )
    if not ordered:
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy candidates V2 are absent"
        )
    fold_index = ordered[0].fold_index
    schedule = manifest.base_manifest.base_manifest.schedule(fold_index)
    lineage = {
        (
            row.manifest_v4_receipt_sha256,
            row.training_manifest_v3_receipt_sha256,
            row.fold_index,
            row.training_forecast_authority_receipt_sha256,
            row.fixed_control_selection_authority_receipt_sha256,
        )
        for row in ordered
    }
    if (
        len(lineage) != 1
        or ordered[0].manifest_v4_receipt_sha256 != manifest.semantic_receipt_sha256
        or ordered[0].training_manifest_v3_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or tuple(row.update_index for row in ordered)
        != schedule.candidate_update_indices
        or len(ordered) != fold_index + 1
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy candidate V2 lineage or schedule differs"
        )
    ranked = tuple(sorted(ordered, key=validation_rank_key_v1))
    eligible = tuple(row for row in ranked if row.economically_eligible)
    selection_pool = eligible if eligible else ranked
    selected = selection_pool[0]
    pool_kind = "eligible" if eligible else "all-no-eligible"
    candidate_receipts = tuple(row.semantic_receipt_sha256 for row in ordered)
    ranked_receipts = tuple(row.semantic_receipt_sha256 for row in ranked)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SCHEMA,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_fit_authority_receipt_sha256": _digest(
            "fold-fit authority receipt",
            fold_fit_authority_receipt_sha256,
        ),
        "fold_index": fold_index,
        "selected_checkpoint_authority_receipt_sha256": (
            selected.checkpoint_authority_receipt_sha256
        ),
        "selected_checkpoint_receipt_sha256": selected.checkpoint_receipt_sha256,
        "selected_model_state_receipt_sha256": selected.model_state_receipt_sha256,
        "selected_update_index": selected.update_index,
        "selected_candidate_receipt_sha256": selected.semantic_receipt_sha256,
        "selected_candidate_validation_eligible": selected.economically_eligible,
        "validation_eligibility_failures": (selected.validation_eligibility_failures),
        "selection_pool_kind": pool_kind,
        "expected_candidate_checkpoint_authority_receipts": expected_receipts,
        "candidate_receipts": candidate_receipts,
        "candidate_inventory_sha256": semantic_sha256(candidate_receipts),
        "ranked_candidate_receipts": ranked_receipts,
        "ranked_candidate_inventory_sha256": semantic_sha256(ranked_receipts),
        "candidate_count": len(ordered),
        "training_forecast_authority_receipt_sha256": (
            selected.training_forecast_authority_receipt_sha256
        ),
        "fixed_control_selection_authority_receipt_sha256": (
            selected.fixed_control_selection_authority_receipt_sha256
        ),
        "source_data_qualified": all(row.source_data_qualified for row in ordered),
        "positive_profitability_authorization_eligible": (
            all(row.source_data_qualified for row in ordered)
            and selected.economically_eligible
        ),
        "validation_selection_specification_sha256": (
            manifest.validation_selection_specification_sha256
        ),
        "candidate_ranking_specification_sha256": (
            manifest.candidate_ranking_specification_sha256
        ),
        "candidate_tie_breaking_specification_sha256": (
            manifest.candidate_tie_breaking_specification_sha256
        ),
        "numerical_comparison_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
        ),
        "no_eligible_candidate_policy": manifest.no_eligible_candidate_policy,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLPolicySelectionV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _validate_fold_fit_lineage_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_fit_authority: MassiveAdaptiveRLFoldFitAuthorityV1,
    selection: MassiveAdaptiveRLPolicySelectionV2 | None = None,
    candidates: Sequence[MassiveAdaptiveRLPolicyCandidateV2] | None = None,
) -> None:
    if (
        not fold_fit_authority.development_stage_authorized
        or fold_fit_authority.manifest_v3_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or fold_fit_authority.experiment_id != manifest.experiment_id
        or fold_fit_authority.candidate_checkpoint_inventory_sha256
        != semantic_sha256(fold_fit_authority.candidate_checkpoint_authority_receipts)
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy selection V2 fold-fit lineage differs"
        )
    if selection is not None and (
        selection.fold_index != fold_fit_authority.outer_fold_index
        or selection.fold_fit_authority_receipt_sha256
        != fold_fit_authority.semantic_receipt_sha256
        or selection.expected_candidate_checkpoint_authority_receipts
        != fold_fit_authority.candidate_checkpoint_authority_receipts
        or selection.training_forecast_authority_receipt_sha256
        != fold_fit_authority.training_forecast_authority.semantic_receipt_sha256
        or selection.fixed_control_selection_authority_receipt_sha256
        != fold_fit_authority.fixed_control_selection_authority_receipt_sha256
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy selection V2 nested fit lineage differs"
        )
    if candidates is not None:
        ordered = _ordered_candidates_v2(
            expected_checkpoint_authority_receipts=(
                fold_fit_authority.candidate_checkpoint_authority_receipts
            ),
            candidates=candidates,
        )
        runtime_workflow = fold_fit_authority.training_workflow.runtime_workflow
        policy_authorities = runtime_workflow.policy_checkpoint_authorities
        checkpoints = tuple(row.runtime_checkpoint for row in policy_authorities)
        fixed_selection = (
            runtime_workflow.fixed_control_selection_authority.runtime_selection
        )
        if (
            tuple(row.semantic_receipt_sha256 for row in policy_authorities)
            != fold_fit_authority.candidate_checkpoint_authority_receipts
            or any(checkpoint is None for checkpoint in checkpoints)
            or fixed_selection is None
        ):
            raise MassiveAdaptiveRLPolicySelectionV2Error(
                "adaptive RL policy selection V2 fit checkpoint replay differs"
            )
        for candidate, checkpoint in zip(ordered, checkpoints, strict=True):
            assert checkpoint is not None
            if (
                candidate.checkpoint_receipt_sha256
                != checkpoint.semantic_receipt_sha256
                or candidate.model_state_receipt_sha256
                != checkpoint.model_state_receipt_sha256
                or candidate.update_index != checkpoint.update_index
                or candidate.training_forecast_authority_receipt_sha256
                != checkpoint.training_forecast_authority_receipt_sha256
                or candidate.selected_fc06_action_receipt_sha256
                != fixed_selection.selected_action_receipt_sha256
            ):
                raise MassiveAdaptiveRLPolicySelectionV2Error(
                    "adaptive RL policy selection V2 candidate checkpoint differs"
                )


def _candidates_from_fold_validation_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV1,
) -> tuple[
    MassiveAdaptiveRLFoldFitAuthorityV1,
    tuple[MassiveAdaptiveRLPolicyCandidateV2, ...],
]:
    """Derive candidates only from persisted and computationally replayed traces."""

    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(validation_authority) is not MassiveAdaptiveRLFoldValidationAuthorityV1
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy selection V2 validation authority type differs"
        )
    manifest.validate()
    validation_authority.validate()
    fold_fit_authority = validation_authority.runtime_fold_fit_authority
    primary_authorities = validation_authority.runtime_primary_trace_authorities
    ladder_authorities = validation_authority.runtime_cost_ladder_authorities
    fixed_authority = validation_authority.runtime_fixed_control_validation_authority
    if (
        not validation_authority.development_stage_authorized
        or fold_fit_authority is None
        or primary_authorities is None
        or ladder_authorities is None
        or fixed_authority is None
        or validation_authority.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy selection V2 validation evidence is not authorized"
        )
    _validate_fold_fit_lineage_v2(
        manifest=manifest,
        fold_fit_authority=fold_fit_authority,
    )
    workflow = fold_fit_authority.training_workflow.runtime_workflow
    checkpoint_authorities = workflow.policy_checkpoint_authorities
    fixed_selection_authority = workflow.fixed_control_selection_authority
    fixed_evaluation = fixed_authority.runtime_evaluation
    if (
        tuple(row.semantic_receipt_sha256 for row in checkpoint_authorities)
        != validation_authority.expected_checkpoint_authority_receipts
        or len(primary_authorities) != len(checkpoint_authorities)
        or len(ladder_authorities) != len(checkpoint_authorities)
        or fixed_evaluation is None
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "adaptive RL policy selection V2 validation inventory differs"
        )
    candidates = []
    for checkpoint_authority, primary_authority, ladder_authority in zip(
        checkpoint_authorities,
        primary_authorities,
        ladder_authorities,
        strict=True,
    ):
        checkpoint = checkpoint_authority.runtime_checkpoint
        primary = primary_authority.runtime_trace
        ladder = ladder_authority.runtime_ladder
        if checkpoint is None or primary is None or ladder is None:
            raise MassiveAdaptiveRLPolicySelectionV2Error(
                "adaptive RL policy selection V2 validation trace is absent"
            )
        candidates.append(
            build_massive_adaptive_rl_policy_candidate_v2(
                manifest=manifest,
                checkpoint_authority_receipt_sha256=(
                    checkpoint_authority.semantic_receipt_sha256
                ),
                checkpoint=checkpoint,
                primary_trace=primary.policy_trace,
                low_cost_trace=ladder.low_cost_trace,
                high_cost_trace=ladder.high_cost_trace,
                fixed_control_selection_authority=fixed_selection_authority,
                fixed_control_validation_trace=fixed_evaluation.policy_trace,
            )
        )
    result = tuple(candidates)
    _validate_fold_fit_lineage_v2(
        manifest=manifest,
        fold_fit_authority=fold_fit_authority,
        candidates=result,
    )
    return fold_fit_authority, result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicySelectionAuthorityV2:
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_fit_authority_receipt_sha256: str
    fold_validation_authority_receipt_sha256: str
    selection_receipt_sha256: str
    candidate_inventory_sha256: str
    candidate_checkpoint_inventory_sha256: str
    source_data_qualified: bool
    selected_candidate_validation_eligible: bool
    positive_profitability_authorization_eligible: bool
    validation_selection_specification_sha256: str
    numerical_comparison_specification_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_manifest: MassiveAdaptiveRLExperimentManifestV4 | None
    runtime_fold_fit_authority: MassiveAdaptiveRLFoldFitAuthorityV1 | None
    runtime_fold_validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV1 | None
    runtime_selection: MassiveAdaptiveRLPolicySelectionV2 | None
    runtime_candidates: tuple[MassiveAdaptiveRLPolicyCandidateV2, ...] | None
    runtime_selection_replayed: bool
    development_selection_computation_authorized: bool
    development_policy_selection_authorized: bool
    policy_freezing_authorized: bool
    outer_diagnostic_preparation_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "manifest_v4_receipt_sha256": self.manifest_v4_receipt_sha256,
            "training_manifest_v3_receipt_sha256": (
                self.training_manifest_v3_receipt_sha256
            ),
            "fold_fit_authority_receipt_sha256": (
                self.fold_fit_authority_receipt_sha256
            ),
            "fold_validation_authority_receipt_sha256": (
                self.fold_validation_authority_receipt_sha256
            ),
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "candidate_checkpoint_inventory_sha256": (
                self.candidate_checkpoint_inventory_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "selected_candidate_validation_eligible": (
                self.selected_candidate_validation_eligible
            ),
            "positive_profitability_authorization_eligible": (
                self.positive_profitability_authorization_eligible
            ),
            "validation_selection_specification_sha256": (
                self.validation_selection_specification_sha256
            ),
            "numerical_comparison_specification_sha256": (
                self.numerical_comparison_specification_sha256
            ),
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    @property
    def selection_computation_replayed(self) -> bool:
        return self.runtime_selection_replayed

    def validate(self) -> None:
        runtime_values = (
            self.runtime_manifest,
            self.runtime_fold_fit_authority,
            self.runtime_fold_validation_authority,
            self.runtime_selection,
            self.runtime_candidates,
        )
        runtime = all(value is not None for value in runtime_values)
        if any(value is not None for value in runtime_values) != runtime:
            raise MassiveAdaptiveRLPolicySelectionV2Error(
                "adaptive RL policy selection V2 runtime is partial"
            )
        expected_computation_authorized = bool(
            runtime
            and self.source_data_qualified
            and self.runtime_fold_validation_authority is not None
            and self.runtime_fold_validation_authority.development_stage_authorized
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SCHEMA
            or not isinstance(self.source_data_qualified, bool)
            or self.positive_profitability_authorization_eligible
            or self.development_selection_computation_authorized
            != expected_computation_authorized
            or self.runtime_selection_replayed != runtime
            or self.development_policy_selection_authorized
            or self.policy_freezing_authorized
            or self.outer_diagnostic_preparation_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.validation_selection_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
            or self.numerical_comparison_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicySelectionV2Error(
                "adaptive RL policy selection authority V2 differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.selection_receipt_sha256
        ):
            raise MassiveAdaptiveRLPolicySelectionV2Error(
                "adaptive RL policy selection V2 source transaction differs"
            )
        if runtime:
            assert self.runtime_manifest is not None
            assert self.runtime_fold_fit_authority is not None
            assert self.runtime_fold_validation_authority is not None
            assert self.runtime_selection is not None
            assert self.runtime_candidates is not None
            if type(self.runtime_manifest) is not MassiveAdaptiveRLExperimentManifestV4:
                raise MassiveAdaptiveRLPolicySelectionV2Error(
                    "adaptive RL runtime Manifest V4 type differs"
                )
            if (
                type(self.runtime_fold_fit_authority)
                is not MassiveAdaptiveRLFoldFitAuthorityV1
            ):
                raise MassiveAdaptiveRLPolicySelectionV2Error(
                    "adaptive RL runtime fold-fit authority type differs"
                )
            if type(self.runtime_fold_validation_authority) is not (
                MassiveAdaptiveRLFoldValidationAuthorityV1
            ):
                raise MassiveAdaptiveRLPolicySelectionV2Error(
                    "adaptive RL runtime fold-validation authority type differs"
                )
            self.runtime_manifest.validate()
            self.runtime_fold_fit_authority.validate()
            self.runtime_fold_validation_authority.validate()
            derived_fold_fit, derived_candidates = _candidates_from_fold_validation_v1(
                manifest=self.runtime_manifest,
                validation_authority=self.runtime_fold_validation_authority,
            )
            if (
                derived_fold_fit.semantic_receipt_sha256
                != self.runtime_fold_fit_authority.semantic_receipt_sha256
                or derived_candidates != self.runtime_candidates
                or self.runtime_fold_validation_authority.semantic_receipt_sha256
                != self.fold_validation_authority_receipt_sha256
            ):
                raise MassiveAdaptiveRLPolicySelectionV2Error(
                    "adaptive RL runtime fold-validation selection differs"
                )
            _validate_fold_fit_lineage_v2(
                manifest=self.runtime_manifest,
                fold_fit_authority=self.runtime_fold_fit_authority,
                selection=self.runtime_selection,
                candidates=self.runtime_candidates,
            )
            rebuilt = select_massive_adaptive_rl_policy_v2(
                manifest=self.runtime_manifest,
                fold_fit_authority_receipt_sha256=(
                    self.runtime_fold_fit_authority.semantic_receipt_sha256
                ),
                expected_candidate_checkpoint_authority_receipts=(
                    self.runtime_fold_fit_authority.candidate_checkpoint_authority_receipts
                ),
                candidates=self.runtime_candidates,
            )
            if (
                rebuilt != self.runtime_selection
                or rebuilt.semantic_receipt_sha256 != self.selection_receipt_sha256
                or rebuilt.candidate_inventory_sha256 != self.candidate_inventory_sha256
                or self.runtime_manifest.semantic_receipt_sha256
                != self.manifest_v4_receipt_sha256
                or self.runtime_manifest.base_manifest.semantic_receipt_sha256
                != self.training_manifest_v3_receipt_sha256
                or self.runtime_fold_fit_authority.semantic_receipt_sha256
                != self.fold_fit_authority_receipt_sha256
                or self.runtime_fold_fit_authority.candidate_checkpoint_inventory_sha256
                != self.candidate_checkpoint_inventory_sha256
                or rebuilt.selected_candidate_validation_eligible
                != self.selected_candidate_validation_eligible
            ):
                raise MassiveAdaptiveRLPolicySelectionV2Error(
                    "adaptive RL runtime policy selection V2 differs"
                )
        for value in (
            self.manifest_v4_receipt_sha256,
            self.training_manifest_v3_receipt_sha256,
            self.fold_fit_authority_receipt_sha256,
            self.fold_validation_authority_receipt_sha256,
            self.selection_receipt_sha256,
            self.candidate_inventory_sha256,
            self.candidate_checkpoint_inventory_sha256,
            self.validation_selection_specification_sha256,
            self.numerical_comparison_specification_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL policy selection authority V2", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _authority_payload_v2(
    *,
    selection: MassiveAdaptiveRLPolicySelectionV2,
    candidates: tuple[MassiveAdaptiveRLPolicyCandidateV2, ...],
    fold_validation_authority_receipt_sha256: str,
) -> dict[str, object]:
    return {
        "fold_validation_authority_receipt_sha256": _digest(
            "fold-validation authority receipt",
            fold_validation_authority_receipt_sha256,
        ),
        "selection": asdict(selection),
        "candidates": tuple(asdict(candidate) for candidate in candidates),
    }


def _load_authority_payload_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> tuple[
    MassiveAdaptiveRLPolicySelectionV2,
    tuple[MassiveAdaptiveRLPolicyCandidateV2, ...],
    str,
]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, Mapping) or raw != canonical_json_file_bytes(payload):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "RL policy selection V2 payload is not canonical JSON"
        )
    selection_payload = dict(payload["selection"])
    for name in (
        "validation_eligibility_failures",
        "expected_candidate_checkpoint_authority_receipts",
        "candidate_receipts",
        "ranked_candidate_receipts",
    ):
        selection_payload[name] = tuple(selection_payload[name])
    selection = MassiveAdaptiveRLPolicySelectionV2(**selection_payload)
    candidates = tuple(
        MassiveAdaptiveRLPolicyCandidateV2(
            **{
                **dict(value),
                "validation_eligibility_failures": tuple(
                    value["validation_eligibility_failures"]
                ),
            }
        )
        for value in payload["candidates"]
    )
    selection.validate()
    for candidate in candidates:
        candidate.validate()
    validation_receipt = _digest(
        "fold-validation authority receipt",
        payload.get("fold_validation_authority_receipt_sha256"),
    )
    return selection, candidates, validation_receipt


def parse_massive_adaptive_rl_policy_selection_authority_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLPolicySelectionAuthorityV2:
    selection, _candidates, validation_receipt = _load_authority_payload_v2(
        root=root,
        loaded_source=loaded_source,
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SCHEMA,
        "manifest_v4_receipt_sha256": selection.manifest_v4_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            selection.training_manifest_v3_receipt_sha256
        ),
        "fold_fit_authority_receipt_sha256": (
            selection.fold_fit_authority_receipt_sha256
        ),
        "fold_validation_authority_receipt_sha256": str(validation_receipt),
        "selection_receipt_sha256": selection.semantic_receipt_sha256,
        "candidate_inventory_sha256": selection.candidate_inventory_sha256,
        "candidate_checkpoint_inventory_sha256": semantic_sha256(
            selection.expected_candidate_checkpoint_authority_receipts
        ),
        "source_data_qualified": selection.source_data_qualified,
        "selected_candidate_validation_eligible": (
            selection.selected_candidate_validation_eligible
        ),
        "positive_profitability_authorization_eligible": False,
        "validation_selection_specification_sha256": (
            selection.validation_selection_specification_sha256
        ),
        "numerical_comparison_specification_sha256": (
            selection.numerical_comparison_specification_sha256
        ),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLPolicySelectionAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        loaded_source=loaded_source,
        runtime_manifest=None,
        runtime_fold_fit_authority=None,
        runtime_fold_validation_authority=None,
        runtime_selection=None,
        runtime_candidates=None,
        runtime_selection_replayed=False,
        development_selection_computation_authorized=False,
        development_policy_selection_authorized=False,
        policy_freezing_authorized=False,
        outer_diagnostic_preparation_authorized=False,
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_policy_selection_authority_v2(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLPolicySelectionAuthorityV2,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV1,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV2:
    fold_fit_authority, ordered = _candidates_from_fold_validation_v1(
        manifest=manifest,
        validation_authority=validation_authority,
    )
    parsed = parse_massive_adaptive_rl_policy_selection_authority_v2(
        root=root,
        loaded_source=authority.loaded_source,
    )
    (
        committed_selection,
        committed_candidates,
        committed_validation_receipt,
    ) = _load_authority_payload_v2(
        root=root,
        loaded_source=authority.loaded_source,
    )
    rebuilt = select_massive_adaptive_rl_policy_v2(
        manifest=manifest,
        fold_fit_authority_receipt_sha256=(fold_fit_authority.semantic_receipt_sha256),
        expected_candidate_checkpoint_authority_receipts=(
            fold_fit_authority.candidate_checkpoint_authority_receipts
        ),
        candidates=ordered,
    )
    _validate_fold_fit_lineage_v2(
        manifest=manifest,
        fold_fit_authority=fold_fit_authority,
        selection=rebuilt,
        candidates=ordered,
    )
    if (
        committed_candidates != ordered
        or committed_selection != rebuilt
        or committed_validation_receipt != validation_authority.semantic_receipt_sha256
        or parsed.fold_validation_authority_receipt_sha256
        != validation_authority.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLPolicySelectionV2Error(
            "RL policy selection authority V2 does not replay"
        )
    result = MassiveAdaptiveRLPolicySelectionAuthorityV2(
        **parsed.semantic_unsigned(),  # type: ignore[arg-type]
        semantic_receipt_sha256=parsed.semantic_receipt_sha256,
        loaded_source=parsed.loaded_source,
        runtime_manifest=manifest,
        runtime_fold_fit_authority=fold_fit_authority,
        runtime_fold_validation_authority=validation_authority,
        runtime_selection=rebuilt,
        runtime_candidates=ordered,
        runtime_selection_replayed=True,
        development_selection_computation_authorized=rebuilt.source_data_qualified,
        development_policy_selection_authorized=False,
        policy_freezing_authorized=False,
        outer_diagnostic_preparation_authorized=False,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_policy_selection_authority_v2(
    *,
    root: str | Path,
    artifact_id: str,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV2:
    artifact = _artifact_id(artifact_id)
    fold_fit_authority, ordered = _candidates_from_fold_validation_v1(
        manifest=manifest,
        validation_authority=validation_authority,
    )
    selection = select_massive_adaptive_rl_policy_v2(
        manifest=manifest,
        fold_fit_authority_receipt_sha256=(fold_fit_authority.semantic_receipt_sha256),
        expected_candidate_checkpoint_authority_receipts=(
            fold_fit_authority.candidate_checkpoint_authority_receipts
        ),
        candidates=ordered,
    )
    _validate_fold_fit_lineage_v2(
        manifest=manifest,
        fold_fit_authority=fold_fit_authority,
        selection=selection,
        candidates=ordered,
    )
    relative = f"massive-adaptive/rl-policy-selection-v2/{artifact}.json"
    publish_massive_source_object(
        stream=BytesIO(
            canonical_json_file_bytes(
                _authority_payload_v2(
                    selection=selection,
                    candidates=ordered,
                    fold_validation_authority_receipt_sha256=(
                        validation_authority.semantic_receipt_sha256
                    ),
                )
            )
        ),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=selection.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-POLICY-SELECTION-V2-{artifact}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_policy_selection_authority_v2(
        root=root,
        authority=parse_massive_adaptive_rl_policy_selection_authority_v2(
            root=root,
            loaded_source=loaded,
        ),
        manifest=manifest,
        validation_authority=validation_authority,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256",
    "MassiveAdaptiveRLPolicyCandidateV2",
    "MassiveAdaptiveRLPolicySelectionAuthorityV2",
    "MassiveAdaptiveRLPolicySelectionV2",
    "MassiveAdaptiveRLPolicySelectionV2Error",
    "authorize_massive_adaptive_rl_policy_selection_authority_v2",
    "build_massive_adaptive_rl_policy_candidate_v2",
    "materialize_massive_adaptive_rl_policy_selection_authority_v2",
    "parse_massive_adaptive_rl_policy_selection_authority_v2",
    "select_massive_adaptive_rl_policy_v2",
    "validation_rank_key_v1",
]
