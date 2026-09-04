"""V5-native validation outcomes derived from one causal release.

The authority persists only package-computed PPO cost-ladder or FC06 evidence.
It never accepts actions, targets, returns, metrics, or an environment from the
caller.  A generic reload proves byte integrity only; exact economic replay is
required before the outcome can contribute to fold validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
import json
import math
from pathlib import Path
import time
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_v1 import (
    MassiveAdaptiveRLCostLadderV1,
    evaluate_massive_adaptive_rl_checkpoint_cost_ladder_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_evaluator_v1 import (
    MassiveAdaptiveRLFixedControlEvaluationV1,
    evaluate_massive_adaptive_rl_fixed_control_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    MassiveAdaptiveRLCheckpointAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    build_massive_adaptive_rl_fixed_control_registry_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SCHEMA,
    MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_v1 import (
    MassiveAdaptiveRLFoldFitAuthorityV1,
)


MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_DATASET = (
    "massive-adaptive-rl-validation-outcome-authority-v3"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SCHEMA,
            "encoding": "canonical-json-v5-native-economic-witness",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_PPO_VALIDATION_OUTCOME_V3 = "ppo-cost-ladder"
MASSIVE_ADAPTIVE_RL_FC06_VALIDATION_OUTCOME_V3 = "fc06-primary"
_OUTCOME_KINDS = frozenset(
    {
        MASSIVE_ADAPTIVE_RL_PPO_VALIDATION_OUTCOME_V3,
        MASSIVE_ADAPTIVE_RL_FC06_VALIDATION_OUTCOME_V3,
    }
)


class MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(ValueError):
    """The causal release, economic replay, or persisted outcome differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: str | None) -> str:
    if value is None:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(f"{name} is absent")
    return _digest(name, value)


def _required_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            f"{name} is absent or invalid"
        )
    return value


def _wall_clock_after(value: int) -> int:
    now = time.time_ns() // 1_000_000
    if isinstance(now, bool) or not isinstance(now, int) or now < 0:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "validation outcome clock differs"
        )
    return max(now, value + 1)


def validation_outcome_authority_relative_path_v3(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    fold_index: int,
    outcome_kind: str,
    subject_receipt_sha256: str,
) -> str:
    manifest.validate()
    if (
        isinstance(fold_index, bool)
        or fold_index not in range(4)
        or outcome_kind not in _OUTCOME_KINDS
    ):
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "validation outcome identity differs"
        )
    subject = _digest("validation outcome subject", subject_receipt_sha256)
    name = (
        "fc06"
        if outcome_kind == MASSIVE_ADAPTIVE_RL_FC06_VALIDATION_OUTCOME_V3
        else f"ppo-{subject}"
    )
    return (
        f"adaptive-rl/{manifest.experiment_id}/validation-outcome-v3/"
        f"fold-{fold_index}/{name}.json"
    )


def _transaction_state(*, root: str | Path, relative: str) -> tuple[bool, bool]:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    return all(present), any(present) and not all(present)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    release_authority_receipt_sha256: str
    release_source_receipt_sha256: str
    release_commit_receipt_sha256: str
    release_committed_at_ms: int
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    fold_fit_authority_receipt_sha256: str
    fold_index: int
    outcome_kind: str
    subject_receipt_sha256: str
    validation_sources_v2_receipt_sha256: str
    validation_registry_v2_receipt_sha256: str
    validation_context_receipt_sha256: str
    checkpoint_authority_receipt_sha256: str | None
    checkpoint_receipt_sha256: str | None
    model_state_receipt_sha256: str | None
    update_index: int | None
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_fc06_action_receipt_sha256: str
    economic_witness_receipt_sha256: str
    primary_trace_receipt_sha256: str
    low_cost_trace_receipt_sha256: str | None
    high_cost_trace_receipt_sha256: str | None
    decision_target_inventory_sha256: str
    primary_transition_inventory_sha256: str
    low_cost_transition_inventory_sha256: str | None
    high_cost_transition_inventory_sha256: str | None
    primary_incremental_log_wealth: float
    primary_active_log_wealth: float
    low_cost_terminal_liquidation_adjusted_return: float | None
    primary_cost_terminal_liquidation_adjusted_return: float
    high_cost_terminal_liquidation_adjusted_return: float | None
    maximum_drawdown: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_economic_witness_replayed: bool = False
    development_fold_validation_authorized: bool = False
    policy_freezing_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV5 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_release: MassiveAdaptiveRLValidationReleaseAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_checkpoint: MassiveAdaptiveRLCheckpointAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_ladder: MassiveAdaptiveRLCostLadderV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_fixed_fit: MassiveAdaptiveRLFixedControlFitAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_fixed_selection: (
        MassiveAdaptiveRLFixedControlSelectionAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_fixed_evaluation: MassiveAdaptiveRLFixedControlEvaluationV1 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name
            not in {
                "semantic_receipt_sha256",
                "runtime_economic_witness_replayed",
                "development_fold_validation_authorized",
            }
        }

    @property
    def source_transaction_verified(self) -> bool:
        return self._loaded_source is not None

    @property
    def source_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.receipt.receipt_sha256
        )

    @property
    def source_transaction_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.receipt_sha256
        )

    @property
    def source_transaction_committed_at_ms(self) -> int | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.committed_at_ms
        )

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_economic_witness_replayed
            and self.development_fold_validation_authorized
            and self.source_data_qualified
        )

    @property
    def runtime_cost_ladder(self) -> MassiveAdaptiveRLCostLadderV1:
        self.validate()
        if self._runtime_ladder is None or not self.development_stage_authorized:
            raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                "PPO validation ladder has not been exactly replayed"
            )
        return self._runtime_ladder

    @property
    def runtime_fixed_control_evaluation(
        self,
    ) -> MassiveAdaptiveRLFixedControlEvaluationV1:
        self.validate()
        if (
            self._runtime_fixed_evaluation is None
            or not self.development_stage_authorized
        ):
            raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                "FC06 validation outcome has not been exactly replayed"
            )
        return self._runtime_fixed_evaluation

    @property
    def runtime_checkpoint_authority(self) -> MassiveAdaptiveRLCheckpointAuthorityV1:
        self.validate()
        if self._runtime_checkpoint is None or not self.development_stage_authorized:
            raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                "validation checkpoint has not been exactly replayed"
            )
        return self._runtime_checkpoint

    def validate(self) -> None:
        is_ppo = self.outcome_kind == MASSIVE_ADAPTIVE_RL_PPO_VALIDATION_OUTCOME_V3
        is_fc06 = self.outcome_kind == MASSIVE_ADAPTIVE_RL_FC06_VALIDATION_OUTCOME_V3
        common_runtime = (
            self._runtime_manifest is not None and self._runtime_release is not None
        )
        ppo_runtime = bool(
            common_runtime
            and self._runtime_checkpoint is not None
            and self._runtime_ladder is not None
            and self._runtime_fixed_fit is not None
            and self._runtime_fixed_selection is not None
            and self._runtime_fixed_evaluation is None
        )
        fc_runtime = bool(
            common_runtime
            and self._runtime_checkpoint is None
            and self._runtime_ladder is None
            and self._runtime_fixed_fit is not None
            and self._runtime_fixed_selection is not None
            and self._runtime_fixed_evaluation is not None
        )
        runtime_present = (is_ppo and ppo_runtime) or (is_fc06 and fc_runtime)
        any_runtime = any(
            value is not None
            for value in (
                self._runtime_manifest,
                self._runtime_release,
                self._runtime_checkpoint,
                self._runtime_ladder,
                self._runtime_fixed_fit,
                self._runtime_fixed_selection,
                self._runtime_fixed_evaluation,
            )
        )
        ppo_optional = (
            self.checkpoint_authority_receipt_sha256,
            self.checkpoint_receipt_sha256,
            self.model_state_receipt_sha256,
            self.update_index,
            self.low_cost_trace_receipt_sha256,
            self.high_cost_trace_receipt_sha256,
            self.low_cost_transition_inventory_sha256,
            self.high_cost_transition_inventory_sha256,
            self.low_cost_terminal_liquidation_adjusted_return,
            self.high_cost_terminal_liquidation_adjusted_return,
        )
        metrics = (
            self.primary_incremental_log_wealth,
            self.primary_active_log_wealth,
            self.primary_cost_terminal_liquidation_adjusted_return,
            self.maximum_drawdown,
        )
        stress_metrics = (
            self.low_cost_terminal_liquidation_adjusted_return,
            self.high_cost_terminal_liquidation_adjusted_return,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or isinstance(self.release_committed_at_ms, bool)
            or not isinstance(self.release_committed_at_ms, int)
            or self.release_committed_at_ms < 0
            or self.outcome_kind not in _OUTCOME_KINDS
            or (is_ppo and any(value is None for value in ppo_optional))
            or (is_fc06 and any(value is not None for value in ppo_optional))
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in metrics
            )
            or is_ppo
            and any(
                type(value) is not float or not math.isfinite(value)
                for value in stress_metrics
            )
            or is_ppo
            and (
                isinstance(self.update_index, bool)
                or not isinstance(self.update_index, int)
                or self.update_index < 0
            )
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or not isinstance(self.source_data_qualified, bool)
            or any_runtime != runtime_present
            or self.runtime_economic_witness_replayed != runtime_present
            or self.development_fold_validation_authorized
            != bool(runtime_present and self.source_data_qualified)
            or self.policy_freezing_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                "validation outcome V3 differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256") and value is not None:
                _digest(name, value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.release_authority_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= self.release_committed_at_ms
            ):
                raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                    "validation outcome source transaction differs"
                )
        if runtime_present:
            assert self._runtime_manifest is not None
            assert self._runtime_release is not None
            assert self._runtime_fixed_fit is not None
            assert self._runtime_fixed_selection is not None
            manifest = self._runtime_manifest
            release = self._runtime_release
            for authority in (
                manifest,
                release,
                self._runtime_fixed_fit,
                self._runtime_fixed_selection,
            ):
                authority.validate()
            sources = release.validation_sources(self.fold_index)
            registry = release.validation_registry(self.fold_index)
            expected_candidates = (
                release.expected_candidate_checkpoint_authority_receipt_inventories[
                    release.released_validation_fold_indices.index(self.fold_index)
                ]
            )
            if (
                not release.development_stage_authorized
                or manifest.semantic_receipt_sha256 != self.manifest_v5_receipt_sha256
                or manifest.scientific_protocol_projection_sha256
                != self.scientific_protocol_projection_sha256
                or release.semantic_receipt_sha256
                != self.release_authority_receipt_sha256
                or release.source_receipt_sha256 != self.release_source_receipt_sha256
                or release.source_transaction_receipt_sha256
                != self.release_commit_receipt_sha256
                or release.source_transaction_committed_at_ms
                != self.release_committed_at_ms
                or release.execution_implementation_registration_authority_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or release.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or release.four_fold_fit_authority_receipt_sha256
                != self.four_fold_fit_authority_receipt_sha256
                or sources.semantic_receipt_sha256
                != self.validation_sources_v2_receipt_sha256
                or registry.semantic_receipt_sha256
                != self.validation_registry_v2_receipt_sha256
                or registry.validation_context_receipt_sha256
                != self.validation_context_receipt_sha256
                or self._runtime_fixed_fit.semantic_receipt_sha256
                != self.fixed_control_fit_authority_receipt_sha256
                or self._runtime_fixed_selection.semantic_receipt_sha256
                != self.fixed_control_selection_authority_receipt_sha256
                or self._runtime_fixed_selection.runtime_selection is None
                or self._runtime_fixed_selection.runtime_selection.selected_action_receipt_sha256
                != self.selected_fc06_action_receipt_sha256
            ):
                raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                    "validation outcome runtime lineage differs"
                )
            if is_ppo:
                assert self._runtime_checkpoint is not None
                assert self._runtime_ladder is not None
                checkpoint = self._runtime_checkpoint
                ladder = self._runtime_ladder
                checkpoint.validate()
                ladder.validate()
                runtime_checkpoint = checkpoint.runtime_checkpoint
                if (
                    runtime_checkpoint is None
                    or checkpoint.semantic_receipt_sha256 not in expected_candidates
                    or self.subject_receipt_sha256 != checkpoint.semantic_receipt_sha256
                    or self.checkpoint_authority_receipt_sha256
                    != checkpoint.semantic_receipt_sha256
                    or self.checkpoint_receipt_sha256
                    != runtime_checkpoint.semantic_receipt_sha256
                    or self.model_state_receipt_sha256
                    != runtime_checkpoint.model_state_receipt_sha256
                    or self.update_index != runtime_checkpoint.update_index
                    or self.economic_witness_receipt_sha256
                    != ladder.semantic_receipt_sha256
                    or self.primary_trace_receipt_sha256
                    != ladder.primary.policy_trace.semantic_receipt_sha256
                    or self.low_cost_trace_receipt_sha256
                    != ladder.low_cost_trace.semantic_receipt_sha256
                    or self.high_cost_trace_receipt_sha256
                    != ladder.high_cost_trace.semantic_receipt_sha256
                    or self.decision_target_inventory_sha256
                    != ladder.decision_target_inventory_sha256
                    or self.primary_transition_inventory_sha256
                    != ladder.primary.transition_inventory_sha256
                    or self.low_cost_transition_inventory_sha256
                    != ladder.low_cost_transition_inventory_sha256
                    or self.high_cost_transition_inventory_sha256
                    != ladder.high_cost_transition_inventory_sha256
                    or self.primary_incremental_log_wealth
                    != ladder.primary.policy_trace.cumulative_incremental_rl_log_return
                    or self.primary_active_log_wealth
                    != ladder.primary.policy_trace.cumulative_strategy_active_log_return
                    or self.low_cost_terminal_liquidation_adjusted_return
                    != ladder.low_cost_trace.terminal_liquidation_adjusted_return
                    or self.primary_cost_terminal_liquidation_adjusted_return
                    != ladder.primary.policy_trace.terminal_liquidation_adjusted_return
                    or self.high_cost_terminal_liquidation_adjusted_return
                    != ladder.high_cost_trace.terminal_liquidation_adjusted_return
                    or self.maximum_drawdown
                    != ladder.primary.policy_trace.maximum_drawdown
                    or self.source_data_qualified
                    != bool(
                        release.source_data_qualified
                        and checkpoint.source_data_qualified
                        and ladder.source_data_qualified
                    )
                ):
                    raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                        "PPO validation outcome replay differs"
                    )
            else:
                assert self._runtime_fixed_evaluation is not None
                evaluation = self._runtime_fixed_evaluation
                evaluation.validate()
                if (
                    self.subject_receipt_sha256
                    != self._runtime_fixed_selection.semantic_receipt_sha256
                    or self.economic_witness_receipt_sha256
                    != evaluation.semantic_receipt_sha256
                    or self.primary_trace_receipt_sha256
                    != evaluation.policy_trace.semantic_receipt_sha256
                    or self.decision_target_inventory_sha256
                    != evaluation.policy_trace.decision_target_inventory_sha256
                    or self.primary_transition_inventory_sha256
                    != evaluation.transition_inventory_sha256
                    or self.primary_incremental_log_wealth
                    != evaluation.policy_trace.cumulative_incremental_rl_log_return
                    or self.primary_active_log_wealth
                    != evaluation.policy_trace.cumulative_strategy_active_log_return
                    or self.primary_cost_terminal_liquidation_adjusted_return
                    != evaluation.policy_trace.terminal_liquidation_adjusted_return
                    or self.maximum_drawdown != evaluation.policy_trace.maximum_drawdown
                    or self.source_data_qualified
                    != bool(
                        release.source_data_qualified
                        and self._runtime_fixed_fit.source_data_qualified
                        and evaluation.source_data_qualified
                    )
                ):
                    raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                        "FC06 validation outcome replay differs"
                    )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _fold_roots(
    release: MassiveAdaptiveRLValidationReleaseAuthorityV1, fold_index: int
) -> tuple[
    MassiveAdaptiveRLFoldFitAuthorityV1,
    tuple[MassiveAdaptiveRLCheckpointAuthorityV1, ...],
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
]:
    release.validate()
    if fold_index not in release.released_validation_fold_indices:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "validation fold is not causally released"
        )
    fold_fit = release.four_fold_fit_authority.fold_fit(fold_index)
    workflow = fold_fit.training_workflow.runtime_workflow
    workflow.validate()
    checkpoints = workflow.policy_checkpoint_authorities
    if (
        type(fold_fit) is not MassiveAdaptiveRLFoldFitAuthorityV1
        or any(
            type(row) is not MassiveAdaptiveRLCheckpointAuthorityV1
            for row in checkpoints
        )
        or type(workflow.fixed_control_fit_authority)
        is not MassiveAdaptiveRLFixedControlFitAuthorityV1
        or type(workflow.fixed_control_selection_authority)
        is not MassiveAdaptiveRLFixedControlSelectionAuthorityV1
        or tuple(row.semantic_receipt_sha256 for row in checkpoints)
        != release.expected_candidate_checkpoint_authority_receipt_inventories[
            release.released_validation_fold_indices.index(fold_index)
        ]
    ):
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "released checkpoint population differs from completed fit"
        )
    return (
        fold_fit,
        checkpoints,
        workflow.fixed_control_fit_authority,
        workflow.fixed_control_selection_authority,
    )


def _base_body(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    fold_index: int,
    outcome_kind: str,
    subject_receipt_sha256: str,
) -> tuple[
    dict[str, object],
    MassiveAdaptiveRLFoldFitAuthorityV1,
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
]:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(release) is not MassiveAdaptiveRLValidationReleaseAuthorityV1
    ):
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "validation outcome requires exact V5 roots"
        )
    manifest.validate()
    release.validate()
    fold_fit, _checkpoints, fixed_fit, fixed_selection = _fold_roots(
        release, fold_index
    )
    sources = release.validation_sources(fold_index)
    registry = release.validation_registry(fold_index)
    release_source = _required_digest(
        "validation release source", release.source_receipt_sha256
    )
    release_commit = _required_digest(
        "validation release commit", release.source_transaction_receipt_sha256
    )
    release_time = _required_time(
        "validation release time", release.source_transaction_committed_at_ms
    )
    fixed_selection_runtime = fixed_selection.runtime_selection
    if (
        not release.development_stage_authorized
        or release.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
        or fixed_selection_runtime is None
        or not fixed_selection.runtime_selection_replayed
    ):
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "validation outcome roots are not authorized"
        )
    body: dict[str, object] = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "release_authority_receipt_sha256": release.semantic_receipt_sha256,
        "release_source_receipt_sha256": release_source,
        "release_commit_receipt_sha256": release_commit,
        "release_committed_at_ms": release_time,
        "execution_implementation_registration_receipt_sha256": release.execution_implementation_registration_authority_receipt_sha256,
        "scientific_execution_fingerprint_sha256": release.scientific_execution_fingerprint_sha256,
        "four_fold_fit_authority_receipt_sha256": release.four_fold_fit_authority_receipt_sha256,
        "fold_fit_authority_receipt_sha256": fold_fit.semantic_receipt_sha256,
        "fold_index": fold_index,
        "outcome_kind": outcome_kind,
        "subject_receipt_sha256": subject_receipt_sha256,
        "validation_sources_v2_receipt_sha256": sources.semantic_receipt_sha256,
        "validation_registry_v2_receipt_sha256": registry.semantic_receipt_sha256,
        "validation_context_receipt_sha256": registry.validation_context_receipt_sha256,
        "fixed_control_fit_authority_receipt_sha256": fixed_fit.semantic_receipt_sha256,
        "fixed_control_selection_authority_receipt_sha256": fixed_selection.semantic_receipt_sha256,
        "selected_fc06_action_receipt_sha256": fixed_selection_runtime.selected_action_receipt_sha256,
        "policy_freezing_authorized": False,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SCHEMA,
    }
    return body, fold_fit, fixed_fit, fixed_selection


def build_massive_adaptive_rl_ppo_validation_outcome_authority_v3(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    fold_index: int,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    if type(checkpoint_authority) is not MassiveAdaptiveRLCheckpointAuthorityV1:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "PPO validation outcome requires exact checkpoint authority"
        )
    body, _fold_fit, fixed_fit, fixed_selection = _base_body(
        manifest=manifest,
        release=release,
        fold_index=fold_index,
        outcome_kind=MASSIVE_ADAPTIVE_RL_PPO_VALIDATION_OUTCOME_V3,
        subject_receipt_sha256=checkpoint_authority.semantic_receipt_sha256,
    )
    checkpoint_authority.validate()
    checkpoint = checkpoint_authority.runtime_checkpoint
    expected = release.expected_candidate_checkpoint_authority_receipt_inventories[
        release.released_validation_fold_indices.index(fold_index)
    ]
    if (
        checkpoint is None
        or checkpoint_authority.semantic_receipt_sha256 not in expected
    ):
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "PPO checkpoint is outside the causal release"
        )
    sources = release.validation_sources(fold_index)
    registry = release.validation_registry(fold_index)
    environments = registry.build_environments()
    ladder = evaluate_massive_adaptive_rl_checkpoint_cost_ladder_v1(
        checkpoint_authority=checkpoint_authority,
        chronology_authority=sources.runtime_chronology_authority,
        primary_environment=environments[20.0],
        low_cost_environment=environments[10.0],
        high_cost_environment=environments[40.0],
        fold_index=fold_index,
        evaluation_role="inner_validation",
    )
    primary = ladder.primary.policy_trace
    values = {
        **body,
        "checkpoint_authority_receipt_sha256": checkpoint_authority.semantic_receipt_sha256,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "update_index": checkpoint.update_index,
        "economic_witness_receipt_sha256": ladder.semantic_receipt_sha256,
        "primary_trace_receipt_sha256": primary.semantic_receipt_sha256,
        "low_cost_trace_receipt_sha256": ladder.low_cost_trace.semantic_receipt_sha256,
        "high_cost_trace_receipt_sha256": ladder.high_cost_trace.semantic_receipt_sha256,
        "decision_target_inventory_sha256": ladder.decision_target_inventory_sha256,
        "primary_transition_inventory_sha256": ladder.primary.transition_inventory_sha256,
        "low_cost_transition_inventory_sha256": ladder.low_cost_transition_inventory_sha256,
        "high_cost_transition_inventory_sha256": ladder.high_cost_transition_inventory_sha256,
        "primary_incremental_log_wealth": primary.cumulative_incremental_rl_log_return,
        "primary_active_log_wealth": primary.cumulative_strategy_active_log_return,
        "low_cost_terminal_liquidation_adjusted_return": ladder.low_cost_trace.terminal_liquidation_adjusted_return,
        "primary_cost_terminal_liquidation_adjusted_return": primary.terminal_liquidation_adjusted_return,
        "high_cost_terminal_liquidation_adjusted_return": ladder.high_cost_trace.terminal_liquidation_adjusted_return,
        "maximum_drawdown": primary.maximum_drawdown,
        "source_data_qualified": bool(
            release.source_data_qualified
            and checkpoint_authority.source_data_qualified
            and ladder.source_data_qualified
        ),
    }
    provisional = MassiveAdaptiveRLValidationOutcomeAuthorityV3(
        **values,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_economic_witness_replayed=True,
        development_fold_validation_authorized=bool(values["source_data_qualified"]),
        _runtime_manifest=manifest,
        _runtime_release=release,
        _runtime_checkpoint=checkpoint_authority,
        _runtime_ladder=ladder,
        _runtime_fixed_fit=fixed_fit,
        _runtime_fixed_selection=fixed_selection,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def build_massive_adaptive_rl_fc06_validation_outcome_authority_v3(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    fold_index: int,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    _fold_fit, _checkpoints, fixed_fit, fixed_selection = _fold_roots(
        release, fold_index
    )
    body, _fold_fit_again, fixed_fit_again, fixed_selection_again = _base_body(
        manifest=manifest,
        release=release,
        fold_index=fold_index,
        outcome_kind=MASSIVE_ADAPTIVE_RL_FC06_VALIDATION_OUTCOME_V3,
        subject_receipt_sha256=fixed_selection.semantic_receipt_sha256,
    )
    if fixed_fit is not fixed_fit_again or fixed_selection is not fixed_selection_again:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "FC06 validation roots changed during replay"
        )
    sources = release.validation_sources(fold_index)
    registry = release.validation_registry(fold_index)
    evaluation = evaluate_massive_adaptive_rl_fixed_control_v1(
        registry=build_massive_adaptive_rl_fixed_control_registry_v1(),
        fit_authority=fixed_fit,
        selection_authority=fixed_selection,
        chronology_authority=sources.runtime_chronology_authority,
        environment=registry.build_environments()[20.0],
    )
    primary = evaluation.policy_trace
    values = {
        **body,
        "checkpoint_authority_receipt_sha256": None,
        "checkpoint_receipt_sha256": None,
        "model_state_receipt_sha256": None,
        "update_index": None,
        "economic_witness_receipt_sha256": evaluation.semantic_receipt_sha256,
        "primary_trace_receipt_sha256": primary.semantic_receipt_sha256,
        "low_cost_trace_receipt_sha256": None,
        "high_cost_trace_receipt_sha256": None,
        "decision_target_inventory_sha256": primary.decision_target_inventory_sha256,
        "primary_transition_inventory_sha256": evaluation.transition_inventory_sha256,
        "low_cost_transition_inventory_sha256": None,
        "high_cost_transition_inventory_sha256": None,
        "primary_incremental_log_wealth": primary.cumulative_incremental_rl_log_return,
        "primary_active_log_wealth": primary.cumulative_strategy_active_log_return,
        "low_cost_terminal_liquidation_adjusted_return": None,
        "primary_cost_terminal_liquidation_adjusted_return": primary.terminal_liquidation_adjusted_return,
        "high_cost_terminal_liquidation_adjusted_return": None,
        "maximum_drawdown": primary.maximum_drawdown,
        "source_data_qualified": bool(
            release.source_data_qualified
            and fixed_fit.source_data_qualified
            and evaluation.source_data_qualified
        ),
    }
    provisional = MassiveAdaptiveRLValidationOutcomeAuthorityV3(
        **values,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_economic_witness_replayed=True,
        development_fold_validation_authorized=bool(values["source_data_qualified"]),
        _runtime_manifest=manifest,
        _runtime_release=release,
        _runtime_fixed_fit=fixed_fit,
        _runtime_fixed_selection=fixed_selection,
        _runtime_fixed_evaluation=evaluation,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "validation outcome source is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    result = MassiveAdaptiveRLValidationOutcomeAuthorityV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_validation_outcome_authority_v3(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    fold_index: int,
    outcome_kind: str,
    subject_receipt_sha256: str,
    verified_at_ms: int | None = None,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    relative = validation_outcome_authority_relative_path_v3(
        manifest=manifest,
        fold_index=fold_index,
        outcome_kind=outcome_kind,
        subject_receipt_sha256=subject_receipt_sha256,
    )
    return _parse(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=(
                time.time_ns() // 1_000_000
                if verified_at_ms is None
                else verified_at_ms
            ),
        ),
    )


def _persist(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    outcome: MassiveAdaptiveRLValidationOutcomeAuthorityV3,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    outcome.validate()
    relative = validation_outcome_authority_relative_path_v3(
        manifest=manifest,
        fold_index=outcome.fold_index,
        outcome_kind=outcome.outcome_kind,
        subject_receipt_sha256=outcome.subject_receipt_sha256,
    )
    committed_at_ms = _wall_clock_after(outcome.release_committed_at_ms)
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(outcome.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=outcome.release_authority_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            "ADAPTIVE-RL-V5-VALIDATION-OUTCOME-"
            f"{outcome.fold_index}-{outcome.outcome_kind}-"
            f"{outcome.subject_receipt_sha256}"
        ),
    )
    return load_massive_adaptive_rl_validation_outcome_authority_v3(
        root=root,
        manifest=manifest,
        fold_index=outcome.fold_index,
        outcome_kind=outcome.outcome_kind,
        subject_receipt_sha256=outcome.subject_receipt_sha256,
        verified_at_ms=committed_at_ms,
    )


def _authorize(
    *,
    persisted: MassiveAdaptiveRLValidationOutcomeAuthorityV3,
    expected: MassiveAdaptiveRLValidationOutcomeAuthorityV3,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    if persisted.semantic_unsigned() != expected.semantic_unsigned():
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "validation outcome does not replay from the released economics"
        )
    result = replace(
        persisted,
        runtime_economic_witness_replayed=True,
        development_fold_validation_authorized=expected.source_data_qualified,
        _runtime_manifest=expected._runtime_manifest,
        _runtime_release=expected._runtime_release,
        _runtime_checkpoint=expected._runtime_checkpoint,
        _runtime_ladder=expected._runtime_ladder,
        _runtime_fixed_fit=expected._runtime_fixed_fit,
        _runtime_fixed_selection=expected._runtime_fixed_selection,
        _runtime_fixed_evaluation=expected._runtime_fixed_evaluation,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_rl_ppo_validation_outcome_v3(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    fold_index: int,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "PPO validation outcome materialization mode differs"
        )
    expected = build_massive_adaptive_rl_ppo_validation_outcome_authority_v3(
        manifest=manifest,
        release=release,
        fold_index=fold_index,
        checkpoint_authority=checkpoint_authority,
    )
    relative = validation_outcome_authority_relative_path_v3(
        manifest=manifest,
        fold_index=fold_index,
        outcome_kind=expected.outcome_kind,
        subject_receipt_sha256=expected.subject_receipt_sha256,
    )
    complete, partial = _transaction_state(root=root, relative=relative)
    if partial:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "validation outcome transaction is incomplete"
        )
    if not complete:
        if not allow_materialize:
            raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                "validation outcome is absent in read-only mode"
            )
        _persist(root=root, manifest=manifest, outcome=expected)
    persisted = load_massive_adaptive_rl_validation_outcome_authority_v3(
        root=root,
        manifest=manifest,
        fold_index=fold_index,
        outcome_kind=expected.outcome_kind,
        subject_receipt_sha256=expected.subject_receipt_sha256,
    )
    return _authorize(persisted=persisted, expected=expected)


def run_or_resume_massive_adaptive_rl_fc06_validation_outcome_v3(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    fold_index: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV3:
    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "FC06 validation outcome materialization mode differs"
        )
    expected = build_massive_adaptive_rl_fc06_validation_outcome_authority_v3(
        manifest=manifest,
        release=release,
        fold_index=fold_index,
    )
    relative = validation_outcome_authority_relative_path_v3(
        manifest=manifest,
        fold_index=fold_index,
        outcome_kind=expected.outcome_kind,
        subject_receipt_sha256=expected.subject_receipt_sha256,
    )
    complete, partial = _transaction_state(root=root, relative=relative)
    if partial:
        raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
            "validation outcome transaction is incomplete"
        )
    if not complete:
        if not allow_materialize:
            raise MassiveAdaptiveRLValidationOutcomeAuthorityV3Error(
                "validation outcome is absent in read-only mode"
            )
        _persist(root=root, manifest=manifest, outcome=expected)
    persisted = load_massive_adaptive_rl_validation_outcome_authority_v3(
        root=root,
        manifest=manifest,
        fold_index=fold_index,
        outcome_kind=expected.outcome_kind,
        subject_receipt_sha256=expected.subject_receipt_sha256,
    )
    return _authorize(persisted=persisted, expected=expected)


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FC06_VALIDATION_OUTCOME_V3",
    "MASSIVE_ADAPTIVE_RL_PPO_VALIDATION_OUTCOME_V3",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_DATASET",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V3_SOURCE_SHA256",
    "MassiveAdaptiveRLValidationOutcomeAuthorityV3",
    "MassiveAdaptiveRLValidationOutcomeAuthorityV3Error",
    "build_massive_adaptive_rl_fc06_validation_outcome_authority_v3",
    "build_massive_adaptive_rl_ppo_validation_outcome_authority_v3",
    "load_massive_adaptive_rl_validation_outcome_authority_v3",
    "run_or_resume_massive_adaptive_rl_fc06_validation_outcome_v3",
    "run_or_resume_massive_adaptive_rl_ppo_validation_outcome_v3",
    "validation_outcome_authority_relative_path_v3",
]
