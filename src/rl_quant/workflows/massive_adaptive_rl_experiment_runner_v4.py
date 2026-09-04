"""Manifest-V4 root through the causally available validation-input boundary.

The registered split is prequential: validation folds 2 and 3 are outer
folds 0 and 1, respectively.  This runner therefore supersedes the legacy
all-four validation root.  It adopts the completed training generation and
commits only validation inputs for folds 0 and 1.  It does not execute a
validation outcome, freeze a policy, or open an outer fold.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Iterator

from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_INPUTS_V1_SOURCE_SHA256,
    MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    run_or_resume_massive_adaptive_rl_initial_validation_inputs_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SPEC_SHA256,
    MassiveAdaptiveRLEndToEndRunV2,
    _run_massive_adaptive_rl_experiment_v2_unlocked,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256,
    MassiveAdaptiveRLExperimentLockV1Error,
    MassiveAdaptiveRLExperimentLockV1Unavailable,
    massive_adaptive_rl_experiment_orchestration_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MassiveAdaptiveRLExperimentStageV2,
    MassiveAdaptiveRLExperimentStateV2,
    load_massive_adaptive_rl_experiment_states_v2,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    load_massive_adaptive_rl_four_fold_fit_authority_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1,
    MassiveAdaptiveRLExperimentManifestV4,
    load_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
    reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
    reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v2,
)


MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V4_SCHEMA = (
    "rl-quant.massive-adaptive-rl-prequential-run-v4"
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SPEC_SHA256 = semantic_sha256(
    {
        "manifest": "v4-diagnostic-continuation-preserved",
        "training_predecessor_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SPEC_SHA256
        ),
        "training_predecessor_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SOURCE_SHA256
        ),
        "training_handoff": "exact-four-fold-fit-from-manifest-v3-runner",
        "runtime_sources": "cold-replayed-validation-complete-v2",
        "initial_validation_inputs_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256
        ),
        "initial_validation_inputs_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_INPUTS_V1_SOURCE_SHA256
        ),
        "initial_validation_folds": (0, 1),
        "withheld_validation_folds": (2, 3),
        "release_edges": ("outer-0-sealed->validation-2", "outer-1-sealed->validation-3"),
        "no_eligible_candidate_policy": (
            MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
        ),
        "diagnostic_continuation": True,
        "legacy_all_four_validation": "rejected-by-initial-input-boundary",
        "future_protocol_ownership": (
            "manifest-v5-registration-disables-legacy-materialization"
        ),
        "experiment_global_lock_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256
        ),
        "experiment_global_lock_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SOURCE_SHA256
        ),
        "next_stage": "prequential-fold-0-and-fold-1-validation-selection-and-freeze",
        "verification": "read-only-cold-replay",
        "validation_outcomes": False,
        "policy_freezing": False,
        "outer_access": False,
        "profitability_reporting": False,
        "lockbox_access": False,
    }
)


class MassiveAdaptiveRLExperimentRunnerV4Error(ValueError):
    """The prequential Manifest-V4 root cannot advance or replay safely."""


class MassiveAdaptiveRLExperimentRunnerV4LeaseUnavailable(RuntimeError):
    """Another process owns the prequential Manifest-V4 root invocation."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExperimentRunnerV4Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _wall_clock_ms() -> int:
    value = time.time_ns() // 1_000_000
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLExperimentRunnerV4Error(
            "adaptive RL prequential-root clock differs"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPrequentialRunV4:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    training_state_receipts: tuple[str, ...]
    four_fold_fit_authority_receipt_sha256: str
    runtime_sources_v2_receipt_sha256: str
    prequential_validation_plan_receipt_sha256: str
    initial_validation_inputs_authority_receipt_sha256: str
    initial_validation_inputs_source_receipt_sha256: str
    initial_validation_inputs_commit_receipt_sha256: str
    released_validation_fold_indices: tuple[int, ...]
    withheld_validation_fold_indices: tuple[int, ...]
    training_evidence_adopted: bool
    source_generation_v2_replayed: bool
    initial_validation_inputs_replayed: bool
    diagnostic_continuation_registered: bool
    validation_execution_complete: bool
    next_required_stage: str
    semantic_receipt_sha256: str
    policy_schedule_disposition: str | None = None
    final_policy_freezing_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    end_to_end_profitability_execution_complete: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V4_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    @property
    def positive_profitability_authorization_eligible(self) -> bool:
        return False

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V4_SCHEMA
            or not self.experiment_id
            or not self.training_state_receipts
            or len(set(self.training_state_receipts)) != len(self.training_state_receipts)
            or self.released_validation_fold_indices != (0, 1)
            or self.withheld_validation_fold_indices != (2, 3)
            or not self.training_evidence_adopted
            or not self.source_generation_v2_replayed
            or not self.initial_validation_inputs_replayed
            or not self.diagnostic_continuation_registered
            or self.validation_execution_complete
            or self.next_required_stage
            != "prequential-fold-0-and-fold-1-validation-selection-and-freeze"
            or self.policy_schedule_disposition is not None
            or self.final_policy_freezing_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.end_to_end_profitability_execution_complete
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentRunnerV4Error(
                "adaptive RL prequential run V4 differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        for receipt in self.training_state_receipts:
            _digest("training state", receipt)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@contextmanager
def _experiment_v4_orchestration_lease(
    *, artifact_root: str | Path, experiment_id: str
) -> Iterator[None]:
    lease = massive_adaptive_rl_experiment_orchestration_lock_v1(
        artifact_root=artifact_root,
        experiment_id=experiment_id,
    )
    try:
        lease.__enter__()
    except MassiveAdaptiveRLExperimentLockV1Unavailable as error:
        raise MassiveAdaptiveRLExperimentRunnerV4LeaseUnavailable(
            "adaptive RL V4 execution is already owned"
        ) from error
    except (MassiveAdaptiveRLExperimentLockV1Error, OSError) as error:
        raise MassiveAdaptiveRLExperimentRunnerV4Error(
            "adaptive RL V4 orchestration lease is invalid"
        ) from error
    try:
        yield
    finally:
        lease.__exit__(None, None, None)


def _validate_training_handoff(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    states: tuple[MassiveAdaptiveRLExperimentStateV2, ...],
) -> str:
    if not states or any(
        state.experiment_id != manifest.experiment_id
        or state.manifest_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        for state in states
    ):
        raise MassiveAdaptiveRLExperimentRunnerV4Error(
            "adaptive RL prequential root training ledger differs"
        )
    matches = tuple(
        state
        for state in states
        if state.stage
        is MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED
    )
    if (
        len(matches) != 1
        or matches[0].stage_artifact_receipt_sha256 is None
        or not matches[0].source_data_qualified
        or not any(
            state.stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
            and state.blocker_code == "inner-validation-backend-required"
            for state in states
        )
    ):
        raise MassiveAdaptiveRLExperimentRunnerV4Error(
            "adaptive RL prequential root has no exact training handoff"
        )
    return matches[0].stage_artifact_receipt_sha256


def _build_result(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    states: tuple[MassiveAdaptiveRLExperimentStateV2, ...],
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
) -> MassiveAdaptiveRLPrequentialRunV4:
    fit_receipt = _validate_training_handoff(manifest=manifest, states=states)
    runtime_sources_v2.validate()
    initial_inputs.validate()
    source_receipt = initial_inputs.source_receipt_sha256
    commit_receipt = initial_inputs.source_transaction_receipt_sha256
    plan = initial_inputs.prequential_validation_plan
    if (
        source_receipt is None
        or commit_receipt is None
        or not initial_inputs.development_stage_authorized
        or initial_inputs.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or initial_inputs.training_manifest_v3_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or initial_inputs.runtime_sources_v2_receipt_sha256
        != runtime_sources_v2.semantic_receipt_sha256
        or initial_inputs.four_fold_fit_authority_receipt_sha256 != fit_receipt
    ):
        raise MassiveAdaptiveRLExperimentRunnerV4Error(
            "adaptive RL prequential root initial-input authority differs"
        )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V4_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "training_state_receipts": tuple(
            state.semantic_receipt_sha256 for state in states
        ),
        "four_fold_fit_authority_receipt_sha256": fit_receipt,
        "runtime_sources_v2_receipt_sha256": runtime_sources_v2.semantic_receipt_sha256,
        "prequential_validation_plan_receipt_sha256": plan.semantic_receipt_sha256,
        "initial_validation_inputs_authority_receipt_sha256": (
            initial_inputs.semantic_receipt_sha256
        ),
        "initial_validation_inputs_source_receipt_sha256": source_receipt,
        "initial_validation_inputs_commit_receipt_sha256": commit_receipt,
        "released_validation_fold_indices": (0, 1),
        "withheld_validation_fold_indices": (2, 3),
        "training_evidence_adopted": True,
        "source_generation_v2_replayed": True,
        "initial_validation_inputs_replayed": True,
        "diagnostic_continuation_registered": True,
        "validation_execution_complete": False,
        "next_required_stage": (
            "prequential-fold-0-and-fold-1-validation-selection-and-freeze"
        ),
        "policy_schedule_disposition": None,
        "final_policy_freezing_authorized": False,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "end_to_end_profitability_execution_complete": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLPrequentialRunV4(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _replay_prequential_root(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
    states: tuple[MassiveAdaptiveRLExperimentStateV2, ...],
    allow_materialize: bool,
) -> MassiveAdaptiveRLPrequentialRunV4:
    fit_receipt = _validate_training_handoff(manifest=manifest, states=states)
    runtime_sources_v2 = (
        reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v2(
            source_root=source_root,
            manifest=manifest.base_manifest,
            committed_at_ms=_wall_clock_ms(),
            allow_materialize=allow_materialize,
        )
    )
    four_fold_fit = load_massive_adaptive_rl_four_fold_fit_authority_v1(
        root=artifact_root,
        manifest=manifest.base_manifest,
        runtime_sources=runtime_sources_v2.base_runtime_sources_v1,
        verified_at_ms=_wall_clock_ms(),
        device=str(device),
    )
    if four_fold_fit.semantic_receipt_sha256 != fit_receipt:
        raise MassiveAdaptiveRLExperimentRunnerV4Error(
            "adaptive RL prequential root fit adoption differs"
        )
    initial_inputs = run_or_resume_massive_adaptive_rl_initial_validation_inputs_v1(
        root=artifact_root,
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit,
        allow_materialize=allow_materialize,
    )
    return _build_result(
        manifest=manifest,
        states=states,
        runtime_sources_v2=runtime_sources_v2,
        initial_inputs=initial_inputs,
    )


def run_massive_adaptive_rl_experiment_v4(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
    resume: bool = True,
) -> MassiveAdaptiveRLPrequentialRunV4 | MassiveAdaptiveRLEndToEndRunV2:
    """Execute training through the initial prequential input commitment."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v4(manifest_path)
    if str(device) != manifest.execution_device_specification:
        raise MassiveAdaptiveRLExperimentRunnerV4Error(
            "requested device differs from the Manifest-V4 training device"
        )
    with _experiment_v4_orchestration_lease(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    ):
        try:
            reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration(
                root=artifact_root,
                experiment_id=manifest.experiment_id,
            )
        except MassiveAdaptiveRLLegacyWriterRejectedByManifestV5 as error:
            raise MassiveAdaptiveRLExperimentRunnerV4Error(
                "Manifest V5 owns this experiment; the V4 writer is disabled"
            ) from error
        training = _run_massive_adaptive_rl_experiment_v2_unlocked(
            manifest=manifest.base_manifest,
            source_root=source_root,
            artifact_root=artifact_root,
            device=device,
            resume=resume,
        )
        if training.four_fold_fit_authority_receipt_sha256 is None:
            return training
        states = load_massive_adaptive_rl_experiment_states_v2(
            artifact_root=artifact_root,
            experiment_id=manifest.experiment_id,
        )
        return _replay_prequential_root(
            manifest=manifest,
            source_root=source_root,
            artifact_root=artifact_root,
            device=device,
            states=states,
            allow_materialize=True,
        )


def verify_massive_adaptive_rl_experiment_v4(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
) -> MassiveAdaptiveRLPrequentialRunV4:
    """Cold-replay the initial prequential boundary without creating artifacts."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v4(manifest_path)
    if str(device) != manifest.execution_device_specification:
        raise MassiveAdaptiveRLExperimentRunnerV4Error(
            "requested device differs from the Manifest-V4 training device"
        )
    states = load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    return _replay_prequential_root(
        manifest=manifest,
        source_root=source_root,
        artifact_root=artifact_root,
        device=device,
        states=states,
        allow_materialize=False,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V4_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RUN_V4_SCHEMA",
    "MassiveAdaptiveRLExperimentRunnerV4Error",
    "MassiveAdaptiveRLExperimentRunnerV4LeaseUnavailable",
    "MassiveAdaptiveRLPrequentialRunV4",
    "run_massive_adaptive_rl_experiment_v4",
    "verify_massive_adaptive_rl_experiment_v4",
]
