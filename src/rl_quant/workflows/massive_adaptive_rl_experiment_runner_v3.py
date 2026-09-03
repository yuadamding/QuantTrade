"""Manifest-V4 root runner through complete four-fold inner validation.

The immutable Manifest-V3 runner remains the training-generation witness and
ends at its registered validation-backend handoff.  This new generation adopts
that exact completed fit, promotes the validation-complete RuntimeSources V2,
and invokes the package-owned four-fold validation runner.  It deliberately
stops before policy freezing and outer access.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import os
from pathlib import Path
import stat
import time
from typing import Iterator

from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_executor_v1 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256,
    run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_four_fold_policy_selection_v1 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_V1_SPEC_SHA256,
    MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
    MassiveAdaptiveRLFourFoldSelectionDispositionV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SPEC_SHA256,
    MassiveAdaptiveRLEndToEndRunV2,
    _run_massive_adaptive_rl_experiment_v2_unlocked,
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
    MassiveAdaptiveRLExperimentManifestV4,
    load_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
    reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v2,
)


MASSIVE_ADAPTIVE_RL_VALIDATION_RUN_V3_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-run-v3"
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SPEC_SHA256 = semantic_sha256(
    {
        "manifest": "v4-validation-selection-preregistered",
        "training_predecessor_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SPEC_SHA256
        ),
        "training_predecessor_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SOURCE_SHA256
        ),
        "training_handoff": "exact-four-fold-fit-from-manifest-v3-runner",
        "runtime_sources": "cold-replayed-validation-complete-v2",
        "four_fold_validation_executor_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256
        ),
        "four_fold_validation_executor_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256
        ),
        "four_fold_selection_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_V1_SPEC_SHA256
        ),
        "qualified_handoff": "selection-v3-aware-walk-forward-policy-freeze-required",
        "ineligible_terminal": "no-qualified-policy",
        "invalid_evidence": "raise-without-outer-access",
        "verification": "read-only-cold-replay",
        "final_policy_freezing": False,
        "outer_access": False,
        "profitability_reporting": False,
        "lockbox_access": False,
    }
)


class MassiveAdaptiveRLExperimentRunnerV3Error(ValueError):
    """The Manifest-V4 validation root cannot advance or replay safely."""


class MassiveAdaptiveRLExperimentRunnerV3LeaseUnavailable(RuntimeError):
    """Another process owns the Manifest-V4 root invocation."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationRunV3:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    training_state_receipts: tuple[str, ...]
    four_fold_fit_authority_receipt_sha256: str
    runtime_sources_v2_receipt_sha256: str
    four_fold_policy_selection_authority_receipt_sha256: str
    four_fold_policy_selection_source_receipt_sha256: str
    four_fold_policy_selection_commit_receipt_sha256: str
    selection_disposition: str
    training_evidence_adopted: bool
    source_generation_v2_replayed: bool
    four_fold_validation_selection_replayed: bool
    validation_execution_complete: bool
    next_required_stage: str | None
    semantic_receipt_sha256: str
    final_policy_freezing_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    end_to_end_profitability_execution_complete: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_RUN_V3_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    @property
    def no_qualified_policy(self) -> bool:
        return (
            self.selection_disposition
            == MassiveAdaptiveRLFourFoldSelectionDispositionV1.NO_QUALIFIED_POLICY.value
        )

    @property
    def positive_profitability_authorization_eligible(self) -> bool:
        return bool(
            self.validation_execution_complete
            and not self.no_qualified_policy
            and self.selection_disposition
            == MassiveAdaptiveRLFourFoldSelectionDispositionV1.FOUR_FOLD_SELECTIONS_QUALIFIED.value
        )

    def validate(self) -> None:
        qualified = (
            self.selection_disposition
            == MassiveAdaptiveRLFourFoldSelectionDispositionV1.FOUR_FOLD_SELECTIONS_QUALIFIED.value
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_VALIDATION_RUN_V3_SCHEMA
            or not self.experiment_id
            or not self.training_state_receipts
            or len(set(self.training_state_receipts))
            != len(self.training_state_receipts)
            or self.selection_disposition
            not in {
                row.value for row in MassiveAdaptiveRLFourFoldSelectionDispositionV1
            }
            or not self.training_evidence_adopted
            or not self.source_generation_v2_replayed
            or not self.four_fold_validation_selection_replayed
            or not self.validation_execution_complete
            or self.next_required_stage
            != ("selection-v3-aware-walk-forward-policy-freeze" if qualified else None)
            or self.final_policy_freezing_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.end_to_end_profitability_execution_complete
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentRunnerV3Error(
                "adaptive RL validation run V3 differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        for receipt in self.training_state_receipts:
            _digest("training state", receipt)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExperimentRunnerV3Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _wall_clock_ms() -> int:
    value = time.time_ns() // 1_000_000
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLExperimentRunnerV3Error(
            "adaptive RL validation-root clock differs"
        )
    return value


@contextmanager
def _experiment_v3_orchestration_lease(
    *, artifact_root: str | Path, experiment_id: str
) -> Iterator[None]:
    directory = (
        Path(artifact_root) / "adaptive-rl" / experiment_id / "orchestration-lease-v1"
    )
    descriptor = -1
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise MassiveAdaptiveRLExperimentRunnerV3Error(
                "adaptive RL V3 orchestration lease directory is a symlink"
            )
        descriptor = os.open(
            directory / "orchestration.lock",
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLExperimentRunnerV3Error(
                "adaptive RL V3 orchestration lease identity differs"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MassiveAdaptiveRLExperimentRunnerV3LeaseUnavailable(
                "adaptive RL V3 execution is already owned"
            ) from error
    except (MassiveAdaptiveRLExperimentRunnerV3Error, OSError):
        if descriptor >= 0:
            os.close(descriptor)
        raise
    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


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
        raise MassiveAdaptiveRLExperimentRunnerV3Error(
            "adaptive RL validation root training ledger differs"
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
        raise MassiveAdaptiveRLExperimentRunnerV3Error(
            "adaptive RL validation root has no exact training handoff"
        )
    return matches[0].stage_artifact_receipt_sha256


def _build_result(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    states: tuple[MassiveAdaptiveRLExperimentStateV2, ...],
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    selection_authority: MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
) -> MassiveAdaptiveRLValidationRunV3:
    fit_receipt = _validate_training_handoff(manifest=manifest, states=states)
    runtime_sources_v2.validate()
    selection_authority.validate()
    source_receipt = selection_authority.source_receipt_sha256
    commit_receipt = selection_authority.source_transaction_receipt_sha256
    if (
        source_receipt is None
        or commit_receipt is None
        or not selection_authority.development_stage_authorized
        or selection_authority.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or selection_authority.training_manifest_v3_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or selection_authority.runtime_sources_v2_receipt_sha256
        != runtime_sources_v2.semantic_receipt_sha256
        or selection_authority.four_fold_fit_authority_receipt_sha256 != fit_receipt
    ):
        raise MassiveAdaptiveRLExperimentRunnerV3Error(
            "adaptive RL validation root aggregate differs"
        )
    disposition = selection_authority.selection_disposition
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_RUN_V3_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "training_state_receipts": tuple(
            state.semantic_receipt_sha256 for state in states
        ),
        "four_fold_fit_authority_receipt_sha256": fit_receipt,
        "runtime_sources_v2_receipt_sha256": (
            runtime_sources_v2.semantic_receipt_sha256
        ),
        "four_fold_policy_selection_authority_receipt_sha256": (
            selection_authority.semantic_receipt_sha256
        ),
        "four_fold_policy_selection_source_receipt_sha256": source_receipt,
        "four_fold_policy_selection_commit_receipt_sha256": commit_receipt,
        "selection_disposition": disposition,
        "training_evidence_adopted": True,
        "source_generation_v2_replayed": True,
        "four_fold_validation_selection_replayed": True,
        "validation_execution_complete": True,
        "next_required_stage": (
            "selection-v3-aware-walk-forward-policy-freeze"
            if disposition
            == MassiveAdaptiveRLFourFoldSelectionDispositionV1.FOUR_FOLD_SELECTIONS_QUALIFIED.value
            else None
        ),
        "final_policy_freezing_authorized": False,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "end_to_end_profitability_execution_complete": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLValidationRunV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _replay_validation_root(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
    states: tuple[MassiveAdaptiveRLExperimentStateV2, ...],
    allow_materialize: bool,
) -> MassiveAdaptiveRLValidationRunV3:
    fit_receipt = _validate_training_handoff(manifest=manifest, states=states)
    now = _wall_clock_ms()
    runtime_sources_v2 = (
        reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v2(
            source_root=source_root,
            manifest=manifest.base_manifest,
            committed_at_ms=now,
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
        raise MassiveAdaptiveRLExperimentRunnerV3Error(
            "adaptive RL validation root fit adoption differs"
        )
    aggregate = run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1(
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
        selection_authority=aggregate,
    )


def run_massive_adaptive_rl_experiment_v3(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
    resume: bool = True,
) -> MassiveAdaptiveRLValidationRunV3 | MassiveAdaptiveRLEndToEndRunV2:
    """Execute training through four-fold validation without outer access."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v4(manifest_path)
    if str(device) != manifest.execution_device_specification:
        raise MassiveAdaptiveRLExperimentRunnerV3Error(
            "requested device differs from the Manifest-V4 training device"
        )
    with _experiment_v3_orchestration_lease(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    ):
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
        return _replay_validation_root(
            manifest=manifest,
            source_root=source_root,
            artifact_root=artifact_root,
            device=device,
            states=states,
            allow_materialize=True,
        )


def verify_massive_adaptive_rl_experiment_v3(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
) -> MassiveAdaptiveRLValidationRunV3:
    """Cold-replay a completed V4 validation root without creating artifacts."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v4(manifest_path)
    if str(device) != manifest.execution_device_specification:
        raise MassiveAdaptiveRLExperimentRunnerV3Error(
            "requested device differs from the Manifest-V4 training device"
        )
    states = load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    return _replay_validation_root(
        manifest=manifest,
        source_root=source_root,
        artifact_root=artifact_root,
        device=device,
        states=states,
        allow_materialize=False,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V3_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_RUN_V3_SCHEMA",
    "MassiveAdaptiveRLExperimentRunnerV3Error",
    "MassiveAdaptiveRLExperimentRunnerV3LeaseUnavailable",
    "MassiveAdaptiveRLValidationRunV3",
    "run_massive_adaptive_rl_experiment_v3",
    "verify_massive_adaptive_rl_experiment_v3",
]
