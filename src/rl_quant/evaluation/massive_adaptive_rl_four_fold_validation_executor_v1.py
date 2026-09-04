"""Run all four attested validation folds and publish one selection barrier."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import time
from typing import Iterator

from rl_quant.evaluation.massive_adaptive_rl_fold_validation_executor_v2 import (
    run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.training.massive_adaptive_rl_four_fold_policy_selection_v1 import (
    MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
    run_or_resume_massive_adaptive_rl_four_fold_policy_selection_authority_v1,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MassiveAdaptiveRLExperimentManifestV4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    legacy_manifest_v5_rejecting_writer_guard_v1,
)


MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-manifest-v4-runtime-sources-v2-and-four-fold-fit",
        "validation_inputs": "package-owned-four-fold-v2-barrier",
        "fold_execution": "exact-fold-order-zero-through-three",
        "output": "four-fold-policy-selection-authority-v1-only",
        "resume": "strict-read-or-create-through-child-authorities",
        "caller_fold": False,
        "caller_validation_inputs": False,
        "caller_environment": False,
        "caller_actions": False,
        "caller_targets": False,
        "caller_metrics": False,
        "caller_selection": False,
        "caller_timestamp": False,
        "final_policy_freezing": False,
        "outer_access": False,
        "profitability_reporting": False,
        "lockbox_access": False,
    }
)


class MassiveAdaptiveRLFourFoldValidationExecutorV1Error(ValueError):
    """The package-owned four-fold validation execution cannot proceed."""


class MassiveAdaptiveRLFourFoldValidationExecutionLeaseUnavailable(
    MassiveAdaptiveRLFourFoldValidationExecutorV1Error
):
    """Another process owns the four-fold validation execution."""


def _wall_clock_ms() -> int:
    value = time.time_ns() // 1_000_000
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
            "four-fold validation publication clock differs"
        )
    return value


@contextmanager
def _four_fold_validation_execution_lease_v1(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> Iterator[None]:
    manifest.validate()
    root_path = Path(root)
    directory = root_path / "massive-adaptive" / "rl-four-fold-validation-execution-v1"
    descriptor = -1
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        if root_path.is_symlink():
            raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
                "four-fold validation root is a symlink"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
                "four-fold validation lease directory is a symlink"
            )
        descriptor = os.open(
            directory / f"v4-{manifest.semantic_receipt_sha256}.lock",
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
                "four-fold validation lease identity differs"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MassiveAdaptiveRLFourFoldValidationExecutionLeaseUnavailable(
                "four-fold validation execution is already owned"
            ) from error
    except (
        MassiveAdaptiveRLFourFoldValidationExecutorV1Error,
        OSError,
    ):
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


def _validate_roots(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
) -> None:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(runtime_sources_v2) is not MassiveAdaptiveRLRuntimeSourcesV2
        or type(four_fold_fit_authority) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
    ):
        raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
            "four-fold validation requires exact root generations"
        )
    manifest.validate()
    runtime_sources_v2.validate()
    four_fold_fit_authority.validate()
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    if (
        manifest.experiment_id != runtime_sources_v2.experiment_id
        or manifest.experiment_id != four_fold_fit_authority.experiment_id
        or manifest.base_manifest.semantic_receipt_sha256
        != four_fold_fit_authority.manifest_v3_receipt_sha256
        or not runtime_sources_v2.source_data_qualified
        or not four_fold_fit_authority.development_stage_authorized
    ):
        raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
            "four-fold validation roots differ or are not authorized"
        )


@legacy_manifest_v5_rejecting_writer_guard_v1(
    materialize_parameter="allow_materialize"
)
def run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1:
    """Execute or exactly replay all four folds as one package-owned stage.

    The function returns only the aggregate selection barrier.  Fold-local
    outcomes and selections are discovered and generated internally, and no
    downstream policy freeze or outer access is authorized here.
    """

    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
            "four-fold validation materialization mode differs"
        )
    _validate_roots(
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    with _four_fold_validation_execution_lease_v1(root=root, manifest=manifest):
        inputs: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2 = (
            run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v2(
                root=root,
                manifest=manifest,
                four_fold_fit_authority=four_fold_fit_authority,
                runtime_sources_v2=runtime_sources_v2,
                committed_at_ms=_wall_clock_ms(),
                allow_materialize=allow_materialize,
            )
        )
        if not inputs.development_stage_authorized:
            raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
                "four-fold validation inputs did not authorize execution"
            )
        executions = tuple(
            run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2(
                root=root,
                manifest=manifest,
                runtime_sources_v2=runtime_sources_v2,
                four_fold_fit_authority=four_fold_fit_authority,
                four_fold_validation_inputs_v2=inputs,
                fold_index=fold_index,
                allow_materialize=allow_materialize,
            )
            for fold_index in range(4)
        )
        if any(not row.development_stage_authorized for row in executions):
            raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
                "four-fold validation did not complete every fold"
            )
        latest_child_commit = max(
            row.source_transaction_committed_at_ms or -1 for row in executions
        )
        if latest_child_commit < 0:
            raise MassiveAdaptiveRLFourFoldValidationExecutorV1Error(
                "four-fold validation execution transaction is absent"
            )
        return (
            run_or_resume_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
                root=root,
                manifest=manifest,
                runtime_sources_v2=runtime_sources_v2,
                four_fold_fit_authority=four_fold_fit_authority,
                four_fold_validation_inputs_v2=inputs,
                fold_executions=executions,
                committed_at_ms=max(_wall_clock_ms(), latest_child_commit + 1),
                allow_materialize=allow_materialize,
            )
        )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFourFoldValidationExecutionLeaseUnavailable",
    "MassiveAdaptiveRLFourFoldValidationExecutorV1Error",
    "run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1",
]
