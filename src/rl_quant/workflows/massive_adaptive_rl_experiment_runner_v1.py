"""Fail-closed root runner for persisted adaptive RL experiments.

The current runner owns registration and composite-source byte replay.  It
intentionally stops before fitting until the persisted typed-runtime loader is
available; a self-hashed JSON source index alone cannot authorize training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v1 import (
    MassiveAdaptiveRLExperimentStageV1,
    MassiveAdaptiveRLExperimentStateV1,
    advance_massive_adaptive_rl_experiment_state_v1,
    fail_massive_adaptive_rl_experiment_state_v1,
    load_massive_adaptive_rl_experiment_states_v1,
    register_massive_adaptive_rl_experiment_state_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    MassiveAdaptiveRLExperimentLockV1Error,
    MassiveAdaptiveRLExperimentLockV1Unavailable,
    massive_adaptive_rl_experiment_orchestration_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MassiveAdaptiveRLSourceBundleV1,
    MassiveAdaptiveRLSourceBundleV1Error,
    load_massive_adaptive_rl_source_bundle_v1,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLExperimentManifestV2,
    load_massive_adaptive_rl_experiment_manifest_v2,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
    reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration,
)


MASSIVE_ADAPTIVE_RL_END_TO_END_RUN_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-end-to-end-run-v1"
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V1_SPEC_SHA256 = semantic_sha256(
    {
        "manifest": "v2-session-derived",
        "source_bundle": "v1-byte-and-runtime-replay-separated",
        "state": "create-only-resumable-ledger-v1",
        "current_runtime_boundary": "typed-persisted-composite-loader-required",
        "caller_actions_or_economics": False,
        "profitability_reporting": False,
        "live_trading": False,
    }
)


class MassiveAdaptiveRLExperimentRunnerV1Error(ValueError):
    """The adaptive RL root run differs or cannot advance safely."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLEndToEndRunV1:
    experiment_id: str
    manifest_receipt_sha256: str
    source_bundle_receipt_sha256: str | None
    state_receipts: tuple[str, ...]
    current_stage: MassiveAdaptiveRLExperimentStageV1
    next_required_stage: MassiveAdaptiveRLExperimentStageV1 | None
    blocker_code: str | None
    execution_complete: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_profitability_reporting_authorized: bool = False
    live_trading_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_END_TO_END_RUN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        failed = self.current_stage is MassiveAdaptiveRLExperimentStageV1.FAILED
        expected_complete = (
            self.current_stage
            is MassiveAdaptiveRLExperimentStageV1.DEVELOPMENT_REPORT_PUBLISHED
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_END_TO_END_RUN_V1_SCHEMA
            or not self.experiment_id
            or not self.state_receipts
            or self.execution_complete != expected_complete
            or expected_complete
            and self.next_required_stage is not None
            or expected_complete
            and self.blocker_code is not None
            or failed
            and self.blocker_code is None
            or failed
            and self.next_required_stage is not None
            or not expected_complete
            and not failed
            and self.next_required_stage is None
            or self.source_data_qualified
            and self.source_bundle_receipt_sha256 is None
            or self.development_profitability_reporting_authorized
            or self.live_trading_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentRunnerV1Error(
                "adaptive RL end-to-end run identity or authorization differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _result(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
    states: tuple[MassiveAdaptiveRLExperimentStateV1, ...],
    source_bundle: MassiveAdaptiveRLSourceBundleV1 | None,
    next_required_stage: MassiveAdaptiveRLExperimentStageV1 | None,
    blocker_code: str | None,
) -> MassiveAdaptiveRLEndToEndRunV1:
    current = states[-1]
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_END_TO_END_RUN_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_receipt_sha256": manifest.semantic_receipt_sha256,
        "source_bundle_receipt_sha256": (
            None if source_bundle is None else source_bundle.semantic_receipt_sha256
        ),
        "state_receipts": tuple(row.semantic_receipt_sha256 for row in states),
        "current_stage": current.stage,
        "next_required_stage": next_required_stage,
        "blocker_code": blocker_code,
        "execution_complete": (
            current.stage
            is MassiveAdaptiveRLExperimentStageV1.DEVELOPMENT_REPORT_PUBLISHED
        ),
        "source_data_qualified": bool(
            source_bundle is not None and source_bundle.source_data_qualified
        ),
        "development_profitability_reporting_authorized": False,
        "live_trading_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V1_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLEndToEndRunV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _load_or_register(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
    artifact_root: str | Path,
    resume: bool,
) -> tuple[MassiveAdaptiveRLExperimentStateV1, ...]:
    states = load_massive_adaptive_rl_experiment_states_v1(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    if states:
        if not resume:
            raise MassiveAdaptiveRLExperimentRunnerV1Error(
                "adaptive RL experiment already exists; use resume"
            )
        if states[-1].manifest_receipt_sha256 != manifest.semantic_receipt_sha256:
            raise MassiveAdaptiveRLExperimentRunnerV1Error(
                "adaptive RL resume manifest differs"
            )
        return states
    registered = register_massive_adaptive_rl_experiment_state_v1(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
        manifest_receipt_sha256=manifest.semantic_receipt_sha256,
    )
    return (registered,)


def _run_massive_adaptive_rl_experiment_v1_unlocked(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
    resume: bool = True,
) -> MassiveAdaptiveRLEndToEndRunV1:
    """Start or resume the package-owned experiment without widening authority.

    `device` is accepted and bound by the future training stages.  The current
    implementation reaches persisted source replay, then returns an explicit
    typed-runtime-loader blocker rather than fabricating Dataset V3 authority.
    """

    del device
    manifest = load_massive_adaptive_rl_experiment_manifest_v2(manifest_path)
    states = _load_or_register(
        manifest=manifest,
        artifact_root=artifact_root,
        resume=resume,
    )
    if states[-1].stage is MassiveAdaptiveRLExperimentStageV1.FAILED:
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=None,
            next_required_stage=None,
            blocker_code=states[-1].failure_code,
        )
    try:
        source_bundle = load_massive_adaptive_rl_source_bundle_v1(
            source_root=source_root,
            manifest=manifest,
        )
    except MassiveAdaptiveRLSourceBundleV1Error:
        if states[-1].stage is not MassiveAdaptiveRLExperimentStageV1.FAILED:
            failure = fail_massive_adaptive_rl_experiment_state_v1(
                artifact_root=artifact_root,
                previous=states[-1],
                failed_stage=(
                    MassiveAdaptiveRLExperimentStageV1.SOURCE_BUNDLE_REPLAYED
                ),
                failure_code="source-bundle-replay-failed",
                failure_evidence_receipt_sha256=semantic_sha256(
                    {
                        "manifest": manifest.semantic_receipt_sha256,
                        "failure_code": "source-bundle-replay-failed",
                    }
                ),
            )
            states = (*states, failure)
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=None,
            next_required_stage=None,
            blocker_code="source-bundle-replay-failed",
        )
    if states[-1].stage is MassiveAdaptiveRLExperimentStageV1.REGISTERED:
        replayed = advance_massive_adaptive_rl_experiment_state_v1(
            artifact_root=artifact_root,
            previous=states[-1],
            stage=MassiveAdaptiveRLExperimentStageV1.SOURCE_BUNDLE_REPLAYED,
            stage_artifact_receipt_sha256=source_bundle.semantic_receipt_sha256,
        )
        states = (*states, replayed)
    if not source_bundle.source_data_qualified:
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=source_bundle,
            next_required_stage=(
                MassiveAdaptiveRLExperimentStageV1.FIT_FORECASTS_AUTHORIZED
            ),
            blocker_code="typed-runtime-source-replay-required",
        )
    raise MassiveAdaptiveRLExperimentRunnerV1Error(
        "typed source replay exists but the four-fold execution backend is not installed"
    )


def run_massive_adaptive_rl_experiment_v1(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
    resume: bool = True,
) -> MassiveAdaptiveRLEndToEndRunV1:
    """Run V1 only while no complete or partial V5 registration exists."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v2(manifest_path)
    try:
        with massive_adaptive_rl_experiment_orchestration_lock_v1(
            artifact_root=artifact_root,
            experiment_id=manifest.experiment_id,
        ):
            try:
                reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration(
                    root=artifact_root,
                    experiment_id=manifest.experiment_id,
                )
            except MassiveAdaptiveRLLegacyWriterRejectedByManifestV5 as error:
                raise MassiveAdaptiveRLExperimentRunnerV1Error(
                    "Manifest V5 owns this experiment; the V1 writer is disabled"
                ) from error
            return _run_massive_adaptive_rl_experiment_v1_unlocked(
                manifest_path=manifest_path,
                source_root=source_root,
                artifact_root=artifact_root,
                device=device,
                resume=resume,
            )
    except MassiveAdaptiveRLExperimentLockV1Unavailable as error:
        raise MassiveAdaptiveRLExperimentRunnerV1Error(
            "adaptive RL V1 execution is already owned"
        ) from error
    except MassiveAdaptiveRLExperimentLockV1Error as error:
        raise MassiveAdaptiveRLExperimentRunnerV1Error(
            "adaptive RL V1 experiment-global lock is invalid"
        ) from error


def verify_massive_adaptive_rl_experiment_v1(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
) -> MassiveAdaptiveRLEndToEndRunV1:
    """Reopen the current ledger and source bytes without creating state."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v2(manifest_path)
    states = load_massive_adaptive_rl_experiment_states_v1(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    if not states:
        raise MassiveAdaptiveRLExperimentRunnerV1Error(
            "adaptive RL experiment has no persisted state"
        )
    if states[-1].stage is MassiveAdaptiveRLExperimentStageV1.FAILED:
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=None,
            next_required_stage=None,
            blocker_code=states[-1].failure_code,
        )
    source_bundle = load_massive_adaptive_rl_source_bundle_v1(
        source_root=source_root,
        manifest=manifest,
    )
    return _result(
        manifest=manifest,
        states=states,
        source_bundle=source_bundle,
        next_required_stage=(
            MassiveAdaptiveRLExperimentStageV1.FIT_FORECASTS_AUTHORIZED
        ),
        blocker_code="typed-runtime-source-replay-required",
    )


__all__ = [
    "MassiveAdaptiveRLEndToEndRunV1",
    "MassiveAdaptiveRLExperimentRunnerV1Error",
    "run_massive_adaptive_rl_experiment_v1",
    "verify_massive_adaptive_rl_experiment_v1",
]
