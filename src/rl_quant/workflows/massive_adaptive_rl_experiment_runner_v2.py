"""Manifest-V3 and state-V2 root runner for persisted adaptive RL experiments.

This generation binds the requested execution device, distinguishes retryable
source or replay-dependency unavailability from integrity failure, and can
represent positive or negative completed reports.  When the package-owned
replay-dependency index exists it reconstructs and authorizes the typed runtime
graph without caller objects.  It now executes or replays all four fit folds
before stopping at the not-yet-installed inner-validation backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2,
    MassiveAdaptiveRLExperimentStageV2,
    MassiveAdaptiveRLExperimentStateV2,
    advance_massive_adaptive_rl_experiment_state_v2,
    block_massive_adaptive_rl_experiment_state_v2,
    fail_massive_adaptive_rl_experiment_state_v2,
    load_massive_adaptive_rl_experiment_states_v2,
    register_massive_adaptive_rl_experiment_state_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
    load_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitExecutionLeaseUnavailable,
    advance_massive_adaptive_rl_four_fold_fit_inputs_state_v1,
    advance_massive_adaptive_rl_four_fold_fit_state_v1,
    run_or_resume_massive_adaptive_rl_four_fold_fit_inputs_v1,
    run_or_resume_massive_adaptive_rl_four_fold_fit_v1,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_v1 import (
    MassiveAdaptiveRLFoldFitExecutionLeaseUnavailable,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
    load_massive_adaptive_rl_runtime_source_graph_authority_v1,
    runtime_source_graph_authority_path_v1,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
    MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable,
    reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1,
    replay_dependency_index_path_v1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MassiveAdaptiveRLSourceBundleV1,
    MassiveAdaptiveRLSourceBundleV1Error,
    load_massive_adaptive_rl_source_bundle_v1,
)


MASSIVE_ADAPTIVE_RL_END_TO_END_RUN_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-end-to-end-run-v2"
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SPEC_SHA256 = semantic_sha256(
    {
        "manifest": "v3-final-profitability-preregistered",
        "source_bundle": "v1-byte-and-runtime-replay-separated",
        "runtime_source_graph": "v1-persisted-generic-reload-nonauthorizing",
        "state": "create-only-blocked-failed-and-report-ledger-v2",
        "device": "manifest-bound",
        "runtime_reconstruction": "package-owned-dependency-index-v1",
        "runtime_reconstruction_unavailability": "retryable-blocker",
        "four_fold_fit": "package-owned-input-first-run-or-resume-v1",
        "current_runtime_boundary": "inner-validation-backend-required",
        "completed_resume": "terminal-idempotent",
        "state_verification": "entire-chain-manifest-bound",
        "verification_surface": "ledger-replay-distinct-from-deep-verification",
        "source_disappearance": "block-current-next-stage-without-regression",
        "valid_negative_result": "execution-complete-report-not-authorized",
        "valid_positive_result": "execution-complete-report-authorized",
        "caller_actions_or_economics": False,
        "live_trading": False,
    }
)


class MassiveAdaptiveRLExperimentRunnerV2Error(ValueError):
    """The adaptive RL V2 root run differs or cannot advance safely."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLEndToEndRunV2:
    experiment_id: str
    manifest_receipt_sha256: str
    execution_device_specification: str
    source_bundle_receipt_sha256: str | None
    runtime_source_graph_authority_receipt_sha256: str | None
    four_fold_fit_inputs_authority_receipt_sha256: str | None
    four_fold_fit_authority_receipt_sha256: str | None
    state_receipts: tuple[str, ...]
    current_stage: MassiveAdaptiveRLExperimentStageV2
    next_required_stage: MassiveAdaptiveRLExperimentStageV2 | None
    blocker_code: str | None
    execution_complete: bool
    source_data_qualified: bool
    ledger_replayed: bool
    completion_authority_replayed: bool
    report_replayed: bool
    outer_evidence_replayed: bool
    runtime_source_graph_replayed: bool
    full_verification_complete: bool
    profitability_report_authority_receipt_sha256: str | None
    profitability_report_receipt_sha256: str | None
    failed_gate_names: tuple[str, ...]
    development_profitability_reporting_authorized: bool
    semantic_receipt_sha256: str
    live_trading_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_END_TO_END_RUN_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        blocked = self.current_stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
        failed = self.current_stage is MassiveAdaptiveRLExperimentStageV2.FAILED
        published = (
            self.current_stage
            is MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED
        )
        expected_full_verification = bool(
            self.ledger_replayed
            and self.completion_authority_replayed
            and self.report_replayed
            and self.outer_evidence_replayed
            and self.runtime_source_graph_replayed
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_END_TO_END_RUN_V2_SCHEMA
            or not self.experiment_id
            or not self.execution_device_specification
            or not self.state_receipts
            or self.execution_complete != published
            or (published or failed) != (self.next_required_stage is None)
            or published
            and self.blocker_code is not None
            or (blocked or failed) != (self.blocker_code is not None)
            or failed
            and self.blocker_code is None
            or self.source_data_qualified
            and (
                self.source_bundle_receipt_sha256 is None
                or self.runtime_source_graph_authority_receipt_sha256 is None
            )
            or self.runtime_source_graph_replayed
            and self.runtime_source_graph_authority_receipt_sha256 is None
            or self.four_fold_fit_authority_receipt_sha256 is not None
            and self.four_fold_fit_inputs_authority_receipt_sha256 is None
            or published
            and not self.source_data_qualified
            or published
            and self.runtime_source_graph_authority_receipt_sha256 is None
            or not self.ledger_replayed
            or self.full_verification_complete != expected_full_verification
            or self.full_verification_complete
            and (not self.execution_complete or not self.source_data_qualified)
            or published
            != (
                self.profitability_report_authority_receipt_sha256 is not None
                and self.profitability_report_receipt_sha256 is not None
            )
            or not published
            and (
                self.failed_gate_names
                or self.development_profitability_reporting_authorized
            )
            or published
            and self.development_profitability_reporting_authorized
            == bool(self.failed_gate_names)
            or self.live_trading_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentRunnerV2Error(
                "adaptive RL V2 end-to-end run identity or authorization differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _result(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    states: tuple[MassiveAdaptiveRLExperimentStateV2, ...],
    source_bundle: MassiveAdaptiveRLSourceBundleV1 | None,
    runtime_source_graph: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1 | None = None,
) -> MassiveAdaptiveRLEndToEndRunV2:
    if any(
        state.experiment_id != manifest.experiment_id
        or state.manifest_receipt_sha256 != manifest.semantic_receipt_sha256
        for state in states
    ):
        raise MassiveAdaptiveRLExperimentRunnerV2Error(
            "adaptive RL V2 state chain belongs to another manifest"
        )
    current = states[-1]
    if current.stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED:
        next_stage = current.blocked_stage
        blocker = current.blocker_code
    elif current.stage is MassiveAdaptiveRLExperimentStageV2.FAILED:
        next_stage = None
        blocker = current.failure_code
    elif current.execution_complete:
        next_stage = None
        blocker = None
    else:
        next_stage = MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[
            current.completed_stage_index + 1
        ]
        blocker = None
    source_bundle_receipt = (
        source_bundle.semantic_receipt_sha256
        if source_bundle is not None
        else current.source_bundle_receipt_sha256
    )
    source_data_qualified = (
        bool(source_bundle is not None and source_bundle.source_data_qualified)
        or bool(
            runtime_source_graph is not None
            and runtime_source_graph.source_data_qualified
        )
        or current.source_data_qualified
    )
    runtime_source_graph_receipt = (
        (
            runtime_source_graph.runtime_authority_receipt_sha256
            or runtime_source_graph.semantic_receipt_sha256
        )
        if runtime_source_graph is not None
        else current.runtime_source_graph_authority_receipt_sha256
    )
    stage_artifact_receipts = {
        state.stage: state.stage_artifact_receipt_sha256
        for state in states
        if state.stage
        in {
            MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED,
            MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED,
        }
    }
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_END_TO_END_RUN_V2_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_receipt_sha256": manifest.semantic_receipt_sha256,
        "execution_device_specification": manifest.execution_device_specification,
        "source_bundle_receipt_sha256": source_bundle_receipt,
        "runtime_source_graph_authority_receipt_sha256": (runtime_source_graph_receipt),
        "four_fold_fit_inputs_authority_receipt_sha256": (
            stage_artifact_receipts.get(
                MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED
            )
        ),
        "four_fold_fit_authority_receipt_sha256": (
            stage_artifact_receipts.get(
                MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED
            )
        ),
        "state_receipts": tuple(row.semantic_receipt_sha256 for row in states),
        "current_stage": current.stage,
        "next_required_stage": next_stage,
        "blocker_code": blocker,
        "execution_complete": current.execution_complete,
        "source_data_qualified": source_data_qualified,
        "ledger_replayed": True,
        "completion_authority_replayed": False,
        "report_replayed": False,
        "outer_evidence_replayed": False,
        "runtime_source_graph_replayed": bool(
            runtime_source_graph is not None
            and runtime_source_graph.runtime_graph_replayed
            and runtime_source_graph.source_data_qualified
        ),
        "full_verification_complete": False,
        "profitability_report_authority_receipt_sha256": (
            current.profitability_report_authority_receipt_sha256
        ),
        "profitability_report_receipt_sha256": (
            current.profitability_report_receipt_sha256
        ),
        "failed_gate_names": current.failed_gate_names,
        "development_profitability_reporting_authorized": (
            current.development_profitability_reporting_authorized
        ),
        "live_trading_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V2_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLEndToEndRunV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _load_or_register(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    artifact_root: str | Path,
    resume: bool,
) -> tuple[MassiveAdaptiveRLExperimentStateV2, ...]:
    states = load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    if states:
        if not resume:
            raise MassiveAdaptiveRLExperimentRunnerV2Error(
                "adaptive RL V2 experiment already exists; use resume"
            )
        if states[-1].manifest_receipt_sha256 != manifest.semantic_receipt_sha256:
            raise MassiveAdaptiveRLExperimentRunnerV2Error(
                "adaptive RL V2 resume manifest differs"
            )
        return states
    registered = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
        manifest_receipt_sha256=manifest.semantic_receipt_sha256,
    )
    return (registered,)


def _source_bundle_path(
    *, source_root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV3
) -> Path:
    return (
        Path(source_root)
        / "adaptive-rl"
        / "source-bundle-v1"
        / f"{manifest.experiment_id}.json"
    )


def _next_required_stage(
    state: MassiveAdaptiveRLExperimentStateV2,
) -> MassiveAdaptiveRLExperimentStageV2:
    next_index = state.completed_stage_index + 1
    if next_index >= len(MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2):
        raise MassiveAdaptiveRLExperimentRunnerV2Error(
            "adaptive RL V2 experiment has no next executable stage"
        )
    return MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[next_index]


def run_massive_adaptive_rl_experiment_v2(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    artifact_root: str | Path,
    device: object,
    resume: bool = True,
) -> MassiveAdaptiveRLEndToEndRunV2:
    """Replay sources, execute four fit folds, and stop before validation."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v3(manifest_path)
    if str(device) != manifest.execution_device_specification:
        raise MassiveAdaptiveRLExperimentRunnerV2Error(
            "requested device differs from preregistered execution device"
        )
    states = _load_or_register(
        manifest=manifest,
        artifact_root=artifact_root,
        resume=resume,
    )
    if states[-1].stage is MassiveAdaptiveRLExperimentStageV2.FAILED:
        return _result(manifest=manifest, states=states, source_bundle=None)
    if (
        states[-1].stage
        is MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED
    ):
        return _result(manifest=manifest, states=states, source_bundle=None)
    bundle_path = _source_bundle_path(source_root=source_root, manifest=manifest)
    if not bundle_path.is_file():
        source_replay_completed = states[-1].completed_stage_index >= (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2.index(
                MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED
            )
        )
        blocked_stage = (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[
                states[-1].completed_stage_index + 1
            ]
            if source_replay_completed
            else MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED
        )
        blocker_code = (
            "previously-replayed-source-temporarily-unavailable"
            if source_replay_completed
            else "source-bundle-temporarily-absent"
        )
        if not (
            states[-1].stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
            and states[-1].blocker_code == blocker_code
        ):
            blocked = block_massive_adaptive_rl_experiment_state_v2(
                artifact_root=artifact_root,
                previous=states[-1],
                blocked_stage=blocked_stage,
                blocker_code=blocker_code,
                blocker_evidence_receipt_sha256=semantic_sha256(
                    {
                        "manifest": manifest.semantic_receipt_sha256,
                        "source_bundle_path": bundle_path.as_posix(),
                        "completed_stage_index": states[-1].completed_stage_index,
                    }
                ),
            )
            states = (*states, blocked)
        return _result(manifest=manifest, states=states, source_bundle=None)
    try:
        source_bundle = load_massive_adaptive_rl_source_bundle_v1(
            source_root=source_root,
            manifest=manifest.base_manifest,
        )
    except MassiveAdaptiveRLSourceBundleV1Error:
        failure = fail_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=states[-1],
            failed_stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
            failure_code="source-bundle-integrity-failed",
            failure_evidence_receipt_sha256=semantic_sha256(
                {
                    "manifest": manifest.semantic_receipt_sha256,
                    "failure_code": "source-bundle-integrity-failed",
                }
            ),
        )
        states = (*states, failure)
        return _result(manifest=manifest, states=states, source_bundle=None)
    if states[-1].completed_stage_index == 0:
        replayed = advance_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=states[-1],
            stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
            stage_artifact_receipt_sha256=source_bundle.semantic_receipt_sha256,
        )
        states = (*states, replayed)
    runtime_graph_path = runtime_source_graph_authority_path_v1(
        source_root=source_root,
        experiment_id=manifest.experiment_id,
    )
    if not runtime_graph_path.is_file():
        if not (
            states[-1].stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
            and states[-1].blocker_code == "typed-runtime-source-replay-required"
        ):
            blocked = block_massive_adaptive_rl_experiment_state_v2(
                artifact_root=artifact_root,
                previous=states[-1],
                blocked_stage=_next_required_stage(states[-1]),
                blocker_code="typed-runtime-source-replay-required",
                blocker_evidence_receipt_sha256=semantic_sha256(
                    {
                        "manifest": manifest.semantic_receipt_sha256,
                        "source_bundle": source_bundle.semantic_receipt_sha256,
                    }
                ),
            )
            states = (*states, blocked)
        return _result(manifest=manifest, states=states, source_bundle=source_bundle)
    try:
        runtime_source_graph = (
            load_massive_adaptive_rl_runtime_source_graph_authority_v1(
                source_root=source_root,
                manifest=manifest,
                source_bundle=source_bundle,
            )
        )
    except MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error:
        failure = fail_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=states[-1],
            failed_stage=MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED,
            failure_code="runtime-source-graph-integrity-failed",
            failure_evidence_receipt_sha256=semantic_sha256(
                {
                    "manifest": manifest.semantic_receipt_sha256,
                    "failure_code": "runtime-source-graph-integrity-failed",
                }
            ),
        )
        states = (*states, failure)
        return _result(manifest=manifest, states=states, source_bundle=source_bundle)
    dependency_index_path = replay_dependency_index_path_v1(
        source_root=source_root,
        experiment_id=manifest.experiment_id,
    )
    runtime_sources = None
    if dependency_index_path.is_file():
        try:
            runtime_sources = (
                reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1(
                    source_root=source_root,
                    manifest=manifest,
                )
            )
        except MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable:
            blocker_code = "runtime-source-temporarily-unavailable"
            if not (
                states[-1].stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
                and states[-1].blocker_code == blocker_code
            ):
                blocked = block_massive_adaptive_rl_experiment_state_v2(
                    artifact_root=artifact_root,
                    previous=states[-1],
                    blocked_stage=_next_required_stage(states[-1]),
                    blocker_code=blocker_code,
                    blocker_evidence_receipt_sha256=semantic_sha256(
                        {
                            "manifest": manifest.semantic_receipt_sha256,
                            "source_bundle": source_bundle.semantic_receipt_sha256,
                            "runtime_source_graph": (
                                runtime_source_graph.semantic_receipt_sha256
                            ),
                            "failure_class": "runtime-source-temporarily-unavailable",
                        }
                    ),
                )
                states = (*states, blocked)
            return _result(
                manifest=manifest,
                states=states,
                source_bundle=source_bundle,
                runtime_source_graph=runtime_source_graph,
            )
        except MassiveAdaptiveRLRuntimeSourceReconstructionV1Error:
            failure = fail_massive_adaptive_rl_experiment_state_v2(
                artifact_root=artifact_root,
                previous=states[-1],
                failed_stage=MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED,
                failure_code="runtime-source-reconstruction-failed",
                failure_evidence_receipt_sha256=semantic_sha256(
                    {
                        "manifest": manifest.semantic_receipt_sha256,
                        "failure_code": "runtime-source-reconstruction-failed",
                    }
                ),
            )
            states = (*states, failure)
            return _result(
                manifest=manifest,
                states=states,
                source_bundle=source_bundle,
                runtime_source_graph=runtime_source_graph,
            )
        runtime_source_graph = runtime_sources.runtime_source_graph_authority
    if runtime_sources is None or not runtime_source_graph.source_data_qualified:
        if not (
            states[-1].stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
            and states[-1].blocker_code
            == "runtime-source-replay-dependency-index-required"
        ):
            blocked = block_massive_adaptive_rl_experiment_state_v2(
                artifact_root=artifact_root,
                previous=states[-1],
                blocked_stage=_next_required_stage(states[-1]),
                blocker_code="runtime-source-replay-dependency-index-required",
                blocker_evidence_receipt_sha256=semantic_sha256(
                    {
                        "manifest": manifest.semantic_receipt_sha256,
                        "source_bundle": source_bundle.semantic_receipt_sha256,
                        "runtime_source_graph": (
                            runtime_source_graph.semantic_receipt_sha256
                        ),
                    }
                ),
            )
            states = (*states, blocked)
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=source_bundle,
            runtime_source_graph=runtime_source_graph,
        )
    fit_inputs_stage_index = MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2.index(
        MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED
    )
    committed_at_ms = time.time_ns() // 1_000_000
    try:
        four_fold_inputs = (
            run_or_resume_massive_adaptive_rl_four_fold_fit_inputs_v1(
                manifest=manifest,
                runtime_sources=runtime_sources,
                artifact_root=artifact_root,
                committed_at_ms=committed_at_ms,
                device=str(device),
                allow_materialize=(
                    states[-1].completed_stage_index < fit_inputs_stage_index
                ),
            )
        )
    except MassiveAdaptiveRLFourFoldFitExecutionLeaseUnavailable:
        blocker_code = "four-fold-fit-input-execution-lease-unavailable"
        if not (
            states[-1].stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
            and states[-1].blocker_code == blocker_code
        ):
            blocked = block_massive_adaptive_rl_experiment_state_v2(
                artifact_root=artifact_root,
                previous=states[-1],
                blocked_stage=_next_required_stage(states[-1]),
                blocker_code=blocker_code,
                blocker_evidence_receipt_sha256=semantic_sha256(
                    {
                        "manifest": manifest.semantic_receipt_sha256,
                        "runtime_sources": runtime_sources.semantic_receipt_sha256,
                    }
                ),
            )
            states = (*states, blocked)
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=source_bundle,
            runtime_source_graph=runtime_source_graph,
        )
    except (ValueError, RuntimeError):
        failure = fail_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=states[-1],
            failed_stage=MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED,
            failure_code="four-fold-fit-input-authorization-failed",
            failure_evidence_receipt_sha256=semantic_sha256(
                {
                    "manifest": manifest.semantic_receipt_sha256,
                    "runtime_sources": runtime_sources.semantic_receipt_sha256,
                    "failure_code": "four-fold-fit-input-authorization-failed",
                }
            ),
        )
        states = (*states, failure)
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=source_bundle,
            runtime_source_graph=runtime_source_graph,
        )
    if states[-1].completed_stage_index < fit_inputs_stage_index:
        advanced = advance_massive_adaptive_rl_four_fold_fit_inputs_state_v1(
            artifact_root=artifact_root,
            previous=states[-1],
            authority=four_fold_inputs,
        )
        states = (*states, advanced)
    elif not any(
        state.stage is MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED
        and state.stage_artifact_receipt_sha256
        == four_fold_inputs.semantic_receipt_sha256
        for state in states
    ):
        failure = fail_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=states[-1],
            failed_stage=MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED,
            failure_code="four-fold-fit-input-state-binding-failed",
            failure_evidence_receipt_sha256=semantic_sha256(
                {
                    "manifest": manifest.semantic_receipt_sha256,
                    "four_fold_fit_inputs": four_fold_inputs.semantic_receipt_sha256,
                }
            ),
        )
        states = (*states, failure)
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=source_bundle,
            runtime_source_graph=runtime_source_graph,
        )
    fit_stage_index = MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2.index(
        MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED
    )
    try:
        four_fold_fit = run_or_resume_massive_adaptive_rl_four_fold_fit_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            fit_inputs_authority=four_fold_inputs,
            artifact_root=artifact_root,
            committed_at_ms=committed_at_ms + 10_000,
            device=str(device),
            allow_materialize=(states[-1].completed_stage_index < fit_stage_index),
        )
    except (
        MassiveAdaptiveRLFourFoldFitExecutionLeaseUnavailable,
        MassiveAdaptiveRLFoldFitExecutionLeaseUnavailable,
    ):
        blocker_code = "four-fold-fit-execution-lease-unavailable"
        if not (
            states[-1].stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
            and states[-1].blocker_code == blocker_code
        ):
            blocked = block_massive_adaptive_rl_experiment_state_v2(
                artifact_root=artifact_root,
                previous=states[-1],
                blocked_stage=_next_required_stage(states[-1]),
                blocker_code=blocker_code,
                blocker_evidence_receipt_sha256=semantic_sha256(
                    {
                        "manifest": manifest.semantic_receipt_sha256,
                        "four_fold_fit_inputs": (
                            four_fold_inputs.semantic_receipt_sha256
                        ),
                    }
                ),
            )
            states = (*states, blocked)
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=source_bundle,
            runtime_source_graph=runtime_source_graph,
        )
    except (ValueError, RuntimeError):
        failure = fail_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=states[-1],
            failed_stage=(
                MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED
            ),
            failure_code="four-fold-fit-execution-failed",
            failure_evidence_receipt_sha256=semantic_sha256(
                {
                    "manifest": manifest.semantic_receipt_sha256,
                    "four_fold_fit_inputs": four_fold_inputs.semantic_receipt_sha256,
                    "failure_code": "four-fold-fit-execution-failed",
                }
            ),
        )
        states = (*states, failure)
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=source_bundle,
            runtime_source_graph=runtime_source_graph,
        )
    if states[-1].completed_stage_index < fit_stage_index:
        advanced = advance_massive_adaptive_rl_four_fold_fit_state_v1(
            artifact_root=artifact_root,
            previous=states[-1],
            authority=four_fold_fit,
        )
        states = (*states, advanced)
    elif not any(
        state.stage
        is MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED
        and state.stage_artifact_receipt_sha256
        == four_fold_fit.semantic_receipt_sha256
        for state in states
    ):
        failure = fail_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=states[-1],
            failed_stage=(
                MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED
            ),
            failure_code="four-fold-fit-state-binding-failed",
            failure_evidence_receipt_sha256=semantic_sha256(
                {
                    "manifest": manifest.semantic_receipt_sha256,
                    "four_fold_fit": four_fold_fit.semantic_receipt_sha256,
                }
            ),
        )
        states = (*states, failure)
        return _result(
            manifest=manifest,
            states=states,
            source_bundle=source_bundle,
            runtime_source_graph=runtime_source_graph,
        )
    if states[-1].completed_stage_index == fit_stage_index and not (
        states[-1].stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
        and states[-1].blocker_code == "inner-validation-backend-required"
    ):
        blocked = block_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=states[-1],
            blocked_stage=MassiveAdaptiveRLExperimentStageV2.INNER_VALIDATION_COMPLETED,
            blocker_code="inner-validation-backend-required",
            blocker_evidence_receipt_sha256=semantic_sha256(
                {
                    "manifest": manifest.semantic_receipt_sha256,
                    "four_fold_fit_inputs": four_fold_inputs.semantic_receipt_sha256,
                    "four_fold_fit": four_fold_fit.semantic_receipt_sha256,
                }
            ),
        )
        states = (*states, blocked)
    return _result(
        manifest=manifest,
        states=states,
        source_bundle=source_bundle,
        runtime_source_graph=runtime_source_graph,
    )


def verify_massive_adaptive_rl_experiment_v2(
    *, manifest_path: str | Path, source_root: str | Path, artifact_root: str | Path
) -> MassiveAdaptiveRLEndToEndRunV2:
    """Replay the current V2 ledger and source bytes without advancing state."""

    manifest = load_massive_adaptive_rl_experiment_manifest_v3(manifest_path)
    states = load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    if not states:
        raise MassiveAdaptiveRLExperimentRunnerV2Error(
            "adaptive RL V2 experiment has no persisted state"
        )
    if any(
        state.experiment_id != manifest.experiment_id
        or state.manifest_receipt_sha256 != manifest.semantic_receipt_sha256
        for state in states
    ):
        raise MassiveAdaptiveRLExperimentRunnerV2Error(
            "adaptive RL V2 verification manifest differs"
        )
    source_bundle = None
    runtime_source_graph = None
    if _source_bundle_path(source_root=source_root, manifest=manifest).is_file():
        source_bundle = load_massive_adaptive_rl_source_bundle_v1(
            source_root=source_root,
            manifest=manifest.base_manifest,
        )
        if runtime_source_graph_authority_path_v1(
            source_root=source_root,
            experiment_id=manifest.experiment_id,
        ).is_file():
            dependency_index = replay_dependency_index_path_v1(
                source_root=source_root,
                experiment_id=manifest.experiment_id,
            )
            if dependency_index.is_file():
                runtime_source_graph = (
                    reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1(
                        source_root=source_root,
                        manifest=manifest,
                    ).runtime_source_graph_authority
                )
            else:
                runtime_source_graph = (
                    load_massive_adaptive_rl_runtime_source_graph_authority_v1(
                        source_root=source_root,
                        manifest=manifest,
                        source_bundle=source_bundle,
                    )
                )
    return _result(
        manifest=manifest,
        states=states,
        source_bundle=source_bundle,
        runtime_source_graph=runtime_source_graph,
    )


def verify_massive_adaptive_rl_experiment_ledger_v1(
    *, manifest_path: str | Path, artifact_root: str | Path
) -> MassiveAdaptiveRLEndToEndRunV2:
    """Replay only Manifest V3 and the canonical state chain.

    This intentionally does not inspect the source root or claim that any
    runtime, report, or outer-evidence authority replayed.
    """

    manifest = load_massive_adaptive_rl_experiment_manifest_v3(manifest_path)
    states = load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    if not states:
        raise MassiveAdaptiveRLExperimentRunnerV2Error(
            "adaptive RL V2 experiment has no persisted state"
        )
    return _result(manifest=manifest, states=states, source_bundle=None)


__all__ = [
    "MassiveAdaptiveRLEndToEndRunV2",
    "MassiveAdaptiveRLExperimentRunnerV2Error",
    "run_massive_adaptive_rl_experiment_v2",
    "verify_massive_adaptive_rl_experiment_ledger_v1",
    "verify_massive_adaptive_rl_experiment_v2",
]
