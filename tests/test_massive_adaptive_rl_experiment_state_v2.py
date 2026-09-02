from __future__ import annotations

from pathlib import Path

import pytest

from rl_quant.protocol.canonical_artifact import semantic_sha256
import rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 as state_module
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2,
    MassiveAdaptiveRLExperimentStageV2,
    MassiveAdaptiveRLExperimentStateV2Error,
    MassiveAdaptiveRLStaleStateError,
    advance_massive_adaptive_rl_experiment_state_v2,
    block_massive_adaptive_rl_experiment_state_v2,
    fail_massive_adaptive_rl_experiment_state_v2,
    load_massive_adaptive_rl_experiment_states_v2,
    register_massive_adaptive_rl_experiment_state_v2,
)


def _receipt(value: object) -> str:
    return semantic_sha256(value)


def _advance_to_evidence(tmp_path: Path, experiment_id: str):
    state = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        experiment_id=experiment_id,
        manifest_receipt_sha256=_receipt((experiment_id, "manifest")),
    )
    for stage in MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[1:-1]:
        state = advance_massive_adaptive_rl_experiment_state_v2(
            artifact_root=tmp_path,
            previous=state,
            stage=stage,
            stage_artifact_receipt_sha256=_receipt((experiment_id, stage.value)),
        )
    return state


def test_blocked_state_can_resume_from_last_completed_stage(tmp_path: Path) -> None:
    registered = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        experiment_id="blocked-source",
        manifest_receipt_sha256=_receipt("manifest"),
    )
    blocked = block_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        previous=registered,
        blocked_stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
        blocker_code="source-temporarily-absent",
        blocker_evidence_receipt_sha256=_receipt("absent"),
    )
    assert not blocked.execution_complete
    replayed = advance_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        previous=blocked,
        stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
        stage_artifact_receipt_sha256=_receipt("source"),
    )
    assert replayed.completed_stage_index == 1
    assert load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=tmp_path,
        experiment_id="blocked-source",
    ) == (registered, blocked, replayed)


def test_state_compare_and_swap_rejects_a_stale_predecessor(tmp_path: Path) -> None:
    registered = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        experiment_id="stale-state-writer",
        manifest_receipt_sha256=_receipt("manifest"),
    )
    blocked = block_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        previous=registered,
        blocked_stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
        blocker_code="source-temporarily-absent",
        blocker_evidence_receipt_sha256=_receipt("absent"),
    )

    with pytest.raises(MassiveAdaptiveRLStaleStateError, match="stale"):
        advance_massive_adaptive_rl_experiment_state_v2(
            artifact_root=tmp_path,
            previous=registered,
            stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
            stage_artifact_receipt_sha256=_receipt("source"),
        )

    assert load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=tmp_path,
        experiment_id=registered.experiment_id,
    ) == (registered, blocked)
    state_directory = (
        tmp_path / "adaptive-rl" / registered.experiment_id / "state-v2"
    )
    assert tuple(sorted(path.name for path in state_directory.glob("*.json"))) == (
        "000-registered.json",
        "001-blocked.json",
    )


def test_raw_v2_report_publisher_is_not_public() -> None:
    assert "publish_massive_adaptive_rl_development_report_state_v2" not in (
        state_module.__all__
    )


def test_interrupted_state_install_leaves_no_partial_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _interrupt_install(_source, _destination) -> None:
        raise RuntimeError("simulated interruption before atomic install")

    monkeypatch.setattr(state_module.os, "link", _interrupt_install)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        register_massive_adaptive_rl_experiment_state_v2(
            artifact_root=tmp_path,
            experiment_id="interrupted-state-install",
            manifest_receipt_sha256=_receipt("manifest"),
        )
    directory = tmp_path / "adaptive-rl" / "interrupted-state-install" / "state-v2"
    assert tuple(directory.iterdir()) == ()


def test_atomic_state_install_is_create_only_and_cleans_collision_temp(
    tmp_path: Path,
) -> None:
    register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        experiment_id="state-install-collision",
        manifest_receipt_sha256=_receipt("manifest"),
    )
    with pytest.raises(MassiveAdaptiveRLExperimentStateV2Error, match="create-only"):
        register_massive_adaptive_rl_experiment_state_v2(
            artifact_root=tmp_path,
            experiment_id="state-install-collision",
            manifest_receipt_sha256=_receipt("manifest"),
        )
    directory = tmp_path / "adaptive-rl" / "state-install-collision" / "state-v2"
    assert tuple(path.name for path in directory.iterdir()) == ("000-registered.json",)
    assert not hasattr(
        state_module,
        "publish_massive_adaptive_rl_development_report_state_v2",
    )


def test_integrity_failure_remains_terminal(tmp_path: Path) -> None:
    registered = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        experiment_id="tampered-source",
        manifest_receipt_sha256=_receipt("manifest"),
    )
    failed = fail_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        previous=registered,
        failed_stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
        failure_code="source-receipt-mismatch",
        failure_evidence_receipt_sha256=_receipt("mismatch"),
    )
    with pytest.raises(MassiveAdaptiveRLExperimentStateV2Error, match="terminal"):
        advance_massive_adaptive_rl_experiment_state_v2(
            artifact_root=tmp_path,
            previous=failed,
            stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
            stage_artifact_receipt_sha256=_receipt("source"),
        )


def test_loader_rejects_any_state_after_a_terminal_failure(tmp_path: Path) -> None:
    registered = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        experiment_id="trailing-terminal-state",
        manifest_receipt_sha256=_receipt("manifest"),
    )
    failed = fail_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path,
        previous=registered,
        failed_stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
        failure_code="source-receipt-mismatch",
        failure_evidence_receipt_sha256=_receipt("mismatch"),
    )
    planted = state_module._build_state(
        previous=failed,
        experiment_id=failed.experiment_id,
        manifest_receipt_sha256=failed.manifest_receipt_sha256,
        stage=MassiveAdaptiveRLExperimentStageV2.BLOCKED,
        completed_stage_index=failed.completed_stage_index,
        stage_artifact_receipt_sha256=_receipt("planted-blocker"),
        blocked_stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
        blocker_code="planted-after-terminal",
    )
    state_module._write_state(artifact_root=tmp_path, state=planted)
    with pytest.raises(
        MassiveAdaptiveRLExperimentStateV2Error,
        match="follows a terminal state",
    ):
        load_massive_adaptive_rl_experiment_states_v2(
            artifact_root=tmp_path,
            experiment_id=registered.experiment_id,
        )
