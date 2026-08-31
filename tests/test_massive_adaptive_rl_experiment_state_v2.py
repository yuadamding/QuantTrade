from __future__ import annotations

from pathlib import Path

import pytest

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2,
    MassiveAdaptiveRLExperimentStageV2,
    MassiveAdaptiveRLExperimentStateV2Error,
    advance_massive_adaptive_rl_experiment_state_v2,
    block_massive_adaptive_rl_experiment_state_v2,
    fail_massive_adaptive_rl_experiment_state_v2,
    load_massive_adaptive_rl_experiment_states_v2,
    publish_massive_adaptive_rl_development_report_state_v2,
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


@pytest.mark.parametrize(
    ("authorized", "failed_gates"),
    ((True, ()), (False, ("incremental-rl-lcb-positive",))),
)
def test_positive_and_negative_reports_are_both_completed_experiments(
    tmp_path: Path,
    authorized: bool,
    failed_gates: tuple[str, ...],
) -> None:
    experiment_id = "positive-report" if authorized else "negative-report"
    evidence = _advance_to_evidence(tmp_path, experiment_id)
    published = publish_massive_adaptive_rl_development_report_state_v2(
        artifact_root=tmp_path,
        previous=evidence,
        profitability_report_authority_receipt_sha256=_receipt("authority"),
        profitability_report_receipt_sha256=_receipt("report"),
        failed_gate_names=failed_gates,
        development_profitability_reporting_authorized=authorized,
    )
    assert published.execution_complete
    assert (
        published.stage
        is MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED
    )
    assert published.development_profitability_reporting_authorized is authorized
    assert not published.live_trading_authorized


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
