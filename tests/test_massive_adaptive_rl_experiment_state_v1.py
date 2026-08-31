from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v1 import (
    MassiveAdaptiveRLExperimentStageV1,
    MassiveAdaptiveRLExperimentStateV1Error,
    advance_massive_adaptive_rl_experiment_state_v1,
    fail_massive_adaptive_rl_experiment_state_v1,
    load_massive_adaptive_rl_experiment_states_v1,
    register_massive_adaptive_rl_experiment_state_v1,
)


def _receipt(value: str) -> str:
    return semantic_sha256(value)


def test_experiment_state_is_create_only_consecutive_and_resumable(
    tmp_path: Path,
) -> None:
    registered = register_massive_adaptive_rl_experiment_state_v1(
        artifact_root=tmp_path,
        experiment_id="synthetic-four-fold",
        manifest_receipt_sha256=_receipt("manifest"),
    )
    replayed = advance_massive_adaptive_rl_experiment_state_v1(
        artifact_root=tmp_path,
        previous=registered,
        stage=MassiveAdaptiveRLExperimentStageV1.SOURCE_BUNDLE_REPLAYED,
        stage_artifact_receipt_sha256=_receipt("source-bundle"),
    )
    states = load_massive_adaptive_rl_experiment_states_v1(
        artifact_root=tmp_path,
        experiment_id="synthetic-four-fold",
    )
    assert states == (registered, replayed)
    assert (
        states[-1].previous_state_receipt_sha256 == registered.semantic_receipt_sha256
    )
    assert not states[-1].development_profitability_reporting_authorized
    assert not states[-1].live_trading_authorized

    with pytest.raises(MassiveAdaptiveRLExperimentStateV1Error, match="consecutive"):
        advance_massive_adaptive_rl_experiment_state_v1(
            artifact_root=tmp_path,
            previous=replayed,
            stage=MassiveAdaptiveRLExperimentStageV1.INNER_VALIDATION_COMPLETED,
            stage_artifact_receipt_sha256=_receipt("skipped-stage"),
        )
    with pytest.raises(MassiveAdaptiveRLExperimentStateV1Error, match="create-only"):
        register_massive_adaptive_rl_experiment_state_v1(
            artifact_root=tmp_path,
            experiment_id="synthetic-four-fold",
            manifest_receipt_sha256=_receipt("manifest"),
        )


def test_failed_experiment_state_is_durable_and_terminal(tmp_path: Path) -> None:
    registered = register_massive_adaptive_rl_experiment_state_v1(
        artifact_root=tmp_path,
        experiment_id="failed-source-run",
        manifest_receipt_sha256=_receipt("manifest"),
    )
    failed = fail_massive_adaptive_rl_experiment_state_v1(
        artifact_root=tmp_path,
        previous=registered,
        failed_stage=MassiveAdaptiveRLExperimentStageV1.SOURCE_BUNDLE_REPLAYED,
        failure_code="source-bundle-absent",
        failure_evidence_receipt_sha256=_receipt("failure"),
    )
    assert failed.stage is MassiveAdaptiveRLExperimentStageV1.FAILED
    assert (
        load_massive_adaptive_rl_experiment_states_v1(
            artifact_root=tmp_path,
            experiment_id="failed-source-run",
        )[-1]
        == failed
    )
    with pytest.raises(MassiveAdaptiveRLExperimentStateV1Error, match="terminal"):
        advance_massive_adaptive_rl_experiment_state_v1(
            artifact_root=tmp_path,
            previous=failed,
            stage=MassiveAdaptiveRLExperimentStageV1.SOURCE_BUNDLE_REPLAYED,
            stage_artifact_receipt_sha256=_receipt("source"),
        )

    forged = replace(failed, failure_code="different")
    with pytest.raises(MassiveAdaptiveRLExperimentStateV1Error, match="differs"):
        forged.validate()
