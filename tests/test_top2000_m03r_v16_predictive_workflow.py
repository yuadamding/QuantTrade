from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rl_quant.training.top2000_m03r_v16_qualification_runtime as qualification
import rl_quant.workflows.top2000_m03r_v16_attestation_gate as attestation_gate
import rl_quant.workflows.top2000_m03r_v16_predictive as predictive
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    render_m03r_v16_fold_geometries,
)
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03RV16PredictiveWorkflowError,
    _await_m03r_v16_qualification_panel_barrier,
    _validate_gathered_update,
    resolve_m03r_v16_completion_index,
)


def _rank_row(rank: int, *, local_count: int = 22) -> dict[str, object]:
    return {
        "update_plan_sha256": "0" * 64,
        "batch_receipt_sha256": str(rank) * 64,
        "step_receipt_sha256": str(rank + 2) * 64,
        "source_array_sha256": "3" * 64,
        "selection_target_operator_root_sha256": "4" * 64,
        "action_operator_root_sha256": "5" * 64,
        "completed_updates_after": 1,
        "distributed_rank": rank,
        "local_origin_count": local_count,
        "global_origin_count": 43,
        "encoder_version_root_before": "6" * 64,
        "encoder_version_root_after": "7" * 64,
        "selection_head_version_root_before": "8" * 64,
        "selection_head_version_root_after": "9" * 64,
    }


def test_v16_completion_index_is_exactly_three_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "2")
    assert resolve_m03r_v16_completion_index(None) == 2
    assert resolve_m03r_v16_completion_index(0) == 0
    for invalid in (-1, 3, True):
        with pytest.raises(M03RV16PredictiveWorkflowError, match="drifted"):
            resolve_m03r_v16_completion_index(invalid)


def test_v16_rank_update_requires_complete_equal_mutation_evidence() -> None:
    rows = [_rank_row(0), _rank_row(1, local_count=21)]
    _validate_gathered_update(rows, 2)

    incomplete = [rows[0], {**rows[1], "local_origin_count": 20}]
    with pytest.raises(M03RV16PredictiveWorkflowError, match="diverged"):
        _validate_gathered_update(incomplete, 2)

    drifted = [
        rows[0],
        {**rows[1], "encoder_version_root_after": "a" * 64},
    ]
    with pytest.raises(M03RV16PredictiveWorkflowError, match="diverged"):
        _validate_gathered_update(drifted, 2)


def test_v16_qualification_risk_state_covers_decisions_and_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SimpleNamespace(
        daily_ohlcv=torch.ones((1001, 4, 5), dtype=torch.float32),
        availability=torch.ones((1001, 4), dtype=torch.bool),
        cache_sha256="a" * 64,
        action_hash="b" * 64,
        validate_unmodified=lambda: None,
    )
    risk_source = SimpleNamespace(validate=lambda: None)
    sentinel = object()
    captured: dict[str, object] = {}

    def build(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(qualification, "build_m03r_v9_device_risk_state", build)
    geometry = render_m03r_v16_fold_geometries(1001)[0]
    result = qualification.build_m03r_v16_qualification_risk_state(
        cache,
        geometry,
        risk_source,
        SimpleNamespace(),
        SimpleNamespace(),
        device=torch.device("cpu"),
    )
    origins = captured["origin_state_indices"]
    assert result is sentinel
    assert isinstance(origins, tuple)
    assert len(origins) == 92
    assert origins[0] == geometry.qualification_origin_start_inclusive
    assert origins[-1] == geometry.qualification_origin_start_inclusive + 91
    assert captured["sequence_asset_axis_sha256"] == cache.action_hash
    assert captured["checkpoint_asset_axis_sha256"] == cache.action_hash


def test_v16_worker_source_requires_terminal_authority_and_is_predictive_only() -> None:
    source = Path(
        "src/rl_quant/workflows/top2000_m03r_v16_predictive.py"
    ).read_text(encoding="utf-8")
    assert "issue_m03r_v16_terminal_checkpoint_authority" in source
    assert "load_m03r_v16_epoch_checkpoint_for_evaluation" in source
    assert "build_m03r_v16_qualification_risk_state" in source
    assert "V16 training requires an immutable activation authority" in source
    assert "V16 qualification requires activation" in source
    assert '"qualification_tail_accessed": False' in source
    assert 'output / "training-terminal.json"' in source
    assert 'output / "launch-consumption.json"' in source
    assert "load_m03r_v16_pod_runtime_attestation" in source
    assert "M03R_V16_CURRENT_POD_UID" in source
    assert "_await_m03r_v16_qualification_panel_barrier" in source
    assert 'output / "training-numerical-failure.json"' in source
    assert '"economic_optimizer_updates": 0' in source
    assert '"reinforcement_learning_updates": 0' in source
    assert '"outer_2026_accessed": False' in source
    assert "top2000_m03r_v15" not in source


@pytest.mark.parametrize("qualification_only", [False, True])
def test_v16_scientific_worker_rejects_direct_invocation_without_phase_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qualification_only: bool,
) -> None:
    package = SimpleNamespace()
    authorization = SimpleNamespace(package_plan_file_sha256="a" * 64)
    monkeypatch.setattr(predictive, "load_m03r_v16_package_plan", lambda *a, **k: package)
    monkeypatch.setattr(
        predictive,
        "load_m03r_v16_execution_authorization",
        lambda *a, **k: authorization,
    )
    monkeypatch.setattr(
        predictive,
        "_validate_runtime_package_members",
        lambda *a, **k: (tmp_path, "b" * 64),
    )

    expected = (
        "qualification requires activation"
        if qualification_only
        else "training requires an immutable activation authority"
    )
    with pytest.raises(M03RV16PredictiveWorkflowError, match=expected):
        predictive.run_m03r_v16_predictive_worker(
            tmp_path / "package-plan.json",
            tmp_path / "execution-authorization.json",
            expected_package_plan_file_sha256="a" * 64,
            expected_authorization_file_sha256="c" * 64,
            qualification_only=qualification_only,
        )


def test_v16_panel_barrier_closes_all_settings_before_outer_access(
    tmp_path: Path,
) -> None:
    package = SimpleNamespace(package_plan_sha256="a" * 64)
    activation = SimpleNamespace(receipt_sha256="b" * 64)
    closure_hashes: list[str] = []
    for setting in range(3):
        output = tmp_path / f"completion-{setting:02d}-setting-{setting:02d}"
        unsigned = {
            "schema": (
                "rl-quant.top2000-dev."
                "m03r-v16-qualification-inputs-complete-v2"
            ),
            "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
            "package_plan_sha256": package.package_plan_sha256,
            "qualification_activation_receipt_sha256": (
                activation.receipt_sha256
            ),
            "setting_index": setting,
            "folds": tuple({"fold_index": fold} for fold in range(5)),
            "outer_qualification_access_started": False,
            "outer_2026_accessed": False,
        }
        payload = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
        closure_hashes.append(
            predictive._write_immutable_json(
                output / "qualification-inputs-complete.json", payload
            )
        )
    barrier = _await_m03r_v16_qualification_panel_barrier(
        phase_root=tmp_path,
        setting_index=0,
        package=package,  # type: ignore[arg-type]
        qualification_activation=activation,  # type: ignore[arg-type]
        local_closure_file_sha256=closure_hashes[0],
        timeout_seconds=0.0,
    )
    assert all(len(value) == 64 for value in barrier)
    assert (tmp_path / "qualification-panel-inputs-complete.json").is_file()
    assert not tuple(tmp_path.glob("**/outer-access-fold-*.json"))


def test_v16_missing_setting_closure_cannot_open_any_outer_fold(
    tmp_path: Path,
) -> None:
    with pytest.raises(M03RV16PredictiveWorkflowError, match="timed out"):
        _await_m03r_v16_qualification_panel_barrier(
            phase_root=tmp_path,
            setting_index=0,
            package=SimpleNamespace(package_plan_sha256="a" * 64),  # type: ignore[arg-type]
            qualification_activation=SimpleNamespace(  # type: ignore[arg-type]
                receipt_sha256="b" * 64
            ),
            local_closure_file_sha256="c" * 64,
            timeout_seconds=0.0,
        )
    assert not tuple(tmp_path.glob("**/outer-access-fold-*.json"))


def test_v16_init_gate_requires_annotations_and_writes_validated_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_path = "pod-runtime/training/job-uid/completion-01.json"
    source_root = Path(attestation_gate.__file__).resolve().parents[2]
    package = SimpleNamespace(
        source_pythonpath=str(source_root),
        worker_output_roots=tuple(
            f"/mnt/output/completion-{index:02d}" for index in range(3)
        ),
    )
    authorization = object()
    admission = SimpleNamespace(job_uid="job-uid")
    launch = SimpleNamespace(
        receipt_sha256="1" * 64,
        relative_path=lambda index: relative_path,
    )
    attestation = SimpleNamespace(
        receipt_sha256="2" * 64,
        pod_uid="pod-uid",
        pod_name="pod-name",
        node_name="node-name",
        relative_path=relative_path,
    )
    monkeypatch.setattr(
        attestation_gate, "load_v16_lifecycle_package", lambda *a, **k: package
    )
    monkeypatch.setattr(
        attestation_gate,
        "load_v16_lifecycle_authorization",
        lambda *a, **k: authorization,
    )
    monkeypatch.setattr(
        attestation_gate,
        "load_v16_lifecycle_admission",
        lambda *a, **k: admission,
    )
    monkeypatch.setattr(
        attestation_gate,
        "load_v16_lifecycle_launch",
        lambda *a, **k: launch,
    )
    monkeypatch.setattr(
        attestation_gate,
        "load_v16_lifecycle_pod_attestation",
        lambda *a, **k: attestation,
    )
    downward = tmp_path / "podinfo"
    downward.mkdir()
    (downward / "pod-runtime-attestation-path").write_text(relative_path)
    (downward / "pod-runtime-attestation-file-sha256").write_text("3" * 64)
    (downward / "pod-runtime-attestation-receipt-sha256").write_text("2" * 64)
    attestation_path = tmp_path / "authority" / relative_path
    attestation_path.parent.mkdir(parents=True)
    attestation_path.write_text("complete")
    monkeypatch.setenv("M03R_V16_CURRENT_POD_UID", "pod-uid")
    monkeypatch.setenv("M03R_V16_CURRENT_POD_NAME", "pod-name")
    monkeypatch.setenv("M03R_V16_CURRENT_NODE_NAME", "node-name")
    marker_path = tmp_path / "marker" / "validated.json"
    marker = attestation_gate.validate_m03r_v16_pod_attestation_gate(
        package_plan_path=tmp_path / "package.json",
        package_plan_file_sha256="4" * 64,
        authorization_path=tmp_path / "authorization.json",
        authorization_file_sha256="5" * 64,
        phase="training",
        prerequisite_authority_receipt_sha256="6" * 64,
        job_contract_sha256="7" * 64,
        pod_contract_sha256="8" * 64,
        launch_authority_path=tmp_path / "launch.json",
        launch_authority_file_sha256="9" * 64,
        launch_authority_receipt_sha256="1" * 64,
        admitted_job_authority_path=tmp_path / "admission.json",
        admitted_job_authority_file_sha256="a" * 64,
        admitted_job_authority_receipt_sha256="b" * 64,
        server_side_dry_run_result_path=tmp_path / "dry.json",
        admitted_manifest_result_path=tmp_path / "admitted.json",
        completion_index=1,
        output_root="/mnt/output/completion-01",
        downward_root=downward,
        authority_root=tmp_path / "authority",
        marker_path=marker_path,
        package_source_root=source_root,
        timeout_seconds=0.0,
    )
    assert marker_path.is_file()
    assert marker["attestation_receipt_sha256"] == "2" * 64
    assert marker["relative_path"] == relative_path
    assert marker["package_source_root"] == str(source_root)
    assert marker["gate_module_path"] == str(Path(attestation_gate.__file__).resolve())
