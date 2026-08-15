from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_SETTING_IDS,
)
from rl_quant.training import top2000_m03r_v15_seadragon_lifecycle as lifecycle
from rl_quant.training import top2000_m03r_v15_seadragon_operator as operator
from rl_quant.training import top2000_m03r_v15_static_gate as static_gate
from rl_quant.training.top2000_m03r_v15_fold import (
    M03RV15PanelEpisodeSchedule,
    render_m03r_v15_fold_geometries,
)
from rl_quant.training.top2000_m03r_v15_package import (
    M03RV15ExecutionAuthorization,
    M03RV15PackageArtifacts,
    build_m03r_v15_package_plan,
    write_m03r_v15_execution_authorization,
    write_m03r_v15_package_plan,
)
from rl_quant.training.top2000_m03r_v15_selection import (
    M03RV15PredictiveQualification,
)


def _package_authorization(tmp_path: Path):
    artifacts = M03RV15PackageArtifacts(
        source_archive_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        dependency_lock_sha256="3" * 64,
        cache_artifact_sha256="4" * 64,
        cache_manifest_sha256="5" * 64,
        asset_axis_sha256="f" * 64,
        risk_artifact_sha256="6" * 64,
        risk_source_manifest_file_sha256="7" * 64,
        risk_source_receipt_sha256="1" * 64,
        exposure_receipt_sha256="2" * 64,
        projector_manifest_file_sha256="8" * 64,
        projector_manifest_sha256="9" * 64,
        projector_binding_sha256="a" * 64,
        worker_source_sha256="b" * 64,
        operator_source_sha256="3" * 64,
        initial_parameter_state_file_sha256="c" * 64,
        initial_parameter_state_sha256="d" * 64,
        initial_parameter_architecture_sha256="e" * 64,
        structural_preflight_file_sha256="f" * 64,
        structural_preflight_receipt_sha256="0" * 64,
        image_reference=f"example/research@sha256:{'a' * 64}",
        image_digest_sha256="a" * 64,
    )
    schedule = M03RV15PanelEpisodeSchedule(
        protocol_common_data_sha256="e" * 64,
        cache_sha256=artifacts.cache_artifact_sha256,
        asset_axis_sha256="f" * 64,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v15_fold_geometries(1001)
        ),
    )
    package = build_m03r_v15_package_plan(artifacts, schedule)
    package_path = tmp_path / "package-plan.json"
    package_file_sha = write_m03r_v15_package_plan(package_path, package)
    authorization = M03RV15ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=package_file_sha,
        source_archive_sha256=artifacts.source_archive_sha256,
        source_manifest_sha256=artifacts.source_manifest_sha256,
        worker_source_sha256=artifacts.worker_source_sha256,
        image_reference=artifacts.image_reference,
    )
    authorization_path = tmp_path / "execution-authorization.json"
    authorization_file_sha = write_m03r_v15_execution_authorization(
        authorization_path, authorization, package
    )
    return (
        package,
        package_path,
        package_file_sha,
        authorization,
        authorization_path,
        authorization_file_sha,
    )


def _qualification(
    *,
    setting_index: int = 0,
    fold_traces: tuple[str, ...] | None = None,
    bootstrap_sha: str = "7" * 64,
) -> M03RV15PredictiveQualification:
    traces = fold_traces or tuple(str(index + 1) * 64 for index in range(6))
    return M03RV15PredictiveQualification(
        setting_index=setting_index,
        setting_id=M03R_V15_SETTING_IDS[setting_index],
        fold_trace_sha256=traces,
        bootstrap_plan_sha256=bootstrap_sha,
        mean_rank_ic=0.0,
        positive_mean_ic_fold_count=0,
        positive_median_ic_fold_count=0,
        positive_date_fraction_fold_count=0,
        positive_spread_fold_count=0,
        annualized_gross_active_return=0.0,
        annualized_net_active_return_10bp=0.0,
        gross_active_lcb_by_block=(0.0, 0.0, 0.0),
        net_10bp_active_lcb_by_block=(0.0, 0.0, 0.0),
        spread_lcb_by_block=(0.0, 0.0, 0.0),
        break_even_category="no-positive-break-even",
        break_even_one_way_cost_basis_points=None,
        median_signal_projection_retention=0.0,
        minimum_fold_median_signal_projection_retention=0.0,
        median_risk_projection_retention=0.0,
        minimum_fold_median_risk_projection_retention=0.0,
        passed=False,
        economic_generation_may_be_minted=False,
    )


def test_v15_attach_config_binds_authorization_and_two_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    package, package_path, package_sha, auth, auth_path, auth_sha = (
        _package_authorization(tmp_path)
    )
    expected = tuple(
        lifecycle.M03RV15ExpectedCompletion(
            completion_index=index,
            setting_index=index,
            setting_id=worker.setting_id,
            worker_plan_sha256=worker.receipt_sha256,
        )
        for index, worker in enumerate(package.panel.workers)
    )
    config = lifecycle.M03RV15AttachSupervisorConfig(
        job_name="qt-m03r-v15-predictive-a04",
        run_id="qt-m03r-v15-predictive-s17-20260814-a04",
        job_uid="job-uid",
        binding_path=str(tmp_path / "binding.json"),
        binding_file_sha256="f" * 64,
        activation_request_path=str(tmp_path / "activation.json"),
        activation_request_file_sha256="0" * 64,
        output_root=str(tmp_path / "output"),
        evidence_root=str(tmp_path / "evidence"),
        package_plan_path=str(package_path),
        package_plan_file_sha256=package_sha,
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_path=str(auth_path),
        execution_authorization_file_sha256=auth_sha,
        execution_authorization_receipt_sha256=auth.receipt_sha256,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        capacity_receipt_sha256="1" * 64,
        lifecycle_source_sha256="2" * 64,
        expected_completions=expected,
        host_python_path="/usr/bin/python3",
        pythonpath=str(tmp_path / "source"),
    )
    loaded_package, loaded_auth = lifecycle._load_package_authorization(config)
    assert loaded_package == package
    assert loaded_auth == auth
    assert config.completions == 2
    assert config.parallelism == 2


def test_v15_lifecycle_reconstructs_exact_typed_qualification() -> None:
    qualification = _qualification()
    value = asdict(qualification)
    lifecycle._validate_qualification(
        value,
        setting_index=0,
        expected_fold_traces=list(qualification.fold_trace_sha256),
        expected_bootstrap_plan_sha256=qualification.bootstrap_plan_sha256,
        expected_receipt_sha256=qualification.receipt_sha256,
    )
    value["economic_generation_may_be_minted"] = True
    with pytest.raises(lifecycle.M03RV15SeadragonLifecycleError):
        lifecycle._validate_qualification(
            value,
            setting_index=0,
            expected_fold_traces=list(qualification.fold_trace_sha256),
            expected_bootstrap_plan_sha256=qualification.bootstrap_plan_sha256,
            expected_receipt_sha256=qualification.receipt_sha256,
        )


def test_v15_workflow_file_hash_convention_is_newline_bound() -> None:
    unsigned = {"schema": "terminal", "training_performed": False}
    value = {**unsigned, "receipt_sha256": lifecycle._content_sha256(unsigned)}
    assert lifecycle._receipt_payload(value, "terminal") == unsigned
    assert lifecycle._content_sha256(unsigned) == hashlib.sha256(
        (
            json.dumps(
                unsigned,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()


def test_v15_operator_and_lifecycle_have_disjoint_mutation_surfaces() -> None:
    operator_source = inspect.getsource(operator.OneCreateKubectl)
    lifecycle_source = inspect.getsource(lifecycle.common.AttachOnlyKubectl)
    assert operator_source.count("create_once") == 1
    assert 'arguments[0] in {"create", "apply", "replace"}' in lifecycle_source
    assert "top2000_m03r_v15_static_validate" not in inspect.getsource(static_gate)


def test_v15_create_config_is_two_worker_predictive_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operator, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    fields = {
        "mode": "predictive",
        "job_name": "qt-m03r-v15-predictive-a04",
        "run_id": "qt-m03r-v15-predictive-s17-20260814-a04",
        "rendered_path": str(tmp_path / "rendered.json"),
        "rendered_file_sha256": "1" * 64,
        "manifest_path": str(tmp_path / "manifest.json"),
        "manifest_file_sha256": "2" * 64,
        "evidence_root": str(tmp_path / "evidence"),
        "binding_output_path": str(tmp_path / "binding.json"),
        "activation_output_path": str(tmp_path / "activation.json"),
        "package_plan_sha256": "3" * 64,
        "execution_authorization_receipt_sha256": "4" * 64,
        "source_archive_sha256": "5" * 64,
        "capacity_receipt_sha256": "6" * 64,
        "operator_source_sha256": "7" * 64,
        "completions": 2,
        "parallelism": 2,
    }
    assert operator.M03RV15CreateOperatorConfig(**fields).completions == 2
    with pytest.raises(operator.M03RV15SeadragonOperatorError, match="identity"):
        operator.M03RV15CreateOperatorConfig(
            **{**fields, "completions": 3, "parallelism": 3}
        )
    with pytest.raises(operator.M03RV15SeadragonOperatorError, match="identity"):
        operator.M03RV15CreateOperatorConfig(
            **{**fields, "mode": "audit"}  # type: ignore[arg-type]
        )


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_v15_worker_validator_matches_exact_v15_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    monkeypatch.setattr(lifecycle, "_validate_rank_runtime", lambda _value: None)
    package, package_path, package_sha, auth, auth_path, auth_sha = (
        _package_authorization(tmp_path)
    )
    worker = package.panel.workers[0]
    expected = tuple(
        lifecycle.M03RV15ExpectedCompletion(
            completion_index=index,
            setting_index=index,
            setting_id=item.setting_id,
            worker_plan_sha256=item.receipt_sha256,
        )
        for index, item in enumerate(package.panel.workers)
    )
    output_root = tmp_path / "output"
    worker_root = output_root / "completion-00-setting-00"
    startup = {
        "schema": lifecycle.M03R_V15_STARTUP_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": package_sha,
        "authorization_receipt_sha256": auth.receipt_sha256,
        "worker_plan_sha256": worker.receipt_sha256,
        "setting_index": 0,
        "setting_id": worker.setting_id,
        "mode": "predictive",
        "rank_runtime": [],
        "exact_h100_80gb_per_rank": True,
        "nccl_process_group_initialized": True,
        "restart_count": 0,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    startup_sha = _write_json(worker_root / "two-h100-startup.json", startup)
    fold_files: list[str] = []
    fold_traces: list[str] = []
    for fold_index, completed_updates in enumerate(worker.fold_optimizer_updates):
        checkpoint = worker_root / "checkpoints" / f"fold-{fold_index:02d}.pt"
        artifact = (
            worker_root
            / "fold-artifacts"
            / f"fold-{fold_index:02d}-horizon-03.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{fold_index}".encode())
        artifact.write_bytes(f"artifact-{fold_index}".encode())
        trace_sha = f"{fold_index + 1:x}" * 64
        fold_traces.append(trace_sha)
        validation_hashes = tuple(
            hashlib.sha256(f"{fold_index}:{epoch}".encode()).hexdigest()
            for epoch in range(8)
        )
        selection = {
            "setting_index": 0,
            "fold_index": fold_index,
            "selected_epoch_index": 3,
            "selected_model_state_sha256": "1" * 64,
            "selected_validation_receipt_sha256": validation_hashes[3],
            "candidate_validation_receipt_sha256": validation_hashes,
            "selection_rule": (
                lifecycle.M03R_V15_PREDICTIVE_SPEC.checkpoint_selection_rule
            ),
            "qualification_tail_accessed": False,
            "protocol_sha256": lifecycle.M03R_V15_PROTOCOL_SHA256,
            "schema": (
                "rl-quant.top2000-dev.m03r-v15-checkpoint-selection-v1"
            ),
        }
        selection_receipt = lifecycle.M03RV15CheckpointSelectionReceipt(
            **selection
        )
        unsigned = {
            "schema": lifecycle.M03R_V15_FOLD_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "authorization_receipt_sha256": auth.receipt_sha256,
            "worker_plan_sha256": worker.receipt_sha256,
            "setting_index": 0,
            "setting_id": worker.setting_id,
            "fold_index": fold_index,
            "completed_updates": completed_updates,
            "training_epoch_count": 8,
            "model_state_sha256": "1" * 64,
            "optimizer_state_sha256": "2" * 64,
            "selected_epoch_index": 3,
            "checkpoint_selection": selection,
            "checkpoint_selection_sha256": selection_receipt.receipt_sha256,
            "inner_validation_receipt_sha256": validation_hashes,
            "qualification_tail_accessed_for_selection": False,
            "training_update_evidence_root_sha256": "3" * 64,
            "training_source_array_root_sha256": "4" * 64,
            "training_target_operator_root_sha256": "5" * 64,
            "training_action_operator_root_sha256": "6" * 64,
            "checkpoint_path": (
                f"/mnt/output/completion-00-setting-00/checkpoints/"
                f"fold-{fold_index:02d}.pt"
            ),
            "checkpoint_file_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
            "qualification_artifact_path": (
                f"/mnt/output/completion-00-setting-00/fold-artifacts/"
                f"fold-{fold_index:02d}-horizon-03.pt"
            ),
            "qualification_artifact_file_sha256": hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest(),
            "qualification_trace_sha256": trace_sha,
            "qualification_batch_receipt_sha256": "7" * 64,
            "fold_risk_state_sha256": "8" * 64,
            "qualification_evaluated_only_after_checkpoint_reload": True,
            "economic_optimizer_updates": 0,
            "outer_2026_accessed": False,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        fold_files.append(
            _write_json(
                worker_root
                / "receipts"
                / f"fold-{fold_index:02d}-terminal.json",
                {**unsigned, "receipt_sha256": lifecycle._content_sha256(unsigned)},
            )
        )
    bootstrap = {
        "chronology_sha256": "9" * 64,
        "fold_lengths": [63] * 6,
        "bootstrap_seed": lifecycle.M03R_V15_PREDICTIVE_SPEC.bootstrap_seed,
        "draw_sha256_by_block": ["a" * 64, "b" * 64, "c" * 64],
        "block_sessions": [10, 21, 30],
        "replicates": lifecycle.M03R_V15_PREDICTIVE_SPEC.bootstrap_replicates,
        "protocol_sha256": lifecycle.M03R_V15_PROTOCOL_SHA256,
        "schema": lifecycle.M03R_V15_BOOTSTRAP_PLAN_SCHEMA,
    }
    bootstrap_sha = lifecycle._compact_sha256(bootstrap)
    qualification = _qualification(
        fold_traces=tuple(fold_traces), bootstrap_sha=bootstrap_sha
    )
    unsigned_terminal = {
        "schema": lifecycle.M03R_V15_WORKER_TERMINAL_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": package_sha,
        "authorization_receipt_sha256": auth.receipt_sha256,
        "worker_plan_sha256": worker.receipt_sha256,
        "startup_file_sha256": startup_sha,
        "setting_index": 0,
        "setting_id": worker.setting_id,
        "fold_terminal_file_sha256": fold_files,
        "bootstrap_plan": bootstrap,
        "bootstrap_plan_sha256": bootstrap_sha,
        "predictive_qualification": asdict(qualification),
        "predictive_qualification_sha256": qualification.receipt_sha256,
        "selected_horizon": None,
        "predictive_gate_passed": False,
        "economic_generation_may_be_minted": False,
        "economic_panel_authorized": False,
        "economic_optimizer_updates": 0,
        "outer_2026_accessed": False,
        "world_size": 2,
        "gpus_per_worker": 2,
        "h100_capacity_evidence": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    terminal_sha = _write_json(
        worker_root / "predictive-terminal.json",
        {
            **unsigned_terminal,
            "receipt_sha256": lifecycle._content_sha256(unsigned_terminal),
        },
    )
    config = lifecycle.M03RV15AttachSupervisorConfig(
        job_name="qt-m03r-v15-predictive-a04",
        run_id="qt-m03r-v15-predictive-s17-20260814-a04",
        job_uid="job-uid",
        binding_path=str(tmp_path / "binding.json"),
        binding_file_sha256="1" * 64,
        activation_request_path=str(tmp_path / "activation.json"),
        activation_request_file_sha256="2" * 64,
        output_root=str(output_root),
        evidence_root=str(tmp_path / "evidence"),
        package_plan_path=str(package_path),
        package_plan_file_sha256=package_sha,
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_path=str(auth_path),
        execution_authorization_file_sha256=auth_sha,
        execution_authorization_receipt_sha256=auth.receipt_sha256,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        capacity_receipt_sha256="3" * 64,
        lifecycle_source_sha256="4" * 64,
        expected_completions=expected,
        host_python_path="/usr/bin/python3",
        pythonpath=str(tmp_path / "source"),
    )
    validated_sha, evidence = lifecycle._validate_one_worker(config, expected[0])
    assert validated_sha == terminal_sha
    assert evidence["predictive_gate_passed"] is False
    assert evidence["selected_horizon"] is None
