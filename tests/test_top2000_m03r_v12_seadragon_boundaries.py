from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_SETTING_IDS,
)
from rl_quant.training import top2000_m03r_v12_seadragon_lifecycle as lifecycle
from rl_quant.training import top2000_m03r_v12_seadragon_operator as operator
from rl_quant.training import top2000_m03r_v12_static_gate as static_gate
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v12_package import (
    M03RV12ExecutionAuthorization,
    M03RV12PackageArtifacts,
    build_m03r_v12_package_plan,
    write_m03r_v12_execution_authorization,
    write_m03r_v12_package_plan,
)
from rl_quant.training.top2000_m03r_v12_schedule import M03RV12PanelEpisodeSchedule
from rl_quant.training.top2000_m03r_v12_selection import (
    M03RV12PredictiveQualification,
)


def _package_authorization(tmp_path: Path):
    artifacts = M03RV12PackageArtifacts(
        source_archive_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        dependency_lock_sha256="3" * 64,
        cache_artifact_sha256="4" * 64,
        cache_manifest_sha256="5" * 64,
        risk_artifact_sha256="6" * 64,
        risk_source_manifest_file_sha256="7" * 64,
        projector_manifest_file_sha256="8" * 64,
        projector_manifest_sha256="9" * 64,
        projector_binding_sha256="a" * 64,
        worker_source_sha256="b" * 64,
        initial_parameter_state_file_sha256="e" * 64,
        initial_parameter_state_sha256="c" * 64,
        structural_preflight_file_sha256="f" * 64,
        structural_preflight_receipt_sha256="0" * 64,
        image_reference=f"example/research@sha256:{'d' * 64}",
        image_digest_sha256="d" * 64,
    )
    folds = render_top2000_m03r_v7_development_folds(1001)
    schedule = M03RV12PanelEpisodeSchedule(
        protocol_common_data_sha256="e" * 64,
        cache_sha256=artifacts.cache_artifact_sha256,
        fold_geometry_sha256=tuple(
            render_m03r_v10_fold_geometry(fold).receipt_sha256 for fold in folds
        ),
    )
    package = build_m03r_v12_package_plan(artifacts, schedule)
    package_path = tmp_path / "package-plan.json"
    package_file_sha = write_m03r_v12_package_plan(package_path, package)
    authorization = M03RV12ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=package_file_sha,
        source_archive_sha256=artifacts.source_archive_sha256,
        source_manifest_sha256=artifacts.source_manifest_sha256,
        worker_source_sha256=artifacts.worker_source_sha256,
        image_reference=artifacts.image_reference,
    )
    authorization_path = tmp_path / "execution-authorization.json"
    authorization_file_sha = write_m03r_v12_execution_authorization(
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


def test_v12_attach_config_binds_package_authorization_and_three_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    package, package_path, package_sha, auth, auth_path, auth_sha = (
        _package_authorization(tmp_path)
    )
    expected = tuple(
        lifecycle.M03RV12ExpectedCompletion(
            completion_index=index,
            setting_index=index,
            setting_id=worker.setting_id,
            worker_plan_sha256=worker.receipt_sha256,
        )
        for index, worker in enumerate(package.panel.workers)
    )
    config = lifecycle.M03RV12AttachSupervisorConfig(
        job_name="qt-m03r-v12-predictive-a02",
        run_id="qt-m03r-v12-predictive-s17-20260812-a02",
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
    assert loaded_package.package_plan_sha256 == package.package_plan_sha256
    assert loaded_auth.receipt_sha256 == auth.receipt_sha256
    with pytest.raises(lifecycle.M03RV12SeadragonLifecycleError, match="authorization"):
        lifecycle._load_package_authorization(
            lifecycle.M03RV12AttachSupervisorConfig(
                **{
                    **asdict(config),
                    "expected_completions": expected,
                    "execution_authorization_receipt_sha256": "3" * 64,
                }
            )
        )


def test_v12_lifecycle_reconstructs_typed_failed_qualification() -> None:
    fold_receipts = [str(index + 1) * 64 for index in range(6)]
    qualification = M03RV12PredictiveQualification(
        setting_index=0,
        setting_id=M03R_V12_SETTING_IDS[0],
        horizon_sessions=3,
        fold_receipt_sha256=tuple(fold_receipts),
        bootstrap_plan_sha256="7" * 64,
        mean_rank_ic=0.0,
        positive_mean_ic_fold_count=0,
        positive_median_ic_fold_count=0,
        positive_date_fraction_fold_count=0,
        positive_spread_fold_count=0,
        annualized_gross_active_return=0.0,
        annualized_net_active_return_10bp=0.0,
        gross_active_return_lcb=0.0,
        net_active_return_10bp_lcb=0.0,
        top_bottom_spread_lcb=0.0,
        prediction_dispersion_gate_passed=False,
        prediction_target_dispersion_ratio_gate_passed=False,
        break_even_category="no-positive-break-even",
        break_even_one_way_cost_basis_points=None,
        median_projection_retention=0.0,
        minimum_fold_median_projection_retention=0.0,
        passed=False,
        economic_generation_may_be_minted=False,
    )
    value = {**asdict(qualification), "receipt_sha256": qualification.receipt_sha256}
    lifecycle._validate_qualification(
        value,
        setting_index=0,
        expected_horizon=3,
        expected_fold_receipts=fold_receipts,
        expected_bootstrap_plan_sha256="7" * 64,
    )
    value["economic_generation_may_be_minted"] = True
    value["receipt_sha256"] = lifecycle._compact_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    with pytest.raises(lifecycle.M03RV12SeadragonLifecycleError):
        lifecycle._validate_qualification(
            value,
            setting_index=0,
            expected_horizon=3,
            expected_fold_receipts=fold_receipts,
            expected_bootstrap_plan_sha256="7" * 64,
        )


def test_v12_lifecycle_uses_workflow_file_hash_for_terminal_receipts() -> None:
    unsigned = {
        "schema": "rl-quant.top2000-dev.m03r-v12-two-h100-capacity-terminal-v1",
        "world_size": 2,
    }
    value = {
        **unsigned,
        "receipt_sha256": lifecycle._content_sha256(unsigned),
    }
    assert lifecycle._receipt_payload(value, "capacity terminal") == unsigned
    value["receipt_sha256"] = lifecycle._compact_sha256(unsigned)
    with pytest.raises(lifecycle.M03RV12SeadragonLifecycleError, match="drifted"):
        lifecycle._receipt_payload(value, "capacity terminal")


def test_v12_workflow_and_lifecycle_terminal_hash_conventions_match() -> None:
    unsigned = {"schema": "terminal", "training_performed": False}
    workflow_bytes = (
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert (
        lifecycle._content_sha256(unsigned)
        == hashlib.sha256(workflow_bytes).hexdigest()
    )


def test_v12_operator_and_lifecycle_have_disjoint_mutation_surfaces() -> None:
    operator_source = inspect.getsource(operator.OneCreateKubectl)
    lifecycle_source = inspect.getsource(lifecycle.common.AttachOnlyKubectl)
    assert operator_source.count("create_once") == 1
    assert (
        "create"
        not in lifecycle_source.split("def _run", 1)[1]
        .split("def get_job", 1)[0]
        .split("rejected", 1)[0]
    )
    assert 'arguments[0] in {"create", "apply", "replace"}' in lifecycle_source
    assert "top2000_m03r_v12_static_validate" not in inspect.getsource(static_gate)


def test_v12_create_config_requires_authorization_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operator, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    fields = {
        "mode": "capacity",
        "job_name": "qt-m03r-v12-cap-a02",
        "run_id": "qt-m03r-v12-predictive-s17-20260812-a02",
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
        "capacity_receipt_sha256": "not-yet-created",
        "operator_source_sha256": "6" * 64,
        "completions": 1,
        "parallelism": 1,
    }
    config = operator.M03RV12CreateOperatorConfig(**fields)
    assert config.execution_authorization_receipt_sha256 == "4" * 64
    with pytest.raises(operator.M03RV12SeadragonOperatorError, match="SHA-256"):
        operator.M03RV12CreateOperatorConfig(
            **{**fields, "execution_authorization_receipt_sha256": "missing"}
        )


def test_v12_create_config_rejects_inherited_audit_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operator, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    with pytest.raises(operator.M03RV12SeadragonOperatorError, match="identity"):
        operator.M03RV12CreateOperatorConfig(
            mode="audit",  # type: ignore[arg-type]
            job_name="qt-m03r-v12-audit-a03",
            run_id="qt-m03r-v12-h3-predictive-s17-20260813-a03",
            rendered_path=str(tmp_path / "rendered.json"),
            rendered_file_sha256="1" * 64,
            manifest_path=str(tmp_path / "manifest.json"),
            manifest_file_sha256="2" * 64,
            evidence_root=str(tmp_path / "evidence"),
            binding_output_path=str(tmp_path / "binding.json"),
            activation_output_path=str(tmp_path / "activation.json"),
            package_plan_sha256="3" * 64,
            execution_authorization_receipt_sha256="4" * 64,
            source_archive_sha256="5" * 64,
            capacity_receipt_sha256="6" * 64,
            operator_source_sha256="7" * 64,
            completions=2,
            parallelism=2,
        )


def test_v12_worker_coverage_uses_the_protocol_selected_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    monkeypatch.setattr(lifecycle, "_validate_rank_runtime", lambda _value: None)
    package, package_path, package_sha, auth, auth_path, auth_sha = (
        _package_authorization(tmp_path)
    )
    worker = package.panel.workers[0]
    completions = tuple(
        lifecycle.M03RV12ExpectedCompletion(
            completion_index=index,
            setting_index=index,
            setting_id=row.setting_id,
            worker_plan_sha256=row.receipt_sha256,
        )
        for index, row in enumerate(package.panel.workers)
    )
    completion = completions[0]
    output_root = tmp_path / "output"
    worker_root = output_root / "completion-00-setting-00"

    def write_json(path: Path, value: object) -> str:
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

    startup = {
        "schema": lifecycle.M03R_V12_STARTUP_SCHEMA,
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
    startup_sha = write_json(worker_root / "two-h100-startup.json", startup)
    fold_file_sha: list[str] = []
    fold_evidence_sha: list[str] = []
    for fold_index in range(6):
        checkpoint = worker_root / "checkpoints" / f"fold-{fold_index:02d}.pt"
        artifact = worker_root / "fold-artifacts" / f"fold-{fold_index:02d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{fold_index}".encode())
        artifact.write_bytes(f"artifact-{fold_index}".encode())
        evidence_sha = f"{fold_index + 1:x}" * 64
        fold_evidence_sha.append(evidence_sha)
        candidate = {
            "checkpoint_path": (
                f"/mnt/output/completion-00-setting-00/checkpoints/"
                f"fold-{fold_index:02d}.pt"
            ),
            "checkpoint_file_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
            "model_state_sha256": "1" * 64,
            "qualification_artifact_path": (
                f"/mnt/output/completion-00-setting-00/fold-artifacts/"
                f"fold-{fold_index:02d}.pt"
            ),
            "qualification_artifact_file_sha256": hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest(),
            "qualification_lineage_sha256": "2" * 64,
            "fold_evidence_sha256": evidence_sha,
            "qualification_source_array_sha256": "3" * 64,
            "qualification_residual_operator_root_sha256": "4" * 64,
            "fold_risk_state_sha256": "5" * 64,
        }
        unsigned_fold = {
            "schema": lifecycle.M03R_V12_FOLD_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "authorization_receipt_sha256": auth.receipt_sha256,
            "worker_plan_sha256": worker.receipt_sha256,
            "setting_index": 0,
            "setting_id": worker.setting_id,
            "fold_index": fold_index,
            "completed_updates": 64,
            "model_state_sha256": "6" * 64,
            "optimizer_state_sha256": "7" * 64,
            "paired_input_receipt_sha256": "8" * 64,
            "rank_step_receipt_sha256": ["9" * 64, "a" * 64],
            "training_source_array_sha256": "b" * 64,
            "training_residual_operator_root_sha256": "c" * 64,
            "horizon_candidates": {"3": candidate},
            "qualification_evaluated_only_after_checkpoint_publication": True,
            "economic_optimizer_updates": 0,
            "outer_2026_accessed": False,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        fold_receipt = {
            **unsigned_fold,
            "receipt_sha256": lifecycle._content_sha256(unsigned_fold),
        }
        fold_file_sha.append(
            write_json(
                worker_root
                / "receipts"
                / f"fold-{fold_index:02d}-terminal.json",
                fold_receipt,
            )
        )
    bootstrap = {
        "chronology_sha256": "d" * 64,
        "fold_lengths": [63] * 6,
        "bootstrap_seed": 17,
        "draw_sha256_by_block": ["e" * 64, "f" * 64, "0" * 64],
        "block_sessions": [10, 21, 30],
        "replicates": lifecycle.M03R_V12_PREDICTIVE_SPEC.bootstrap_replicates,
        "rule": lifecycle.M03R_V12_BOOTSTRAP_RULE,
        "protocol_sha256": lifecycle.M03R_V12_PROTOCOL_SHA256,
        "schema": lifecycle.M03R_V12_BOOTSTRAP_PLAN_SCHEMA,
    }
    bootstrap_sha = lifecycle._compact_sha256(bootstrap)
    qualification = M03RV12PredictiveQualification(
        setting_index=0,
        setting_id=worker.setting_id,
        horizon_sessions=3,
        fold_receipt_sha256=tuple(fold_evidence_sha),
        bootstrap_plan_sha256=bootstrap_sha,
        mean_rank_ic=0.0,
        positive_mean_ic_fold_count=0,
        positive_median_ic_fold_count=0,
        positive_date_fraction_fold_count=0,
        positive_spread_fold_count=0,
        annualized_gross_active_return=0.0,
        annualized_net_active_return_10bp=0.0,
        gross_active_return_lcb=0.0,
        net_active_return_10bp_lcb=0.0,
        top_bottom_spread_lcb=0.0,
        prediction_dispersion_gate_passed=False,
        prediction_target_dispersion_ratio_gate_passed=False,
        break_even_category="no-positive-break-even",
        break_even_one_way_cost_basis_points=None,
        median_projection_retention=0.0,
        minimum_fold_median_projection_retention=0.0,
        passed=False,
        economic_generation_may_be_minted=False,
    )
    qualification_row = {
        **asdict(qualification),
        "receipt_sha256": qualification.receipt_sha256,
    }
    unsigned_terminal = {
        "schema": lifecycle.M03R_V12_WORKER_TERMINAL_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": package_sha,
        "authorization_receipt_sha256": auth.receipt_sha256,
        "worker_plan_sha256": worker.receipt_sha256,
        "startup_file_sha256": startup_sha,
        "setting_index": 0,
        "setting_id": worker.setting_id,
        "fold_terminal_file_sha256": fold_file_sha,
        "bootstrap_plan": bootstrap,
        "bootstrap_plan_sha256": bootstrap_sha,
        "horizon_qualification": [qualification_row],
        "selected_horizon": None,
        "selected_qualification_sha256": None,
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
    terminal = {
        **unsigned_terminal,
        "receipt_sha256": lifecycle._content_sha256(unsigned_terminal),
    }
    terminal_sha = write_json(worker_root / "predictive-terminal.json", terminal)
    config = lifecycle.M03RV12AttachSupervisorConfig(
        job_name="qt-m03r-v12-predictive-a05",
        run_id="qt-m03r-v12-h3-predictive-s17-20260813-a05",
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
        expected_completions=completions,
        host_python_path="/usr/bin/python3",
        pythonpath=str(tmp_path / "source"),
    )
    validated_terminal_sha, evidence = lifecycle._validate_one_worker(
        config, completion
    )
    assert validated_terminal_sha == terminal_sha
    assert evidence["selected_horizon"] is None
    assert lifecycle.M03R_V12_EXECUTION_HORIZONS == (3,)

    original_read = lifecycle._read_bound_json

    def stale_inventory(path: Path, expected_sha256: str, label: str):
        value = original_read(path, expected_sha256, label)
        if path.name == "fold-00-terminal.json":
            unsigned = dict(value)
            del unsigned["receipt_sha256"]
            candidates = dict(unsigned["horizon_candidates"])
            candidates["21"] = candidates["3"]
            unsigned["horizon_candidates"] = candidates
            return {
                **unsigned,
                "receipt_sha256": lifecycle._content_sha256(unsigned),
            }
        return value

    monkeypatch.setattr(lifecycle, "_read_bound_json", stale_inventory)
    with pytest.raises(
        lifecycle.M03RV12SeadragonLifecycleError,
        match="fold horizon inventory drifted",
    ):
        lifecycle._validate_one_worker(config, completion)
