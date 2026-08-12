from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_SETTING_IDS,
)
from rl_quant.training import top2000_m03r_v11_seadragon_lifecycle as lifecycle
from rl_quant.training import top2000_m03r_v11_seadragon_operator as operator
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v11_package import (
    M03RV11ExecutionAuthorization,
    M03RV11PackageArtifacts,
    build_m03r_v11_package_plan,
    write_m03r_v11_execution_authorization,
    write_m03r_v11_package_plan,
)
from rl_quant.training.top2000_m03r_v11_schedule import M03RV11PanelEpisodeSchedule
from rl_quant.training.top2000_m03r_v11_selection import (
    M03RV11PredictiveQualification,
)


def _package_authorization(tmp_path: Path):
    artifacts = M03RV11PackageArtifacts(
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
        image_reference=f"example/research@sha256:{'d' * 64}",
        image_digest_sha256="d" * 64,
    )
    folds = render_top2000_m03r_v7_development_folds(1001)
    schedule = M03RV11PanelEpisodeSchedule(
        protocol_common_data_sha256="e" * 64,
        cache_sha256=artifacts.cache_artifact_sha256,
        fold_geometry_sha256=tuple(
            render_m03r_v10_fold_geometry(fold).receipt_sha256 for fold in folds
        ),
    )
    package = build_m03r_v11_package_plan(artifacts, schedule)
    package_path = tmp_path / "package-plan.json"
    package_file_sha = write_m03r_v11_package_plan(package_path, package)
    authorization = M03RV11ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=package_file_sha,
        source_archive_sha256=artifacts.source_archive_sha256,
        source_manifest_sha256=artifacts.source_manifest_sha256,
        worker_source_sha256=artifacts.worker_source_sha256,
        image_reference=artifacts.image_reference,
    )
    authorization_path = tmp_path / "execution-authorization.json"
    authorization_file_sha = write_m03r_v11_execution_authorization(
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


def test_v11_attach_config_binds_package_authorization_and_three_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    package, package_path, package_sha, auth, auth_path, auth_sha = (
        _package_authorization(tmp_path)
    )
    expected = tuple(
        lifecycle.M03RV11ExpectedCompletion(
            completion_index=index,
            setting_index=index,
            setting_id=worker.setting_id,
            worker_plan_sha256=worker.receipt_sha256,
        )
        for index, worker in enumerate(package.panel.workers)
    )
    config = lifecycle.M03RV11AttachSupervisorConfig(
        job_name="qt-m03r-v11-predictive-a02",
        run_id="qt-m03r-v11-predictive-s17-20260812-a02",
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
    with pytest.raises(lifecycle.M03RV11SeadragonLifecycleError, match="authorization"):
        lifecycle._load_package_authorization(
            lifecycle.M03RV11AttachSupervisorConfig(
                **{
                    **asdict(config),
                    "expected_completions": expected,
                    "execution_authorization_receipt_sha256": "3" * 64,
                }
            )
        )


def test_v11_lifecycle_reconstructs_typed_failed_qualification() -> None:
    fold_receipts = [str(index + 1) * 64 for index in range(6)]
    qualification = M03RV11PredictiveQualification(
        setting_index=0,
        setting_id=M03R_V11_SETTING_IDS[0],
        horizon_sessions=21,
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
        expected_horizon=21,
        expected_fold_receipts=fold_receipts,
        expected_bootstrap_plan_sha256="7" * 64,
    )
    value["economic_generation_may_be_minted"] = True
    value["receipt_sha256"] = lifecycle._compact_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    with pytest.raises(lifecycle.M03RV11SeadragonLifecycleError):
        lifecycle._validate_qualification(
            value,
            setting_index=0,
            expected_horizon=21,
            expected_fold_receipts=fold_receipts,
            expected_bootstrap_plan_sha256="7" * 64,
        )


def test_v11_lifecycle_uses_workflow_file_hash_for_terminal_receipts() -> None:
    unsigned = {
        "schema": "rl-quant.top2000-dev.m03r-v11-two-h100-capacity-terminal-v1",
        "world_size": 2,
    }
    value = {
        **unsigned,
        "receipt_sha256": lifecycle._content_sha256(unsigned),
    }
    assert lifecycle._receipt_payload(value, "capacity terminal") == unsigned
    value["receipt_sha256"] = lifecycle._compact_sha256(unsigned)
    with pytest.raises(lifecycle.M03RV11SeadragonLifecycleError, match="drifted"):
        lifecycle._receipt_payload(value, "capacity terminal")


def test_v11_workflow_and_lifecycle_terminal_hash_conventions_match() -> None:
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


def test_v11_operator_and_lifecycle_have_disjoint_mutation_surfaces() -> None:
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


def test_v11_create_config_requires_authorization_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operator, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    fields = {
        "mode": "capacity",
        "job_name": "qt-m03r-v11-cap-a02",
        "run_id": "qt-m03r-v11-predictive-s17-20260812-a02",
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
    config = operator.M03RV11CreateOperatorConfig(**fields)
    assert config.execution_authorization_receipt_sha256 == "4" * 64
    with pytest.raises(operator.M03RV11SeadragonOperatorError, match="SHA-256"):
        operator.M03RV11CreateOperatorConfig(
            **{**fields, "execution_authorization_receipt_sha256": "missing"}
        )
