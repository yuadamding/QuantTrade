from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_kubernetes import (
    M03RV16CapacityGateQualification,
    M03RV16KubernetesError,
    M03RV16StaticGateQualification,
    render_m03r_v16_suspended_capacity_job,
    render_m03r_v16_suspended_predictive_job,
    render_m03r_v16_suspended_static_job,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackageArtifacts,
    M03RV16PackagePlan,
    build_m03r_v16_package_plan,
    write_m03r_v16_execution_authorization,
    write_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_selection import (
    M03RV16PredictiveQualification,
    build_m03r_v16_bootstrap_plan,
)
from rl_quant.workflows.top2000_m03r_v16_aggregate import (
    aggregate_m03r_v16_panel,
)
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03R_V16_WORKER_TERMINAL_SCHEMA,
)


def _surfaces() -> tuple[
    M03RV16PackagePlan,
    M03RV16ExecutionAuthorization,
    str,
    str,
]:
    digest = "a" * 64
    artifacts = M03RV16PackageArtifacts(
        source_archive_sha256=digest,
        source_manifest_sha256=digest,
        dependency_lock_sha256=digest,
        cache_artifact_sha256=digest,
        cache_manifest_sha256=digest,
        asset_axis_sha256=digest,
        risk_artifact_sha256=digest,
        risk_source_manifest_file_sha256=digest,
        risk_source_receipt_sha256=digest,
        exposure_receipt_sha256=digest,
        projector_manifest_file_sha256=digest,
        projector_manifest_sha256=digest,
        projector_binding_sha256=digest,
        worker_source_sha256=digest,
        operator_source_sha256=digest,
        initial_parameter_state_file_sha256=digest,
        initial_parameter_state_sha256=digest,
        initial_parameter_architecture_sha256=digest,
        structural_slab_file_sha256=digest,
        structural_slab_receipt_sha256=digest,
        structural_action_operator_root_sha256=digest,
        structural_target_operator_root_sha256=digest,
        structural_target_root_sha256=(digest, digest, digest),
        image_reference=f"registry.invalid/quanttrade@sha256:{digest}",
        image_digest_sha256=digest,
    )
    schedule = M03RV16PanelSchedule(
        protocol_common_data_sha256=digest,
        cache_sha256=digest,
        asset_axis_sha256=digest,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v16_fold_geometries(1001)
        ),
    )
    package = build_m03r_v16_package_plan(artifacts, schedule)
    plan_file = "b" * 64
    authorization_file = "c" * 64
    authorization = M03RV16ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=plan_file,
        source_archive_sha256=artifacts.source_archive_sha256,
        source_manifest_sha256=artifacts.source_manifest_sha256,
        worker_source_sha256=artifacts.worker_source_sha256,
        structural_slab_file_sha256=artifacts.structural_slab_file_sha256,
        structural_slab_receipt_sha256=artifacts.structural_slab_receipt_sha256,
        image_reference=artifacts.image_reference,
    )
    authorization.validate(package)
    return package, authorization, plan_file, authorization_file


def _template(name: str) -> M03RV7KubernetesTemplateConfig:
    return M03RV7KubernetesTemplateConfig(
        job_name=name,
        run_id="m03r-v16-v5-local-contract",
        service_account_name="default",
        pvc_claim_name="research-pvc",
        package_mount_path="/mnt/package",
        output_mount_path="/mnt/output",
    )


def test_v16_jobs_are_suspended_and_gate_predictive_h100_panel() -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static_job = render_m03r_v16_suspended_static_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-static"),
    )
    assert static_job.maximum_gpu_requests == 0
    assert static_job.manifest["spec"]["suspend"] is True
    static = M03RV16StaticGateQualification(
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[attr-defined]
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        rendered_manifest_sha256=static_job.manifest_sha256,
        result_file_sha256="d" * 64,
        result_receipt_sha256="e" * 64,
        image_digest_sha256=package.artifacts.image_digest_sha256,  # type: ignore[attr-defined]
    )
    capacity_job = render_m03r_v16_suspended_capacity_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-capacity"),
        static=static,
    )
    assert capacity_job.maximum_gpu_requests == 2
    capacity = M03RV16CapacityGateQualification(
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[attr-defined]
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        static_gate_receipt_sha256=static.receipt_sha256,
        rendered_manifest_sha256=capacity_job.manifest_sha256,
        terminal_file_sha256="f" * 64,
        terminal_receipt_sha256="1" * 64,
        image_digest_sha256=package.artifacts.image_digest_sha256,  # type: ignore[attr-defined]
    )
    predictive = render_m03r_v16_suspended_predictive_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-predictive"),
        static=static,
        capacity=capacity,
    )
    assert predictive.completions == 3
    assert predictive.parallelism == 3
    assert predictive.maximum_gpu_requests == 6
    assert predictive.manifest["spec"]["suspend"] is True
    assert predictive.activation_authorized is False


def test_v16_predictive_job_rejects_missing_or_drifted_capacity() -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static_job = render_m03r_v16_suspended_static_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-static-two"),
    )
    static = M03RV16StaticGateQualification(
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[attr-defined]
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        rendered_manifest_sha256=static_job.manifest_sha256,
        result_file_sha256="d" * 64,
        result_receipt_sha256="e" * 64,
        image_digest_sha256=package.artifacts.image_digest_sha256,  # type: ignore[attr-defined]
    )
    capacity = M03RV16CapacityGateQualification(
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[attr-defined]
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        static_gate_receipt_sha256=static.receipt_sha256,
        rendered_manifest_sha256="f" * 64,
        terminal_file_sha256="1" * 64,
        terminal_receipt_sha256="2" * 64,
        image_digest_sha256=package.artifacts.image_digest_sha256,  # type: ignore[attr-defined]
    )
    with pytest.raises(M03RV16KubernetesError, match="capacity"):
        render_m03r_v16_suspended_predictive_job(
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
            package_plan_file_sha256=plan_file,
            authorization_file_sha256=authorization_file,
            template=_template("m03r-v16-predictive-two"),
            static=static,
            capacity=replace(capacity, passed=False),
        )


def _qualification(
    setting_index: int,
    bootstrap_sha256: str,
    schedule_sha256: str,
) -> M03RV16PredictiveQualification:
    primary = setting_index == 2
    return M03RV16PredictiveQualification(
        setting_index=setting_index,
        setting_id=(
            "V16-R0-h21-selection-control",
            "V16-R1-h30-selection-control",
            "V16-R2-hold30-prior-selection-primary",
        )[setting_index],
        fold_trace_sha256=tuple(f"{index:x}" * 64 for index in range(1, 6)),
        terminal_checkpoint_authority_sha256=tuple(
            f"{index:x}" * 64 for index in range(6, 11)
        ),
        qualified_score_authority_sha256=tuple(
            f"{index:x}" * 64 for index in range(11, 16)
        ),
        panel_schedule_sha256=schedule_sha256,
        bootstrap_plan_sha256=bootstrap_sha256,
        mean_projected_rank_ic=0.025,
        positive_mean_ic_fold_count=5,
        positive_spread_fold_count=5,
        annualized_gross_active_return=0.04,
        annualized_net_active_return_10bp=0.02,
        gross_active_lcb_by_block=(0.01, 0.008, 0.006),
        net_10bp_active_lcb_by_block=(0.005, 0.004, 0.003),
        spread_lcb_by_block=(0.001, 0.0008, 0.0006),
        break_even_category="finite-positive",
        break_even_one_way_cost_basis_points=12.0,
        absolute_policy_break_even_one_way_cost_basis_points=8.0,
        median_risk_projection_retention=0.9,
        minimum_fold_median_risk_projection_retention=0.8,
        median_weighted_cohort_age=15.0,
        gates_passed=True,
        primary_hypothesis_passed=primary,
        three_seed_confirmation_may_be_minted=primary,
    )


def test_v16_file_aggregate_joins_exact_three_worker_terminals(
    tmp_path: Path,
) -> None:
    package, authorization, _plan_file, _authorization_file = _surfaces()
    plans = tmp_path / "plans"
    plan_path = plans / "package-plan.json"
    plan_file = write_m03r_v16_package_plan(plan_path, package)
    authorization = replace(authorization, package_plan_file_sha256=plan_file)
    authorization_path = plans / "execution-authorization.json"
    authorization_file = write_m03r_v16_execution_authorization(
        authorization_path,
        authorization,
        package,
    )
    geometries = render_m03r_v16_fold_geometries(1001)
    decisions = tuple(
        torch.arange(
            row.qualification_origin_start_inclusive,
            row.qualification_origin_stop_exclusive,
            dtype=torch.int64,
        )
        for row in geometries
    )
    executions = tuple(
        torch.arange(
            row.qualification_origin_start_inclusive,
            row.qualification_origin_stop_exclusive + 29,
            dtype=torch.int64,
        )
        for row in geometries
    )
    bootstrap = build_m03r_v16_bootstrap_plan(decisions, executions)
    terminal_paths: list[Path] = []
    terminal_hashes: list[str] = []
    for index, worker in enumerate(package.panel.workers):
        qualification = _qualification(
            index,
            bootstrap.receipt_sha256,
            package.schedule.receipt_sha256,
        )
        unsigned = {
            "schema": M03R_V16_WORKER_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "package_plan_file_sha256": plan_file,
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "worker_plan_sha256": worker.receipt_sha256,
            "startup_file_sha256": "e" * 64,
            "setting_index": index,
            "setting_id": worker.setting_id,
            "fold_terminal_file_sha256": tuple("f" * 64 for _ in range(5)),
            "bootstrap_plan": asdict(bootstrap),
            "bootstrap_plan_sha256": bootstrap.receipt_sha256,
            "predictive_qualification": asdict(qualification),
            "predictive_qualification_sha256": qualification.receipt_sha256,
            "primary_hypothesis_passed": (
                qualification.primary_hypothesis_passed
            ),
            "three_seed_confirmation_may_be_minted": (
                qualification.three_seed_confirmation_may_be_minted
            ),
            "economic_generation_may_be_minted": False,
            "reinforcement_learning_authorized": False,
            "outer_2026_accessed": False,
            "world_size": 2,
            "gpus_per_worker": 2,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        receipt_sha = hashlib.sha256(
            json.dumps(
                unsigned,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        payload = {**unsigned, "receipt_sha256": receipt_sha}
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        path = tmp_path / f"terminal-{index}.json"
        path.write_bytes(encoded)
        terminal_paths.append(path)
        terminal_hashes.append(hashlib.sha256(encoded).hexdigest())
    aggregate = aggregate_m03r_v16_panel(
        package_plan_path=plan_path,
        package_plan_file_sha256=plan_file,
        execution_authorization_path=authorization_path,
        execution_authorization_file_sha256=authorization_file,
        worker_terminal_paths=(
            terminal_paths[0],
            terminal_paths[1],
            terminal_paths[2],
        ),
        worker_terminal_file_sha256=(
            terminal_hashes[0],
            terminal_hashes[1],
            terminal_hashes[2],
        ),
        output_root=tmp_path / "aggregate",
    )
    assert aggregate["primary_hypothesis_passed"] is True
    assert aggregate["next_research_action"] == "three-seed-predictive-confirmation"
    assert aggregate["economic_generation_may_be_minted"] is False
    assert aggregate["reinforcement_learning_authorized"] is False
