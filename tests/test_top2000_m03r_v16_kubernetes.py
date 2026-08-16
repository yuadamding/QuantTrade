from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import rl_quant.training.top2000_m03r_v16_kubernetes as kubernetes_runtime
import rl_quant.training.top2000_m03r_v16_cohort_runtime as cohort_runtime

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes, file_sha256
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_SETTINGS,
)

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
    load_and_issue_m03r_v16_capacity_gate,
    load_and_issue_m03r_v16_static_gate,
    issue_m03r_v16_training_activation_from_gates,
    render_m03r_v16_suspended_capacity_job,
    render_m03r_v16_suspended_qualification_job,
    render_m03r_v16_suspended_static_job,
    render_m03r_v16_suspended_training_job,
)
from rl_quant.training.top2000_m03r_v16_capacity import (
    M03RV16CapacityRankEvidence,
    build_m03r_v16_capacity_terminal,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    issue_m03r_v16_qualification_activation,
    write_m03r_v16_qualification_activation,
)
from rl_quant.training.top2000_m03r_v16_static_contract import (
    M03R_V16_STATIC_RESULT_SCHEMA,
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
    M03RV16ReconciledFoldEvidence,
    build_m03r_v16_bootstrap_plan,
    qualify_m03r_v16_reconciled_evidence,
)
from rl_quant.training.top2000_m03r_v16_cohort_runtime import (
    M03RV16CohortTrace,
)
from rl_quant.workflows.top2000_m03r_v16_aggregate import (
    M03RV16AggregateError,
    aggregate_m03r_v16_panel,
)
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03R_V16_FOLD_TERMINAL_SCHEMA,
    M03R_V16_QUALIFICATION_ARTIFACT_SCHEMA,
    M03R_V16_WORKER_TERMINAL_SCHEMA,
    _write_immutable_json,
    _write_immutable_torch,
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
        run_id="m03r-v16-v7-local-contract",
        service_account_name="default",
        pvc_claim_name="research-pvc",
        package_mount_path="/mnt/package",
        output_mount_path="/mnt/output",
    )


def _static_gate(
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    manifest_sha256: str,
) -> M03RV16StaticGateQualification:
    return M03RV16StaticGateQualification(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        rendered_manifest_sha256=manifest_sha256,
        result_file_sha256="d" * 64,
        result_receipt_sha256="e" * 64,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        source_tree_root_sha256="9" * 64,
        _issuer=kubernetes_runtime._STATIC_GATE_ISSUER,
    )


def _capacity_gate(
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    static: M03RV16StaticGateQualification,
    manifest_sha256: str,
) -> M03RV16CapacityGateQualification:
    return M03RV16CapacityGateQualification(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        static_gate_receipt_sha256=static.receipt_sha256,
        rendered_manifest_sha256=manifest_sha256,
        terminal_file_sha256="f" * 64,
        terminal_receipt_sha256="1" * 64,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        source_tree_root_sha256=static.source_tree_root_sha256,
        _issuer=kubernetes_runtime._CAPACITY_GATE_ISSUER,
    )


def _capacity_rank(rank: int) -> M03RV16CapacityRankEvidence:
    return M03RV16CapacityRankEvidence(
        setting_index=2,
        distributed_rank=rank,
        distributed_world_size=2,
        cuda_device_name="NVIDIA H100 80GB HBM3",
        cuda_total_memory_bytes=80 * 1024**3,
        peak_allocated_bytes=40 * 1024**3 + rank,
        peak_reserved_bytes=50 * 1024**3 + rank,
        pre_validation_update_receipt_sha256=("1" if rank == 0 else "2") * 64,
        validation_batch_receipt_sha256=("3" if rank == 0 else "4") * 64,
        update_plan_sha256="5" * 64,
        batch_receipt_sha256=("6" if rank == 0 else "7") * 64,
        score_step_receipt_sha256=("8" if rank == 0 else "9") * 64,
        structural_slab_receipt_sha256="a" * 64,
        qualification_projection_receipt_sha256=(
            ("b" if rank == 0 else "c") * 64
        ),
        qualification_requested_active_one_way_mass=0.01,
        qualification_projected_active_one_way_mass=0.0025,
        qualification_requested_to_executed_retention=0.25,
        post_update_model_state_sha256="d" * 64,
        post_update_optimizer_state_sha256="e" * 64,
        episode_state_rows=345,
        global_origin_count=43,
        local_origin_count=22 if rank == 0 else 21,
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
    resources = static_job.manifest["spec"]["template"]["spec"]["containers"][0][
        "resources"
    ]
    assert "nvidia.com/gpu" not in resources["requests"]
    assert "nvidia.com/gpu" not in resources["limits"]
    static = _static_gate(package, authorization, static_job.manifest_sha256)
    capacity_job = render_m03r_v16_suspended_capacity_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-capacity"),
        static=static,
    )
    assert capacity_job.maximum_gpu_requests == 2
    capacity = _capacity_gate(
        package, authorization, static, capacity_job.manifest_sha256
    )
    activation = issue_m03r_v16_training_activation_from_gates(
        package=package,
        authorization=authorization,
        static=static,
        capacity=capacity,
    )
    predictive = render_m03r_v16_suspended_training_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-predictive"),
        static=static,
        capacity=capacity,
        training_activation=activation,
        training_activation_file_sha256="2" * 64,
    )
    assert predictive.completions == 3
    assert predictive.parallelism == 3
    assert predictive.maximum_gpu_requests == 6
    assert predictive.manifest["spec"]["suspend"] is True
    assert predictive.activation_authorized is False
    qualification_activation = issue_m03r_v16_qualification_activation(
        package=package,
        authorization=authorization,
        training_panel_receipt_sha256="6" * 64,
        training_terminal_file_sha256=("3" * 64, "4" * 64, "5" * 64),
        primary_training_adequacy_receipt_sha256=tuple(
            f"{index + 20:064x}" for index in range(5)
        ),
        source_tree_root_sha256=capacity.source_tree_root_sha256,
    )
    qualification_job = render_m03r_v16_suspended_qualification_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-qualification"),
        static=static,
        capacity=capacity,
        qualification_activation=qualification_activation,
        qualification_activation_file_sha256="8" * 64,
    )
    assert qualification_job.mode == "qualification"
    mounts = qualification_job.manifest["spec"]["template"]["spec"][
        "containers"
    ][0]["volumeMounts"]
    assert any(
        row.get("mountPath") == "/mnt/training" and row.get("readOnly") is True
        for row in mounts
    )


def test_v16_predictive_job_rejects_missing_or_drifted_capacity() -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static_job = render_m03r_v16_suspended_static_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-static-two"),
    )
    static = _static_gate(package, authorization, static_job.manifest_sha256)
    capacity = _capacity_gate(package, authorization, static, "f" * 64)
    with pytest.raises(M03RV16KubernetesError, match="capacity"):
        activation = issue_m03r_v16_training_activation_from_gates(
            package=package,
            authorization=authorization,
            static=static,
            capacity=capacity,
        )
        render_m03r_v16_suspended_training_job(
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
            package_plan_file_sha256=plan_file,
            authorization_file_sha256=authorization_file,
            template=_template("m03r-v16-predictive-two"),
            static=static,
            capacity=replace(capacity, passed=False),
            training_activation=activation,
            training_activation_file_sha256="2" * 64,
        )


def test_v16_gate_authorities_are_issued_from_exact_result_files(
    tmp_path: Path,
) -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static_job = render_m03r_v16_suspended_static_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-static-issued"),
    )
    source_tree_root = "f" * 64
    unsigned_static = {
        "schema": M03R_V16_STATIC_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "source_tree_root_sha256": source_tree_root,
        "gpu_mask": "none",
        "gpu_requests": 0,
        "gpu_limits": 0,
        "unmasked_visibility_claimed": False,
        "training_performed": False,
        "initial_state_strict_loaded_all_settings": True,
        "development_only": True,
    }
    static_result = {
        **unsigned_static,
        "receipt_sha256": semantic_sha256(unsigned_static),
    }
    static_path = tmp_path / "static-result.json"
    static_path.write_bytes(canonical_json_file_bytes(static_result))
    static = load_and_issue_m03r_v16_static_gate(
        static_path,
        expected_result_file_sha256=file_sha256(static_path),
        rendered=static_job,
        package=package,
        authorization=authorization,
    )
    static.validate_for(package, authorization)

    capacity_job = render_m03r_v16_suspended_capacity_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-capacity-issued"),
        static=static,
    )
    terminal = build_m03r_v16_capacity_terminal(
        (_capacity_rank(0), _capacity_rank(1))
    )
    capacity_result = {
        "schema": terminal.schema,
        "package_plan_sha256": package.package_plan_sha256,
        "authorization_receipt_sha256": authorization.receipt_sha256,
        "source_tree_root_sha256": source_tree_root,
        "capacity": asdict(terminal),
        "capacity_receipt_sha256": terminal.receipt_sha256,
        "scientific_training_performed": False,
        "disposable_optimizer_update_executed": True,
        "disposable_train_validate_train_executed": True,
        "scientific_checkpoint_published": False,
        "development_only": True,
    }
    capacity_path = tmp_path / "two-h100-capacity-terminal.json"
    capacity_path.write_bytes(canonical_json_file_bytes(capacity_result))
    capacity = load_and_issue_m03r_v16_capacity_gate(
        capacity_path,
        expected_terminal_file_sha256=file_sha256(capacity_path),
        rendered=capacity_job,
        package=package,
        authorization=authorization,
        static=static,
    )
    capacity.validate_for(package, authorization, static)

    with pytest.raises(TypeError):
        M03RV16StaticGateQualification(  # type: ignore[call-arg]
            package_plan_sha256=package.package_plan_sha256,
            execution_authorization_receipt_sha256=authorization.receipt_sha256,
            rendered_manifest_sha256=static_job.manifest_sha256,
            result_file_sha256=file_sha256(static_path),
            result_receipt_sha256=semantic_sha256(unsigned_static),
            image_digest_sha256=package.artifacts.image_digest_sha256,
            source_tree_root_sha256=source_tree_root,
        )


def _synthetic_evidence(
    setting_index: int,
    fold_index: int,
    decision_origins: torch.Tensor,
    schedule_sha256: str,
) -> M03RV16ReconciledFoldEvidence:
    decisions = M03R_V16_PREDICTIVE_SPEC.qualification_origins_per_fold
    steps = decisions + M03R_V16_PREDICTIVE_SPEC.cohort_no_new_decision_tail_sessions
    execution_origins = torch.arange(
        int(decision_origins[0]),
        int(decision_origins[0]) + steps,
        dtype=torch.int64,
    )
    score = torch.tensor((-2.0, -1.0, 1.0, 2.0), dtype=torch.float64).expand(
        decisions, -1
    ).clone()
    target = score * 0.001
    valid = torch.ones_like(score, dtype=torch.bool)
    positive = torch.full((steps,), 0.0003, dtype=torch.float64)
    zero = torch.zeros(steps, dtype=torch.float64)
    turnover = torch.full((steps,), 0.01, dtype=torch.float64)
    retention = torch.full((steps,), 0.9, dtype=torch.float64)
    age = torch.full((steps,), 15.0, dtype=torch.float64)
    terminal_authority = semantic_sha256(
        ["terminal", setting_index, fold_index]
    )
    score_authority = semantic_sha256(["score", setting_index, fold_index])
    costs = tuple(
        turnover * (basis_points / 10_000.0)
        for basis_points in M03R_V16_PREDICTIVE_SPEC.evaluation_cost_basis_points
    )
    zeros_by_cost = tuple(torch.zeros_like(row) for row in costs)
    provisional = M03RV16CohortTrace(
        setting_index=setting_index,
        setting_id=M03R_V16_SETTINGS[setting_index].setting_id,
        fold_index=fold_index,
        checkpoint_file_sha256=semantic_sha256(
            ["checkpoint-file", setting_index, fold_index]
        ),
        checkpoint_model_state_sha256=semantic_sha256(
            ["checkpoint-model", setting_index, fold_index]
        ),
        terminal_checkpoint_authority_sha256=terminal_authority,
        qualified_score_authority_sha256=score_authority,
        panel_schedule_sha256=schedule_sha256,
        qualification_batch_receipt_sha256=semantic_sha256(
            ["batch", setting_index, fold_index]
        ),
        asset_axis_sha256="a" * 64,
        action_valid_sha256=cohort_runtime._tensor_sha256(valid),
        diagnostic_valid_sha256=cohort_runtime._tensor_sha256(valid),
        risk_manifest_sha256="b" * 64,
        risk_state_sha256=semantic_sha256(["risk", fold_index]),
        decision_origin_indices=decision_origins,
        execution_origin_indices=execution_origins,
        policy_gross_returns=positive,
        benchmark_gross_returns=zero,
        policy_one_way_turnover=turnover,
        benchmark_one_way_turnover=zero,
        active_one_way_mass=turnover,
        cohort_entry_one_way_mass=turnover,
        signal_cohort_mass_reduction_after_execution=zero,
        weighted_mean_cohort_age=age,
        requested_to_executed_retention=retention,
        risk_repair_active_one_way_mass=zero,
        prior_risk_repair_unwind_one_way_mass=zero,
        risk_projection_request_to_execution_one_way_distance=zero,
        absolute_policy_cost_by_cost=costs,
        benchmark_cost_by_cost=zeros_by_cost,
        incremental_active_cost_by_cost=costs,
        net_policy_return_by_cost=tuple(positive - row for row in costs),
        net_benchmark_return_by_cost=zeros_by_cost,
        net_active_return_by_cost=tuple(positive - row for row in costs),
        terminal_liquidation_one_way_turnover=0.0,
        terminal_preliquidation_active_one_way_mass=0.0,
        array_sha256=(),
        trace_sha256="0" * 64,
    )
    trace = replace(
        provisional,
        array_sha256=tuple(
            cohort_runtime._tensor_sha256(row) for row in provisional.arrays
        ),
    )
    trace = replace(
        trace,
        trace_sha256=cohort_runtime._sha256(trace.unsigned_payload()),
    )
    evidence = M03RV16ReconciledFoldEvidence(
        trace=trace,
        executable_selection_mean=score,
        selection_target_economic=target,
        selection_valid=valid,
        terminal_checkpoint_authority_sha256=terminal_authority,
        qualified_score_authority_sha256=score_authority,
        panel_schedule_sha256=schedule_sha256,
    )
    evidence.validate()
    return evidence


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
    training_terminal_hashes = ("3" * 64, "4" * 64, "5" * 64)
    qualification_activation = issue_m03r_v16_qualification_activation(
        package=package,
        authorization=authorization,
        training_panel_receipt_sha256="6" * 64,
        training_terminal_file_sha256=training_terminal_hashes,
        primary_training_adequacy_receipt_sha256=tuple(
            f"{index + 20:064x}" for index in range(5)
        ),
        source_tree_root_sha256="7" * 64,
    )
    activation_path = plans / "qualification-activation.json"
    activation_file_sha = write_m03r_v16_qualification_activation(
        activation_path, qualification_activation
    )
    terminal_paths: list[Path] = []
    terminal_hashes: list[str] = []
    for index, worker in enumerate(package.panel.workers):
        worker_root = tmp_path / f"worker-{index}"
        evidence = tuple(
            _synthetic_evidence(
                index,
                fold_index,
                decisions[fold_index],
                package.schedule.receipt_sha256,
            )
            for fold_index in range(5)
        )
        qualification = qualify_m03r_v16_reconciled_evidence(evidence, bootstrap)
        fold_terminal_hashes: list[str] = []
        for fold_index, row in enumerate(evidence):
            artifact = {
                "schema": M03R_V16_QUALIFICATION_ARTIFACT_SCHEMA,
                "terminal_checkpoint_authority_sha256": (
                    row.terminal_checkpoint_authority_sha256
                ),
                "qualified_score_authority_sha256": (
                    row.qualified_score_authority_sha256
                ),
                "trace_unsigned_payload": row.trace.unsigned_payload(),
                "trace_arrays": row.trace.arrays,
                "decision_origin_indices": row.trace.decision_origin_indices,
                "executable_selection_mean": row.executable_selection_mean,
                "selection_target_economic": row.selection_target_economic,
                "selection_valid": row.selection_valid,
                "action_valid": row.selection_valid,
                "outer_2026_accessed": False,
                "economic_optimizer_updates": 0,
                "reinforcement_learning_updates": 0,
            }
            artifact_sha = _write_immutable_torch(
                worker_root
                / "fold-artifacts"
                / f"fold-{fold_index:02d}-qualification.pt",
                artifact,
            )
            unsigned_fold = {
                "schema": M03R_V16_FOLD_TERMINAL_SCHEMA,
                "package_plan_sha256": package.package_plan_sha256,
                "authorization_receipt_sha256": authorization.receipt_sha256,
                "qualification_activation_receipt_sha256": (
                    qualification_activation.receipt_sha256
                ),
                "training_terminal_file_sha256": training_terminal_hashes[index],
                "worker_plan_sha256": worker.receipt_sha256,
                "setting_index": index,
                "setting_id": worker.setting_id,
                "fold_index": fold_index,
                "terminal_checkpoint_authority_sha256": (
                    row.terminal_checkpoint_authority_sha256
                ),
                "qualified_score_authority_sha256": (
                    row.qualified_score_authority_sha256
                ),
                "panel_schedule_sha256": package.schedule.receipt_sha256,
                "qualification_artifact_file_sha256": artifact_sha,
                "qualification_trace_sha256": row.trace.trace_sha256,
                "qualification_after_strict_terminal_reload": True,
                "economic_optimizer_updates": 0,
                "reinforcement_learning_updates": 0,
                "outer_2026_accessed": False,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
            fold_terminal_hashes.append(
                _write_immutable_json(
                    worker_root
                    / "receipts"
                    / f"fold-{fold_index:02d}-terminal.json",
                    {
                        **unsigned_fold,
                        "receipt_sha256": semantic_sha256(unsigned_fold),
                    },
                )
            )
        unsigned = {
            "schema": M03R_V16_WORKER_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "package_plan_file_sha256": plan_file,
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "qualification_activation_receipt_sha256": (
                qualification_activation.receipt_sha256
            ),
            "training_terminal_file_sha256": training_terminal_hashes[index],
            "worker_plan_sha256": worker.receipt_sha256,
            "startup_file_sha256": "e" * 64,
            "setting_index": index,
            "setting_id": worker.setting_id,
            "fold_terminal_file_sha256": tuple(fold_terminal_hashes),
            "bootstrap_plan": asdict(bootstrap),
            "bootstrap_plan_sha256": bootstrap.receipt_sha256,
            "predictive_qualification": asdict(qualification),
            "predictive_qualification_sha256": qualification.receipt_sha256,
            "raw_predictive_gates_passed": (
                qualification.primary_hypothesis_passed
            ),
            "three_seed_confirmation_may_be_minted": False,
            "economic_generation_may_be_minted": False,
            "reinforcement_learning_authorized": False,
            "outer_2026_accessed": False,
            "world_size": 2,
            "gpus_per_worker": 2,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        payload = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
        path = worker_root / "predictive-terminal.json"
        file_sha = _write_immutable_json(path, payload)
        terminal_paths.append(path)
        terminal_hashes.append(file_sha)
    aggregate = aggregate_m03r_v16_panel(
        package_plan_path=plan_path,
        package_plan_file_sha256=plan_file,
        execution_authorization_path=authorization_path,
        execution_authorization_file_sha256=authorization_file,
        qualification_activation_path=activation_path,
        qualification_activation_file_sha256=activation_file_sha,
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

    terminal_paths[0].chmod(0o640)
    terminal_paths[0].write_bytes(terminal_paths[0].read_bytes() + b"\n")
    with pytest.raises(M03RV16AggregateError, match="hash drifted"):
        aggregate_m03r_v16_panel(
            package_plan_path=plan_path,
            package_plan_file_sha256=plan_file,
            execution_authorization_path=authorization_path,
            execution_authorization_file_sha256=authorization_file,
            qualification_activation_path=activation_path,
            qualification_activation_file_sha256=activation_file_sha,
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
            output_root=tmp_path / "aggregate-tampered",
        )
