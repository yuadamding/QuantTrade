from __future__ import annotations

from datetime import UTC, datetime

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v13_fold import (
    M03RV13PanelEpisodeSchedule,
    render_m03r_v13_fold_geometries,
)
from rl_quant.training.top2000_m03r_v13_kubernetes import (
    M03RV13LiveEvidence,
    M03RV13TwoH100CapacityQualification,
    build_m03r_v13_live_evidence,
    render_m03r_v13_suspended_capacity_job,
    render_m03r_v13_suspended_predictive_job,
    render_m03r_v13_suspended_static_job,
)
from rl_quant.training.top2000_m03r_v13_package import (
    M03RV13ExecutionAuthorization,
    M03RV13PackageArtifacts,
    M03RV13PackagePlan,
    build_m03r_v13_package_plan,
)

NOW = datetime(2026, 8, 14, 18, 0, tzinfo=UTC)
PLAN_FILE_SHA = "7" * 64
AUTH_FILE_SHA = "8" * 64


def _package_and_authorization() -> tuple[
    M03RV13PackagePlan,
    M03RV13ExecutionAuthorization,
]:
    image = "f" * 64
    package = build_m03r_v13_package_plan(
        M03RV13PackageArtifacts(
            source_archive_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            dependency_lock_sha256="c" * 64,
            cache_artifact_sha256="d" * 64,
            cache_manifest_sha256="e" * 64,
            risk_artifact_sha256="1" * 64,
            risk_source_manifest_file_sha256="2" * 64,
            projector_manifest_file_sha256="3" * 64,
            projector_manifest_sha256="4" * 64,
            projector_binding_sha256="5" * 64,
            worker_source_sha256="6" * 64,
            initial_parameter_state_file_sha256="8" * 64,
            initial_parameter_state_sha256="9" * 64,
            initial_parameter_architecture_sha256="0" * 64,
            structural_preflight_file_sha256="a" * 64,
            structural_preflight_receipt_sha256="b" * 64,
            image_reference=f"registry/research@sha256:{image}",
            image_digest_sha256=image,
        ),
        M03RV13PanelEpisodeSchedule(
            protocol_common_data_sha256="0" * 64,
            cache_sha256="d" * 64,
            asset_axis_sha256="1" * 64,
            fold_geometry_sha256=tuple(
                row.receipt_sha256 for row in render_m03r_v13_fold_geometries(1001)
            ),
        ),
    )
    authorization = M03RV13ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=PLAN_FILE_SHA,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        source_manifest_sha256=package.artifacts.source_manifest_sha256,
        worker_source_sha256=package.artifacts.worker_source_sha256,
        image_reference=package.artifacts.image_reference,
    )
    return package, authorization


def _live(*, committed: int = 0) -> M03RV13LiveEvidence:
    return build_m03r_v13_live_evidence(
        observed_at_utc=NOW.isoformat(),
        rbac=M03RV7KubernetesRBACEvidence(
            jobs_get=True,
            jobs_list=True,
            jobs_create=True,
            jobs_patch=True,
            jobs_delete=True,
            pods_get=True,
            pods_list=True,
            pods_watch=True,
            pod_logs_get=True,
        ),
        protected_or_other_committed_h100_count=committed,
        live_schedulable_free_h100_count=max(0, 16 - committed),
        gpu_product_label_key=M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
        gpu_product_label_values=M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
        live_h100_cap_verified=True,
        gpu_selector_observed_live=True,
    )


def _template(name: str) -> M03RV7KubernetesTemplateConfig:
    return M03RV7KubernetesTemplateConfig(
        job_name=name,
        run_id="qt-m03r-v13-context-h3-s17-20260814-a03",
        service_account_name="default",
        pvc_claim_name="research-pvc",
        package_mount_path="/mnt/package",
        output_mount_path="/mnt/output",
    )


def _capacity(
    package: M03RV13PackagePlan,
    authorization: M03RV13ExecutionAuthorization,
) -> M03RV13TwoH100CapacityQualification:
    return M03RV13TwoH100CapacityQualification(
        static_gate_file_sha256="0" * 64,
        static_gate_receipt_sha256="9" * 64,
        terminal_file_sha256="a" * 64,
        terminal_receipt_sha256="b" * 64,
        startup_file_sha256="c" * 64,
        terminal_evidence_file_sha256="d" * 64,
        cleanup_receipt_file_sha256="e" * 64,
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        worker_plan_sha256=package.panel.workers[0].receipt_sha256,
        initial_parameter_state_file_sha256=(
            package.artifacts.initial_parameter_state_file_sha256
        ),
        initial_parameter_state_sha256=(
            package.artifacts.initial_parameter_state_sha256
        ),
        initial_parameter_architecture_sha256=(
            package.artifacts.initial_parameter_architecture_sha256
        ),
    )


def test_v13_static_capacity_and_predictive_shapes_are_bounded() -> None:
    package, authorization = _package_and_authorization()
    live = _live()
    static = render_m03r_v13_suspended_static_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=PLAN_FILE_SHA,
        authorization_file_sha256=AUTH_FILE_SHA,
        live=live,
        template=_template("qt-m03r-v13-static-a03"),
        now_utc=NOW,
    )
    capacity = render_m03r_v13_suspended_capacity_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=PLAN_FILE_SHA,
        authorization_file_sha256=AUTH_FILE_SHA,
        live=live,
        template=_template("qt-m03r-v13-cap-a03"),
        now_utc=NOW,
    )
    predictive = render_m03r_v13_suspended_predictive_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=PLAN_FILE_SHA,
        authorization_file_sha256=AUTH_FILE_SHA,
        live=live,
        capacity=_capacity(package, authorization),
        template=_template("qt-m03r-v13-pred-a03"),
        now_utc=NOW,
    )
    assert static.request_ceiling_h100 == 0
    assert capacity.completions == capacity.parallelism == 1
    assert capacity.request_ceiling_h100 == 2
    assert predictive.completions == predictive.parallelism == 2
    assert predictive.request_ceiling_h100 == 4
    assert predictive.manifest["spec"]["suspend"] is True
    assert predictive.economic_panel_authorized is False
    container = static.manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "0"
    assert {row["name"]: row["value"] for row in container["env"]}[
        "NVIDIA_VISIBLE_DEVICES"
    ] == "none"


def test_v13_live_cap_reduces_predictive_parallelism_without_changing_coverage() -> None:
    package, authorization = _package_and_authorization()
    rendered = render_m03r_v13_suspended_predictive_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=PLAN_FILE_SHA,
        authorization_file_sha256=AUTH_FILE_SHA,
        capacity=_capacity(package, authorization),
        live=_live(committed=14),
        template=_template("qt-m03r-v13-pred-a04"),
        now_utc=NOW,
    )
    assert rendered.completions == 2
    assert rendered.parallelism == 1
    assert rendered.request_ceiling_h100 == 2
