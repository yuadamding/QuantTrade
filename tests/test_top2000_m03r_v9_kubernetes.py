from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v9_kubernetes import (
    M03RV9KubernetesError,
    M03RV9TwoH100CapacityQualification,
    bind_m03r_v9_admitted_suspended_job,
    build_m03r_v9_live_evidence,
    render_m03r_v9_suspended_capacity_job,
    render_m03r_v9_suspended_predictive_job,
)
from rl_quant.training.top2000_m03r_v9_package import (
    M03RV9PackageArtifacts,
    build_m03r_v9_package_plan,
)

NOW = datetime(2026, 8, 11, 18, 0, tzinfo=UTC)


def _package():
    image = "f" * 64
    return build_m03r_v9_package_plan(
        artifacts=M03RV9PackageArtifacts(
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
            image_reference=f"registry/research@sha256:{image}",
            image_digest_sha256=image,
        )
    )


def _live(*, committed: int = 0):
    return build_m03r_v9_live_evidence(
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


def _template(name: str):
    return M03RV7KubernetesTemplateConfig(
        job_name=name,
        run_id="qt-m03r-v9-predictive-20260811-a01",
        service_account_name="default",
        pvc_claim_name="research-pvc",
        package_mount_path="/mnt/package",
        output_mount_path="/mnt/output",
    )


def _capacity(package):
    return M03RV9TwoH100CapacityQualification(
        terminal_file_sha256="7" * 64,
        terminal_receipt_sha256="8" * 64,
        startup_file_sha256="9" * 64,
        terminal_evidence_file_sha256="a" * 64,
        cleanup_receipt_file_sha256="b" * 64,
        package_plan_sha256=package.package_plan_sha256,
        worker_plan_sha256=package.panel.workers[0].receipt_sha256,
    )


def test_capacity_then_three_row_predictive_manifests_are_suspended() -> None:
    package = _package()
    capacity = render_m03r_v9_suspended_capacity_job(
        package=package,
        live=_live(),
        template=_template("qt-m03r-v9-cap-a01"),
        now_utc=NOW,
    )
    predictive = render_m03r_v9_suspended_predictive_job(
        package=package,
        capacity=_capacity(package),
        live=_live(),
        template=_template("qt-m03r-v9-pred-a01"),
        now_utc=NOW,
    )
    assert capacity.manifest["spec"]["suspend"] is True
    assert capacity.completions == capacity.parallelism == 1
    assert capacity.request_ceiling_h100 == 2
    assert predictive.manifest["spec"]["suspend"] is True
    assert predictive.completions == predictive.parallelism == 3
    assert predictive.request_ceiling_h100 == 6
    assert predictive.manifest["spec"]["backoffLimit"] == 0
    container = predictive.manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert "--startup-only" not in container["args"]
    assert "--max-restarts=0" in container["args"]
    assert predictive.economic_panel_authorized is False


def test_cap_can_queue_one_completion_without_duplicating_the_job() -> None:
    package = _package()
    predictive = render_m03r_v9_suspended_predictive_job(
        package=package,
        capacity=_capacity(package),
        live=_live(committed=12),
        template=_template("qt-m03r-v9-pred-a01"),
        now_utc=NOW,
    )
    assert predictive.completions == 3
    assert predictive.parallelism == 2
    assert predictive.request_ceiling_h100 == 4


def test_predictive_job_requires_exact_capacity_receipt() -> None:
    package = _package()
    with pytest.raises(M03RV9KubernetesError, match="does not bind"):
        render_m03r_v9_suspended_predictive_job(
            package=package,
            capacity=M03RV9TwoH100CapacityQualification(
                terminal_file_sha256="7" * 64,
                terminal_receipt_sha256="8" * 64,
                startup_file_sha256="9" * 64,
                terminal_evidence_file_sha256="a" * 64,
                cleanup_receipt_file_sha256="b" * 64,
                package_plan_sha256=package.package_plan_sha256,
                worker_plan_sha256="0" * 64,
            ),
            live=_live(),
            template=_template("qt-m03r-v9-pred-a01"),
            now_utc=NOW,
        )


def test_v9_admitted_binding_reuses_strict_shared_contract() -> None:
    package = _package()
    rendered = render_m03r_v9_suspended_capacity_job(
        package=package,
        live=_live(),
        template=_template("qt-m03r-v9-cap-a01"),
        now_utc=NOW,
    )
    job_uid = "capacity-job-uid"
    admitted = json.loads(json.dumps(rendered.manifest))
    admitted["metadata"].update(
        {"uid": job_uid, "resourceVersion": "41", "creationTimestamp": None}
    )
    selector = {"matchLabels": {"batch.kubernetes.io/controller-uid": job_uid}}
    admitted["spec"]["selector"] = selector
    template_metadata = admitted["spec"]["template"]["metadata"]
    template_metadata["creationTimestamp"] = None
    template_metadata["labels"].update(
        {
            "batch.kubernetes.io/controller-uid": job_uid,
            "batch.kubernetes.io/job-name": "qt-m03r-v9-cap-a01",
            "controller-uid": job_uid,
            "job-name": "qt-m03r-v9-cap-a01",
        }
    )
    second = json.loads(json.dumps(admitted))
    second["metadata"]["resourceVersion"] = "42"
    binding = bind_m03r_v9_admitted_suspended_job(
        rendered=rendered,
        first_read=admitted,
        second_read=second,
        attached_owned_pod_uids=(),
    )
    assert binding.job_uid == job_uid
    assert binding.first_resource_version == "41"
    assert binding.second_resource_version == "42"
    assert binding.suspended is True

    drifted = json.loads(json.dumps(second))
    drifted["spec"]["template"]["spec"]["runtimeClassName"] = "injected"
    with pytest.raises(M03RV9KubernetesError, match="rendered|admitted|unknown"):
        bind_m03r_v9_admitted_suspended_job(
            rendered=rendered,
            first_read=admitted,
            second_read=drifted,
            attached_owned_pod_uids=(),
        )
