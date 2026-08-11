from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v8_pretraining_kubernetes import (
    M03RV8PretrainingKubernetesError,
    M03RV8PretrainingQualification,
    build_m03r_v8_pretraining_live_evidence,
    render_m03r_v8_suspended_pretraining_batch_job,
    render_m03r_v8_suspended_pretraining_qualification_job,
)
from rl_quant.training.top2000_m03r_v8_pretraining_package import (
    M03RV8PretrainingArtifactBindings,
    build_m03r_v8_pretraining_package_plan,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _package():
    image = "f" * 64
    return build_m03r_v8_pretraining_package_plan(
        artifacts=M03RV8PretrainingArtifactBindings(
            source_archive_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            dependency_lock_sha256="c" * 64,
            cache_artifact_sha256="d" * 64,
            cache_manifest_sha256="e" * 64,
            worker_source_sha256="1" * 64,
            image_reference=f"registry/research@sha256:{image}",
            image_digest_sha256=image,
        )
    )


def _live(*, committed: int = 0):
    return build_m03r_v8_pretraining_live_evidence(
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
        live_schedulable_free_h100_count=16 - committed,
        gpu_product_label_key=M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
        gpu_product_label_values=M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
        live_h100_cap_verified=True,
        gpu_selector_observed_live=True,
    )


def _template(job_name: str):
    return M03RV7KubernetesTemplateConfig(
        job_name=job_name,
        run_id="qt-m03r-v8-pretrain-20260810-a01",
        service_account_name="default",
        pvc_claim_name="research-pvc",
        package_mount_path="/mnt/package",
        output_mount_path="/mnt/output",
    )


def _qualification(package):
    return M03RV8PretrainingQualification(
        terminal_file_sha256="2" * 64,
        terminal_receipt_sha256="3" * 64,
        startup_receipt_sha256="4" * 64,
        setting_plan_sha256=package.plans[0].receipt_sha256,
    )


def test_qualification_and_full_jobs_are_suspended_and_phase_disjoint() -> None:
    package = _package()
    qualification = render_m03r_v8_suspended_pretraining_qualification_job(
        package=package,
        live_evidence=_live(),
        template=_template("qt-m03r-v8-pretrain-q-a01"),
        now_utc=NOW,
    )
    full = render_m03r_v8_suspended_pretraining_batch_job(
        package=package,
        qualification=_qualification(package),
        live_evidence=_live(),
        template=_template("qt-m03r-v8-pretrain-a01"),
        now_utc=NOW,
    )

    assert qualification.manifest["spec"]["suspend"] is True
    assert qualification.completions == qualification.parallelism == 1
    assert qualification.request_ceiling_h100 == 2
    assert full.completions == full.parallelism == 7
    assert full.request_ceiling_h100 == 14
    assert full.manifest["spec"]["backoffLimit"] == 0
    q_container = qualification.manifest["spec"]["template"]["spec"]["containers"][0]
    f_container = full.manifest["spec"]["template"]["spec"]["containers"][0]
    assert "--qualification-updates" in q_container["args"]
    assert "--qualification-updates" not in f_container["args"]
    q_output = q_container["volumeMounts"][1]["subPath"]
    f_output = f_container["volumeMounts"][1]["subPath"]
    assert q_output.endswith("/phases/pretraining-qualification")
    assert f_output.endswith("/phases/pretraining-full")
    assert q_output != f_output
    assert q_container["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert f_container["resources"]["limits"]["nvidia.com/gpu"] == "2"


def test_live_cap_reduces_parallelism_without_changing_completions() -> None:
    package = _package()
    rendered = render_m03r_v8_suspended_pretraining_batch_job(
        package=package,
        qualification=_qualification(package),
        live_evidence=_live(committed=10),
        template=_template("qt-m03r-v8-pretrain-a01"),
        now_utc=NOW,
    )
    assert rendered.completions == 7
    assert rendered.parallelism == 3
    assert rendered.request_ceiling_h100 == 6


def test_full_job_rejects_missing_or_wrong_qualification() -> None:
    package = _package()
    with pytest.raises(M03RV8PretrainingKubernetesError, match="does not bind"):
        render_m03r_v8_suspended_pretraining_batch_job(
            package=package,
            qualification=M03RV8PretrainingQualification(
                terminal_file_sha256="2" * 64,
                terminal_receipt_sha256="3" * 64,
                startup_receipt_sha256="4" * 64,
                setting_plan_sha256="5" * 64,
            ),
            live_evidence=_live(),
            template=_template("qt-m03r-v8-pretrain-a01"),
            now_utc=NOW,
        )


def test_render_evidence_cannot_claim_a_future_server_dry_run() -> None:
    evidence = _live()
    assert evidence.exact_manifest_server_dry_run_deferred is True
    assert "server_dry_run_passed" not in evidence.canonical_payload()
    rendered = render_m03r_v8_suspended_pretraining_qualification_job(
        package=_package(),
        live_evidence=evidence,
        template=_template("qt-m03r-v8-pretrain-q-a01"),
        now_utc=NOW,
    )
    assert rendered.live_evidence_sha256 == evidence.receipt_sha256


def test_qualification_rejects_a_fully_committed_h100_cap() -> None:
    with pytest.raises(M03RV8PretrainingKubernetesError, match="does not allow"):
        render_m03r_v8_suspended_pretraining_qualification_job(
            package=_package(),
            live_evidence=_live(committed=16),
            template=_template("qt-m03r-v8-pretrain-q-a01"),
            now_utc=NOW,
        )
