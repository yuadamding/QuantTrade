from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v12_kubernetes import (
    M03RV12KubernetesError,
    M03RV12TwoH100CapacityQualification,
    build_m03r_v12_live_evidence,
    render_m03r_v12_suspended_capacity_job,
    render_m03r_v12_suspended_predictive_job,
    render_m03r_v12_suspended_static_job,
)
from rl_quant.training.top2000_m03r_v12_package import (
    M03RV12ExecutionAuthorization,
    M03RV12PackageArtifacts,
    build_m03r_v12_package_plan,
)
from rl_quant.training.top2000_m03r_v12_schedule import M03RV12PanelEpisodeSchedule
from rl_quant.training.top2000_m03r_v12_seadragon_lifecycle import (
    M03RV12SeadragonLifecycleError,
    _compact_sha256,
    _validate_static_gate_lineage,
)
from rl_quant.training.top2000_m03r_v12_static_gate import (
    M03RV12StaticGateError,
    bind_m03r_v12_static_admitted_suspended_job,
    validate_m03r_v12_static_actual_pod,
    validate_m03r_v12_static_log,
)
from rl_quant.workflows.top2000_m03r_v12_static_validate import (
    M03R_V12_STATIC_RESULT_SCHEMA,
)

NOW = datetime(2026, 8, 12, 18, 0, tzinfo=UTC)
PLAN_FILE_SHA = "7" * 64
AUTH_FILE_SHA = "8" * 64


def test_v12_capacity_lineage_requires_exact_static_gate_receipt() -> None:
    package, authorization = _package_and_authorization()
    value = {
        "actual_pod_proof_file_sha256": "1" * 64,
        "cleanup_receipt_file_sha256": "2" * 64,
        "created_binding_file_sha256": "3" * 64,
        "development_only": True,
        "economic_training_authorized": False,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "gpu_limits": 0,
        "gpu_requests": 0,
        "h100_capacity_evidence": False,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "outer_2026_access_authorized": False,
        "package_plan_sha256": package.package_plan_sha256,
        "passed": True,
        "promotion_eligible": False,
        "rendered_manifest_sha256": "4" * 64,
        "reportable": False,
        "schema": "rl-quant.top2000-dev.m03r-v12-static-gate-v1",
        "server_dry_run_file_sha256": "5" * 64,
        "source_archive_sha256": package.artifacts.source_archive_sha256,
        "static_log_file_sha256": "6" * 64,
        "terminal_evidence_file_sha256": "7" * 64,
        "training_performed": False,
        "unmasked_visibility_claimed": False,
        "visibility_mask": "none",
    }
    expected = _compact_sha256(value)
    _validate_static_gate_lineage(
        value,
        expected_receipt_sha256=expected,
        package_plan_sha256=package.package_plan_sha256,
        authorization_receipt_sha256=authorization.receipt_sha256,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
    )
    tampered = {**value, "training_performed": True}
    with pytest.raises(M03RV12SeadragonLifecycleError):
        _validate_static_gate_lineage(
            tampered,
            expected_receipt_sha256=_compact_sha256(tampered),
            package_plan_sha256=package.package_plan_sha256,
            authorization_receipt_sha256=authorization.receipt_sha256,
            source_archive_sha256=package.artifacts.source_archive_sha256,
            image_digest_sha256=package.artifacts.image_digest_sha256,
        )


def _package_and_authorization():
    image = "f" * 64
    folds = render_top2000_m03r_v7_development_folds(1001)
    package = build_m03r_v12_package_plan(
        M03RV12PackageArtifacts(
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
            structural_preflight_file_sha256="a" * 64,
            structural_preflight_receipt_sha256="b" * 64,
            image_reference=f"registry/research@sha256:{image}",
            image_digest_sha256=image,
        ),
        M03RV12PanelEpisodeSchedule(
            protocol_common_data_sha256="0" * 64,
            cache_sha256="d" * 64,
            fold_geometry_sha256=tuple(
                render_m03r_v10_fold_geometry(fold).receipt_sha256 for fold in folds
            ),
        ),
    )
    authorization = M03RV12ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=PLAN_FILE_SHA,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        source_manifest_sha256=package.artifacts.source_manifest_sha256,
        worker_source_sha256=package.artifacts.worker_source_sha256,
        image_reference=package.artifacts.image_reference,
    )
    return package, authorization


def _live(*, committed: int = 0):
    return build_m03r_v12_live_evidence(
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
        run_id="qt-m03r-v12-predictive-s17-20260812-a02",
        service_account_name="default",
        pvc_claim_name="research-pvc",
        package_mount_path="/mnt/package",
        output_mount_path="/mnt/output",
    )


def _capacity(package, authorization):
    return M03RV12TwoH100CapacityQualification(
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
    )


def test_v12_capacity_and_predictive_manifests_bind_exact_authorization() -> None:
    package, authorization = _package_and_authorization()
    common = {
        "package": package,
        "authorization": authorization,
        "package_plan_file_sha256": PLAN_FILE_SHA,
        "authorization_file_sha256": AUTH_FILE_SHA,
        "live": _live(),
        "now_utc": NOW,
    }
    capacity = render_m03r_v12_suspended_capacity_job(
        **common,
        template=_template("qt-m03r-v12-cap-a02"),
    )
    predictive = render_m03r_v12_suspended_predictive_job(
        **common,
        capacity=_capacity(package, authorization),
        template=_template("qt-m03r-v12-pred-a02"),
    )
    assert capacity.completions == capacity.parallelism == 1
    assert predictive.completions == predictive.parallelism == 3
    assert predictive.request_ceiling_h100 == 6
    args = predictive.manifest["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "--package-plan-file-sha256" in args
    assert "--execution-authorization-file-sha256" in args
    assert "--startup-only" not in args
    capacity_args = capacity.manifest["spec"]["template"]["spec"]["containers"][0][
        "args"
    ]
    assert "/mnt/output/capacity-sentinel" in capacity_args
    assert predictive.manifest["spec"]["suspend"] is True
    assert predictive.economic_panel_authorized is False


def _admitted_static(rendered, *, uid: str = "job-uid", dry: bool = False):
    value = copy.deepcopy(rendered.manifest)
    value["metadata"]["uid"] = uid
    value["metadata"]["resourceVersion"] = "17"
    spec = value["spec"]
    spec["selector"] = {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}}
    labels = dict(spec["template"]["metadata"]["labels"])
    spec["template"]["metadata"]["labels"] = labels
    labels.update(
        {
            "batch.kubernetes.io/controller-uid": uid,
            "batch.kubernetes.io/job-name": value["metadata"]["name"],
            "controller-uid": uid,
            "job-name": value["metadata"]["name"],
            "runai/queue": "yding4-yn-gpu-workload-queue",
        }
    )
    spec["template"]["metadata"]["creationTimestamp"] = None
    pod_spec = spec["template"]["spec"]
    pod_spec["schedulerName"] = "kai-scheduler"
    pod_spec["nodeSelector"] = {"gpu-type": "A100"}
    pod_spec["priorityClassName"] = "high-nonpreempting"
    container = pod_spec["containers"][0]
    for side in ("requests", "limits"):
        container["resources"][side]["cpu"] = "0"
        container["resources"][side]["memory"] = "0" if dry else "4Gi"
    return value


def test_v12_static_manifest_binds_zero_gpu_and_actual_pod_surface() -> None:
    package, authorization = _package_and_authorization()
    rendered = render_m03r_v12_suspended_static_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=PLAN_FILE_SHA,
        authorization_file_sha256=AUTH_FILE_SHA,
        live=_live(),
        template=_template("qt-m03r-v12-static-a03"),
        now_utc=NOW,
    )
    desired = rendered.manifest["spec"]["template"]["spec"]
    container = desired["containers"][0]
    assert rendered.request_ceiling_h100 == 0
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "0"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "0"
    assert {row["name"]: row["value"] for row in container["env"]}[
        "NVIDIA_VISIBLE_DEVICES"
    ] == "none"
    assert "nodeSelector" not in desired
    admitted = _admitted_static(rendered)
    binding = bind_m03r_v12_static_admitted_suspended_job(
        rendered=rendered,
        first_read=admitted,
        second_read=admitted,
        attached_owned_pod_uids=(),
    )
    pod_name = rendered.manifest["metadata"]["name"] + "-abcde"
    actual_spec = copy.deepcopy(admitted["spec"]["template"]["spec"])
    actual_spec.update(
        {
            "hostname": rendered.manifest["metadata"]["name"] + "-0",
            "nodeName": "bound-node",
            "preemptionPolicy": "Never",
            "priority": 1000000,
            "tolerations": [],
        }
    )
    actual_spec["containers"][0]["env"].append(
        {
            "name": "JOB_COMPLETION_INDEX",
            "valueFrom": {
                "fieldRef": {
                    "apiVersion": "v1",
                    "fieldPath": (
                        "metadata.labels['batch.kubernetes.io/job-completion-index']"
                    ),
                }
            },
        }
    )
    base_annotations = rendered.manifest["metadata"]["annotations"]
    pod = {
        "metadata": {
            "name": pod_name,
            "uid": "pod-uid",
            "labels": {
                **admitted["spec"]["template"]["metadata"]["labels"],
                "runai/queue": "yding4-yn-gpu-workload-queue",
                "batch.kubernetes.io/job-completion-index": "0",
            },
            "annotations": {
                **base_annotations,
                "batch.kubernetes.io/job-completion-index": "0",
                "cni.projectcalico.org/containerID": "a" * 64,
                "cni.projectcalico.org/podIP": "10.1.2.3/32",
                "cni.projectcalico.org/podIPs": "10.1.2.3/32",
                "pod-group-name": f"pg-{pod_name}-{binding.job_uid}",
                "received-resource-type": "Regular",
                "runai-job-id": binding.job_uid,
            },
            "ownerReferences": [
                {
                    "uid": binding.job_uid,
                    "name": binding.job_name,
                    "kind": "Job",
                    "controller": True,
                }
            ],
        },
        "spec": actual_spec,
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "name": "validator",
                    "imageID": (
                        "registry/research@sha256:"
                        + package.artifacts.image_digest_sha256
                    ),
                    "state": {"terminated": {"exitCode": 0}},
                }
            ],
        },
    }
    proof = validate_m03r_v12_static_actual_pod(
        pod=pod,
        terminal_job=admitted,
        rendered=rendered,
        job_uid=binding.job_uid,
    )
    assert proof["profile"]["gpu_requests"] == 0
    result = {
        "schema": M03R_V12_STATIC_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": PLAN_FILE_SHA,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "execution_authorization_file_sha256": AUTH_FILE_SHA,
        "source_archive_sha256": package.artifacts.source_archive_sha256,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "gpu_mask": "none",
        "gpu_requests": 0,
        "gpu_limits": 0,
        "unmasked_visibility_claimed": False,
        "output_empty": True,
        "container_started": True,
        "training_performed": False,
        "economic_training_authorized": False,
        "outer_2026_access_authorized": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    assert (
        validate_m03r_v12_static_log(
            (json.dumps(result, sort_keys=True) + "\n").encode(), rendered=rendered
        )
        == result
    )

    lifecycle = copy.deepcopy(admitted)
    lifecycle["spec"]["template"]["spec"]["containers"][0]["lifecycle"] = {
        "postStart": {"exec": {"command": ["sh", "-c", "touch /tmp/pwned"]}}
    }
    with pytest.raises(M03RV12StaticGateError):
        bind_m03r_v12_static_admitted_suspended_job(
            rendered=rendered,
            first_read=lifecycle,
            second_read=lifecycle,
            attached_owned_pod_uids=(),
        )


def test_v12_renderer_rejects_authorization_or_capacity_drift() -> None:
    package, authorization = _package_and_authorization()
    with pytest.raises(M03RV12KubernetesError, match="package-plan file"):
        render_m03r_v12_suspended_capacity_job(
            package=package,
            authorization=authorization,
            package_plan_file_sha256="0" * 64,
            authorization_file_sha256=AUTH_FILE_SHA,
            live=_live(),
            template=_template("qt-m03r-v12-cap-a02"),
            now_utc=NOW,
        )
    wrong = replace(
        _capacity(package, authorization),
        execution_authorization_receipt_sha256="0" * 64,
    )
    with pytest.raises(M03RV12KubernetesError, match="does not bind"):
        render_m03r_v12_suspended_predictive_job(
            package=package,
            authorization=authorization,
            package_plan_file_sha256=PLAN_FILE_SHA,
            authorization_file_sha256=AUTH_FILE_SHA,
            capacity=wrong,
            live=_live(),
            template=_template("qt-m03r-v12-pred-a02"),
            now_utc=NOW,
        )
