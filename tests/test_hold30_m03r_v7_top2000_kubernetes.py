"""Receipt and lifecycle tests for the development-only TOP2000 Indexed Job."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v7_schedule import (
    M03R_V7_ADMISSION_ORDER,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_POOL_NODE_SELECTOR,
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS,
    M03R_TOP2000_MULTI_GPU_TOLERATION,
    M03R_TOP2000_PRIORITY_CLASS_NAME,
    M03RV7AdmittedJobBinding,
    M03RV7FoldSeedReceiptRef,
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
    M03RV7LiveAdmissionEvidence,
    M03RV7RenderedSuspendedJob,
    M03RV7Top2000KubernetesError,
    bind_m03r_v7_top2000_admitted_suspended_job,
    build_m03r_v7_exact_cleanup_receipt,
    build_m03r_v7_exact_job_activation_request,
    build_m03r_v7_exact_job_cleanup_request,
    build_m03r_v7_index_completion_receipt,
    build_m03r_v7_indexed_batch_receipt,
    build_m03r_v7_live_admission_evidence,
    render_m03r_v7_top2000_suspended_indexed_job,
    render_m03r_v7_top2000_suspended_qualification_batch_job,
    render_m03r_v7_top2000_suspended_qualification_pilot_job,
)
from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03RV7Top2000ArtifactBindings,
    M03RV7Top2000PackageError,
    M03RV7Top2000QualifiedPackage,
    M03RV7Top2000RuntimeProfile,
    M03RV7Top2000VerifiedQualificationArtifact,
    build_m03r_v7_top2000_capacity_receipt,
    build_m03r_v7_top2000_package_plan,
    build_m03r_v7_top2000_worker_receipt_from_qualifications,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _artifacts(prefix: str = "base") -> M03RV7Top2000ArtifactBindings:
    image_digest = _digest(prefix + "-image")
    return M03RV7Top2000ArtifactBindings(
        source_archive_sha256=_digest(prefix + "-source-archive"),
        source_manifest_sha256=_digest(prefix + "-source-manifest"),
        dependency_lock_sha256=_digest(prefix + "-dependency-lock"),
        cache_artifact_sha256=_digest(prefix + "-cache-artifact"),
        cache_manifest_sha256=_digest(prefix + "-cache-manifest"),
        data_manifest_sha256=_digest(prefix + "-data-manifest"),
        execution_model_sha256=_digest(prefix + "-execution-model"),
        image_reference=f"registry.example/research/quanttrade@sha256:{image_digest}",
        image_digest_sha256=image_digest,
    )


def _qualified_package(
    prefix: str = "base",
) -> M03RV7Top2000QualifiedPackage:
    plan = build_m03r_v7_top2000_package_plan(
        artifacts=_artifacts(prefix),
        plan_artifact_path="/mnt/package/package-plan.json",
    )
    qualifications = tuple(
        _qualification(plan, completion_index)
        for completion_index in range(12)
    )
    worker = build_m03r_v7_top2000_worker_receipt_from_qualifications(
        plan=plan,
        qualifications=qualifications,
        worker_argv_prefix=(
            "/opt/quanttrade/bin/python",
            "-m",
            "rl_quant.workflows.top2000_m03r_v7_index_worker",
        ),
        worker_entrypoint_sha256=_digest(prefix + "-worker"),
        runtime_manifest_sha256=_digest(prefix + "-runtime"),
    )
    capacity = build_m03r_v7_top2000_capacity_receipt(
        plan=plan,
        worker=worker,
        qualifications=qualifications,
    )
    return M03RV7Top2000QualifiedPackage(
        plan=plan,
        worker_receipt=worker,
        capacity_receipt=capacity,
    )


def _qualification(
    plan: Any,
    completion_index: int,
) -> M03RV7Top2000VerifiedQualificationArtifact:
    row = plan.indices[completion_index]
    overlay = (
        (_digest(f"overlay-{completion_index}"),) * 2
        if row.development_setting_id == "A06-sharpe-overlay-top2000-dev-v1"
        else (None, None)
    )
    fields = {
        "completion_index": completion_index,
        "setting_index": row.setting_index,
        "setting_id": row.development_setting_id,
        "qualification_receipt_sha256": _digest(f"qual-{completion_index}"),
        "cell_receipt_sha256": _digest(f"cell-{completion_index}"),
        "execution_plan_binding_sha256": _digest(f"binding-{completion_index}"),
        "rank_model_state_sha256": (_digest(f"model-{completion_index}"),) * 2,
        "rank_alpha_optimizer_state_sha256": (
            _digest(f"optimizer-{completion_index}"),
        )
        * 2,
        "rank_overlay_optimizer_state_sha256": overlay,
        "rank_peak_allocated_bytes": (64 * 1024**3, 65 * 1024**3),
        "rank_peak_reserved_bytes": (70 * 1024**3, 71 * 1024**3),
        "rank_total_memory_bytes": (80 * 1024**3, 80 * 1024**3),
        "rank_elapsed_seconds": (1.0, 1.1),
        "gpu_names": ("NVIDIA H100 80GB HBM3",) * 2,
        "compute_capabilities": ((9, 0), (9, 0)),
        "runtime_identity_sha256": _digest(f"runtime-{completion_index}"),
        "qualification_steps": 4,
        "schema": "rl-quant.m03r-v7-top2000-verified-qualification-v1",
    }
    unsigned = M03RV7Top2000VerifiedQualificationArtifact.__new__(
        M03RV7Top2000VerifiedQualificationArtifact
    )
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    payload = unsigned.canonical_payload()
    evidence = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return M03RV7Top2000VerifiedQualificationArtifact(
        **fields,
        evidence_sha256=evidence,
    )
def _rbac(**overrides: bool) -> M03RV7KubernetesRBACEvidence:
    values = {
        "jobs_get": True,
        "jobs_list": True,
        "jobs_create": True,
        "jobs_patch": True,
        "jobs_delete": True,
        "pods_get": True,
        "pods_list": True,
        "pods_watch": True,
        "pod_logs_get": True,
    }
    values.update(overrides)
    return M03RV7KubernetesRBACEvidence(**values)


def _live_evidence(
    *,
    protected: int = 0,
    free: int = 16,
    rbac: M03RV7KubernetesRBACEvidence | None = None,
    selector_proven: bool = True,
) -> M03RV7LiveAdmissionEvidence:
    return build_m03r_v7_live_admission_evidence(
        observed_at_utc="2026-08-05T12:00:00+00:00",
        rbac=rbac or _rbac(),
        protected_or_other_committed_h100_count=protected,
        live_schedulable_free_h100_count=free,
        gpu_product_label_key=M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
        gpu_product_label_values=M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
        live_h100_cap_verified=True,
        gpu_selector_observed_live=selector_proven,
        indexed_job_server_dry_run_passed=True,
    )


def _template() -> M03RV7KubernetesTemplateConfig:
    return M03RV7KubernetesTemplateConfig(
        job_name="qt-m03r-v7-top2000-dev",
        run_id="m03r-v7-top2000-dev-001",
        service_account_name="default",
        pvc_claim_name="yding4-gpu-home",
        package_mount_path="/mnt/package",
        output_mount_path="/mnt/output",
    )


def _rendered() -> M03RV7RenderedSuspendedJob:
    return render_m03r_v7_top2000_suspended_indexed_job(
        package=_qualified_package(),
        live_evidence=_live_evidence(),
        template=_template(),
        now_utc=datetime(2026, 8, 5, 12, 1, tzinfo=UTC),
    )


def _admitted_reads(
    rendered: Any,
    *,
    uid: str = "uid-m03r-v7-001",
    first_resource_version: str = "1001",
    second_resource_version: str = "1002",
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = deepcopy(rendered.manifest)
    first["metadata"].update(
        {"uid": uid, "resourceVersion": first_resource_version}
    )
    first["spec"]["selector"] = {
        "matchLabels": {"batch.kubernetes.io/controller-uid": uid}
    }
    template_metadata = first["spec"]["template"]["metadata"]
    template_metadata["creationTimestamp"] = None
    template_metadata["labels"].update(
        {
            "batch.kubernetes.io/controller-uid": uid,
            "batch.kubernetes.io/job-name": first["metadata"]["name"],
            "controller-uid": uid,
            "job-name": first["metadata"]["name"],
        }
    )
    second = deepcopy(first)
    second["metadata"]["resourceVersion"] = second_resource_version
    return first, second


def _binding(rendered: M03RV7RenderedSuspendedJob) -> M03RV7AdmittedJobBinding:
    first, second = _admitted_reads(rendered)
    return bind_m03r_v7_top2000_admitted_suspended_job(
        rendered=rendered,
        first_read=first,
        second_read=second,
        attached_owned_pod_uids=(),
    )


def _cell_receipts(index: int) -> tuple[M03RV7FoldSeedReceiptRef, ...]:
    package = _qualified_package()
    plan = package.plan.indices[index]
    return tuple(
        M03RV7FoldSeedReceiptRef(
            fold_index=fold,
            seed=seed,
            receipt_sha256=_digest(f"index-{index}-fold-{fold}-seed-{seed}"),
        )
        for fold in plan.fold_indices
        for seed in plan.paired_seeds
    )


def test_package_separates_artifacts_and_maps_all_twelve_indices() -> None:
    plan = _qualified_package().plan
    payload = plan.canonical_payload()
    assert set(payload["artifacts"]) >= {
        "source_archive_sha256",
        "source_manifest_sha256",
        "dependency_lock_sha256",
        "cache_artifact_sha256",
        "cache_manifest_sha256",
        "image_reference",
        "image_digest_sha256",
    }
    assert tuple(value.completion_index for value in plan.indices) == tuple(range(12))
    assert tuple(value.setting_index for value in plan.indices) == M03R_V7_ADMISSION_ORDER
    assert all(value.fold_seed_cell_count == 30 for value in plan.indices)
    assert plan.runtime_profile == M03RV7Top2000RuntimeProfile()
    assert plan.source_pythonpath == "/mnt/package/source/src"
    assert payload["source_pythonpath"] == plan.source_pythonpath
    assert payload["runtime_profile"]["token_dim"] == 512
    assert payload["runtime_profile"]["max_origin_batch"] == 22
    assert not plan.promotion_eligible
    assert not plan.outer_evaluation_authorized


def test_runtime_profile_is_content_bound_and_rejects_unqualified_shapes() -> None:
    base = build_m03r_v7_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path="/mnt/package/package-plan.json",
    )
    smaller = build_m03r_v7_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path="/mnt/package/package-plan.json",
        runtime_profile=M03RV7Top2000RuntimeProfile(token_dim=448),
    )
    assert base.package_plan_sha256 != smaller.package_plan_sha256
    assert M03RV7Top2000RuntimeProfile(max_origin_batch=22).max_origin_batch == 22
    with pytest.raises(M03RV7Top2000PackageError, match="runtime profile"):
        M03RV7Top2000RuntimeProfile(token_dim=513)
    with pytest.raises(M03RV7Top2000PackageError, match="runtime profile"):
        M03RV7Top2000RuntimeProfile(max_origin_batch=21)


def test_package_fails_closed_without_worker_and_capacity_and_rejects_stale_surface() -> None:
    plan = build_m03r_v7_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path="/mnt/package/package-plan.json",
    )
    unresolved = M03RV7Top2000QualifiedPackage(plan=plan)
    assert unresolved.launch_blockers == (
        "executable-worker-qualification-receipt-missing",
        "matching-two-h100-capacity-receipt-missing",
    )
    with pytest.raises(M03RV7Top2000PackageError, match="remains blocked"):
        unresolved.require_launch_ready()

    qualified = _qualified_package()
    other = _qualified_package("other")
    assert qualified.worker_receipt is not None
    assert other.capacity_receipt is not None
    with pytest.raises(M03RV7Top2000PackageError, match="different execution surface"):
        M03RV7Top2000QualifiedPackage(
            plan=qualified.plan,
            worker_receipt=qualified.worker_receipt,
            capacity_receipt=other.capacity_receipt,
        )


@pytest.mark.parametrize(
    ("protected", "free", "expected"),
    ((0, 16, 8), (8, 16, 4), (0, 8, 8), (15, 16, 0)),
)
def test_parallelism_is_authorized_cap_derived_and_never_exceeds_eight(
    protected: int,
    free: int,
    expected: int,
) -> None:
    assert _live_evidence(protected=protected, free=free).allowed_parallelism == expected

    incomplete = _live_evidence(rbac=_rbac(jobs_patch=False))
    assert incomplete.allowed_parallelism == 0


def test_renderer_requires_qualified_surface_and_fresh_live_evidence() -> None:
    plan = build_m03r_v7_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path="/mnt/package/package-plan.json",
    )
    with pytest.raises(M03RV7Top2000KubernetesError, match="remains blocked"):
        render_m03r_v7_top2000_suspended_indexed_job(
            package=M03RV7Top2000QualifiedPackage(plan=plan),
            live_evidence=_live_evidence(),
            template=_template(),
            now_utc=datetime(2026, 8, 5, 12, 1, tzinfo=UTC),
        )


def test_unqualified_plan_renders_suspended_single_and_all_setting_pilots() -> None:
    plan = build_m03r_v7_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path="/mnt/package/package-plan.json",
    )
    now = datetime(2026, 8, 5, 12, 1, tzinfo=UTC)
    single = render_m03r_v7_top2000_suspended_qualification_pilot_job(
        plan=plan,
        completion_index=10,
        live_evidence=_live_evidence(),
        template=_template(),
        now_utc=now,
    )
    assert single.manifest["spec"]["suspend"] is True
    assert single.manifest["spec"]["completions"] == 1
    assert single.manifest["spec"]["activeDeadlineSeconds"] == 86400
    assert "backoffLimitPerIndex" not in single.manifest["spec"]
    assert "maxFailedIndexes" not in single.manifest["spec"]
    single_args = single.manifest["spec"]["template"]["spec"]["containers"][0][
        "args"
    ]
    assert single_args[-6:] == [
        "--completion-index",
        "10",
        "--qualification-only",
        "--qualification-steps",
        "4",
        "--qualification-restart-after-step1",
    ]

    batch = render_m03r_v7_top2000_suspended_qualification_batch_job(
        plan=plan,
        live_evidence=_live_evidence(),
        template=_template(),
        now_utc=now,
    )
    batch_spec = batch.manifest["spec"]
    assert batch_spec["completions"] == 12
    assert batch_spec["parallelism"] == 8
    assert batch_spec["activeDeadlineSeconds"] == 86400
    assert "backoffLimitPerIndex" not in batch_spec
    assert "maxFailedIndexes" not in batch_spec
    assert batch.parallelism * 2 == 16
    container = batch_spec["template"]["spec"]["containers"][0]
    assert "--completion-index" not in container["args"]
    assert any(row["name"] == "JOB_COMPLETION_INDEX" for row in container["env"])

    first, second = _admitted_reads(
        batch,
        uid="uid-m03r-v7-pilot",
        first_resource_version="2001",
        second_resource_version="2002",
    )
    binding = bind_m03r_v7_top2000_admitted_suspended_job(
        rendered=batch,
        first_read=first,
        second_read=second,
        attached_owned_pod_uids=(),
    )
    assert binding.suspended


def test_package_mount_is_read_only_and_output_mount_omits_read_only() -> None:
    plan = build_m03r_v7_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path="/mnt/package/package-plan.json",
    )
    pilot = render_m03r_v7_top2000_suspended_qualification_pilot_job(
        plan=plan,
        completion_index=0,
        live_evidence=_live_evidence(),
        template=_template(),
        now_utc=datetime(2026, 8, 5, 12, 1, tzinfo=UTC),
    )
    for rendered in (pilot, _rendered()):
        container = rendered.manifest["spec"]["template"]["spec"]["containers"][0]
        mounts = {row["mountPath"]: row for row in container["volumeMounts"]}
        assert mounts["/mnt/package"]["readOnly"] is True
        assert "readOnly" not in mounts["/mnt/output"]


def test_pilot_can_establish_runtime_selector_proof_but_final_requires_it() -> None:
    plan = build_m03r_v7_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path="/mnt/package/package-plan.json",
    )
    evidence = _live_evidence(selector_proven=False)
    pilot = render_m03r_v7_top2000_suspended_qualification_batch_job(
        plan=plan,
        live_evidence=evidence,
        template=_template(),
        now_utc=datetime(2026, 8, 5, 12, 1, tzinfo=UTC),
    )
    assert pilot.parallelism == 8
    with pytest.raises(M03RV7Top2000KubernetesError, match="selector proof"):
        render_m03r_v7_top2000_suspended_indexed_job(
            package=_qualified_package(),
            live_evidence=evidence,
            template=_template(),
            now_utc=datetime(2026, 8, 5, 12, 1, tzinfo=UTC),
        )
    with pytest.raises(M03RV7Top2000KubernetesError, match="stale"):
        render_m03r_v7_top2000_suspended_indexed_job(
            package=_qualified_package(),
            live_evidence=_live_evidence(),
            template=_template(),
            now_utc=datetime(2026, 8, 5, 13, 0, tzinfo=UTC),
        )


def test_suspended_indexed_manifest_uses_proven_two_h100_pool_profile() -> None:
    rendered = _rendered()
    manifest = rendered.manifest
    spec = manifest["spec"]
    pod_spec = spec["template"]["spec"]
    container = pod_spec["containers"][0]
    assert spec == {
        **spec,
        "suspend": True,
        "completionMode": "Indexed",
        "completions": 12,
        "parallelism": 8,
        "backoffLimit": 0,
    }
    assert "backoffLimitPerIndex" not in spec
    assert "maxFailedIndexes" not in spec
    assert spec["activeDeadlineSeconds"] == M03R_TOP2000_MAX_ACTIVE_DEADLINE_SECONDS
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    match = pod_spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"][0]
    assert match == {
        "key": M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
        "operator": "In",
        "values": list(M03R_TOP2000_H100_PRODUCT_LABEL_VALUES),
    }
    assert pod_spec["nodeSelector"] == M03R_TOP2000_H100_POOL_NODE_SELECTOR
    assert pod_spec["priorityClassName"] == M03R_TOP2000_PRIORITY_CLASS_NAME
    assert pod_spec["tolerations"] == [M03R_TOP2000_MULTI_GPU_TOLERATION]
    assert pod_spec["dnsPolicy"] == "ClusterFirst"
    assert pod_spec["terminationGracePeriodSeconds"] == 60
    assert not any("$(JOB_COMPLETION_INDEX)" in arg for arg in container["args"])
    assert any(item["name"] == "JOB_COMPLETION_INDEX" for item in container["env"])
    environment = {item["name"]: item.get("value") for item in container["env"]}
    assert environment["PYTHONPATH"] == "/mnt/package/source/src"
    assert pod_spec["automountServiceAccountToken"] is False
    assert pod_spec["serviceAccount"] == "default"
    assert pod_spec["serviceAccountName"] == "default"
    assert pod_spec["schedulerName"] == "kai-scheduler"
    assert pod_spec["securityContext"] == {
        "runAsNonRoot": True,
        "runAsUser": 307469,
        "runAsGroup": 600815,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    assert manifest["spec"]["template"]["metadata"]["labels"]["runai/queue"] == (
        "yding4-yn-gpu-workload-queue"
    )
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }
    assert container["terminationMessagePath"] == "/dev/termination-log"
    assert container["terminationMessagePolicy"] == "File"
    mounts = {row["name"]: row for row in container["volumeMounts"]}
    assert mounts["research-data"]["subPath"] == (
        "quant/training/runs/m03r-v7-top2000-dev-001"
    )
    assert mounts["dshm"]["mountPath"] == "/dev/shm"
    assert any(row["name"] == "dshm" for row in pod_spec["volumes"])
    rendered_text = json.dumps(manifest, sort_keys=True)
    assert "nodeName" not in rendered_text
    assert "kubernetes.io/hostname" not in rendered_text
    assert "hostPath" not in rendered_text
    assert rendered.activation_authorized is False


def test_admitted_binding_requires_two_stable_suspended_reads_and_zero_pods() -> None:
    rendered = _rendered()
    first, second = _admitted_reads(rendered)
    binding = bind_m03r_v7_top2000_admitted_suspended_job(
        rendered=rendered,
        first_read=first,
        second_read=second,
        attached_owned_pod_uids=(),
    )
    assert binding.job_uid == "uid-m03r-v7-001"
    assert binding.second_resource_version == "1002"
    assert binding.suspended

    with pytest.raises(M03RV7Top2000KubernetesError, match="already owns Pods"):
        bind_m03r_v7_top2000_admitted_suspended_job(
            rendered=rendered,
            first_read=first,
            second_read=second,
            attached_owned_pod_uids=("pod-uid",),
        )
    changed = deepcopy(second)
    changed["spec"]["suspend"] = False
    with pytest.raises(M03RV7Top2000KubernetesError, match="spec changed"):
        bind_m03r_v7_top2000_admitted_suspended_job(
            rendered=rendered,
            first_read=first,
            second_read=changed,
            attached_owned_pod_uids=(),
        )


def test_admitted_binding_rejects_unknown_metadata_and_pool_mutations() -> None:
    rendered = _rendered()
    first, second = _admitted_reads(rendered)

    extra_label_first = deepcopy(first)
    extra_label_second = deepcopy(second)
    for value in (extra_label_first, extra_label_second):
        value["spec"]["template"]["metadata"]["labels"]["unknown"] = "value"
    with pytest.raises(M03RV7Top2000KubernetesError, match="unknown mutation"):
        bind_m03r_v7_top2000_admitted_suspended_job(
            rendered=rendered,
            first_read=extra_label_first,
            second_read=extra_label_second,
            attached_owned_pod_uids=(),
        )

    wrong_pool_first = deepcopy(first)
    wrong_pool_second = deepcopy(second)
    for value in (wrong_pool_first, wrong_pool_second):
        value["spec"]["template"]["spec"]["nodeSelector"] = {
            "gpu-type": "A100"
        }
    with pytest.raises(M03RV7Top2000KubernetesError, match="Pod field"):
        bind_m03r_v7_top2000_admitted_suspended_job(
            rendered=rendered,
            first_read=wrong_pool_first,
            second_read=wrong_pool_second,
            attached_owned_pod_uids=(),
        )


def test_template_rejects_unproven_deadline_or_service_account() -> None:
    with pytest.raises(M03RV7Top2000KubernetesError, match="proven bounded"):
        replace(_template(), active_deadline_seconds=216001)
    with pytest.raises(M03RV7Top2000KubernetesError, match="service account default"):
        replace(_template(), service_account_name="other")


def test_activation_request_is_full_content_bound_json_patch() -> None:
    rendered = _rendered()
    binding = _binding(rendered)
    _, fresh = _admitted_reads(rendered)
    fresh["metadata"]["resourceVersion"] = "1003"
    request = build_m03r_v7_exact_job_activation_request(binding, fresh)

    spec = fresh["spec"]
    assert request.content_type == "application/json-patch+json"
    assert request.resource_version == "1003"
    assert request.json_patch == [
        {"op": "test", "path": "/metadata/uid", "value": binding.job_uid},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": "1003",
        },
        {
            "op": "test",
            "path": "/metadata/annotations/rl-quant~1run-id",
            "value": binding.run_id,
        },
        {"op": "test", "path": "/spec/suspend", "value": True},
        {
            "op": "test",
            "path": "/spec/parallelism",
            "value": binding.parallelism,
        },
        {"op": "test", "path": "/spec/selector", "value": spec["selector"]},
        {
            "op": "test",
            "path": "/spec/template/metadata",
            "value": spec["template"]["metadata"],
        },
        {
            "op": "test",
            "path": "/spec/template/spec",
            "value": spec["template"]["spec"],
        },
        {"op": "replace", "path": "/spec/suspend", "value": False},
    ]
    assert all(operation["op"] == "test" for operation in request.json_patch[:-1])
    assert request.json_patch[-1]["op"] == "replace"
    with pytest.raises(M03RV7Top2000KubernetesError, match="Patch hash mismatch"):
        replace(request, json_patch_sha256=_digest("tampered-patch"))


def test_activation_request_rejects_tamper_stale_read_and_uid_mismatch() -> None:
    rendered = _rendered()
    binding = _binding(rendered)
    _, fresh = _admitted_reads(rendered)
    fresh["metadata"]["resourceVersion"] = "1003"

    tampered = deepcopy(fresh)
    tampered["spec"]["template"]["spec"]["schedulerName"] = "tampered"
    with pytest.raises(M03RV7Top2000KubernetesError, match="spec does not match"):
        build_m03r_v7_exact_job_activation_request(binding, tampered)

    stale = deepcopy(fresh)
    stale["metadata"]["resourceVersion"] = binding.first_resource_version
    with pytest.raises(M03RV7Top2000KubernetesError, match="known-stale"):
        build_m03r_v7_exact_job_activation_request(binding, stale)

    wrong_uid = deepcopy(fresh)
    wrong_uid["metadata"]["uid"] = "replacement-uid"
    with pytest.raises(M03RV7Top2000KubernetesError, match="name/namespace/UID"):
        build_m03r_v7_exact_job_activation_request(binding, wrong_uid)

    wrong_run = deepcopy(fresh)
    wrong_run["metadata"]["annotations"]["rl-quant/run-id"] = "other-run"
    with pytest.raises(M03RV7Top2000KubernetesError, match="run-ID"):
        build_m03r_v7_exact_job_activation_request(binding, wrong_run)


def test_exact_cleanup_is_uid_and_resource_version_preconditioned() -> None:
    rendered = _rendered()
    binding = _binding(rendered)
    _, post_run = _admitted_reads(rendered)
    post_run["metadata"]["resourceVersion"] = "2001"
    post_run["spec"]["suspend"] = False
    request = build_m03r_v7_exact_job_cleanup_request(binding, post_run)
    assert request.delete_options == {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "propagationPolicy": "Foreground",
        "preconditions": {
            "uid": binding.job_uid,
            "resourceVersion": "2001",
        },
    }
    receipt = build_m03r_v7_exact_cleanup_receipt(
        request=request,
        first_job_absent=True,
        second_job_absent=True,
        first_owned_pod_uids=(),
        second_owned_pod_uids=(),
        verification_evidence_sha256=_digest("cleanup-evidence"),
    )
    assert receipt.receipt_sha256
    with pytest.raises(M03RV7Top2000KubernetesError, match="cleanup is incomplete"):
        build_m03r_v7_exact_cleanup_receipt(
            request=request,
            first_job_absent=True,
            second_job_absent=True,
            first_owned_pod_uids=(),
            second_owned_pod_uids=("remaining-pod",),
            verification_evidence_sha256=_digest("cleanup-evidence"),
        )


def test_cleanup_request_rejects_stale_and_uid_or_run_id_mismatch() -> None:
    rendered = _rendered()
    binding = _binding(rendered)
    _, post_run = _admitted_reads(rendered)
    post_run["metadata"]["resourceVersion"] = "2001"

    stale = deepcopy(post_run)
    stale["metadata"]["resourceVersion"] = binding.second_resource_version
    with pytest.raises(M03RV7Top2000KubernetesError, match="stale pre-run"):
        build_m03r_v7_exact_job_cleanup_request(binding, stale)

    wrong_uid = deepcopy(post_run)
    wrong_uid["metadata"]["uid"] = "replacement-uid"
    with pytest.raises(M03RV7Top2000KubernetesError, match="name/namespace/UID"):
        build_m03r_v7_exact_job_cleanup_request(binding, wrong_uid)

    wrong_run = deepcopy(post_run)
    wrong_run["metadata"]["annotations"]["rl-quant/run-id"] = "other-run"
    with pytest.raises(M03RV7Top2000KubernetesError, match="run-ID"):
        build_m03r_v7_exact_job_cleanup_request(binding, wrong_run)


def test_per_index_receipts_require_exact_twelve_by_thirty_coverage() -> None:
    package = _qualified_package()
    binding = _binding(_rendered())
    receipts = tuple(
        build_m03r_v7_index_completion_receipt(
            package=package,
            binding=binding,
            completion_index=index,
            fold_seed_receipts=_cell_receipts(index),
            output_manifest_sha256=_digest(f"output-{index}"),
        )
        for index in range(12)
    )
    batch = build_m03r_v7_indexed_batch_receipt(
        package=package,
        binding=binding,
        index_receipts=receipts,
    )
    assert batch.all_twelve_complete
    assert tuple(value.completion_index for value in batch.index_receipts) == tuple(
        range(12)
    )
    assert all(len(value.fold_seed_receipts) == 30 for value in receipts)
    assert not batch.promotion_eligible

    with pytest.raises(M03RV7Top2000KubernetesError, match="all twelve"):
        build_m03r_v7_indexed_batch_receipt(
            package=package,
            binding=binding,
            index_receipts=receipts[:-1],
        )
    bad_cells = _cell_receipts(0)[1:] + _cell_receipts(0)[:1]
    with pytest.raises(M03RV7Top2000KubernetesError, match="exact admitted"):
        build_m03r_v7_index_completion_receipt(
            package=package,
            binding=binding,
            completion_index=0,
            fold_seed_receipts=bad_cells,
            output_manifest_sha256=_digest("bad-output"),
        )


def test_capacity_receipt_rejects_legacy_or_underfilled_hbm_profile() -> None:
    package = _qualified_package()
    assert package.worker_receipt is not None
    with pytest.raises(TypeError, match="unexpected keyword"):
        build_m03r_v7_top2000_capacity_receipt(
            plan=package.plan,
            worker=package.worker_receipt,
            rank_peak_hbm_gib=(20.0, 20.0),
            measurement_artifact_sha256=_digest("legacy-small-profile"),
        )  # type: ignore[call-arg]
    assert package.capacity_receipt is not None
    with pytest.raises(M03RV7Top2000PackageError, match="hash mismatch"):
        replace(
            package.capacity_receipt,
            execution_surface_sha256=_digest("legacy-surface"),
        )
