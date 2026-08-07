"""Seed-17 Kubernetes rendering and exact six-fold receipt coverage."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03R_TOP2000_INDEX_RECEIPT_SCHEMA,
    M03RV7FoldSeedReceiptRef,
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
    bind_m03r_v7_top2000_admitted_suspended_job,
    build_m03r_v7_live_admission_evidence,
)
from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03RV7Top2000ArtifactBindings,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_kubernetes import (
    M03R_SEED17_TOP2000_INDEX_RECEIPT_SCHEMA,
    M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX,
    M03RV7Seed17FoldReceiptRef,
    M03RV7Seed17KubernetesError,
    M03RV7Seed17QualifiedPackage,
    build_m03r_v7_seed17_capacity_receipt,
    build_m03r_v7_seed17_execution_qualification,
    build_m03r_v7_seed17_index_completion_receipt,
    build_m03r_v7_seed17_indexed_batch_receipt,
    build_m03r_v7_seed17_qualification_artifact_ref,
    render_m03r_v7_seed17_top2000_suspended_indexed_job,
    render_m03r_v7_seed17_top2000_suspended_qualification_batch_job,
    render_m03r_v7_seed17_top2000_suspended_validation_sentinel_job,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_package import (
    M03RV7Seed17PackagePlan,
    build_m03r_v7_seed17_top2000_package_plan,
)
from rl_quant.workflows.top2000_m03r_v7_seed17_operator import (
    Top2000M03RV7Seed17OperatorError,
)
from rl_quant.workflows.top2000_m03r_v7_seed17_operator import (
    main as operator_main,
)

_PACKAGE_MOUNT = "/" + "mnt/package"
_OUTPUT_MOUNT = "/" + "mnt/output"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _artifacts() -> M03RV7Top2000ArtifactBindings:
    image = _digest("seed17-image")
    return M03RV7Top2000ArtifactBindings(
        source_archive_sha256=_digest("seed17-source-archive"),
        source_manifest_sha256=_digest("seed17-source-manifest"),
        dependency_lock_sha256=_digest("seed17-dependency-lock"),
        cache_artifact_sha256=_digest("seed17-cache"),
        cache_manifest_sha256=_digest("seed17-cache-manifest"),
        data_manifest_sha256=_digest("seed17-data-manifest"),
        execution_model_sha256=_digest("seed17-execution-model"),
        image_reference=f"registry.example/quanttrade@sha256:{image}",
        image_digest_sha256=image,
    )


def _package() -> M03RV7Seed17QualifiedPackage:
    plan = build_m03r_v7_seed17_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path=f"{_PACKAGE_MOUNT}/package-plan.json",
        benchmark_preflight_sha256=_digest("seed17-benchmark-preflight"),
    )
    runtime_manifest = _digest("seed17-runtime-manifest")
    artifacts = tuple(
        build_m03r_v7_seed17_qualification_artifact_ref(
            plan=plan,
            completion_index=index,
            runtime_manifest_sha256=runtime_manifest,
            qualification_receipt_sha256=_digest(f"qualification-{index}"),
            validation_receipt_sha256=_digest(f"qualified-validation-{index}"),
            fold_execution_receipt_sha256=_digest(f"qualified-execution-{index}"),
        )
        for index in range(12)
    )
    sentinel = build_m03r_v7_seed17_qualification_artifact_ref(
        plan=plan,
        completion_index=3,
        runtime_manifest_sha256=runtime_manifest,
        qualification_receipt_sha256=_digest("sentinel-qualification"),
        validation_receipt_sha256=_digest("sentinel-validation"),
        fold_execution_receipt_sha256=_digest("sentinel-fold-execution"),
    )
    capacity = build_m03r_v7_seed17_capacity_receipt(
        plan=plan,
        worker_entrypoint_sha256=_digest("seed17-worker-entrypoint"),
        runtime_manifest_sha256=runtime_manifest,
        sentinel=sentinel,
        all_setting_qualifications=artifacts,
    )
    qualification = build_m03r_v7_seed17_execution_qualification(
        plan=plan,
        capacity_receipt=capacity,
    )
    return M03RV7Seed17QualifiedPackage(
        plan=plan,
        qualification=qualification,
    )


def _live_evidence():
    return build_m03r_v7_live_admission_evidence(
        observed_at_utc="2026-08-07T12:00:00+00:00",
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
        protected_or_other_committed_h100_count=0,
        live_schedulable_free_h100_count=16,
        gpu_product_label_key=M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
        gpu_product_label_values=M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
        live_h100_cap_verified=True,
        gpu_selector_observed_live=True,
        indexed_job_server_dry_run_passed=True,
    )


def _template() -> M03RV7KubernetesTemplateConfig:
    return M03RV7KubernetesTemplateConfig(
        job_name="qt-m03r-v7-seed17-dev",
        run_id="m03r-v7-seed17-dev-001",
        service_account_name="default",
        pvc_claim_name="yding4-gpu-home",
        package_mount_path=_PACKAGE_MOUNT,
        output_mount_path=_OUTPUT_MOUNT,
    )


def _rendered():
    return render_m03r_v7_seed17_top2000_suspended_indexed_job(
        package=_package(),
        live_evidence=_live_evidence(),
        template=_template(),
        now_utc=datetime(2026, 8, 7, 12, 1, tzinfo=UTC),
    )


def _binding():
    rendered = _rendered()
    first = deepcopy(rendered.manifest)
    first["metadata"].update({"uid": "seed17-job-uid", "resourceVersion": "1"})
    first["spec"]["selector"] = {
        "matchLabels": {"batch.kubernetes.io/controller-uid": "seed17-job-uid"}
    }
    metadata = first["spec"]["template"]["metadata"]
    metadata["creationTimestamp"] = None
    metadata["labels"].update(
        {
            "batch.kubernetes.io/controller-uid": "seed17-job-uid",
            "batch.kubernetes.io/job-name": first["metadata"]["name"],
            "controller-uid": "seed17-job-uid",
            "job-name": first["metadata"]["name"],
        }
    )
    second = deepcopy(first)
    second["metadata"]["resourceVersion"] = "2"
    return bind_m03r_v7_top2000_admitted_suspended_job(
        rendered=rendered,
        first_read=first,
        second_read=second,
        attached_owned_pod_uids=(),
    )


def _fold_receipts(completion_index: int):
    return tuple(
        M03RV7Seed17FoldReceiptRef(
            fold_index=fold,
            seed=17,
            completion_receipt_sha256=_digest(
                f"completion-{completion_index}-fold-{fold}"
            ),
            validation_receipt_sha256=_digest(
                f"validation-{completion_index}-fold-{fold}"
            ),
            fold_execution_receipt_sha256=_digest(
                f"execution-{completion_index}-fold-{fold}"
            ),
        )
        for fold in range(6)
    )


def test_seed17_renderer_is_disjoint_and_uses_exact_two_h100_worker() -> None:
    rendered = _rendered()
    manifest = rendered.manifest
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert manifest["spec"]["suspend"] is True
    assert manifest["spec"]["completions"] == 12
    assert manifest["spec"]["parallelism"] == 8
    assert container["command"] == [M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX[0]]
    argv_tail = M03R_SEED17_TOP2000_WORKER_ARGV_PREFIX[1:]
    assert container["args"][: len(argv_tail)] == list(argv_tail)
    assert "rl_quant.workflows.top2000_m03r_v7_seed17_dev" in container["args"]
    assert "--validation-sentinel" not in container["args"]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    annotations = manifest["metadata"]["annotations"]
    assert annotations["rl-quant/paired-seeds"] == "17"
    assert annotations["rl-quant/fold-count"] == "6"
    assert annotations["rl-quant/five-seed-ensemble-eligible"] == "false"
    volumes = manifest["spec"]["template"]["spec"]["volumes"]
    assert [row["name"] for row in volumes].count("tmp") == 1

    assert M03R_SEED17_TOP2000_INDEX_RECEIPT_SCHEMA != (
        M03R_TOP2000_INDEX_RECEIPT_SCHEMA
    )
    legacy_ref = M03RV7FoldSeedReceiptRef(
        fold_index=0,
        seed=101,
        receipt_sha256=_digest("legacy-cell"),
    )
    assert legacy_ref.seed == 101


def test_seed17_validation_renderers_cross_real_fold_execution_boundary() -> None:
    plan = _package().plan
    now = datetime(2026, 8, 7, 12, 1, tzinfo=UTC)
    sentinel = render_m03r_v7_seed17_top2000_suspended_validation_sentinel_job(
        plan=plan,
        completion_index=3,
        live_evidence=_live_evidence(),
        template=_template(),
        now_utc=now,
    )
    sentinel_container = sentinel.manifest["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert sentinel.completions == 1
    assert sentinel.parallelism == 1
    assert sentinel.completion_index == 3
    assert sentinel_container["args"][-3:] == [
        "--completion-index",
        "3",
        "--validation-sentinel",
    ]
    assert sentinel_container["env"][0] == {
        "name": "JOB_COMPLETION_INDEX",
        "value": "3",
    }
    with pytest.raises(M03RV7Seed17KubernetesError, match="completion index 3"):
        render_m03r_v7_seed17_top2000_suspended_validation_sentinel_job(
            plan=plan,
            completion_index=2,
            live_evidence=_live_evidence(),
            template=_template(),
            now_utc=now,
        )

    qualification = (
        render_m03r_v7_seed17_top2000_suspended_qualification_batch_job(
            plan=plan,
            live_evidence=_live_evidence(),
            template=_template(),
            now_utc=now,
        )
    )
    container = qualification.manifest["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert qualification.completions == 12
    assert qualification.parallelism == 8
    assert qualification.completion_index is None
    assert container["args"][-1] == "--validation-sentinel"
    assert container["env"][0]["valueFrom"]["fieldRef"]["apiVersion"] == "v1"


def test_seed17_capacity_is_typed_all_setting_h100_evidence() -> None:
    package = _package()
    capacity = package.qualification.capacity_receipt
    assert capacity.sentinel_completion_index == 3
    assert len(capacity.all_setting_qualifications) == 12
    assert tuple(
        row.completion_index for row in capacity.all_setting_qualifications
    ) == tuple(range(12))
    assert all(row.world_size == 2 for row in capacity.all_setting_qualifications)
    assert all(
        row.rank_gpu_names == ("NVIDIA H100 80GB HBM3",) * 2
        for row in capacity.all_setting_qualifications
    )
    with pytest.raises(M03RV7Seed17KubernetesError, match="two H100"):
        replace(
            capacity.all_setting_qualifications[0],
            rank_gpu_names=("NVIDIA A100-SXM4-40GB",) * 2,
        )
    with pytest.raises(M03RV7Seed17KubernetesError, match="all 12"):
        replace(
            capacity,
            all_setting_qualifications=capacity.all_setting_qualifications[:-1],
        )


def test_seed17_index_receipt_requires_exactly_six_fold_executions() -> None:
    package = _package()
    binding = _binding()
    receipt = build_m03r_v7_seed17_index_completion_receipt(
        package=package,
        binding=binding,
        completion_index=0,
        fold_receipts=_fold_receipts(0),
        output_manifest_sha256=_digest("output-manifest-0"),
    )
    assert len(receipt.fold_receipts) == 6
    assert tuple((row.fold_index, row.seed) for row in receipt.fold_receipts) == (
        (0, 17),
        (1, 17),
        (2, 17),
        (3, 17),
        (4, 17),
        (5, 17),
    )
    assert receipt.one_member_fold_execution
    assert not receipt.five_seed_ensemble_eligible

    with pytest.raises(M03RV7Seed17KubernetesError, match="six ordered"):
        replace(receipt, fold_receipts=receipt.fold_receipts[:-1])
    with pytest.raises(M03RV7Seed17KubernetesError, match="seed 17"):
        M03RV7Seed17FoldReceiptRef(
            fold_index=0,
            seed=101,
            completion_receipt_sha256=_digest("bad-completion"),
            validation_receipt_sha256=_digest("bad-validation"),
            fold_execution_receipt_sha256=_digest("bad-execution"),
        )


def test_seed17_batch_receipt_binds_twelve_settings_and_72_folds() -> None:
    package = _package()
    binding = _binding()
    indices = tuple(
        build_m03r_v7_seed17_index_completion_receipt(
            package=package,
            binding=binding,
            completion_index=index,
            fold_receipts=_fold_receipts(index),
            output_manifest_sha256=_digest(f"output-manifest-{index}"),
        )
        for index in range(12)
    )
    batch = build_m03r_v7_seed17_indexed_batch_receipt(
        package=package,
        binding=binding,
        index_receipts=tuple(reversed(indices)),
    )
    assert tuple(row.completion_index for row in batch.index_receipts) == tuple(
        range(12)
    )
    assert batch.total_fold_executions == 72
    assert batch.all_twelve_complete
    assert batch.one_member_fold_execution
    assert not batch.five_seed_ensemble_eligible

    with pytest.raises(M03RV7Seed17KubernetesError, match="all twelve"):
        build_m03r_v7_seed17_indexed_batch_receipt(
            package=package,
            binding=binding,
            index_receipts=indices[:-1],
        )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _operator_artifact(
    *,
    tmp_path: Path,
    plan: M03RV7Seed17PackagePlan,
    package_path: Path,
    completion_index: int,
    label: str,
    source_archive_path: Path,
    runtime_manifest_path: Path,
) -> Path:
    row = plan.indices[completion_index]
    root = tmp_path / label
    root.mkdir()
    validation_path = root / "validation.json"
    fold_path = root / "fold-execution.json"
    qualification_path = root / "qualification.json"
    binding_path = root / "execution-plan-binding.json"
    artifact_path = root / "artifact.json"
    validation = {
        "schema": "rl-quant.top2000-dev.m03r-v7-seed17-validation-v1",
        "protocol_sha256": plan.protocol_sha256,
        "setting_index": row.setting_index,
        "setting_id": row.setting_id,
        "fold_index": 0,
        "seed": 17,
        "metrics": {"decision_count": 63},
        "development_only": True,
        "promotion_eligible": False,
    }
    _write_json(validation_path, validation)
    validation_sha = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    fold = {
        "schema": "rl-quant.top2000-dev.m03r-v7-seed17-fold-execution-v1",
        "protocol_sha256": plan.protocol_sha256,
        "setting_index": row.setting_index,
        "setting_id": row.setting_id,
        "runtime_setting_id": row.runtime_setting_id,
        "fold_index": 0,
        "ordered_seeds": [17],
        "member_count": 1,
        "seed_validation_receipt_sha256s": [validation_sha],
        "one_member_fold_execution": True,
        "five_seed_ensemble_eligible": False,
        "development_only": True,
        "promotion_eligible": False,
    }
    _write_json(fold_path, fold)
    fold_sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()
    qualification = {
        "schema": (
            "rl-quant.top2000-dev.m03r-v7-seed17-validation-sentinel-v1"
        ),
        "protocol_sha256": plan.protocol_sha256,
        "setting_index": row.setting_index,
        "setting_id": row.setting_id,
        "runtime_setting_id": row.runtime_setting_id,
        "world_size": 2,
        "fold_count": 1,
        "paired_seeds": [17],
        "completed_cells": 1,
        "seed_validation_receipt_sha256": {"validation.json": validation_sha},
        "fold_execution_receipt_sha256": {"fold.json": fold_sha},
        "rank_peak_cuda_memory": [
            {
                "rank": rank,
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "gpu_total_memory_bytes": 80 * 1024**3,
                "compute_capability": [9, 0],
                "allocator_oom_count": 0,
                "allocator_retry_count": 0,
            }
            for rank in range(2)
        ],
        "complete": True,
        "development_only": True,
        "promotion_eligible": False,
    }
    _write_json(qualification_path, qualification)
    binding = {
        "package_plan_sha256": plan.package_plan_sha256,
        "completion": asdict(row),
        "training_plan": {
            "protocol_sha256": plan.protocol_sha256,
            "setting_index": row.setting_index,
            "setting_id": row.setting_id,
            "runtime_setting_id": row.runtime_setting_id,
            "paired_seeds": [17],
        },
        "prior_training_evidence_imported": False,
        "one_member_fold_execution": True,
        "promotion_eligible": False,
    }
    _write_json(binding_path, binding)
    assert operator_main(
        [
            "build-artifact",
            "--package-plan",
            str(package_path),
            "--package-plan-sha256",
            plan.package_plan_sha256,
            "--completion-index",
            str(completion_index),
            "--qualification-receipt",
            str(qualification_path),
            "--validation-receipt",
            str(validation_path),
            "--fold-execution-receipt",
            str(fold_path),
            "--execution-plan-binding",
            str(binding_path),
            "--source-archive",
            str(source_archive_path),
            "--runtime-manifest",
            str(runtime_manifest_path),
            "--output",
            str(artifact_path),
        ]
    ) == 0
    return artifact_path


def test_local_operator_preserves_container_plan_path(tmp_path: Path) -> None:
    """A local plan copy must retain the immutable in-container mount path."""

    plan = _package().plan
    package_path = tmp_path / "package-plan-copy.json"
    payload = asdict(plan)
    payload["schema"] = M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA
    _write_json(package_path, payload)
    live_path = tmp_path / "live.json"
    template_path = tmp_path / "template.json"
    manifest_path = tmp_path / "sentinel.json"
    rendered_path = tmp_path / "sentinel-receipt.json"
    _write_json(live_path, asdict(_live_evidence()))
    _write_json(template_path, asdict(_template()))

    assert operator_main(
        [
            "render-sentinel",
            "--package-plan",
            str(package_path),
            "--package-plan-sha256",
            plan.package_plan_sha256,
            "--live-evidence",
            str(live_path),
            "--template",
            str(template_path),
            "--now-utc",
            "2026-08-07T12:01:00+00:00",
            "--manifest-output",
            str(manifest_path),
            "--rendered-output",
            str(rendered_path),
            "--completion-index",
            "3",
        ]
    ) == 0
    manifest = json.loads(manifest_path.read_bytes())
    arguments = manifest["spec"]["template"]["spec"]["containers"][0]["args"]
    package_argument = arguments.index("--package-plan")
    assert arguments[package_argument + 1] == plan.plan_artifact_path


def test_seed17_operator_renders_qualifies_final_and_binds_activation(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "package-plan.json"
    plan = build_m03r_v7_seed17_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path=str(package_path),
        benchmark_preflight_sha256=_digest("operator-benchmark-preflight"),
    )
    plan_payload = asdict(plan)
    plan_payload["schema"] = M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA
    _write_json(package_path, plan_payload)

    live_path = tmp_path / "live.json"
    template_path = tmp_path / "template.json"
    _write_json(live_path, asdict(_live_evidence()))
    template = replace(
        _template(),
        package_mount_path=str(tmp_path),
        output_mount_path=str(tmp_path / "output"),
    )
    _write_json(template_path, asdict(template))
    common = [
        "--package-plan",
        str(package_path),
        "--package-plan-sha256",
        plan.package_plan_sha256,
        "--live-evidence",
        str(live_path),
        "--template",
        str(template_path),
        "--now-utc",
        "2026-08-07T12:01:00+00:00",
    ]
    sentinel_manifest = tmp_path / "sentinel-manifest.json"
    sentinel_render = tmp_path / "sentinel-render.json"
    qualification_manifest = tmp_path / "qualification-manifest.json"
    qualification_render = tmp_path / "qualification-render.json"
    assert operator_main(
        [
            "render-sentinel",
            *common,
            "--manifest-output",
            str(sentinel_manifest),
            "--rendered-output",
            str(sentinel_render),
        ]
    ) == 0
    assert operator_main(
        [
            "render-qualification",
            *common,
            "--manifest-output",
            str(qualification_manifest),
            "--rendered-output",
            str(qualification_render),
        ]
    ) == 0
    assert _read_manifest(sentinel_manifest)["spec"]["completions"] == 1
    assert _read_manifest(qualification_manifest)["spec"]["completions"] == 12

    source_archive_path = tmp_path / "source.tar"
    source_archive_path.write_bytes(b"seed17-source-archive")
    runtime_manifest_path = tmp_path / "runtime-manifest.json"
    runtime_manifest_path.write_bytes(b"operator-runtime-manifest")
    runtime_manifest = hashlib.sha256(
        runtime_manifest_path.read_bytes()
    ).hexdigest()
    artifact_paths: list[Path] = []
    for index in range(12):
        path = _operator_artifact(
            tmp_path=tmp_path,
            plan=plan,
            package_path=package_path,
            completion_index=index,
            label=f"all-setting-{index}",
            source_archive_path=source_archive_path,
            runtime_manifest_path=runtime_manifest_path,
        )
        artifact_paths.append(path)
    sentinel_artifact_path = _operator_artifact(
        tmp_path=tmp_path,
        plan=plan,
        package_path=package_path,
        completion_index=3,
        label="sentinel",
        source_archive_path=source_archive_path,
        runtime_manifest_path=runtime_manifest_path,
    )
    qualification_path = tmp_path / "qualification.json"
    capacity_args = [
        "build-capacity",
        "--package-plan",
        str(package_path),
        "--package-plan-sha256",
        plan.package_plan_sha256,
        "--sentinel-artifact",
        str(sentinel_artifact_path),
        "--worker-entrypoint-sha256",
        _digest("operator-worker"),
        "--runtime-manifest-sha256",
        runtime_manifest,
        "--output",
        str(qualification_path),
    ]
    for path in artifact_paths:
        capacity_args.extend(["--setting-artifact", str(path)])
    assert operator_main(capacity_args) == 0

    final_manifest_path = tmp_path / "final-manifest.json"
    final_path = tmp_path / "final-render.json"
    assert operator_main(
        [
            "render-final",
            *common,
            "--qualification",
            str(qualification_path),
            "--manifest-output",
            str(final_manifest_path),
            "--rendered-output",
            str(final_path),
        ]
    ) == 0
    final_manifest = _read_manifest(final_manifest_path)
    first, second = _admitted_pair(final_manifest)
    fresh = deepcopy(second)
    fresh["metadata"]["resourceVersion"] = "3"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    fresh_path = tmp_path / "fresh.json"
    _write_json(first_path, first)
    _write_json(second_path, second)
    _write_json(fresh_path, fresh)
    binding_path = tmp_path / "binding.json"
    activation_path = tmp_path / "activation.json"
    assert operator_main(
        [
            "bind-activation",
            "--rendered-receipt",
            str(final_path),
            "--first-read",
            str(first_path),
            "--second-read",
            str(second_path),
            "--fresh-read",
            str(fresh_path),
            "--binding-output",
            str(binding_path),
            "--activation-output",
            str(activation_path),
        ]
    ) == 0
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    assert binding["job_uid"] == "seed17-job-uid"
    assert activation["job_uid"] == "seed17-job-uid"


def test_seed17_operator_artifact_rejects_unbound_source_archive(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "package-plan.json"
    plan = build_m03r_v7_seed17_top2000_package_plan(
        artifacts=_artifacts(),
        plan_artifact_path=str(package_path),
        benchmark_preflight_sha256=_digest("artifact-benchmark-preflight"),
    )
    payload = asdict(plan)
    payload["schema"] = M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA
    _write_json(package_path, payload)
    wrong_source = tmp_path / "wrong-source.tar"
    wrong_source.write_bytes(b"not-the-bound-source")
    runtime_manifest = tmp_path / "runtime-manifest.json"
    runtime_manifest.write_bytes(b"runtime")
    with pytest.raises(
        Top2000M03RV7Seed17OperatorError,
        match="package/source/execution-plan binding drifted",
    ):
        _operator_artifact(
            tmp_path=tmp_path,
            plan=plan,
            package_path=package_path,
            completion_index=0,
            label="unbound-source",
            source_archive_path=wrong_source,
            runtime_manifest_path=runtime_manifest,
        )


def _read_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _admitted_pair(
    manifest: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    first = deepcopy(manifest)
    metadata = first["metadata"]
    spec = first["spec"]
    assert isinstance(metadata, dict) and isinstance(spec, dict)
    metadata.update({"uid": "seed17-job-uid", "resourceVersion": "1"})
    spec["selector"] = {
        "matchLabels": {"batch.kubernetes.io/controller-uid": "seed17-job-uid"}
    }
    template = spec["template"]
    assert isinstance(template, dict)
    template_metadata = template["metadata"]
    assert isinstance(template_metadata, dict)
    template_metadata["creationTimestamp"] = None
    labels = template_metadata["labels"]
    assert isinstance(labels, dict)
    labels.update(
        {
            "batch.kubernetes.io/controller-uid": "seed17-job-uid",
            "batch.kubernetes.io/job-name": metadata["name"],
            "controller-uid": "seed17-job-uid",
            "job-name": metadata["name"],
        }
    )
    second = deepcopy(first)
    second_metadata = second["metadata"]
    assert isinstance(second_metadata, dict)
    second_metadata["resourceVersion"] = "2"
    return first, second
