from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rl_quant.training import top2000_m03r_v9_seadragon_lifecycle as lifecycle
from rl_quant.training import top2000_m03r_v9_seadragon_operator as operator
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
    build_m03r_v7_exact_job_activation_request,
)
from rl_quant.training.top2000_m03r_v9_kubernetes import (
    M03RV9TwoH100CapacityQualification,
    bind_m03r_v9_admitted_suspended_job,
    build_m03r_v9_live_evidence,
    render_m03r_v9_suspended_capacity_job,
)
from rl_quant.training.top2000_m03r_v9_package import (
    M03RV9PackageArtifacts,
    build_m03r_v9_package_plan,
    package_plan_file_payload,
)
from rl_quant.workflows import top2000_m03r_v9_seadragon_prepare as prepare

NOW = datetime(2026, 8, 11, 18, 0, tzinfo=UTC)


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _bytes(value)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _package():
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
            image_reference="registry/research@sha256:" + "f" * 64,
            image_digest_sha256="f" * 64,
        )
    )


def _live():
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
        protected_or_other_committed_h100_count=0,
        live_schedulable_free_h100_count=16,
        gpu_product_label_key=M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
        gpu_product_label_values=M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
        live_h100_cap_verified=True,
        gpu_selector_observed_live=True,
    )


def _template() -> M03RV7KubernetesTemplateConfig:
    return M03RV7KubernetesTemplateConfig(
        job_name="qt-m03r-v9-cap-prepare",
        run_id="qt-m03r-v9-prepare-run",
        service_account_name="default",
        pvc_claim_name="research-pvc",
        package_mount_path="/mnt/package",
        output_mount_path="/mnt/output",
    )


def _admitted(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    uid = "prepare-job-uid"
    first = json.loads(json.dumps(manifest))
    first["metadata"].update({"uid": uid, "resourceVersion": "1"})
    first["spec"]["selector"] = {
        "matchLabels": {"batch.kubernetes.io/controller-uid": uid}
    }
    metadata = first["spec"]["template"]["metadata"]
    metadata["creationTimestamp"] = None
    metadata["labels"].update(
        {
            "batch.kubernetes.io/controller-uid": uid,
            "batch.kubernetes.io/job-name": "qt-m03r-v9-cap-prepare",
            "controller-uid": uid,
            "job-name": "qt-m03r-v9-cap-prepare",
        }
    )
    second = json.loads(json.dumps(first))
    second["metadata"]["resourceVersion"] = "2"
    return first, second


def test_render_and_build_all_lifecycle_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operator, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    package = _package()
    package_path = tmp_path / "package-plan.json"
    package_file_sha = _write(package_path, package_plan_file_payload(package))
    live_path = tmp_path / "live.json"
    live_file_sha = _write(live_path, asdict(_live()))
    template_path = tmp_path / "template.json"
    template_file_sha = _write(template_path, asdict(_template()))
    manifest_path = tmp_path / "manifest.json"
    rendered_path = tmp_path / "rendered.json"
    assert (
        prepare.main(
            [
                "render",
                "--mode",
                "capacity",
                "--package-plan",
                str(package_path),
                "--package-plan-sha256",
                package.package_plan_sha256,
                "--live-evidence",
                str(live_path),
                "--live-evidence-file-sha256",
                live_file_sha,
                "--template",
                str(template_path),
                "--template-file-sha256",
                template_file_sha,
                "--now-utc",
                NOW.isoformat(),
                "--manifest-output",
                str(manifest_path),
                "--rendered-output",
                str(rendered_path),
            ]
        )
        == 0
    )
    rendered_sha = hashlib.sha256(rendered_path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    create_config_path = tmp_path / "create-config.json"
    prepare.main(
        [
            "build-create-config",
            "--rendered",
            str(rendered_path),
            "--rendered-file-sha256",
            rendered_sha,
            "--manifest",
            str(manifest_path),
            "--manifest-file-sha256",
            manifest_sha,
            "--evidence-root",
            str(tmp_path / "create-evidence"),
            "--binding-output",
            str(tmp_path / "binding.json"),
            "--activation-output",
            str(tmp_path / "activation.json"),
            "--output",
            str(create_config_path),
        ]
    )
    create_config = operator.M03RV9CreateOperatorConfig(
        **json.loads(create_config_path.read_bytes())
    )
    assert create_config.mode == "capacity"
    assert create_config.completions == create_config.parallelism == 1

    rendered = render_m03r_v9_suspended_capacity_job(
        package=package, live=_live(), template=_template(), now_utc=NOW
    )
    first, second = _admitted(rendered.manifest)
    binding = bind_m03r_v9_admitted_suspended_job(
        rendered=rendered,
        first_read=first,
        second_read=second,
        attached_owned_pod_uids=(),
    )
    activation = build_m03r_v7_exact_job_activation_request(binding, second)
    binding_path = tmp_path / "binding.json"
    activation_path = tmp_path / "activation.json"
    binding_sha = _write(binding_path, asdict(binding))
    activation_sha = _write(activation_path, asdict(activation))
    capacity_config_path = tmp_path / "capacity-config.json"
    prepare.main(
        [
            "build-capacity-attach-config",
            "--package-plan",
            str(package_path),
            "--package-plan-file-sha256",
            package_file_sha,
            "--package-plan-sha256",
            package.package_plan_sha256,
            "--binding",
            str(binding_path),
            "--binding-file-sha256",
            binding_sha,
            "--activation",
            str(activation_path),
            "--activation-file-sha256",
            activation_sha,
            "--output-root",
            str(tmp_path / "capacity-output"),
            "--evidence-root",
            str(tmp_path / "capacity-evidence"),
            "--host-python",
            sys.executable,
            "--pythonpath",
            str(tmp_path),
            "--output",
            str(capacity_config_path),
        ]
    )
    capacity_config = lifecycle.M03RV9CapacityAttachConfig(
        **json.loads(capacity_config_path.read_bytes())
    )
    assert capacity_config.job_uid == binding.job_uid
    assert capacity_config.capacity_receipt_sha256 == "not-yet-created"

    qualification = M03RV9TwoH100CapacityQualification(
        terminal_file_sha256="7" * 64,
        terminal_receipt_sha256="8" * 64,
        startup_file_sha256="9" * 64,
        terminal_evidence_file_sha256="a" * 64,
        cleanup_receipt_file_sha256="b" * 64,
        package_plan_sha256=package.package_plan_sha256,
        worker_plan_sha256=package.panel.workers[0].receipt_sha256,
    )
    qualification_path = tmp_path / "capacity-qualification.json"
    qualification_sha = _write(qualification_path, asdict(qualification))
    predictive_config_path = tmp_path / "predictive-config.json"
    prepare.main(
        [
            "build-predictive-attach-config",
            "--package-plan",
            str(package_path),
            "--package-plan-file-sha256",
            package_file_sha,
            "--package-plan-sha256",
            package.package_plan_sha256,
            "--binding",
            str(binding_path),
            "--binding-file-sha256",
            binding_sha,
            "--activation",
            str(activation_path),
            "--activation-file-sha256",
            activation_sha,
            "--output-root",
            str(tmp_path / "predictive-output"),
            "--evidence-root",
            str(tmp_path / "predictive-evidence"),
            "--host-python",
            sys.executable,
            "--pythonpath",
            str(tmp_path),
            "--capacity-qualification",
            str(qualification_path),
            "--capacity-qualification-file-sha256",
            qualification_sha,
            "--output",
            str(predictive_config_path),
        ]
    )
    predictive_config = lifecycle.M03RV9AttachSupervisorConfig(
        **{
            **json.loads(predictive_config_path.read_bytes()),
            "expected_completions": tuple(
                lifecycle.M03RV9ExpectedCompletion(**row)
                for row in json.loads(predictive_config_path.read_bytes())[
                    "expected_completions"
                ]
            ),
        }
    )
    assert len(predictive_config.expected_completions) == 3
    assert predictive_config.capacity_receipt_sha256 == "8" * 64
    assert predictive_config.parallelism == 1
