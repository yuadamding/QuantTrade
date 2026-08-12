from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rl_quant.training import top2000_m03r_v9_seadragon_operator as operator
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_H100_PRODUCT_LABEL_KEY,
    M03R_TOP2000_H100_PRODUCT_LABEL_VALUES,
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v9_kubernetes import (
    build_m03r_v9_live_evidence,
    render_m03r_v9_suspended_capacity_job,
)
from rl_quant.training.top2000_m03r_v9_package import (
    M03RV9PackageArtifacts,
    build_m03r_v9_package_plan,
)

NOW = datetime(2026, 8, 11, 18, 0, tzinfo=UTC)
JOB_NAME = "qt-m03r-v9-cap-test"
RUN_ID = "qt-m03r-v9-cap-test-run"
JOB_UID = "v9-capacity-uid"
SOURCE_SHA = "a" * 64


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _bytes(value)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _package():
    return build_m03r_v9_package_plan(
        artifacts=M03RV9PackageArtifacts(
            source_archive_sha256=SOURCE_SHA,
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


def _rendered():
    package = _package()
    live = build_m03r_v9_live_evidence(
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
    rendered = render_m03r_v9_suspended_capacity_job(
        package=package,
        live=live,
        template=M03RV7KubernetesTemplateConfig(
            job_name=JOB_NAME,
            run_id=RUN_ID,
            service_account_name="default",
            pvc_claim_name="research-pvc",
            package_mount_path="/mnt/package",
            output_mount_path="/mnt/output",
        ),
        now_utc=NOW,
    )
    return package, rendered


def _admitted(manifest: dict[str, Any], *, resource_version: str) -> dict[str, Any]:
    value = json.loads(json.dumps(manifest))
    value["metadata"].update(
        {"uid": JOB_UID, "resourceVersion": resource_version, "creationTimestamp": None}
    )
    value["spec"]["selector"] = {
        "matchLabels": {"batch.kubernetes.io/controller-uid": JOB_UID}
    }
    template = value["spec"]["template"]["metadata"]
    template["creationTimestamp"] = None
    template["labels"].update(
        {
            "batch.kubernetes.io/controller-uid": JOB_UID,
            "batch.kubernetes.io/job-name": JOB_NAME,
            "controller-uid": JOB_UID,
            "job-name": JOB_NAME,
        }
    )
    return value


def _config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[operator.M03RV9CreateOperatorConfig, dict[str, Any]]:
    monkeypatch.setattr(operator, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    package, rendered = _rendered()
    rendered_path = tmp_path / "rendered.json"
    manifest_path = tmp_path / "manifest.json"
    rendered_sha = _write_json(rendered_path, asdict(rendered))
    manifest_sha = _write_json(manifest_path, rendered.manifest)
    config = operator.M03RV9CreateOperatorConfig(
        mode="capacity",
        job_name=JOB_NAME,
        run_id=RUN_ID,
        rendered_path=str(rendered_path),
        rendered_file_sha256=rendered_sha,
        manifest_path=str(manifest_path),
        manifest_file_sha256=manifest_sha,
        evidence_root=str(tmp_path / "evidence"),
        binding_output_path=str(tmp_path / "binding.json"),
        activation_output_path=str(tmp_path / "activation.json"),
        package_plan_sha256=package.package_plan_sha256,
        source_archive_sha256=SOURCE_SHA,
        capacity_receipt_sha256="not-yet-created",
        operator_source_sha256=operator._file_sha256(Path(operator.__file__)),
        completions=1,
        parallelism=1,
        request_timeout_seconds=5,
    )
    return config, rendered.manifest


class _Transport:
    def __init__(
        self,
        manifest: dict[str, Any],
        *,
        transport_error: bool = False,
        never_created: bool = False,
        unexpected_pod: bool = False,
        dry_drift: bool = False,
        empty_success_response: bool = False,
    ) -> None:
        self.manifest = manifest
        self.dry = _admitted(manifest, resource_version="dry")
        if dry_drift:
            self.dry["spec"]["template"]["spec"]["runtimeClassName"] = "injected"
        self.created: dict[str, Any] | None = None
        self.transport_error = transport_error
        self.never_created = never_created
        self.unexpected_pod = unexpected_pod
        self.empty_success_response = empty_success_response
        self.create_count = 0
        self.delete_count = 0
        self.read_count = 0

    def server_dry_run(self, manifest_path: Path) -> dict[str, Any]:
        del manifest_path
        return self.dry

    def get_job(self, *, allow_absent: bool = False) -> dict[str, Any] | None:
        del allow_absent
        if self.created is None:
            return None
        self.read_count += 1
        value = json.loads(json.dumps(self.created))
        value["metadata"]["resourceVersion"] = str(self.read_count + 1)
        return value

    def get_pods_by_job_name(self) -> tuple[dict[str, Any], ...]:
        return ()

    def get_owned_pods(self, job_uid: str) -> tuple[dict[str, Any], ...]:
        del job_uid
        if not self.unexpected_pod:
            return ()
        return ({"metadata": {"uid": "unexpected"}},)

    def create_once(self, manifest_path: Path) -> operator.M03RV9CreateAttempt:
        del manifest_path
        self.create_count += 1
        if not self.never_created:
            self.created = _admitted(self.manifest, resource_version="1")
        if self.transport_error:
            raise operator.M03RV9SeadragonOperatorError("simulated timeout")
        if self.never_created:
            return operator.M03RV9CreateAttempt(1, b"", b"rejected")
        if self.empty_success_response:
            return operator.M03RV9CreateAttempt(0, b"", b"")
        return operator.M03RV9CreateAttempt(0, _bytes(self.created), b"")

    def delete(self, request: Any, options_path: Path) -> None:
        del request, options_path
        self.delete_count += 1
        self.created = None


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _run(
    tmp_path: Path,
    config: operator.M03RV9CreateOperatorConfig,
    transport: _Transport,
    *,
    clock: _Clock | None = None,
) -> None:
    Path(config.evidence_root).mkdir()
    config_path = tmp_path / "config.json"
    config_sha = _write_json(config_path, asdict(config))
    active_clock = clock or _Clock()
    operator.prepare_suspended_job_once(
        config_path,
        config_sha,
        transport=transport,
        sleep=active_clock.sleep,
        monotonic=active_clock.monotonic,
    )


def test_one_create_then_two_read_zero_pod_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _config(tmp_path, monkeypatch)
    transport = _Transport(manifest)
    _run(tmp_path, config, transport)
    assert transport.create_count == 1
    assert Path(config.binding_output_path).is_file()
    assert Path(config.activation_output_path).is_file()
    success = json.loads(
        (Path(config.evidence_root) / "prepare-success.json").read_text()
    )
    assert success["create_attempt_count"] == 1
    assert success["create_retried"] is False
    assert success["suspended"] is True
    assert success["zero_owned_pods"] is True
    assert success["economic_panel_authorized"] is False


def test_transport_error_reconciles_accepted_job_without_create_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _config(tmp_path, monkeypatch)
    transport = _Transport(manifest, transport_error=True)
    _run(tmp_path, config, transport)
    assert transport.create_count == 1
    assert Path(config.binding_output_path).is_file()


def test_server_dry_run_mutation_rejects_before_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _config(tmp_path, monkeypatch)
    transport = _Transport(manifest, dry_drift=True)
    with pytest.raises(operator.M03RV9SeadragonOperatorError, match="dry-run"):
        _run(tmp_path, config, transport)
    assert transport.create_count == 0


def test_stable_absence_consumes_identity_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _config(tmp_path, monkeypatch)
    transport = _Transport(manifest, never_created=True)
    clock = _Clock()
    with pytest.raises(operator.M03RV9SeadragonOperatorError, match="stable absence"):
        _run(tmp_path, config, transport, clock=clock)
    assert transport.create_count == 1
    reconciliation = json.loads(
        (Path(config.evidence_root) / "create-reconciliation.json").read_text()
    )
    assert reconciliation["outcome"] == "stable-absence"
    assert reconciliation["create_retried"] is False


def test_success_without_response_uid_is_attach_required_and_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _config(tmp_path, monkeypatch)
    transport = _Transport(manifest, empty_success_response=True)
    with pytest.raises(operator.M03RV9CreateAttachRequired):
        _run(tmp_path, config, transport)
    assert transport.create_count == 1
    assert (
        Path(config.evidence_root) / "create-response-attach-required.json"
    ).is_file()
    assert not Path(config.binding_output_path).exists()


def test_postaccept_unexpected_pod_retains_attach_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _config(tmp_path, monkeypatch)
    transport = _Transport(manifest, unexpected_pod=True)
    with pytest.raises(operator.M03RV9CreateAttachRequired):
        _run(tmp_path, config, transport)
    assert transport.create_count == 1
    assert (
        Path(config.evidence_root) / "postaccept-binding-attach-required.json"
    ).is_file()
    assert not Path(config.activation_output_path).exists()


def test_postbinding_handoff_failure_exact_cleans_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest = _config(tmp_path, monkeypatch)
    transport = _Transport(manifest)
    monkeypatch.setattr(
        operator,
        "build_m03r_v7_exact_job_activation_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            operator.M03RV9SeadragonOperatorError("simulated handoff failure")
        ),
    )
    with pytest.raises(operator.M03RV9SeadragonOperatorError, match="exact cleanup"):
        _run(tmp_path, config, transport)
    assert transport.create_count == 1
    assert transport.delete_count == 1
    assert transport.created is None
    root = Path(config.evidence_root)
    assert (root / "cleanup-receipt.json").is_file()
    assert (root / "postaccept-cleanup-error.json").is_file()
    assert not (root / "postaccept-cleanup-attach-required.json").exists()
