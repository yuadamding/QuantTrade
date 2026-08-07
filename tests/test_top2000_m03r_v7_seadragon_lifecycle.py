from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from rl_quant.training import top2000_m03r_v7_seadragon_lifecycle as lifecycle
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03RV7AdmittedJobBinding,
    M03RV7ExactJobActivationRequest,
    M03RV7ExactJobCleanupRequest,
    build_m03r_v7_exact_job_activation_request,
)

JOB_NAME = "qt-seed17-test"
RUN_ID = "qt-seed17-test-run"
JOB_UID = "job-uid-123"
NAMESPACE = "yn-gpu-workload"
SETTING_IDS = lifecycle.M03R_SEED17_TOP2000_SETTING_IDS
PACKAGE_SHA = "d" * 64
SOURCE_SHA = "e" * 64
CAPACITY_SHA = "f" * 64


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _pod_sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value) + b"\n").hexdigest()


def _rank_proof() -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "device": f"cuda:{rank}",
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_total_memory_bytes": 80 * 1024**3,
            "compute_capability": [9, 0],
            "allocator_oom_count": 0,
            "torchrun_restart_count": 0,
        }
        for rank in range(2)
    ]


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _bytes(value)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _detached_process_identity(pid: int) -> dict[str, Any]:
    """Provide the child-session shape without depending on the pytest launcher."""

    return {
        "pid": pid,
        "pgrp": pid,
        "session": pid,
        "tty_nr": 0,
        "start_ticks": 123456,
        "cmdline_sha256": "a" * 64,
    }


def _job(
    *,
    resource_version: str,
    suspended: bool,
    condition: str | None = None,
) -> dict[str, Any]:
    selector = {"matchLabels": {"controller-uid": JOB_UID}}
    template_metadata = {
        "labels": {"controller-uid": JOB_UID, "job-name": JOB_NAME},
        "annotations": {
            "rl-quant/run-id": RUN_ID,
            "rl-quant/package-plan-sha256": PACKAGE_SHA,
            "rl-quant/source-archive-sha256": SOURCE_SHA,
            "rl-quant/capacity-receipt-sha256": CAPACITY_SHA,
        },
    }
    pod_spec = {
        "restartPolicy": "Never",
        "serviceAccountName": "default",
        "containers": [
            {
                "name": "trainer",
                "image": "example.invalid/quanttrade@sha256:" + "a" * 64,
                "resources": {
                    "limits": {"nvidia.com/gpu": "2"},
                    "requests": {"nvidia.com/gpu": "2"},
                },
            }
        ],
    }
    status: dict[str, Any] = {}
    if condition is not None:
        status["conditions"] = [{"type": condition, "status": "True"}]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
            "uid": JOB_UID,
            "resourceVersion": resource_version,
            "annotations": dict(template_metadata["annotations"]),
        },
        "spec": {
            "suspend": suspended,
            "parallelism": 8,
            "selector": selector,
            "template": {"metadata": template_metadata, "spec": pod_spec},
        },
        "status": status,
    }


def _binding(suspended_job: dict[str, Any]) -> M03RV7AdmittedJobBinding:
    spec = suspended_job["spec"]
    template = spec["template"]
    fields: dict[str, Any] = {
        "job_name": JOB_NAME,
        "namespace": NAMESPACE,
        "job_uid": JOB_UID,
        "run_id": RUN_ID,
        "first_resource_version": "1",
        "second_resource_version": "2",
        "parallelism": 8,
        "admitted_spec_sha256": _sha(spec),
        "admitted_pod_template_sha256": _pod_sha(template["spec"]),
        "admitted_selector_sha256": _sha(spec["selector"]),
        "admitted_template_metadata_sha256": _sha(template["metadata"]),
        "desired_manifest_sha256": "b" * 64,
        "attached_owned_pod_uids": (),
        "suspended": True,
    }
    canonical = {
        "schema": M03RV7AdmittedJobBinding.__dataclass_fields__["schema"].default,
        **fields,
        "attached_owned_pod_uids": [],
    }
    return M03RV7AdmittedJobBinding(
        **fields,
        receipt_sha256=_sha(canonical),
    )


def _pod(index: int, *, phase: str = "Succeeded") -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"qt-seed17-test-{index}-abcde",
            "uid": f"pod-uid-{index}",
            "annotations": {
                "batch.kubernetes.io/job-completion-index": str(index)
            },
            "ownerReferences": [
                {"uid": JOB_UID, "controller": True, "kind": "Job"}
            ],
        },
        "status": {"phase": phase},
    }


class FakeTransport:
    def __init__(
        self,
        *,
        failed: bool = False,
        cleanup_conflict_once: bool = False,
        fresh_resource_version: str = "3",
    ) -> None:
        self.fresh = _job(
            resource_version=fresh_resource_version,
            suspended=True,
        )
        self.activated = False
        self.deleted = False
        self.failed = failed
        self.cleanup_conflict_once = cleanup_conflict_once
        self.cleanup_resource_version = "5"
        self.activation_requests: list[M03RV7ExactJobActivationRequest] = []
        self.cleanup_requests: list[M03RV7ExactJobCleanupRequest] = []

    def get_job(self, *, allow_absent: bool = False) -> dict[str, Any] | None:
        del allow_absent
        if self.deleted:
            return None
        if self.failed:
            return _job(
                resource_version=self.cleanup_resource_version,
                suspended=False,
                condition="Failed",
            )
        if self.activated:
            return _job(
                resource_version=self.cleanup_resource_version,
                suspended=False,
                condition="Complete",
            )
        return self.fresh

    def get_owned_pods(self) -> tuple[dict[str, Any], ...]:
        if self.deleted or (not self.activated and not self.failed):
            return ()
        if self.failed:
            return (_pod(3, phase="Failed"),)
        return tuple(_pod(index) for index in range(12))

    def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
        assert pod_name.startswith("qt-seed17-test-")
        assert limit_bytes >= 4096
        return b"bounded terminal log\n"

    def activate(
        self, request: M03RV7ExactJobActivationRequest
    ) -> dict[str, Any]:
        self.activation_requests.append(request)
        self.activated = True
        return _job(resource_version="4", suspended=False)

    def delete(
        self,
        request: M03RV7ExactJobCleanupRequest,
        options_path: Path,
    ) -> None:
        options = json.loads(options_path.read_bytes())
        assert options["preconditions"] == {
            "uid": JOB_UID,
            "resourceVersion": self.cleanup_resource_version,
        }
        assert options["propagationPolicy"] == "Foreground"
        self.cleanup_requests.append(request)
        if self.cleanup_conflict_once:
            self.cleanup_conflict_once = False
            self.cleanup_resource_version = "6"
            raise lifecycle.SeadragonLifecycleError("simulated RV conflict")
        self.deleted = True


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[lifecycle.AttachSupervisorConfig, M03RV7AdmittedJobBinding]:
    project = tmp_path / "training"
    project.mkdir()
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(project))
    monkeypatch.setattr(lifecycle, "_process_identity", _detached_process_identity)
    output_root = project / "runs" / RUN_ID
    output_root.mkdir(parents=True)
    evidence_root = project / "launches" / RUN_ID
    evidence_root.mkdir(parents=True)
    package_root = project / "packages" / RUN_ID
    package_root.mkdir(parents=True)
    suspended = _job(resource_version="3", suspended=True)
    binding = _binding(suspended)
    activation = build_m03r_v7_exact_job_activation_request(binding, suspended)
    binding_path = package_root / "final-binding-receipt.json"
    activation_path = package_root / "activation-request.json"
    binding_sha = _write_json(binding_path, asdict(binding))
    activation_sha = _write_json(activation_path, asdict(activation))
    process = lifecycle._process_identity(os.getpid())
    _write_json(
        evidence_root / "spawn-process.json",
        {
            "schema": "test",
            "config_sha256": "c" * 64,
            **process,
        },
    )
    config = lifecycle.AttachSupervisorConfig(
        job_name=JOB_NAME,
        run_id=RUN_ID,
        job_uid=JOB_UID,
        binding_path=str(binding_path),
        binding_file_sha256=binding_sha,
        activation_request_path=str(activation_path),
        activation_request_file_sha256=activation_sha,
        output_root=str(output_root),
        evidence_root=str(evidence_root),
        package_plan_sha256=PACKAGE_SHA,
        source_archive_sha256=SOURCE_SHA,
        capacity_receipt_sha256=CAPACITY_SHA,
        expected_completions=lifecycle.canonical_one_seed_completions(SETTING_IDS),
        host_python_path="/usr/bin/python3",
        pythonpath=str(package_root),
        poll_interval_seconds=5,
        request_timeout_seconds=5,
        handshake_timeout_seconds=30,
        hard_wall_seconds=60,
        log_limit_bytes=4096,
    )
    for row in config.expected_completions:
        base = (
            output_root
            / f"completion-{row.completion_index:02d}-setting-{row.setting_index:02d}"
        )
        _write_json(
            base / "execution-plan-binding.json",
            {
                "package_plan_sha256": config.package_plan_sha256,
                "completion": {
                    "completion_index": row.completion_index,
                    "setting_index": row.setting_index,
                    "setting_id": row.setting_id,
                    "fold_indices": list(range(6)),
                    "paired_seeds": [17],
                    "one_member_fold_execution": True,
                    "development_only": True,
                    "promotion_eligible": False,
                },
            },
        )
        run_root = base / "training"
        cell_hashes: dict[str, str] = {}
        seed_hashes: dict[str, str] = {}
        fold_hashes: dict[str, str] = {}
        for fold_index in range(6):
            seed_key = (
                f"receipts/seed-validation/fold-{fold_index:02d}-seed-17.json"
            )
            seed_hash = _write_json(
                run_root / seed_key,
                {
                    "schema": lifecycle.M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA,
                    "protocol_sha256": (
                        lifecycle.M03R_SEED17_TOP2000_PROTOCOL_SHA256
                    ),
                    "setting_index": row.setting_index,
                    "setting_id": row.setting_id,
                    "fold_index": fold_index,
                    "seed": 17,
                    "metrics": {"decision_count": 63},
                    "development_only": True,
                    "outer_evaluation_authorized": False,
                    "promotion_eligible": False,
                },
            )
            seed_hashes[seed_key] = seed_hash
            cell_key = f"fold-{fold_index:02d}-seed-17.json"
            cell_hashes[cell_key] = _write_json(
                run_root / "receipts" / cell_key,
                {
                    "schema": lifecycle._CELL_RECEIPT_SCHEMA,
                    "protocol_sha256": (
                        lifecycle.M03R_SEED17_TOP2000_PROTOCOL_SHA256
                    ),
                    "setting_index": row.setting_index,
                    "setting_id": row.setting_id,
                    "fold_index": fold_index,
                    "seed": 17,
                    "seed_validation_required": True,
                    "seed_validation_receipt_sha256": seed_hash,
                    "rank_peak_cuda_memory": _rank_proof(),
                    "development_only": True,
                    "promotion_eligible": False,
                },
            )
            fold_key = f"receipts/fold-execution/fold-{fold_index:02d}.json"
            fold_hashes[fold_key] = _write_json(
                run_root / fold_key,
                {
                    "schema": lifecycle.M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA,
                    "protocol_sha256": (
                        lifecycle.M03R_SEED17_TOP2000_PROTOCOL_SHA256
                    ),
                    "setting_index": row.setting_index,
                    "setting_id": row.setting_id,
                    "fold_index": fold_index,
                    "ordered_seeds": [17],
                    "seed_validation_receipt_sha256s": [seed_hash],
                    "member_count": 1,
                    "chronological_return_path_count": 1,
                    "one_member_fold_execution": True,
                    "output_space_ensemble": False,
                    "five_seed_ensemble_eligible": False,
                    "development_only": True,
                    "outer_evaluation_authorized": False,
                    "promotion_eligible": False,
                },
            )
        _write_json(
            run_root / "completion-receipt.json",
            {
                "schema": lifecycle.M03R_SEED17_TOP2000_COMPLETION_SCHEMA,
                "protocol_sha256": lifecycle.M03R_SEED17_TOP2000_PROTOCOL_SHA256,
                "setting_index": row.setting_index,
                "setting_id": row.setting_id,
                "world_size": 2,
                "fold_count": 6,
                "paired_seeds": [17],
                "completed_cells": 6,
                "seed_validation_receipt_count": 6,
                "fold_ensemble_receipt_count": 0,
                "fold_execution_receipt_count": 6,
                "inference_path_count": 6,
                "one_member_fold_execution_required": True,
                "five_seed_ensemble_eligible": False,
                "output_space_ensemble_required": False,
                "cell_receipt_sha256": cell_hashes,
                "seed_validation_receipt_sha256": seed_hashes,
                "fold_execution_receipt_sha256": fold_hashes,
                "rank_peak_cuda_memory": _rank_proof(),
                "complete": True,
                "development_only": True,
                "future_selected_universe": True,
                "outer_evaluation_authorized": False,
                "promotion_eligible": False,
            },
        )
    return config, binding


def test_attach_transport_has_no_job_creation_surface() -> None:
    transport = lifecycle.AttachOnlyKubectl(
        kubectl_path=lifecycle.SEADRAGON_KUBECTL,
        kubeconfig_path=lifecycle.SEADRAGON_KUBECONFIG,
        context="yding4_yn-gpu-workload@kubernetes-admin@kubernetes",
        namespace=NAMESPACE,
        job_name=JOB_NAME,
        job_uid=JOB_UID,
        request_timeout_seconds=5,
    )
    assert not hasattr(transport, "create")
    assert not hasattr(transport, "apply")
    assert not hasattr(transport, "replace")
    with pytest.raises(lifecycle.SeadragonLifecycleError, match="rejected"):
        transport._run(("create", "job"))


def test_spawned_identity_rejects_an_attached_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = 4242
    attached = {
        **_detached_process_identity(pid),
        "pgrp": pid - 1,
        "session": pid - 1,
    }
    monkeypatch.setattr(lifecycle, "_process_identity", lambda _: attached)
    with pytest.raises(
        lifecycle.SeadragonLifecycleError,
        match="not a detached session leader",
    ):
        lifecycle._validate_spawned_identity(attached, pid=pid)


def test_launch_handshake_waits_for_complete_receipt_and_binds_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _prepare(tmp_path, monkeypatch)
    root = Path(config.evidence_root)
    launch_path = root / "launch-success.json"
    launch_path.write_bytes(b'{"schema":')
    process_receipt = json.loads((root / "spawn-process.json").read_bytes())

    assert lifecycle._validated_launch_success(
        path=launch_path,
        root=root,
        config=config,
        process_receipt=process_receipt,
        pid=os.getpid(),
    ) is None

    arm_sha = _write_json(root / "arm.json", {"schema": "test-arm"})
    runtime_activation = json.loads(
        Path(config.activation_request_path).read_bytes()
    )
    _write_json(root / "activation-request-runtime.json", runtime_activation)
    activation_sha = _write_json(
        root / "activation.json",
        {
            "schema": "rl-quant.top2000-m03r-v7-supervisor-activation-v1",
            "activated_at_utc": "2026-08-07T12:00:00+00:00",
            "arm_file_sha256": arm_sha,
            "activation_request_sha256": runtime_activation["request_sha256"],
            "activated_job_sha256": "a" * 64,
        },
    )
    launch_path.write_bytes(
        _bytes(
            {
                "schema": (
                    "rl-quant.top2000-m03r-v7-supervisor-launch-success-v1"
                ),
                "launched_at_utc": "2026-08-07T12:00:01+00:00",
                "activation_file_sha256": activation_sha,
                "job_name": config.job_name,
                "job_uid": config.job_uid,
                "run_id": config.run_id,
                "parallelism": config.parallelism,
                "gpus_per_worker": config.gpus_per_worker,
                "request_ceiling": (
                    config.parallelism * config.gpus_per_worker
                ),
                "capacity_receipt_sha256": config.capacity_receipt_sha256,
                "quota_pending_backfill_accepted": True,
            }
        )
    )

    launch = lifecycle._validated_launch_success(
        path=launch_path,
        root=root,
        config=config,
        process_receipt=process_receipt,
        pid=os.getpid(),
    )
    assert launch is not None
    assert launch["activation_file_sha256"] == activation_sha


def test_hard_wall_starts_before_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _prepare(tmp_path, monkeypatch)
    clock = [0.0]

    class SlowActivationTransport(FakeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.nonterminal_reads = 0

        def activate(
            self, request: M03RV7ExactJobActivationRequest
        ) -> dict[str, Any]:
            self.activation_requests.append(request)
            self.activated = True
            clock[0] = 61.0
            return _job(resource_version="4", suspended=False)

        def get_job(
            self, *, allow_absent: bool = False
        ) -> dict[str, Any] | None:
            if self.deleted:
                return None
            if self.activated:
                self.nonterminal_reads += 1
                return _job(
                    resource_version=self.cleanup_resource_version,
                    suspended=False,
                )
            return super().get_job(allow_absent=allow_absent)

    transport = SlowActivationTransport()
    with pytest.raises(
        lifecycle.SeadragonLifecycleError,
        match="supervisor-hard-wall",
    ):
        lifecycle.run_attach_supervisor(
            config,
            config_sha256="c" * 64,
            transport=transport,
            sleep=lambda _: None,
            monotonic=lambda: clock[0],
            supervisor_started_monotonic=0.0,
        )

    # One terminal observation and one cleanup read are enough.  A wall that
    # began after activation would instead poll for a further sixty seconds.
    assert transport.nonterminal_reads == 2


def test_supervisor_activates_validates_all_twelve_and_exact_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _prepare(tmp_path, monkeypatch)
    transport = FakeTransport()
    lifecycle.run_attach_supervisor(
        config,
        config_sha256="c" * 64,
        transport=transport,
        sleep=lambda _: None,
    )
    root = Path(config.evidence_root)
    for name in (
        "readiness.json",
        "arm.json",
        "activation.json",
        "launch-success.json",
        "terminal-evidence.json",
        "completion-coverage.json",
        "cleanup-request.json",
        "cleanup-receipt.json",
    ):
        assert (root / name).is_file()
    assert len(transport.activation_requests) == 1
    assert len(transport.cleanup_requests) == 1
    coverage = json.loads((root / "completion-coverage.json").read_bytes())
    assert coverage["completion_count"] == 12
    assert coverage["expected_seed"] == 17


def test_supervisor_rebuilds_activation_after_harmless_resource_version_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _prepare(tmp_path, monkeypatch)
    transport = FakeTransport(fresh_resource_version="4")
    lifecycle.run_attach_supervisor(
        config,
        config_sha256="c" * 64,
        transport=transport,
        sleep=lambda _: None,
    )
    assert len(transport.activation_requests) == 1
    assert transport.activation_requests[0].resource_version == "4"
    readiness = json.loads(
        (Path(config.evidence_root) / "readiness.json").read_bytes()
    )
    assert readiness["configured_resource_version"] == "3"
    assert readiness["runtime_resource_version"] == "4"
    assert readiness["configured_activation_request_sha256"] != readiness[
        "runtime_activation_request_sha256"
    ]


def test_invalid_completion_is_preserved_then_exact_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _prepare(tmp_path, monkeypatch)
    first = config.expected_completions[0]
    receipt = (
        Path(config.output_root)
        / f"completion-{first.completion_index:02d}-setting-{first.setting_index:02d}"
        / "training"
        / "completion-receipt.json"
    )
    payload = json.loads(receipt.read_bytes())
    payload["completed_cells"] = 5
    receipt.write_bytes(_bytes(payload))
    transport = FakeTransport()
    with pytest.raises(lifecycle.SeadragonLifecycleError, match="coverage"):
        lifecycle.run_attach_supervisor(
            config,
            config_sha256="c" * 64,
            transport=transport,
            sleep=lambda _: None,
        )
    root = Path(config.evidence_root)
    assert (root / "terminal-evidence.json").is_file()
    assert (root / "supervisor-error.json").is_file()
    assert (root / "cleanup-receipt.json").is_file()
    assert transport.deleted


def test_cleanup_refreshes_resource_version_after_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _prepare(tmp_path, monkeypatch)
    transport = FakeTransport(cleanup_conflict_once=True)
    lifecycle.run_attach_supervisor(
        config,
        config_sha256="c" * 64,
        transport=transport,
        sleep=lambda _: None,
    )
    root = Path(config.evidence_root)
    assert [row.resource_version for row in transport.cleanup_requests] == ["5", "6"]
    assert (root / "cleanup-delete-error-attempt-01.json").is_file()
    assert (root / "cleanup-request-attempt-02.json").is_file()
    assert (root / "cleanup-receipt.json").is_file()


def test_old_failed_job_capture_and_exact_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, binding = _prepare(tmp_path, monkeypatch)
    failed_root = Path(config.evidence_root).parent / "failed-cleanup"
    cleanup = lifecycle.FailedJobCleanupConfig(
        job_name=JOB_NAME,
        run_id=RUN_ID,
        job_uid=JOB_UID,
        binding_path=config.binding_path,
        binding_file_sha256=config.binding_file_sha256,
        evidence_root=str(failed_root),
        request_timeout_seconds=5,
        log_limit_bytes=4096,
    )
    assert binding.job_uid == JOB_UID
    transport = FakeTransport(failed=True)
    lifecycle.capture_and_cleanup_failed_job(
        cleanup,
        transport=transport,
        sleep=lambda _: None,
    )
    assert (failed_root / "terminal-evidence.json").is_file()
    assert (failed_root / "cleanup-receipt.json").is_file()
    assert len(transport.cleanup_requests) == 1


def test_receipts_are_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "training"
    project.mkdir()
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(project))
    evidence = project / "launches" / "attempt"
    evidence.parent.mkdir(parents=True)
    lifecycle._create_evidence_root(evidence)
    lifecycle._exclusive_json(evidence / "receipt.json", {"value": 1})
    with pytest.raises(lifecycle.SeadragonLifecycleError, match="overwrite"):
        lifecycle._exclusive_json(evidence / "receipt.json", {"value": 2})
