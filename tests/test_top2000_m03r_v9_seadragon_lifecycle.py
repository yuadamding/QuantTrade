from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_SETTING_IDS,
)
from rl_quant.training import top2000_m03r_v7_seadragon_lifecycle as common
from rl_quant.training import top2000_m03r_v9_seadragon_lifecycle as lifecycle
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03RV7AdmittedJobBinding,
    M03RV7ExactJobActivationRequest,
    M03RV7ExactJobCleanupRequest,
    build_m03r_v7_exact_job_activation_request,
)
from rl_quant.training.top2000_m03r_v9_package import (
    M03RV9PackageArtifacts,
    build_m03r_v9_package_plan,
    package_plan_file_payload,
)

JOB_NAME = "qt-m03r-v9-test"
RUN_ID = "qt-m03r-v9-test-run"
JOB_UID = "v9-job-uid"
PACKAGE_SHA = "a" * 64
SOURCE_SHA = "b" * 64
CAPACITY_SHA = "c" * 64


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


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_receipt(unsigned: dict[str, Any]) -> dict[str, Any]:
    return {**unsigned, "receipt_sha256": _sha_bytes(_bytes(unsigned))}


def _semantic_receipt(unsigned: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return {**unsigned, "receipt_sha256": _sha_bytes(encoded)}


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _bytes(value)
    path.write_bytes(content)
    return _sha_bytes(content)


def _write_binary(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return _sha_bytes(value)


def _job(
    *,
    resource_version: str,
    suspended: bool,
    parallelism: int = 3,
    capacity_receipt_sha256: str = CAPACITY_SHA,
    package_plan_sha256: str = PACKAGE_SHA,
) -> dict[str, Any]:
    selector = {"matchLabels": {"batch.kubernetes.io/controller-uid": JOB_UID}}
    annotations = {
        "rl-quant/run-id": RUN_ID,
        "rl-quant/package-plan-sha256": package_plan_sha256,
        "rl-quant/source-archive-sha256": SOURCE_SHA,
        "rl-quant/capacity-receipt-sha256": capacity_receipt_sha256,
    }
    template_metadata = {
        "labels": {
            "batch.kubernetes.io/controller-uid": JOB_UID,
            "batch.kubernetes.io/job-name": JOB_NAME,
            "controller-uid": JOB_UID,
            "job-name": JOB_NAME,
        },
        "annotations": annotations,
        "creationTimestamp": None,
    }
    pod_spec = {
        "restartPolicy": "Never",
        "serviceAccountName": "default",
        "containers": [
            {
                "name": "trainer",
                "image": "example.invalid/v9@sha256:" + "d" * 64,
                "terminationMessagePath": "/dev/termination-log",
                "terminationMessagePolicy": "File",
                "resources": {
                    "requests": {"nvidia.com/gpu": "2"},
                    "limits": {"nvidia.com/gpu": "2"},
                },
            }
        ],
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": "yn-gpu-workload",
            "uid": JOB_UID,
            "resourceVersion": resource_version,
            "annotations": annotations,
        },
        "spec": {
            "suspend": suspended,
            "parallelism": parallelism,
            "selector": selector,
            "template": {"metadata": template_metadata, "spec": pod_spec},
        },
        "status": {},
    }


def _binding(job: dict[str, Any]) -> M03RV7AdmittedJobBinding:
    spec = job["spec"]
    fields: dict[str, Any] = {
        "job_name": JOB_NAME,
        "namespace": "yn-gpu-workload",
        "job_uid": JOB_UID,
        "run_id": RUN_ID,
        "first_resource_version": "1",
        "second_resource_version": "2",
        "parallelism": spec["parallelism"],
        "admitted_spec_sha256": common._content_sha256(spec),
        "admitted_pod_template_sha256": common._content_sha256(
            spec["template"]["spec"]
        ),
        "admitted_selector_sha256": hashlib.sha256(
            json.dumps(spec["selector"], separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "admitted_template_metadata_sha256": hashlib.sha256(
            json.dumps(
                spec["template"]["metadata"],
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "desired_manifest_sha256": "e" * 64,
        "attached_owned_pod_uids": (),
        "suspended": True,
    }
    # The admitted spec and selector use the v7 canonical implementation,
    # which omits the trailing newline except for the Pod-spec digest.
    fields["admitted_spec_sha256"] = hashlib.sha256(
        json.dumps(spec, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    canonical = {
        "schema": M03RV7AdmittedJobBinding.__dataclass_fields__["schema"].default,
        **fields,
        "attached_owned_pod_uids": [],
    }
    return M03RV7AdmittedJobBinding(
        **fields,
        receipt_sha256=hashlib.sha256(
            json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
    )


def _pod(index: int) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"v9-pod-{index}",
            "uid": f"v9-pod-uid-{index}",
            "annotations": {"batch.kubernetes.io/job-completion-index": str(index)},
            "ownerReferences": [{"uid": JOB_UID, "kind": "Job", "controller": True}],
        },
        "status": {"phase": "Succeeded"},
    }


def _config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> lifecycle.M03RV9AttachSupervisorConfig:
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    expected = tuple(
        lifecycle.M03RV9ExpectedCompletion(
            completion_index=index,
            setting_index=index,
            setting_id=M03R_V9_SETTING_IDS[index],
            worker_plan_sha256=str(index + 1) * 64,
        )
        for index in range(3)
    )
    return lifecycle.M03RV9AttachSupervisorConfig(
        job_name=JOB_NAME,
        run_id=RUN_ID,
        job_uid=JOB_UID,
        binding_path=str(tmp_path / "binding.json"),
        binding_file_sha256="1" * 64,
        activation_request_path=str(tmp_path / "activation.json"),
        activation_request_file_sha256="2" * 64,
        output_root=str(tmp_path / "output"),
        evidence_root=str(tmp_path / "evidence"),
        package_plan_sha256=PACKAGE_SHA,
        source_archive_sha256=SOURCE_SHA,
        capacity_receipt_sha256=CAPACITY_SHA,
        lifecycle_source_sha256=lifecycle._file_sha256(Path(lifecycle.__file__)),
        expected_completions=expected,
        host_python_path=sys.executable,
        pythonpath=str(tmp_path),
    )


def _capacity_package():
    return build_m03r_v9_package_plan(
        artifacts=M03RV9PackageArtifacts(
            source_archive_sha256=SOURCE_SHA,
            source_manifest_sha256="3" * 64,
            dependency_lock_sha256="4" * 64,
            cache_artifact_sha256="5" * 64,
            cache_manifest_sha256="6" * 64,
            risk_artifact_sha256="7" * 64,
            risk_source_manifest_file_sha256="8" * 64,
            projector_manifest_file_sha256="9" * 64,
            projector_manifest_sha256="a" * 64,
            projector_binding_sha256="b" * 64,
            worker_source_sha256="c" * 64,
            image_reference="registry/research@sha256:" + "d" * 64,
            image_digest_sha256="d" * 64,
        )
    )


def _capacity_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[lifecycle.M03RV9CapacityAttachConfig, Any]:
    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    package = _capacity_package()
    package_path = tmp_path / "package-plan.json"
    package_file_sha = _write_json(package_path, package_plan_file_payload(package))
    config = lifecycle.M03RV9CapacityAttachConfig(
        job_name=JOB_NAME,
        run_id=RUN_ID,
        job_uid=JOB_UID,
        binding_path=str(tmp_path / "capacity-binding.json"),
        binding_file_sha256="1" * 64,
        activation_request_path=str(tmp_path / "capacity-activation.json"),
        activation_request_file_sha256="2" * 64,
        package_plan_path=str(package_path),
        package_plan_file_sha256=package_file_sha,
        package_plan_sha256=package.package_plan_sha256,
        source_archive_sha256=SOURCE_SHA,
        lifecycle_source_sha256=lifecycle._file_sha256(Path(lifecycle.__file__)),
        output_root=str(tmp_path / "capacity-output"),
        evidence_root=str(tmp_path / "capacity-evidence"),
        host_python_path=sys.executable,
        pythonpath=str(tmp_path),
        request_timeout_seconds=5,
        handshake_timeout_seconds=30,
        hard_wall_seconds=60,
    )
    return config, package


def _write_capacity_output(
    config: lifecycle.M03RV9CapacityAttachConfig, package: Any
) -> None:
    root = Path(config.output_root) / "completion-00-setting-00"
    worker = package.panel.workers[0]
    startup = {
        "schema": lifecycle.M03R_V9_STARTUP_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "worker_plan_sha256": worker.receipt_sha256,
        "setting_index": 0,
        "setting_id": M03R_V9_SETTING_IDS[0],
        "mode": "two-h100-capacity",
        "rank_runtime": [
            {
                "rank": rank,
                "local_rank": rank,
                "world_size": 2,
                "visible_device_count": 2,
                "device_name": "NVIDIA H100 80GB HBM3",
                "device_total_memory": 80 * 1024**3,
                "compute_capability": [9, 0],
            }
            for rank in range(2)
        ],
        "exact_h100_80gb_per_rank": True,
        "nccl_process_group_initialized": True,
        "restart_count": 0,
        "research_only": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    startup_sha = _write_json(root / "two-h100-startup.json", startup)
    terminal = _json_receipt(
        {
            "schema": lifecycle.M03R_V9_CAPACITY_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "worker_plan_sha256": worker.receipt_sha256,
            "startup_file_sha256": startup_sha,
            "setting_index": 0,
            "setting_id": M03R_V9_SETTING_IDS[0],
            "world_size": 2,
            "gpus_per_worker": 2,
            "exact_h100_80gb_per_rank": True,
            "nccl_process_group_initialized": True,
            "training_performed": False,
            "economic_optimizer_updates": 0,
            "h100_capacity_evidence": True,
            "research_only": True,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
    )
    _write_json(root / "two-h100-capacity-terminal.json", terminal)


def _write_worker_evidence(
    config: lifecycle.M03RV9AttachSupervisorConfig,
    completion_index: int,
) -> None:
    row = config.expected_completions[completion_index]
    root = (
        Path(config.output_root)
        / f"completion-{completion_index:02d}-setting-{completion_index:02d}"
    )
    startup = {
        "schema": lifecycle.M03R_V9_STARTUP_SCHEMA,
        "package_plan_sha256": PACKAGE_SHA,
        "worker_plan_sha256": row.worker_plan_sha256,
        "setting_index": completion_index,
        "setting_id": row.setting_id,
        "mode": "predictive",
        "rank_runtime": [
            {
                "rank": rank,
                "local_rank": rank,
                "world_size": 2,
                "visible_device_count": 2,
                "device_name": "NVIDIA H100 80GB HBM3",
                "device_total_memory": 80 * 1024**3,
                "compute_capability": [9, 0],
            }
            for rank in range(2)
        ],
        "exact_h100_80gb_per_rank": True,
        "nccl_process_group_initialized": True,
        "restart_count": 0,
        "research_only": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    startup_sha = _write_json(root / "two-h100-startup.json", startup)
    fold_hashes: list[str] = []
    alpha_receipts: dict[int, list[str]] = {21: [], 30: []}
    sleeve_receipts: dict[int, list[str]] = {21: [], 30: []}
    for fold_index in range(6):
        candidates: dict[str, Any] = {}
        for horizon in (21, 30):
            checkpoint = (
                root
                / "checkpoints"
                / f"fold-{fold_index:02d}-horizon-{horizon:02d}-update-0064.pt"
            )
            artifact = (
                root
                / "fold-artifacts"
                / f"fold-{fold_index:02d}-horizon-{horizon:02d}.pt"
            )
            checkpoint_sha = _write_binary(
                checkpoint,
                f"checkpoint-{completion_index}-{fold_index}-{horizon}".encode(),
            )
            artifact_sha = _write_binary(
                artifact, f"artifact-{completion_index}-{fold_index}-{horizon}".encode()
            )
            alpha_sha = hashlib.sha256(
                f"alpha-{completion_index}-{fold_index}-{horizon}".encode()
            ).hexdigest()
            sleeve_sha = hashlib.sha256(
                f"sleeve-{completion_index}-{fold_index}-{horizon}".encode()
            ).hexdigest()
            alpha_receipts[horizon].append(alpha_sha)
            sleeve_receipts[horizon].append(sleeve_sha)
            candidates[str(horizon)] = {
                "horizon_binding_sha256": ("3" * 64 if horizon == 21 else "d" * 64),
                "alpha_head_identity": {},
                "checkpoint_path": (
                    f"/mnt/output/completion-{completion_index:02d}-setting-"
                    f"{completion_index:02d}/checkpoints/{checkpoint.name}"
                ),
                "checkpoint_file_sha256": checkpoint_sha,
                "qualification_artifact_path": (
                    f"/mnt/output/completion-{completion_index:02d}-setting-"
                    f"{completion_index:02d}/fold-artifacts/{artifact.name}"
                ),
                "qualification_artifact_file_sha256": artifact_sha,
                "alpha_evidence_sha256": alpha_sha,
                "sleeve_evidence_sha256": sleeve_sha,
                "sleeve_trace_sha256": "6" * 64,
                "fold_risk_state_sha256": "7" * 64,
            }
        fold = _json_receipt(
            {
                "schema": lifecycle.M03R_V9_FOLD_RESULT_SCHEMA,
                "package_plan_sha256": PACKAGE_SHA,
                "worker_plan_sha256": row.worker_plan_sha256,
                "setting_index": completion_index,
                "setting_id": row.setting_id,
                "fold_index": fold_index,
                "completed_updates": 64,
                "early_stopping_enabled": False,
                "qualification_evaluated_only_after_update64": True,
                "rank_state_equal": True,
                "model_state_sha256": "8" * 64,
                "optimizer_state_sha256": "9" * 64,
                "step_receipt_sha256": ["a" * 64 for _ in range(64)],
                "horizon_candidates": candidates,
                "economic_optimizer_updates": 0,
                "research_only": True,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
        )
        fold_hashes.append(
            _write_json(root / "receipts" / f"fold-{fold_index:02d}.json", fold)
        )
    qualifications = [
        _semantic_receipt(
            {
                "setting_id": row.setting_id,
                "selected_horizon_sessions": horizon,
                "horizon_binding_sha256": ("3" * 64 if horizon == 21 else "d" * 64),
                "fold_alpha_receipt_sha256": alpha_receipts[horizon],
                "fold_sleeve_receipt_sha256": sleeve_receipts[horizon],
                "mean_rank_ic": 0.0,
                "positive_rank_ic_fold_count": 0,
                "mean_top_bottom_spread": 0.0,
                "positive_spread_fold_count": 0,
                "mean_simple_sleeve_gross_active_return": 0.0,
                "mean_simple_sleeve_net_active_return_10bp": 0.0,
                "mean_simple_sleeve_net_active_return_10bp_lcb": -0.1,
                "gross_positive_fold_count": 0,
                "mean_break_even_one_way_cost_basis_points": None,
                "passed": False,
                "economic_generation_may_be_minted": False,
                "economic_panel_authorized": False,
                "protocol_sha256": lifecycle.M03R_V9_PROTOCOL_SHA256,
                "schema": lifecycle.M03R_V9_PREDICTIVE_QUALIFICATION_SCHEMA,
            }
        )
        for horizon in (21, 30)
    ]
    terminal = _json_receipt(
        {
            "schema": lifecycle.M03R_V9_TERMINAL_SCHEMA,
            "package_plan_sha256": PACKAGE_SHA,
            "worker_plan_sha256": row.worker_plan_sha256,
            "startup_file_sha256": startup_sha,
            "setting_index": completion_index,
            "setting_id": row.setting_id,
            "fold_receipt_file_sha256": fold_hashes,
            "horizon_qualification": qualifications,
            "selected_horizon": None,
            "selected_qualification_sha256": None,
            "predictive_gate_passed": False,
            "economic_generation_may_be_minted": False,
            "economic_panel_authorized": False,
            "economic_optimizer_updates": 0,
            "h100_capacity_evidence": True,
            "world_size": 2,
            "gpus_per_worker": 2,
            "research_only": True,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
    )
    _write_json(root / "predictive-terminal.json", terminal)


def test_lifecycle_import_does_not_require_torch() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    program = """
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch deliberately unavailable")
        return None

sys.meta_path.insert(0, BlockTorch())
import rl_quant.training.top2000_m03r_v9_seadragon_lifecycle as lifecycle
import rl_quant.training.top2000_m03r_v9_seadragon_operator as operator
assert lifecycle.ATTACH_CONFIG_SCHEMA
assert operator.CREATE_CONFIG_SCHEMA
assert "torch" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_predictive_coverage_binds_three_workers_and_rejects_artifact_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    for index in range(3):
        _write_worker_evidence(config, index)
    coverage = lifecycle.validate_m03r_v9_predictive_coverage(
        config, owned_pods=tuple(_pod(index) for index in range(3))
    )
    assert coverage["completion_count"] == 3
    assert coverage["economic_panel_authorized"] is False
    assert set(coverage["worker_runtime_proof"]) == {"0", "1", "2"}

    artifact = (
        Path(config.output_root)
        / "completion-00-setting-00/fold-artifacts/fold-00-horizon-21.pt"
    )
    artifact.write_bytes(b"tampered")
    with pytest.raises(
        lifecycle.M03RV9SeadragonLifecycleError, match="artifact file hash drifted"
    ):
        lifecycle.validate_m03r_v9_predictive_coverage(
            config, owned_pods=tuple(_pod(index) for index in range(3))
        )


class _AmbiguousActivationTransport:
    def __init__(self, job: dict[str, Any]) -> None:
        self.job = job
        self.deleted = False

    def get_job(self, *, allow_absent: bool = False) -> dict[str, Any] | None:
        del allow_absent
        return self.job

    def get_owned_pods(self) -> tuple[dict[str, Any], ...]:
        return ()

    def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
        del pod_name, limit_bytes
        return b""

    def activate(self, request: M03RV7ExactJobActivationRequest) -> dict[str, Any]:
        del request
        raise common.SeadragonLifecycleError("simulated transport timeout")

    def delete(
        self,
        request: M03RV7ExactJobCleanupRequest,
        options_path: Path,
    ) -> None:
        del request, options_path
        self.deleted = True


class _PreactivationCleanupTransport(_AmbiguousActivationTransport):
    def __init__(self, job: dict[str, Any], *, ambiguous: bool = False) -> None:
        super().__init__(job)
        self.ambiguous = ambiguous
        self.delete_count = 0

    def get_job(self, *, allow_absent: bool = False) -> dict[str, Any] | None:
        del allow_absent
        return None if self.deleted else self.job

    def delete(
        self,
        request: M03RV7ExactJobCleanupRequest,
        options_path: Path,
    ) -> None:
        del request, options_path
        self.delete_count += 1
        if self.ambiguous:
            raise common.SeadragonLifecycleError("ambiguous delete")
        self.deleted = True


class _CapacitySuccessTransport:
    def __init__(self, suspended_job: dict[str, Any]) -> None:
        self.job = suspended_job
        self.deleted = False
        self.delete_count = 0

    def get_job(self, *, allow_absent: bool = False) -> dict[str, Any] | None:
        del allow_absent
        if self.deleted:
            return None
        return self.job

    def get_owned_pods(self) -> tuple[dict[str, Any], ...]:
        return () if self.deleted or self.job["spec"]["suspend"] else (_pod(0),)

    def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
        del pod_name, limit_bytes
        return b"capacity startup complete\n"

    def activate(self, request: M03RV7ExactJobActivationRequest) -> dict[str, Any]:
        del request
        self.job = json.loads(json.dumps(self.job))
        self.job["spec"]["suspend"] = False
        self.job["metadata"]["resourceVersion"] = "4"
        self.job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
        return self.job

    def delete(
        self,
        request: M03RV7ExactJobCleanupRequest,
        options_path: Path,
    ) -> None:
        del request, options_path
        self.delete_count += 1
        self.deleted = True


class _PostactivationCleanupTransport(_AmbiguousActivationTransport):
    def __init__(self, job: dict[str, Any], *, ambiguous: bool = False) -> None:
        super().__init__(job)
        self.ambiguous = ambiguous
        self.delete_count = 0

    def get_job(self, *, allow_absent: bool = False) -> dict[str, Any] | None:
        del allow_absent
        return None if self.deleted else self.job

    def delete(
        self,
        request: M03RV7ExactJobCleanupRequest,
        options_path: Path,
    ) -> None:
        del request, options_path
        self.delete_count += 1
        if self.ambiguous:
            raise common.SeadragonLifecycleError("ambiguous post-run delete")
        self.deleted = True


def test_activation_transport_error_publishes_attach_and_never_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    Path(config.evidence_root).mkdir()
    job = _job(resource_version="3", suspended=True)
    binding = _binding(job)
    activation = build_m03r_v7_exact_job_activation_request(binding, job)
    binding_sha = _write_json(Path(config.binding_path), asdict(binding))
    activation_sha = _write_json(
        Path(config.activation_request_path), asdict(activation)
    )
    config = lifecycle.M03RV9AttachSupervisorConfig(
        **{
            **asdict(config),
            "binding_file_sha256": binding_sha,
            "activation_request_file_sha256": activation_sha,
            "expected_completions": config.expected_completions,
        }
    )
    config_path = tmp_path / "config.json"
    config_sha = _write_json(config_path, asdict(config))
    _write_json(Path(config.evidence_root) / "spawn-process.json", {"pid": os.getpid()})
    monkeypatch.setattr(
        common, "_validate_spawned_identity", lambda *args, **kwargs: None
    )
    transport = _AmbiguousActivationTransport(job)

    with pytest.raises(lifecycle.M03RV9ActivationAttachRequired):
        lifecycle.run_attach_supervisor(
            config_path,
            config_sha,
            transport=transport,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )
    assert not transport.deleted
    assert (Path(config.evidence_root) / "activation-attach-required.json").is_file()
    assert not (Path(config.evidence_root) / "launch-success.json").exists()


def test_preactivation_cleanup_allows_current_rv_equality_and_deletes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    root = Path(config.evidence_root)
    root.mkdir()
    job = _job(resource_version="2", suspended=True)
    binding = _binding(job)
    transport = _PreactivationCleanupTransport(job)

    lifecycle._cleanup_preactivation_exact(
        root=root,
        config=config,
        binding=binding,
        transport=transport,
        sleep=lambda _: None,
    )
    assert transport.delete_count == 1
    assert transport.deleted
    assert (root / "cleanup-receipt.json").is_file()
    reads = json.loads((root / "preactivation-cleanup-reads.json").read_bytes())
    assert reads["bound_resource_version_equality_permitted"] is True


def test_preactivation_delete_ambiguity_retains_job_and_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    root = Path(config.evidence_root)
    root.mkdir()
    job = _job(resource_version="2", suspended=True)
    binding = _binding(job)
    transport = _PreactivationCleanupTransport(job, ambiguous=True)

    with pytest.raises(lifecycle.M03RV9ActivationAttachRequired):
        lifecycle._cleanup_preactivation_exact(
            root=root,
            config=config,
            binding=binding,
            transport=transport,
            sleep=lambda _: None,
        )
    assert transport.delete_count == 1
    assert not transport.deleted
    assert (root / "preactivation-cleanup-attach-required.json").is_file()
    assert not (root / "cleanup-receipt.json").exists()


def test_capacity_supervisor_publishes_only_after_terminal_and_exact_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, package = _capacity_config(tmp_path, monkeypatch)
    Path(config.evidence_root).mkdir()
    _write_capacity_output(config, package)
    job = _job(
        resource_version="3",
        suspended=True,
        parallelism=1,
        capacity_receipt_sha256="not-yet-created",
        package_plan_sha256=package.package_plan_sha256,
    )
    binding = _binding(job)
    activation = build_m03r_v7_exact_job_activation_request(binding, job)
    binding_sha = _write_json(Path(config.binding_path), asdict(binding))
    activation_sha = _write_json(
        Path(config.activation_request_path), asdict(activation)
    )
    config = lifecycle.M03RV9CapacityAttachConfig(
        **{
            **asdict(config),
            "binding_file_sha256": binding_sha,
            "activation_request_file_sha256": activation_sha,
        }
    )
    config_path = tmp_path / "capacity-config.json"
    config_sha = _write_json(config_path, asdict(config))
    transport = _CapacitySuccessTransport(job)
    _write_json(Path(config.evidence_root) / "spawn-process.json", {"pid": os.getpid()})
    monkeypatch.setattr(
        common, "_validate_spawned_identity", lambda *args, **kwargs: None
    )

    lifecycle.run_capacity_supervisor(
        config_path,
        config_sha,
        transport=transport,
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.0,
    )

    evidence = Path(config.evidence_root)
    assert transport.delete_count == 1
    assert transport.deleted is True
    assert (evidence / "terminal-evidence.json").is_file()
    assert (evidence / "cleanup-receipt.json").is_file()
    qualification = json.loads(
        (evidence / "capacity-qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["passed"] is True
    assert qualification["training_performed"] is False
    assert qualification["cleanup_receipt_file_sha256"] == _sha_bytes(
        (evidence / "cleanup-receipt.json").read_bytes()
    )
    assert qualification["terminal_evidence_file_sha256"] == _sha_bytes(
        (evidence / "terminal-evidence.json").read_bytes()
    )


@pytest.mark.parametrize("ambiguous", [False, True])
def test_postactivation_cleanup_issues_only_one_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambiguous: bool,
) -> None:
    config = _config(tmp_path, monkeypatch)
    root = Path(config.evidence_root)
    root.mkdir()
    job = _job(resource_version="5", suspended=False)
    binding = _binding(job)
    transport = _PostactivationCleanupTransport(job, ambiguous=ambiguous)

    def validate(fresh: dict[str, Any]) -> None:
        common._job_identity(
            fresh,
            job_name=config.job_name,
            run_id=config.run_id,
            job_uid=config.job_uid,
        )

    if ambiguous:
        with pytest.raises(lifecycle.M03RV9ActivationAttachRequired):
            lifecycle._cleanup_postactivation_exact(
                root=root,
                binding=binding,
                transport=transport,
                request_timeout_seconds=5,
                validate_job=validate,
                sleep=lambda _seconds: None,
            )
        assert (root / "cleanup-attach-required.json").is_file()
        assert not (root / "cleanup-receipt.json").exists()
    else:
        lifecycle._cleanup_postactivation_exact(
            root=root,
            binding=binding,
            transport=transport,
            request_timeout_seconds=5,
            validate_job=validate,
            sleep=lambda _seconds: None,
        )
        assert (root / "cleanup-receipt.json").is_file()
    assert transport.delete_count == 1
