from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

import rl_quant.training.top2000_m03r_v16_cohort_runtime as cohort_runtime
import rl_quant.training.top2000_m03r_v16_kubernetes as kubernetes_runtime
import rl_quant.training.top2000_m03r_v16_seadragon_controller as seadragon_controller_runtime
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    _TRAINING_PANEL_ISSUER,
    M03R_V16_ADMITTED_MANIFEST_SCHEMA,
    M03R_V16_DRY_RUN_RESULT_SCHEMA,
    M03R_V16_PREQUALIFICATION_CLOSURE_SCHEMA,
    M03R_V16_TRAINING_PANEL_SCHEMA,
    M03RV16ActivationError,
    M03RV16TrainingPanelAuthority,
    _issue_m03r_v16_admitted_job_authority,
    _issue_m03r_v16_pod_runtime_attestation,
    _issue_m03r_v16_qualification_activation_from_panel_authority,
    _issue_m03r_v16_qualification_outer_access_authority,
    admitted_job_authority_file_sha256,
    load_m03r_v16_admitted_job_authority,
    load_m03r_v16_phase_launch_authority,
    load_m03r_v16_pod_runtime_attestation,
    load_m03r_v16_training_panel_authority,
    write_m03r_v16_admitted_job_authority,
    write_m03r_v16_pod_runtime_attestation,
    write_m03r_v16_qualification_activation,
    write_m03r_v16_qualification_outer_access_authority,
)
from rl_quant.training.top2000_m03r_v16_capacity import (
    M03RV16CapacityRankEvidence,
    build_m03r_v16_capacity_terminal,
)
from rl_quant.training.top2000_m03r_v16_cohort_runtime import (
    M03RV16CohortTrace,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_kubernetes import (
    M03RV16CapacityGateQualification,
    M03RV16KubernetesError,
    M03RV16StaticGateQualification,
    bind_m03r_v16_admitted_launch_authority,
    issue_m03r_v16_training_activation_from_gates,
    load_and_issue_m03r_v16_capacity_gate,
    load_and_issue_m03r_v16_static_gate,
    m03r_v16_pod_runtime_attestation_annotations,
    render_m03r_v16_suspended_capacity_job,
    render_m03r_v16_suspended_qualification_job,
    render_m03r_v16_suspended_qualification_preflight_job,
    render_m03r_v16_suspended_static_job,
    render_m03r_v16_suspended_storage_job,
    render_m03r_v16_suspended_training_job,
    write_m03r_v16_rendered_launch_authority,
)
from rl_quant.training.top2000_m03r_v16_lifecycle import (
    M03RV16PodAnnotationReadback,
    M03RV16PodObservation,
    M03RV16StorageSemanticsEvidence,
    publish_m03r_v16_pod_runtime_attestation_after_annotation_patch,
    qualify_m03r_v16_append_only_storage,
    write_m03r_v16_storage_semantics_evidence,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackageArtifacts,
    M03RV16PackagePlan,
    build_m03r_v16_package_plan,
    write_m03r_v16_execution_authorization,
    write_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_seadragon_controller import (
    M03R_V16_CONTROLLER_TERMINAL_SCHEMA,
    M03R_V16_KUBECTL,
    M03RV16KubectlTransport,
    M03RV16KubernetesConflictError,
    M03RV16SeadragonControllerError,
    admit_m03r_v16_suspended_job,
    cleanup_m03r_v16_exact_job,
    launch_m03r_v16_zero_gpu_gate,
    load_m03r_v16_controller_job_plan,
    resume_and_attest_m03r_v16_job,
    write_m03r_v16_controller_job_plan,
)
from rl_quant.training.top2000_m03r_v16_selection import (
    M03RV16ReconciledFoldEvidence,
    build_m03r_v16_bootstrap_plan,
    qualify_m03r_v16_reconciled_evidence,
)
from rl_quant.training.top2000_m03r_v16_static_contract import (
    M03R_V16_STATIC_RESULT_SCHEMA,
)
from rl_quant.workflows.top2000_m03r_v16_aggregate import (
    M03RV16AggregateError,
    aggregate_m03r_v16_panel,
)
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03R_V16_FOLD_TERMINAL_SCHEMA,
    M03R_V16_QUALIFICATION_ARTIFACT_SCHEMA,
    M03R_V16_TRAINING_TERMINAL_SCHEMA,
    M03R_V16_WORKER_TERMINAL_SCHEMA,
    _write_immutable_json,
    _write_immutable_torch,
)


def test_v16_kubectl_transport_uses_the_supported_filename_stdin_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"apiVersion": "batch/v1", "kind": "Job"}
    response = {"kind": "Job", "metadata": {"name": "contract-test"}}
    observed: dict[str, object] = {}

    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=canonical_json_file_bytes(response),
            stderr=b"",
        )

    monkeypatch.setattr(seadragon_controller_runtime.subprocess, "run", fake_run)
    transport = M03RV16KubectlTransport()
    arguments = (
        "create",
        "--dry-run=server",
        "-f",
        "-",
        "--output",
        "json",
    )

    assert transport.invoke(arguments, payload=payload) == response
    assert observed["command"] == (
        M03R_V16_KUBECTL,
        "--kubeconfig",
        seadragon_controller_runtime.M03R_V16_KUBECONFIG,
        "--context",
        seadragon_controller_runtime.M03R_V16_CONTEXT,
        "--namespace",
        kubernetes_runtime.M03R_V16_NAMESPACE,
        *arguments,
    )
    assert observed["input"] == canonical_json_file_bytes(payload)
    assert observed["capture_output"] is True
    assert observed["check"] is False


def test_v16_controller_never_emits_the_unsupported_file_flag() -> None:
    controller_source = Path(seadragon_controller_runtime.__file__).read_text(
        encoding="utf-8"
    )
    assert '"--file"' not in controller_source
    assert controller_source.count('"-f", "-"') == 4


def test_v16_controller_reads_patch_payloads_from_dev_stdin() -> None:
    controller_source = Path(seadragon_controller_runtime.__file__).read_text(
        encoding="utf-8"
    )
    assert '"--patch-file", "-"' not in controller_source
    assert controller_source.count('"--patch-file", "/dev/stdin"') == 4


def test_v16_kubectl_transport_classifies_api_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            subprocess.CompletedProcess(
                (), 1, stdout=b"", stderr=b"Error from server (NotFound): not found"
            ),
            subprocess.CompletedProcess(
                (), 1, stdout=b"", stderr=b"Error from server: 409 Conflict"
            ),
            subprocess.CompletedProcess(
                (), 1, stdout=b"", stderr=b"error: unknown flag: --file"
            ),
        )
    )

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return next(responses)

    monkeypatch.setattr(seadragon_controller_runtime.subprocess, "run", fake_run)
    transport = M03RV16KubectlTransport()
    get_arguments = ("get", "job", "missing", "--output", "json")
    assert transport.invoke(get_arguments, allow_not_found=True) is None
    with pytest.raises(M03RV16KubernetesConflictError, match="conflict"):
        transport.invoke(get_arguments)
    with pytest.raises(M03RV16SeadragonControllerError, match="unknown flag"):
        transport.invoke(get_arguments)


def test_v16_kubectl_transport_rejects_bad_responses_and_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = M03RV16KubectlTransport()
    arguments = ("get", "job", "contract-test", "--output", "json")

    monkeypatch.setattr(
        seadragon_controller_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            (), 0, stdout=b"not-json", stderr=b""
        ),
    )
    with pytest.raises(M03RV16SeadragonControllerError, match="malformed"):
        transport.invoke(arguments)

    monkeypatch.setattr(seadragon_controller_runtime, "_MAX_KUBECTL_RESPONSE_BYTES", 8)
    monkeypatch.setattr(
        seadragon_controller_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            (), 0, stdout=b'{"long":1}', stderr=b""
        ),
    )
    with pytest.raises(M03RV16SeadragonControllerError, match="oversized"):
        transport.invoke(arguments)

    def time_out(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(M03R_V16_KUBECTL, 120.0)

    monkeypatch.setattr(seadragon_controller_runtime.subprocess, "run", time_out)
    with pytest.raises(M03RV16SeadragonControllerError, match="timed out"):
        transport.invoke(arguments)


@pytest.mark.skipif(
    not Path(M03R_V16_KUBECTL).is_file()
    or not os.access(M03R_V16_KUBECTL, os.X_OK),
    reason="the pinned Seadragon kubectl binary is unavailable locally",
)
def test_v16_pinned_kubectl_create_parser_exposes_filename_flag() -> None:
    completed = subprocess.run(
        (M03R_V16_KUBECTL, "create", "--help"),
        capture_output=True,
        check=False,
        text=True,
        timeout=10.0,
    )
    assert completed.returncode == 0
    assert any(
        "-f" in line and "--filename" in line
        for line in completed.stdout.splitlines()
    )


def _storage_evidence(
    tmp_path: Path,
) -> tuple[M03RV16StorageSemanticsEvidence, Path, str, Path, Path]:
    authority = tmp_path / "authority"
    observer = tmp_path / "authority-observer"
    observer.mkdir(parents=True)

    def publish(_source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(_source, target)

    evidence = qualify_m03r_v16_append_only_storage(
        authority,
        observer_root=observer,
        publish_observer_view=publish,
    )
    path = authority / "storage-semantics.json"
    file_sha = write_m03r_v16_storage_semantics_evidence(
        path,
        evidence,
        authority_root=authority,
        observer_root=observer,
    )
    return evidence, path, file_sha, authority, observer


def _surfaces() -> tuple[
    M03RV16PackagePlan,
    M03RV16ExecutionAuthorization,
    str,
    str,
]:
    digest = "a" * 64
    artifacts = M03RV16PackageArtifacts(
        source_archive_sha256=digest,
        source_manifest_sha256=digest,
        dependency_lock_sha256=digest,
        cache_artifact_sha256=digest,
        cache_manifest_sha256=digest,
        asset_axis_sha256=digest,
        risk_artifact_sha256=digest,
        risk_source_manifest_file_sha256=digest,
        risk_source_receipt_sha256=digest,
        exposure_receipt_sha256=digest,
        projector_manifest_file_sha256=digest,
        projector_manifest_sha256=digest,
        projector_binding_sha256=digest,
        worker_source_sha256=digest,
        operator_source_sha256=digest,
        initial_parameter_state_file_sha256=digest,
        initial_parameter_state_sha256=digest,
        initial_parameter_architecture_sha256=digest,
        structural_slab_file_sha256=digest,
        structural_slab_receipt_sha256=digest,
        structural_action_operator_root_sha256=digest,
        structural_target_operator_root_sha256=digest,
        structural_target_root_sha256=(digest, digest, digest),
        image_reference=f"registry.invalid/quanttrade@sha256:{digest}",
        image_digest_sha256=digest,
    )
    schedule = M03RV16PanelSchedule(
        protocol_common_data_sha256=digest,
        cache_sha256=digest,
        asset_axis_sha256=digest,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v16_fold_geometries(1001)
        ),
    )
    package = build_m03r_v16_package_plan(artifacts, schedule)
    plan_file = "b" * 64
    authorization_file = "c" * 64
    authorization = M03RV16ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256=plan_file,
        source_archive_sha256=artifacts.source_archive_sha256,
        source_manifest_sha256=artifacts.source_manifest_sha256,
        worker_source_sha256=artifacts.worker_source_sha256,
        structural_slab_file_sha256=artifacts.structural_slab_file_sha256,
        structural_slab_receipt_sha256=artifacts.structural_slab_receipt_sha256,
        image_reference=artifacts.image_reference,
    )
    authorization.validate(package)
    return package, authorization, plan_file, authorization_file


def _template(name: str) -> M03RV7KubernetesTemplateConfig:
    return M03RV7KubernetesTemplateConfig(
        job_name=name,
        run_id="m03r-v16-v15-local-contract",
        service_account_name="default",
        pvc_claim_name="research-pvc",
        package_mount_path="/mnt/package",
        output_mount_path="/mnt/output",
    )


def test_v16_exact_isolated_init_bootstrap_prefers_mount_and_stays_torch_free(
    tmp_path: Path,
) -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static = _static_gate(package, authorization, "d" * 64)
    rendered = render_m03r_v16_suspended_capacity_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-init-bootstrap"),
        static=static,
    )
    bootstrap = rendered.manifest["spec"]["template"]["spec"][
        "initContainers"
    ][0]["args"][3]
    source_root = Path(__file__).resolve().parents[1] / "src"
    probe = bootstrap.replace(
        "pathlib.Path('/mnt/package/source/src')", f"pathlib.Path({str(source_root)!r})"
    ).replace(
        "raise SystemExit(gate.main())",
        (
            "import json;"
            "hwm_kib=next(int(line.split()[1]) for line in "
            "pathlib.Path('/proc/self/status').read_text().splitlines() "
            "if line.startswith('VmHWM:'));"
            "print(json.dumps({'path':str(resolved),'torch':('torch' in sys.modules),"
            "'maxrss_kib':hwm_kib}))"
        ),
    )
    decoy = tmp_path / "decoy"
    decoy_gate = (
        decoy
        / "rl_quant/workflows/top2000_m03r_v16_attestation_gate.py"
    )
    decoy_gate.parent.mkdir(parents=True)
    (decoy / "rl_quant/__init__.py").write_text("", encoding="utf-8")
    (decoy / "rl_quant/workflows/__init__.py").write_text("", encoding="utf-8")
    decoy_gate.write_text("raise RuntimeError('decoy imported')\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(decoy)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    evidence = json.loads(result.stdout)
    assert Path(evidence["path"]).resolve().is_relative_to(source_root.resolve())
    assert evidence["torch"] is False
    assert int(evidence["maxrss_kib"]) < 128 * 1024


def _static_gate(
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    manifest_sha256: str,
) -> M03RV16StaticGateQualification:
    return M03RV16StaticGateQualification(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        rendered_manifest_sha256=manifest_sha256,
        result_file_sha256="d" * 64,
        result_receipt_sha256="e" * 64,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        source_tree_root_sha256="9" * 64,
        _issuer=kubernetes_runtime._STATIC_GATE_ISSUER,
    )


def _capacity_gate(
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    static: M03RV16StaticGateQualification,
    manifest_sha256: str,
) -> M03RV16CapacityGateQualification:
    return M03RV16CapacityGateQualification(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        static_gate_receipt_sha256=static.receipt_sha256,
        rendered_manifest_sha256=manifest_sha256,
        terminal_file_sha256="f" * 64,
        terminal_receipt_sha256="1" * 64,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        source_tree_root_sha256=static.source_tree_root_sha256,
        _issuer=kubernetes_runtime._CAPACITY_GATE_ISSUER,
    )


def _capacity_rank(rank: int) -> M03RV16CapacityRankEvidence:
    return M03RV16CapacityRankEvidence(
        setting_index=2,
        distributed_rank=rank,
        distributed_world_size=2,
        cuda_device_name="NVIDIA H100 80GB HBM3",
        cuda_total_memory_bytes=80 * 1024**3,
        peak_allocated_bytes=40 * 1024**3 + rank,
        peak_reserved_bytes=50 * 1024**3 + rank,
        pre_validation_update_receipt_sha256=("1" if rank == 0 else "2") * 64,
        validation_batch_receipt_sha256=("3" if rank == 0 else "4") * 64,
        update_plan_sha256="5" * 64,
        batch_receipt_sha256=("6" if rank == 0 else "7") * 64,
        score_step_receipt_sha256=("8" if rank == 0 else "9") * 64,
        structural_slab_receipt_sha256="a" * 64,
        qualification_projection_receipt_sha256=(
            ("b" if rank == 0 else "c") * 64
        ),
        qualification_requested_active_one_way_mass=0.01,
        qualification_projected_active_one_way_mass=0.0025,
        qualification_requested_to_executed_retention=0.25,
        post_update_model_state_sha256="d" * 64,
        post_update_optimizer_state_sha256="e" * 64,
        episode_state_rows=345,
        global_origin_count=43,
        local_origin_count=22 if rank == 0 else 21,
    )


def test_v16_controller_admits_and_binds_one_exact_suspended_job(
    tmp_path: Path,
) -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static = _static_gate(package, authorization, "d" * 64)
    rendered = render_m03r_v16_suspended_capacity_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-capacity-controller"),
        static=static,
    )
    admitted = json.loads(json.dumps(rendered.manifest))
    admitted["metadata"]["uid"] = "capacity-controller-job-uid"
    admitted["metadata"]["resourceVersion"] = "17"
    live_job = json.loads(json.dumps(admitted))
    calls: list[tuple[str, ...]] = []
    deletion_started = False
    deletion_polls = 0

    class FakeTransport:
        def invoke(
            self,
            arguments: tuple[str, ...],
            *,
            payload: dict | None = None,
            allow_not_found: bool = False,
        ) -> dict | None:
            nonlocal deletion_started, deletion_polls
            del allow_not_found
            calls.append(arguments)
            if arguments[:2] == ("get", "pods"):
                return {"items": []}
            if arguments[:2] == ("create", "--dry-run=server"):
                return json.loads(json.dumps(rendered.manifest))
            if arguments[:2] == ("create", "-f"):
                return json.loads(json.dumps(live_job))
            if arguments[:2] == ("get", "job"):
                if deletion_started:
                    deletion_polls += 1
                    if deletion_polls >= 2:
                        return None
                return json.loads(json.dumps(live_job))
            if arguments[:2] == ("patch", "job"):
                assert payload is not None
                patched = json.loads(json.dumps(live_job))
                patched["metadata"]["resourceVersion"] = "18"
                patched["metadata"]["annotations"].update(
                    payload["metadata"]["annotations"]
                )
                patched["spec"]["template"]["metadata"]["annotations"].update(
                    payload["spec"]["template"]["metadata"]["annotations"]
                )
                live_job.clear()
                live_job.update(json.loads(json.dumps(patched)))
                return patched
            if arguments[:2] == ("delete", "--raw"):
                assert payload is not None
                assert payload["preconditions"]["uid"] == (
                    "capacity-controller-job-uid"
                )
                deletion_started = True
                return {"status": "Success"}
            raise AssertionError(arguments)

    storage, _storage_path, storage_file, authority_root, observer_root = (
        _storage_evidence(tmp_path)
    )
    transport = FakeTransport()
    result = admit_m03r_v16_suspended_job(
        transport=transport,
        rendered=rendered,
        package=package,
        authorization=authorization,
        storage_evidence=storage,
        storage_evidence_file_sha256=storage_file,
        authority_root=authority_root,
        source_tree_root_sha256=static.source_tree_root_sha256,
        run_id=rendered.manifest["metadata"]["labels"]["rl-quant/run-id"],
        storage_authority_identity_root=authority_root,
        storage_observer_identity_root=observer_root,
    )
    assert result.admission.job_uid == "capacity-controller-job-uid"
    assert result.prelaunch_job_resource_version == "18"
    assert result.rendered.launch_authority is not None
    assert result.dry_run_path.is_file()
    assert result.admission_path.is_file()
    assert result.launch_path.is_file()
    assert result.controller_receipt_path.is_file()
    recovered = admit_m03r_v16_suspended_job(
        transport=transport,
        rendered=rendered,
        package=package,
        authorization=authorization,
        storage_evidence=storage,
        storage_evidence_file_sha256=storage_file,
        authority_root=authority_root,
        source_tree_root_sha256=static.source_tree_root_sha256,
        run_id=rendered.manifest["metadata"]["labels"]["rl-quant/run-id"],
        storage_authority_identity_root=authority_root,
        storage_observer_identity_root=observer_root,
    )
    assert recovered.admission.receipt_sha256 == result.admission.receipt_sha256
    assert calls.count(("get", "pods", "--selector", (
        "job-name=m03r-v16-capacity-controller"
    ), "--output", "json")) == 3
    cleanup = cleanup_m03r_v16_exact_job(
        transport=transport,
        controller_admission=recovered,
        package=package,
        authorization=authorization,
        authority_root=authority_root,
        cleanup_authorized=True,
        timeout_seconds=1e-9,
        poll_seconds=0.001,
    )
    assert cleanup["delete_submitted"] is True
    assert cleanup["cleanup_complete"] is False
    assert not (
        authority_root / "capacity-controller-cleanup-complete.json"
    ).exists()
    assert tuple(
        (authority_root / "cleanup-attempts").rglob("attempt-*.json")
    )
    cleanup = cleanup_m03r_v16_exact_job(
        transport=transport,
        controller_admission=recovered,
        package=package,
        authorization=authorization,
        authority_root=authority_root,
        cleanup_authorized=True,
        timeout_seconds=1.0,
        poll_seconds=0.001,
    )
    assert cleanup["cleanup_complete"] is True
    assert (
        authority_root / "capacity-controller-cleanup-complete.json"
    ).is_file()
    assert deletion_polls >= 2

    class ReplacementTransport:
        deleted = False

        def invoke(
            self,
            arguments: tuple[str, ...],
            *,
            payload: dict | None = None,
            allow_not_found: bool = False,
        ) -> dict | None:
            del payload, allow_not_found
            if arguments[:2] == ("get", "job"):
                replacement = json.loads(json.dumps(live_job))
                replacement["metadata"]["uid"] = "replacement-job-uid"
                return replacement
            if arguments[:2] == ("get", "pods"):
                return {"items": []}
            if arguments[:2] == ("delete", "--raw"):
                self.deleted = True
            raise AssertionError(arguments)

    replacement_transport = ReplacementTransport()
    replacement_cleanup = cleanup_m03r_v16_exact_job(
        transport=replacement_transport,
        controller_admission=recovered,
        package=package,
        authorization=authorization,
        authority_root=authority_root / "replacement-cleanup",
        cleanup_authorized=True,
        timeout_seconds=1.0,
        poll_seconds=0.001,
    )
    assert replacement_cleanup["target_already_absent"] is True
    assert replacement_cleanup["cleanup_complete"] is True
    assert replacement_transport.deleted is False


def test_v16_controller_job_plan_round_trips_exact_rendered_bytes(
    tmp_path: Path,
) -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static = _static_gate(package, authorization, "d" * 64)
    rendered = render_m03r_v16_suspended_capacity_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-capacity-controller-plan"),
        static=static,
    )
    path = tmp_path / "capacity-controller-plan.json"
    observed_file_sha256 = write_m03r_v16_controller_job_plan(path, rendered)
    loaded = load_m03r_v16_controller_job_plan(
        path,
        expected_file_sha256=observed_file_sha256,
    )

    assert loaded == rendered
    with pytest.raises(M03RV16SeadragonControllerError, match="hash drifted"):
        load_m03r_v16_controller_job_plan(
            path,
            expected_file_sha256="0" * 64,
        )


def test_v16_zero_gpu_launch_recovers_after_resume_before_receipt(
    tmp_path: Path,
) -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    rendered = render_m03r_v16_suspended_static_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-static-restart"),
    )
    live_job: dict | None = None
    crash_after_resume = True
    resume_calls = 0

    class FakeTransport:
        def invoke(
            self,
            arguments: tuple[str, ...],
            *,
            payload: dict | None = None,
            allow_not_found: bool = False,
        ) -> dict | None:
            nonlocal live_job, crash_after_resume, resume_calls
            del allow_not_found
            if arguments[:2] == ("create", "--dry-run=server"):
                return json.loads(json.dumps(rendered.manifest))
            if arguments[:2] == ("get", "job"):
                return None if live_job is None else json.loads(json.dumps(live_job))
            if arguments[:2] == ("create", "-f"):
                live_job = json.loads(json.dumps(rendered.manifest))
                live_job["metadata"]["uid"] = "static-restart-job-uid"
                live_job["metadata"]["resourceVersion"] = "1"
                return json.loads(json.dumps(live_job))
            if arguments[:2] == ("get", "pods"):
                return {"items": []}
            if arguments[:2] == ("patch", "job"):
                assert live_job is not None and payload is not None
                resume_calls += 1
                live_job["spec"]["suspend"] = False
                live_job["metadata"]["resourceVersion"] = "2"
                if crash_after_resume:
                    raise KeyboardInterrupt("crash after zero-GPU resume")
                return json.loads(json.dumps(live_job))
            raise AssertionError(arguments)

    authority_root = tmp_path / "zero-gpu-authority"
    with pytest.raises(KeyboardInterrupt, match="zero-GPU resume"):
        launch_m03r_v16_zero_gpu_gate(
            transport=FakeTransport(),
            rendered=rendered,
            authority_root=authority_root,
        )
    assert live_job is not None and live_job["spec"]["suspend"] is False
    assert (
        authority_root / "static-zero-gpu-admitted.json"
    ).is_file()
    assert not (
        authority_root / "static-controller-admission.json"
    ).exists()
    crash_after_resume = False
    recovered = launch_m03r_v16_zero_gpu_gate(
        transport=FakeTransport(),
        rendered=rendered,
        authority_root=authority_root,
    )
    assert recovered.job_uid == "static-restart-job-uid"
    assert resume_calls == 1
    assert (
        authority_root / "static-zero-gpu-resumed.json"
    ).is_file()
    assert (
        authority_root / "static-controller-admission.json"
    ).is_file()


def test_v16_zero_gpu_launch_rejects_admission_gpu_injection(
    tmp_path: Path,
) -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    rendered = render_m03r_v16_suspended_static_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-static-gpu-injection"),
    )

    class FakeTransport:
        def invoke(
            self,
            arguments: tuple[str, ...],
            *,
            payload: dict | None = None,
            allow_not_found: bool = False,
        ) -> dict | None:
            del payload, allow_not_found
            if arguments[:2] == ("create", "--dry-run=server"):
                admitted = json.loads(json.dumps(rendered.manifest))
                resources = admitted["spec"]["template"]["spec"]["containers"][0][
                    "resources"
                ]
                resources["requests"]["nvidia.com/gpu"] = "1"
                resources["limits"]["nvidia.com/gpu"] = "1"
                return admitted
            raise AssertionError(arguments)

    with pytest.raises(M03RV16SeadragonControllerError, match="admitted a GPU"):
        launch_m03r_v16_zero_gpu_gate(
            transport=FakeTransport(),
            rendered=rendered,
            authority_root=tmp_path / "zero-gpu-injection-authority",
        )


def test_v16_controller_attests_partial_pods_and_retries_conflict(
    tmp_path: Path,
) -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static = _static_gate(package, authorization, "d" * 64)
    capacity = _capacity_gate(package, authorization, static, "e" * 64)
    training_activation = issue_m03r_v16_training_activation_from_gates(
        package=package,
        authorization=authorization,
        static=static,
        capacity=capacity,
    )
    rendered = render_m03r_v16_suspended_training_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-training-controller-partial"),
        static=static,
        capacity=capacity,
        training_activation=training_activation,
        training_activation_file_sha256="f" * 64,
    )
    live_job = json.loads(json.dumps(rendered.manifest))
    live_job["metadata"]["uid"] = "training-controller-job-uid"
    live_job["metadata"]["resourceVersion"] = "10"
    pods: dict[int, dict] = {}
    events: list[str] = []
    pod_list_calls = 0
    conflict_injected = False
    crash_after_first_attestation = True

    def build_pod(index: int) -> dict:
        return {
            "metadata": {
                "uid": f"pod-uid-{index}",
                "name": f"training-pod-{index}",
                "resourceVersion": "20",
                "annotations": {
                    "batch.kubernetes.io/job-completion-index": str(index)
                },
                "ownerReferences": [
                    {
                        "kind": "Job",
                        "controller": True,
                        "uid": "training-controller-job-uid",
                        "name": "m03r-v16-training-controller-partial",
                    }
                ],
            },
            "spec": {
                "nodeName": f"node-{index}",
                "initContainers": [
                    {
                        "name": "runtime-attestation-gate",
                        "image": package.artifacts.image_reference,
                    }
                ],
                "containers": [
                    {
                        "name": "trainer",
                        "image": package.artifacts.image_reference,
                    }
                ],
            },
            "status": {
                "phase": "Pending",
                "initContainerStatuses": [
                    {
                        "name": "runtime-attestation-gate",
                        "image": package.artifacts.image_reference,
                        "imageID": (
                            "containerd://sha256:"
                            + package.artifacts.image_digest_sha256
                        ),
                    }
                ],
            },
        }

    class FakeTransport:
        def invoke(
            self,
            arguments: tuple[str, ...],
            *,
            payload: dict | None = None,
            allow_not_found: bool = False,
        ) -> dict | None:
            nonlocal pod_list_calls, conflict_injected
            nonlocal crash_after_first_attestation
            del allow_not_found
            if arguments[:2] == ("create", "--dry-run=server"):
                return json.loads(json.dumps(rendered.manifest))
            if arguments[:2] == ("create", "-f"):
                return json.loads(json.dumps(live_job))
            if arguments[:2] == ("get", "job"):
                return json.loads(json.dumps(live_job))
            if arguments[:2] == ("patch", "job"):
                assert payload is not None
                live_job["metadata"]["resourceVersion"] = str(
                    int(live_job["metadata"]["resourceVersion"]) + 1
                )
                live_job["metadata"]["annotations"].update(
                    payload["metadata"].get("annotations", {})
                )
                if "template" in payload.get("spec", {}):
                    live_job["spec"]["template"]["metadata"][
                        "annotations"
                    ].update(
                        payload["spec"]["template"]["metadata"][
                            "annotations"
                        ]
                    )
                live_job["spec"]["suspend"] = payload["spec"]["suspend"]
                return json.loads(json.dumps(live_job))
            if arguments[:2] == ("get", "pods"):
                if live_job["spec"]["suspend"]:
                    return {"items": []}
                if crash_after_first_attestation and "attest:0" in events:
                    raise KeyboardInterrupt("crash after completion-0 attestation")
                pod_list_calls += 1
                visible = (0,) if pod_list_calls <= 2 else (0, 1, 2)
                for index in visible:
                    pods.setdefault(index, build_pod(index))
                events.append("list:" + ",".join(str(index) for index in visible))
                return {
                    "items": [
                        json.loads(json.dumps(pods[index])) for index in visible
                    ]
                }
            if arguments[:2] == ("patch", "pod"):
                assert payload is not None
                index = int(arguments[2].rsplit("-", 1)[-1])
                if index == 0 and not conflict_injected:
                    conflict_injected = True
                    pods[index]["metadata"]["resourceVersion"] = "21"
                    events.append("conflict:0")
                    raise M03RV16KubernetesConflictError("synthetic conflict")
                pods[index]["metadata"]["resourceVersion"] = str(
                    int(pods[index]["metadata"]["resourceVersion"]) + 1
                )
                pods[index]["metadata"]["annotations"].update(
                    payload["metadata"]["annotations"]
                )
                events.append(f"attest:{index}")
                return json.loads(json.dumps(pods[index]))
            if arguments[:2] == ("get", "pod"):
                index = int(arguments[2].rsplit("-", 1)[-1])
                return json.loads(json.dumps(pods[index]))
            raise AssertionError(arguments)

    storage, _storage_path, storage_file, authority_root, observer_root = (
        _storage_evidence(tmp_path)
    )
    controller_admission = admit_m03r_v16_suspended_job(
        transport=FakeTransport(),
        rendered=rendered,
        package=package,
        authorization=authorization,
        storage_evidence=storage,
        storage_evidence_file_sha256=storage_file,
        authority_root=authority_root,
        source_tree_root_sha256=static.source_tree_root_sha256,
        run_id=rendered.manifest["metadata"]["labels"]["rl-quant/run-id"],
        storage_authority_identity_root=authority_root,
        storage_observer_identity_root=observer_root,
    )
    with pytest.raises(KeyboardInterrupt, match="completion-0"):
        resume_and_attest_m03r_v16_job(
            transport=FakeTransport(),
            controller_admission=controller_admission,
            package=package,
            authorization=authorization,
            storage_evidence=storage,
            authority_root=authority_root,
            storage_authority_identity_root=authority_root,
            storage_observer_identity_root=observer_root,
            timeout_seconds=1.0,
            poll_seconds=0.001,
        )
    journal = (
        authority_root
        / "controller-attestations"
        / "training"
        / "training-controller-job-uid"
        / "completion-00.json"
    )
    assert journal.is_file()
    # Exercise the narrower crash boundary where the immutable attestation
    # exists and Pod annotations are visible, but the per-completion journal
    # was not durably published.
    journal.unlink()
    assert not (authority_root / "training-controller-attestations.json").exists()
    completion_zero_annotations = dict(pods[0]["metadata"]["annotations"])
    completion_zero_resource_version = pods[0]["metadata"]["resourceVersion"]
    crash_after_first_attestation = False
    result = resume_and_attest_m03r_v16_job(
        transport=FakeTransport(),
        controller_admission=controller_admission,
        package=package,
        authorization=authorization,
        storage_evidence=storage,
        authority_root=authority_root,
        storage_authority_identity_root=authority_root,
        storage_observer_identity_root=observer_root,
        timeout_seconds=1.0,
        poll_seconds=0.001,
    )

    assert len(result.rows) == 3
    assert events.index("attest:0") < events.index("list:0,1,2")
    assert events.count("conflict:0") == 1
    assert events.count("attest:0") == 1
    assert pods[0]["metadata"]["annotations"] == completion_zero_annotations
    assert (
        pods[0]["metadata"]["resourceVersion"]
        == completion_zero_resource_version
    )
    recovered_journal = json.loads(journal.read_text())
    assert recovered_journal["recovery_mode"] == (
        "recovered-after-final-publication"
    )
    controller_receipt = json.loads(result.controller_receipt_path.read_text())
    assert controller_receipt["scheduling_policy"] == (
        "independent-per-completion"
    )
    assert controller_receipt["attestation_release_mode"] == (
        "as-each-completion-becomes-observable"
    )
    assert controller_receipt["gang_scheduling_required"] is False
    recovered = resume_and_attest_m03r_v16_job(
        transport=FakeTransport(),
        controller_admission=controller_admission,
        package=package,
        authorization=authorization,
        storage_evidence=storage,
        authority_root=authority_root,
        storage_authority_identity_root=authority_root,
        storage_observer_identity_root=observer_root,
        timeout_seconds=1.0,
        poll_seconds=0.001,
    )
    assert recovered.controller_receipt_path == result.controller_receipt_path
    assert tuple(row.file_sha256 for row in recovered.rows) == tuple(
        row.file_sha256 for row in result.rows
    )


def _qualification_activation(package, authorization, source_root):
    adequacy = tuple(
        tuple(f"{20 + setting * 5 + fold:064x}" for fold in range(5))
        for setting in range(3)
    )
    checkpoints = tuple(
        tuple(f"{40 + setting * 5 + fold:064x}" for fold in range(5))
        for setting in range(3)
    )
    panel = M03RV16TrainingPanelAuthority(
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        training_panel_receipt_sha256="6" * 64,
        training_panel_file_sha256="7" * 64,
        training_terminal_file_sha256=("3" * 64, "4" * 64, "5" * 64),
        training_terminal_receipt_sha256=("a" * 64, "b" * 64, "c" * 64),
        setting_fold_training_adequacy_receipt_sha256=adequacy,
        setting_fold_training_adequacy_status=(("adequate",) * 5,) * 3,
        terminal_checkpoint_file_sha256=checkpoints,
        prequalification_closure_receipt_sha256="9" * 64,
        prequalification_closure_file_sha256="8" * 64,
        source_tree_root_sha256=source_root,
        _issuer=_TRAINING_PANEL_ISSUER,
    )
    return _issue_m03r_v16_qualification_activation_from_panel_authority(
        package=package,
        authorization=authorization,
        panel=panel,
    )


def test_v16_jobs_are_suspended_and_gate_predictive_h100_panel(
    tmp_path: Path,
) -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static_job = render_m03r_v16_suspended_static_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-static"),
    )
    assert static_job.maximum_gpu_requests == 0
    assert static_job.manifest["spec"]["suspend"] is True
    resources = static_job.manifest["spec"]["template"]["spec"]["containers"][0][
        "resources"
    ]
    assert resources["requests"]["nvidia.com/gpu"] == "0"
    assert resources["limits"]["nvidia.com/gpu"] == "0"
    static = _static_gate(package, authorization, static_job.manifest_sha256)
    storage_job = render_m03r_v16_suspended_storage_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-storage"),
        static=static,
    )
    assert storage_job.maximum_gpu_requests == 0
    storage_pod = storage_job.manifest["spec"]["template"]["spec"]
    assert "initContainers" not in storage_pod
    storage_mounts = storage_pod["containers"][0]["volumeMounts"]
    assert any(
        row["mountPath"] == "/mnt/authority" and not row["readOnly"]
        for row in storage_mounts
    )
    assert any(
        row["mountPath"] == "/mnt/authority-observer" and row["readOnly"]
        for row in storage_mounts
    )
    capacity_job = render_m03r_v16_suspended_capacity_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-capacity"),
        static=static,
    )
    dry_unsigned = {
        "schema": M03R_V16_DRY_RUN_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "phase": "capacity",
        "job_contract_sha256": capacity_job.job_contract_sha256,
        "pod_contract_sha256": capacity_job.pod_contract_sha256,
        "passed": True,
    }
    dry_result = {
        **dry_unsigned,
        "receipt_sha256": semantic_sha256(dry_unsigned),
    }
    dry_path = tmp_path / "capacity-dry-run.json"
    dry_path.write_bytes(canonical_json_file_bytes(dry_result))
    admitted_unsigned = {
        "schema": M03R_V16_ADMITTED_MANIFEST_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "phase": "capacity",
        "job_contract_sha256": capacity_job.job_contract_sha256,
        "pod_contract_sha256": capacity_job.pod_contract_sha256,
        "job_uid": "capacity-job-uid",
        "job_name": "m03r-v16-capacity",
        "image_reference": package.artifacts.image_reference,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "suspended_at_admission": True,
    }
    admitted_result = {
        **admitted_unsigned,
        "receipt_sha256": semantic_sha256(admitted_unsigned),
    }
    admitted_path = tmp_path / "capacity-admitted-manifest.json"
    admitted_path.write_bytes(canonical_json_file_bytes(admitted_result))
    admission = _issue_m03r_v16_admitted_job_authority(
        package=package,
        authorization=authorization,
        phase="capacity",
        run_id="m03r-v16-v15-local-contract",
        job_contract_sha256=capacity_job.job_contract_sha256,
        pod_contract_sha256=capacity_job.pod_contract_sha256,
        server_side_dry_run_file_sha256=file_sha256(dry_path),
        server_side_dry_run_receipt_sha256=dry_result["receipt_sha256"],
        admitted_manifest_file_sha256=file_sha256(admitted_path),
        admitted_manifest_sha256=admitted_result["receipt_sha256"],
        job_uid="capacity-job-uid",
        job_name="m03r-v16-capacity",
    )
    admission_file = admitted_job_authority_file_sha256(admission)
    storage, _storage_path, storage_file, authority_root, observer_root = (
        _storage_evidence(tmp_path)
    )
    capacity_job = bind_m03r_v16_admitted_launch_authority(
        rendered=capacity_job,
        package=package,
        authorization=authorization,
        admission=admission,
        admission_file_sha256=admission_file,
        source_tree_root_sha256=static.source_tree_root_sha256,
        storage_evidence=storage,
        storage_evidence_file_sha256=storage_file,
        authority_root=authority_root,
        observer_root=observer_root,
    )
    assert capacity_job.maximum_gpu_requests == 2
    assert admission.completions == 1
    assert "pod_uids" not in asdict(admission)
    container = capacity_job.manifest["spec"]["template"]["spec"][
        "containers"
    ][0]
    environment_names = {row["name"] for row in container["env"]}
    assert {
        "M03R_V16_CURRENT_POD_UID",
        "M03R_V16_CURRENT_POD_NAME",
        "M03R_V16_CURRENT_NODE_NAME",
        "M03R_V16_POD_ATTESTATION_FILE_SHA256",
        "M03R_V16_POD_ATTESTATION_RECEIPT_SHA256",
        "M03R_V16_POD_ATTESTATION_PATH",
    }.issubset(environment_names)
    init_container = capacity_job.manifest["spec"]["template"]["spec"][
        "initContainers"
    ][0]
    assert init_container["name"] == "runtime-attestation-gate"
    assert init_container["args"][:3] == ["-I", "-B", "-c"]
    assert "sys.path.insert(0,str(root))" in init_container["args"][3]
    assert "top2000_m03r_v16_attestation_gate" in init_container["args"][3]
    assert "PYTHONPATH" not in {row["name"] for row in init_container["env"]}
    assert init_container["resources"] == {
        "requests": {"cpu": "50m", "memory": "512Mi"},
        "limits": {"cpu": "250m", "memory": "1Gi"},
    }
    assert init_container["image"] == container["image"]
    assert any(
        row["mountPath"] == "/etc/podinfo"
        for row in init_container["volumeMounts"]
    )
    assert "--pod-runtime-attestation-marker" in container["args"]
    output_root_sha = semantic_sha256(
        {"output_root": "/mnt/output/capacity-sentinel"}
    )
    assert capacity_job.launch_authority is not None
    relative_attestation_path = (
        capacity_job.launch_authority.pod_runtime_attestation_relative_path(0)
    )
    package_image_repository = package.artifacts.image_reference.split(
        "@sha256:", 1
    )[0]
    runtime_image_id_repository = package_image_repository.rsplit(":", 1)[0]
    pod_attestation = _issue_m03r_v16_pod_runtime_attestation(
        package=package,
        authorization=authorization,
        admission=admission,
        launch=capacity_job.launch_authority,
        completion_index=0,
        pod_uid="capacity-pod-uid",
        pod_name="capacity-pod-0",
        node_name="capacity-node",
        observed_owner_job_uid=admission.job_uid,
        observed_owner_job_name="m03r-v16-capacity",
        observed_completion_index=0,
        observed_pod_resource_version="17",
        relative_path=relative_attestation_path,
        attested_container_name="runtime-attestation-gate",
        attested_container_kind="init",
        observed_spec_image=package.artifacts.image_reference,
        observed_status_image="sha256:" + "c" * 64,
        observed_status_image_id=(
            runtime_image_id_repository
            + "@sha256:"
            + package.artifacts.image_digest_sha256
        ),
        output_root_sha256=output_root_sha,
    )
    pod_attestation_path = tmp_path / relative_attestation_path
    pod_attestation_file = write_m03r_v16_pod_runtime_attestation(
        pod_attestation_path, pod_attestation
    )
    annotations = m03r_v16_pod_runtime_attestation_annotations(pod_attestation)
    assert annotations["rl-quant/pod-runtime-attestation-path"] == (
        relative_attestation_path
    )
    assert annotations["rl-quant/pod-runtime-attestation-file-sha256"] == (
        pod_attestation_file
    )
    assert not tuple(pod_attestation_path.parent.glob("*.tmp"))
    assert write_m03r_v16_pod_runtime_attestation(
        pod_attestation_path, pod_attestation
    ) == pod_attestation_file
    controller_root = authority_root
    patched: dict[str, str] = {}

    def patch_annotations(precondition: object, values: dict[str, str]) -> None:
        assert not (controller_root / relative_attestation_path).exists()
        assert precondition.pod_uid == "capacity-pod-uid"
        assert precondition.pod_resource_version == "17"
        patched.update(values)

    stale_temporary = (
        controller_root
        / relative_attestation_path
    ).with_name(
        f".completion-00.json.{pod_attestation.receipt_sha256}.stale.tmp"
    )
    stale_temporary.parent.mkdir(parents=True, exist_ok=True)
    stale_temporary.write_text("abandoned-controller-attempt", encoding="utf-8")

    published = publish_m03r_v16_pod_runtime_attestation_after_annotation_patch(
        package=package,
        authorization=authorization,
        admission=admission,
        launch=capacity_job.launch_authority,
        observation=M03RV16PodObservation(
            completion_index=0,
            pod_uid="capacity-pod-uid",
            pod_name="capacity-pod-0",
            node_name="capacity-node",
            observed_owner_job_uid=admission.job_uid,
            observed_owner_job_name="m03r-v16-capacity",
            observed_completion_index=0,
            observed_pod_resource_version="17",
            attested_container_name="runtime-attestation-gate",
            attested_container_kind="init",
            observed_spec_image=package.artifacts.image_reference,
            observed_status_image="sha256:" + "c" * 64,
            observed_status_image_id=(
                runtime_image_id_repository
                + "@sha256:"
                + package.artifacts.image_digest_sha256
            ),
        ),
        output_root_sha256=output_root_sha,
        authority_root=controller_root,
        observer_root=observer_root,
        storage_evidence=storage,
        patch_annotations=patch_annotations,
        read_annotations=lambda _precondition: M03RV16PodAnnotationReadback(
            pod_uid="capacity-pod-uid",
            pod_resource_version="18",
            annotations=patched,
        ),
    )
    assert published.final_path.is_file()
    assert not stale_temporary.exists()
    assert published.relative_path == relative_attestation_path
    assert len(published.controller_transaction_receipt_sha256) == 64
    assert load_m03r_v16_pod_runtime_attestation(
        pod_attestation_path,
        expected_file_sha256=pod_attestation_file,
        expected_receipt_sha256=pod_attestation.receipt_sha256,
        package=package,
        authorization=authorization,
        admission=admission,
        launch=capacity_job.launch_authority,
        expected_completion_index=0,
        expected_output_root_sha256=output_root_sha,
        current_pod_uid="capacity-pod-uid",
        current_pod_name="capacity-pod-0",
        current_node_name="capacity-node",
        expected_relative_path=relative_attestation_path,
    ) == pod_attestation
    with pytest.raises(M03RV16ActivationError, match="Pod identity"):
        load_m03r_v16_pod_runtime_attestation(
            pod_attestation_path,
            expected_file_sha256=pod_attestation_file,
            expected_receipt_sha256=pod_attestation.receipt_sha256,
            package=package,
            authorization=authorization,
            admission=admission,
            launch=capacity_job.launch_authority,
            expected_completion_index=0,
            expected_output_root_sha256=output_root_sha,
            current_pod_uid="different-pod",
            current_pod_name="capacity-pod-0",
            current_node_name="capacity-node",
            expected_relative_path=relative_attestation_path,
        )
    with pytest.raises(M03RV16ActivationError, match="runtime attestation"):
        _issue_m03r_v16_pod_runtime_attestation(
            package=package,
            authorization=authorization,
            admission=admission,
            launch=capacity_job.launch_authority,
            completion_index=0,
            pod_uid="capacity-pod-uid",
            pod_name="capacity-pod-0",
            node_name="capacity-node",
            observed_owner_job_uid=admission.job_uid,
            observed_owner_job_name="m03r-v16-capacity",
            observed_completion_index=0,
            observed_pod_resource_version="17",
            relative_path=relative_attestation_path,
            attested_container_name="runtime-attestation-gate",
            attested_container_kind="init",
            observed_spec_image=(
                "different.invalid/q@sha256:"
                + package.artifacts.image_digest_sha256
            ),
            observed_status_image=(
                "different.invalid/q@sha256:"
                + package.artifacts.image_digest_sha256
            ),
            observed_status_image_id=(
                "containerd://sha256:" + package.artifacts.image_digest_sha256
            ),
            output_root_sha256=output_root_sha,
        )
    with pytest.raises(M03RV16ActivationError, match="runtime attestation"):
        _issue_m03r_v16_pod_runtime_attestation(
            package=package,
            authorization=authorization,
            admission=admission,
            launch=capacity_job.launch_authority,
            completion_index=0,
            pod_uid="capacity-pod-uid",
            pod_name="capacity-pod-0",
            node_name="capacity-node",
            observed_owner_job_uid=admission.job_uid,
            observed_owner_job_name="m03r-v16-capacity",
            observed_completion_index=0,
            observed_pod_resource_version="17",
            relative_path=(
                "pod-runtime/capacity/stale-job-uid/completion-00.json"
            ),
            attested_container_name="runtime-attestation-gate",
            attested_container_kind="init",
            observed_spec_image=package.artifacts.image_reference,
            observed_status_image=package.artifacts.image_reference,
            observed_status_image_id=(
                "docker-pullable://" + package.artifacts.image_reference
            ),
            output_root_sha256=output_root_sha,
        )
    capacity = _capacity_gate(
        package, authorization, static, capacity_job.manifest_sha256
    )
    activation = issue_m03r_v16_training_activation_from_gates(
        package=package,
        authorization=authorization,
        static=static,
        capacity=capacity,
    )
    predictive = render_m03r_v16_suspended_training_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-predictive"),
        static=static,
        capacity=capacity,
        training_activation=activation,
        training_activation_file_sha256="2" * 64,
    )
    assert predictive.completions == 3
    assert predictive.parallelism == 3
    assert predictive.maximum_gpu_requests == 6
    assert predictive.manifest["spec"]["suspend"] is True
    assert predictive.activation_authorized is False
    assert predictive.launch_authority is None
    dry_run_unsigned = {
        "schema": M03R_V16_DRY_RUN_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "phase": "training",
        "job_contract_sha256": predictive.job_contract_sha256,
        "pod_contract_sha256": predictive.pod_contract_sha256,
        "passed": True,
    }
    dry_run = {
        **dry_run_unsigned,
        "receipt_sha256": semantic_sha256(dry_run_unsigned),
    }
    dry_run_path = tmp_path / "training-dry-run.json"
    dry_run_file_sha256 = _write_immutable_json(dry_run_path, dry_run)
    admitted_unsigned = {
        "schema": M03R_V16_ADMITTED_MANIFEST_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "phase": "training",
        "job_contract_sha256": predictive.job_contract_sha256,
        "pod_contract_sha256": predictive.pod_contract_sha256,
        "job_uid": "job-uid-1",
        "job_name": "m03r-v16-training",
        "image_reference": package.artifacts.image_reference,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "suspended_at_admission": True,
    }
    admitted_result = {
        **admitted_unsigned,
        "receipt_sha256": semantic_sha256(admitted_unsigned),
    }
    admitted_result_path = tmp_path / "training-admitted-manifest.json"
    admitted_result_file_sha256 = _write_immutable_json(
        admitted_result_path, admitted_result
    )
    admission = _issue_m03r_v16_admitted_job_authority(
        package=package,
        authorization=authorization,
        phase="training",
        run_id="m03r-v16-v15-local-contract",
        job_contract_sha256=predictive.job_contract_sha256,
        pod_contract_sha256=predictive.pod_contract_sha256,
        server_side_dry_run_file_sha256=dry_run_file_sha256,
        server_side_dry_run_receipt_sha256=dry_run["receipt_sha256"],
        admitted_manifest_file_sha256=admitted_result_file_sha256,
        admitted_manifest_sha256=admitted_result["receipt_sha256"],
        job_uid="job-uid-1",
        job_name="m03r-v16-training",
    )
    admission_path = tmp_path / "training-admission.json"
    admission_file_sha256 = write_m03r_v16_admitted_job_authority(
        admission_path, admission
    )
    assert admission_file_sha256 == admitted_job_authority_file_sha256(admission)
    assert load_m03r_v16_admitted_job_authority(
        admission_path,
        expected_file_sha256=admission_file_sha256,
        expected_receipt_sha256=admission.receipt_sha256,
        package=package,
        authorization=authorization,
        expected_phase="training",
        expected_job_contract_sha256=predictive.job_contract_sha256,
        expected_pod_contract_sha256=predictive.pod_contract_sha256,
        server_side_dry_run_path=dry_run_path,
        admitted_manifest_path=admitted_result_path,
    ) == admission
    predictive = bind_m03r_v16_admitted_launch_authority(
        rendered=predictive,
        package=package,
        authorization=authorization,
        admission=admission,
        admission_file_sha256=admission_file_sha256,
        source_tree_root_sha256=capacity.source_tree_root_sha256,
        storage_evidence=storage,
        storage_evidence_file_sha256=storage_file,
        authority_root=authority_root,
        observer_root=observer_root,
    )
    assert predictive.launch_authority is not None
    qualification_launch = replace(
        predictive.launch_authority,
        phase="qualification",
        job_uid="qualification-job-uid",
        pod_runtime_attestation_path_template=(
            "pod-runtime/qualification/qualification-job-uid/"
            "completion-{completion_index:02d}.json"
        ),
    )
    sequential_paths = {
        relative_attestation_path,
        *(
            predictive.launch_authority.pod_runtime_attestation_relative_path(index)
            for index in range(3)
        ),
        *(
            qualification_launch.pod_runtime_attestation_relative_path(index)
            for index in range(3)
        ),
    }
    assert len(sequential_paths) == 7
    assert all("job-uid-1" in path for path in sequential_paths if "training" in path)
    assert predictive.launch_authority.job_contract_sha256 == (
        predictive.job_contract_sha256
    )
    launch_path = tmp_path / "training-launch.json"
    assert write_m03r_v16_rendered_launch_authority(
        launch_path, predictive
    ) == predictive.launch_authority_file_sha256
    assert load_m03r_v16_phase_launch_authority(
        launch_path,
        expected_file_sha256=predictive.launch_authority_file_sha256,
        expected_receipt_sha256=predictive.launch_authority.receipt_sha256,
        package=package,
        authorization=authorization,
        expected_phase="training",
        expected_prerequisite_receipt_sha256=activation.receipt_sha256,
        expected_job_contract_sha256=predictive.job_contract_sha256,
        expected_pod_contract_sha256=predictive.pod_contract_sha256,
        admission=admission,
        expected_admission_file_sha256=admission_file_sha256,
    ) == predictive.launch_authority
    worker_args = predictive.manifest["spec"]["template"]["spec"]["containers"][0][
        "args"
    ]
    assert "--launch-authority" in worker_args
    assert "--rendered-manifest-sha256" in worker_args
    qualification_activation = _qualification_activation(
        package, authorization, capacity.source_tree_root_sha256
    )
    preflight_job = render_m03r_v16_suspended_qualification_preflight_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-qualification-preflight"),
        static=static,
        capacity=capacity,
        qualification_activation=qualification_activation,
        qualification_activation_file_sha256="8" * 64,
    )
    assert preflight_job.maximum_gpu_requests == 0
    assert preflight_job.manifest["spec"]["template"]["spec"]["containers"][0][
        "resources"
    ]["requests"]["nvidia.com/gpu"] == "0"
    preflight_resources = preflight_job.manifest["spec"]["template"]["spec"][
        "containers"
    ][0]["resources"]
    assert preflight_resources["requests"] == {
        "cpu": "12",
        "memory": "64Gi",
        "ephemeral-storage": "8Gi",
        "nvidia.com/gpu": "0",
    }
    assert preflight_resources["limits"] == {
        "cpu": "16",
        "memory": "128Gi",
        "ephemeral-storage": "16Gi",
        "nvidia.com/gpu": "0",
    }
    outer_access = _issue_m03r_v16_qualification_outer_access_authority(
        package=package,
        authorization=authorization,
        activation=qualification_activation,
        setting_input_closure_file_sha256=("1" * 64,) * 3,
        setting_input_closure_receipt_sha256=("2" * 64,) * 3,
        setting_preflight_terminal_file_sha256=("4" * 64,) * 3,
        setting_preflight_terminal_receipt_sha256=("5" * 64,) * 3,
        qualification_risk_input_root_sha256="3" * 64,
    )
    qualification_job = render_m03r_v16_suspended_qualification_job(
        package=package,
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-qualification"),
        static=static,
        capacity=capacity,
        qualification_activation=qualification_activation,
        qualification_activation_file_sha256="8" * 64,
        qualification_outer_access=outer_access,
        qualification_outer_access_file_sha256="9" * 64,
    )
    assert qualification_job.mode == "qualification"
    mounts = qualification_job.manifest["spec"]["template"]["spec"][
        "containers"
    ][0]["volumeMounts"]
    assert any(
        row.get("mountPath") == "/mnt/training" and row.get("readOnly") is True
        for row in mounts
    )


def test_v16_predictive_job_rejects_missing_or_drifted_capacity() -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static_job = render_m03r_v16_suspended_static_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-static-two"),
    )
    static = _static_gate(package, authorization, static_job.manifest_sha256)
    capacity = _capacity_gate(package, authorization, static, "f" * 64)
    with pytest.raises(M03RV16KubernetesError, match="capacity"):
        activation = issue_m03r_v16_training_activation_from_gates(
            package=package,
            authorization=authorization,
            static=static,
            capacity=capacity,
        )
        render_m03r_v16_suspended_training_job(
            package=package,  # type: ignore[arg-type]
            authorization=authorization,
            package_plan_file_sha256=plan_file,
            authorization_file_sha256=authorization_file,
            template=_template("m03r-v16-predictive-two"),
            static=static,
            capacity=replace(capacity, passed=False),
            training_activation=activation,
            training_activation_file_sha256="2" * 64,
        )


def test_v16_gate_authorities_are_issued_from_exact_result_files(
    tmp_path: Path,
) -> None:
    package, authorization, plan_file, authorization_file = _surfaces()
    static_job = render_m03r_v16_suspended_static_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-static-issued"),
    )
    source_tree_root = "f" * 64
    unsigned_static = {
        "schema": M03R_V16_STATIC_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "source_tree_root_sha256": source_tree_root,
        "gpu_mask": "none",
        "gpu_requests": 0,
        "gpu_limits": 0,
        "unmasked_visibility_claimed": False,
        "training_performed": False,
        "initial_state_strict_loaded_all_settings": True,
        "development_only": True,
    }
    static_result = {
        **unsigned_static,
        "receipt_sha256": semantic_sha256(unsigned_static),
    }
    static_path = tmp_path / "static-result.json"
    static_path.write_bytes(canonical_json_file_bytes(static_result))
    static = load_and_issue_m03r_v16_static_gate(
        static_path,
        expected_result_file_sha256=file_sha256(static_path),
        rendered=static_job,
        package=package,
        authorization=authorization,
    )
    static.validate_for(package, authorization)

    capacity_job = render_m03r_v16_suspended_capacity_job(
        package=package,  # type: ignore[arg-type]
        authorization=authorization,
        package_plan_file_sha256=plan_file,
        authorization_file_sha256=authorization_file,
        template=_template("m03r-v16-capacity-issued"),
        static=static,
    )
    dry_unsigned = {
        "schema": M03R_V16_DRY_RUN_RESULT_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "phase": "capacity",
        "job_contract_sha256": capacity_job.job_contract_sha256,
        "pod_contract_sha256": capacity_job.pod_contract_sha256,
        "passed": True,
    }
    dry_result = {
        **dry_unsigned,
        "receipt_sha256": semantic_sha256(dry_unsigned),
    }
    dry_path = tmp_path / "capacity-issued-dry-run.json"
    dry_path.write_bytes(canonical_json_file_bytes(dry_result))
    admitted_unsigned = {
        "schema": M03R_V16_ADMITTED_MANIFEST_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "phase": "capacity",
        "job_contract_sha256": capacity_job.job_contract_sha256,
        "pod_contract_sha256": capacity_job.pod_contract_sha256,
        "job_uid": "capacity-issued-job-uid",
        "job_name": "m03r-v16-capacity-issued",
        "image_reference": package.artifacts.image_reference,
        "image_digest_sha256": package.artifacts.image_digest_sha256,
        "suspended_at_admission": True,
    }
    admitted_result = {
        **admitted_unsigned,
        "receipt_sha256": semantic_sha256(admitted_unsigned),
    }
    admitted_path = tmp_path / "capacity-issued-admitted-manifest.json"
    admitted_path.write_bytes(canonical_json_file_bytes(admitted_result))
    admission = _issue_m03r_v16_admitted_job_authority(
        package=package,
        authorization=authorization,
        phase="capacity",
        run_id="m03r-v16-v15-local-contract",
        job_contract_sha256=capacity_job.job_contract_sha256,
        pod_contract_sha256=capacity_job.pod_contract_sha256,
        server_side_dry_run_file_sha256=file_sha256(dry_path),
        server_side_dry_run_receipt_sha256=dry_result["receipt_sha256"],
        admitted_manifest_file_sha256=file_sha256(admitted_path),
        admitted_manifest_sha256=admitted_result["receipt_sha256"],
        job_uid="capacity-issued-job-uid",
        job_name="m03r-v16-capacity-issued",
    )
    storage, _storage_path, storage_file, authority_root, observer_root = (
        _storage_evidence(tmp_path)
    )
    capacity_job = bind_m03r_v16_admitted_launch_authority(
        rendered=capacity_job,
        package=package,
        authorization=authorization,
        admission=admission,
        admission_file_sha256=admitted_job_authority_file_sha256(admission),
        source_tree_root_sha256=source_tree_root,
        storage_evidence=storage,
        storage_evidence_file_sha256=storage_file,
        authority_root=authority_root,
        observer_root=observer_root,
    )
    terminal = build_m03r_v16_capacity_terminal(
        (_capacity_rank(0), _capacity_rank(1))
    )
    capacity_result = {
        "schema": terminal.schema,
        "package_plan_sha256": package.package_plan_sha256,
        "authorization_receipt_sha256": authorization.receipt_sha256,
        "source_tree_root_sha256": source_tree_root,
        "rendered_manifest_sha256": capacity_job.job_contract_sha256,
        "pod_template_sha256": capacity_job.pod_contract_sha256,
        "launch_authority_receipt_sha256": (
            capacity_job.launch_authority.receipt_sha256
        ),
        "admitted_job_authority_receipt_sha256": admission.receipt_sha256,
        "job_uid": admission.job_uid,
        "pod_runtime_attestation_receipt_sha256": "8" * 64,
        "pod_uid": "capacity-issued-pod-uid",
        "capacity": asdict(terminal),
        "capacity_receipt_sha256": terminal.receipt_sha256,
        "scientific_training_performed": False,
        "disposable_optimizer_update_executed": True,
        "disposable_train_validate_train_executed": True,
        "scientific_checkpoint_published": False,
        "development_only": True,
    }
    capacity_path = tmp_path / "two-h100-capacity-terminal.json"
    capacity_path.write_bytes(canonical_json_file_bytes(capacity_result))
    capacity = load_and_issue_m03r_v16_capacity_gate(
        capacity_path,
        expected_terminal_file_sha256=file_sha256(capacity_path),
        rendered=capacity_job,
        package=package,
        authorization=authorization,
        static=static,
    )
    capacity.validate_for(package, authorization, static)

    with pytest.raises(TypeError):
        M03RV16StaticGateQualification(  # type: ignore[call-arg]
            package_plan_sha256=package.package_plan_sha256,
            execution_authorization_receipt_sha256=authorization.receipt_sha256,
            rendered_manifest_sha256=static_job.manifest_sha256,
            result_file_sha256=file_sha256(static_path),
            result_receipt_sha256=semantic_sha256(unsigned_static),
            image_digest_sha256=package.artifacts.image_digest_sha256,
            source_tree_root_sha256=source_tree_root,
        )


def _synthetic_evidence(
    setting_index: int,
    fold_index: int,
    decision_origins: torch.Tensor,
    schedule_sha256: str,
) -> M03RV16ReconciledFoldEvidence:
    decisions = M03R_V16_PREDICTIVE_SPEC.qualification_origins_per_fold
    steps = decisions + M03R_V16_PREDICTIVE_SPEC.cohort_no_new_decision_tail_sessions
    execution_origins = torch.arange(
        int(decision_origins[0]),
        int(decision_origins[0]) + steps,
        dtype=torch.int64,
    )
    score = torch.tensor((-2.0, -1.0, 1.0, 2.0), dtype=torch.float64).expand(
        decisions, -1
    ).clone()
    target = score * 0.001
    valid = torch.ones_like(score, dtype=torch.bool)
    positive = torch.full((steps,), 0.0003, dtype=torch.float64)
    zero = torch.zeros(steps, dtype=torch.float64)
    turnover = torch.full((steps,), 0.01, dtype=torch.float64)
    retention = torch.full((steps,), 0.9, dtype=torch.float64)
    age = torch.full((steps,), 15.0, dtype=torch.float64)
    terminal_authority = semantic_sha256(
        ["terminal", setting_index, fold_index]
    )
    score_authority = semantic_sha256(["score", setting_index, fold_index])
    costs = tuple(
        turnover * (basis_points / 10_000.0)
        for basis_points in M03R_V16_PREDICTIVE_SPEC.evaluation_cost_basis_points
    )
    zeros_by_cost = tuple(torch.zeros_like(row) for row in costs)
    provisional = M03RV16CohortTrace(
        setting_index=setting_index,
        setting_id=M03R_V16_SETTINGS[setting_index].setting_id,
        fold_index=fold_index,
        checkpoint_file_sha256=semantic_sha256(
            ["checkpoint-file", setting_index, fold_index]
        ),
        checkpoint_model_state_sha256=semantic_sha256(
            ["checkpoint-model", setting_index, fold_index]
        ),
        terminal_checkpoint_authority_sha256=terminal_authority,
        qualified_score_authority_sha256=score_authority,
        panel_schedule_sha256=schedule_sha256,
        qualification_batch_receipt_sha256=semantic_sha256(
            ["batch", setting_index, fold_index]
        ),
        asset_axis_sha256="a" * 64,
        action_valid_sha256=cohort_runtime._tensor_sha256(valid),
        diagnostic_valid_sha256=cohort_runtime._tensor_sha256(valid),
        risk_manifest_sha256="b" * 64,
        risk_state_sha256=semantic_sha256(["risk", fold_index]),
        decision_origin_indices=decision_origins,
        execution_origin_indices=execution_origins,
        policy_gross_returns=positive,
        benchmark_gross_returns=zero,
        policy_one_way_turnover=turnover,
        benchmark_one_way_turnover=zero,
        active_one_way_mass=turnover,
        cohort_entry_one_way_mass=turnover,
        signal_cohort_mass_reduction_after_execution=zero,
        weighted_mean_cohort_age=age,
        requested_to_executed_retention=retention,
        risk_repair_active_one_way_mass=zero,
        prior_risk_repair_unwind_one_way_mass=zero,
        risk_projection_request_to_execution_one_way_distance=zero,
        absolute_policy_cost_by_cost=costs,
        benchmark_cost_by_cost=zeros_by_cost,
        incremental_active_cost_by_cost=costs,
        net_policy_return_by_cost=tuple(positive - row for row in costs),
        net_benchmark_return_by_cost=zeros_by_cost,
        net_active_return_by_cost=tuple(positive - row for row in costs),
        terminal_liquidation_one_way_turnover=0.0,
        terminal_preliquidation_active_one_way_mass=0.0,
        array_sha256=(),
        trace_sha256="0" * 64,
    )
    trace = replace(
        provisional,
        array_sha256=tuple(
            cohort_runtime._tensor_sha256(row) for row in provisional.arrays
        ),
    )
    trace = replace(
        trace,
        trace_sha256=cohort_runtime._sha256(trace.unsigned_payload()),
    )
    evidence = M03RV16ReconciledFoldEvidence(
        trace=trace,
        executable_selection_mean=score,
        selection_target_economic=target,
        selection_valid=valid,
        terminal_checkpoint_authority_sha256=terminal_authority,
        qualified_score_authority_sha256=score_authority,
        panel_schedule_sha256=schedule_sha256,
    )
    evidence.validate()
    return evidence


def test_v16_file_aggregate_joins_exact_three_worker_terminals(
    tmp_path: Path,
) -> None:
    package, authorization, _plan_file, _authorization_file = _surfaces()
    plans = tmp_path / "plans"
    plan_path = plans / "package-plan.json"
    plan_file = write_m03r_v16_package_plan(plan_path, package)
    authorization = replace(authorization, package_plan_file_sha256=plan_file)
    authorization_path = plans / "execution-authorization.json"
    authorization_file = write_m03r_v16_execution_authorization(
        authorization_path,
        authorization,
        package,
    )
    geometries = render_m03r_v16_fold_geometries(1001)
    decisions = tuple(
        torch.arange(
            row.qualification_origin_start_inclusive,
            row.qualification_origin_stop_exclusive,
            dtype=torch.int64,
        )
        for row in geometries
    )
    executions = tuple(
        torch.arange(
            row.qualification_origin_start_inclusive,
            row.qualification_origin_stop_exclusive + 29,
            dtype=torch.int64,
        )
        for row in geometries
    )
    bootstrap = build_m03r_v16_bootstrap_plan(decisions, executions)
    source_root = "7" * 64
    training_evidence_paths = tuple(
        tmp_path / f"training-terminal-{index}.json" for index in range(3)
    )
    training_terminal_receipts: list[str] = []
    for setting, path in enumerate(training_evidence_paths):
        training_unsigned = {
            "schema": M03R_V16_TRAINING_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "setting_index": setting,
            "job_uid": "training-job-uid",
            "admitted_job_authority_receipt_sha256": "1" * 64,
            "source_tree_root_sha256": source_root,
            "fold_training_adequacy_status": ("adequate",) * 5,
            "qualification_tail_accessed": False,
            "outer_qualification_authorized": False,
            "three_seed_confirmation_may_be_minted": False,
        }
        training_payload = {
            **training_unsigned,
            "receipt_sha256": semantic_sha256(training_unsigned),
        }
        training_terminal_receipts.append(training_payload["receipt_sha256"])
        path.write_bytes(canonical_json_file_bytes(training_payload))
    training_terminal_hashes = tuple(
        file_sha256(path) for path in training_evidence_paths
    )
    adequacy_matrix = tuple(
        tuple(f"{20 + setting * 5 + fold:064x}" for fold in range(5))
        for setting in range(3)
    )
    checkpoint_matrix = tuple(
        tuple(f"{40 + setting * 5 + fold:064x}" for fold in range(5))
        for setting in range(3)
    )
    closure_unsigned = {
        "schema": M03R_V16_PREQUALIFICATION_CLOSURE_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "package_plan_sha256": package.package_plan_sha256,
        "training_terminal_file_sha256": training_terminal_hashes,
        "terminal_checkpoint_file_sha256": checkpoint_matrix,
        "all_setting_folds_adequate": True,
        "outer_qualification_outcomes_accessed": False,
    }
    closure_payload = {
        **closure_unsigned,
        "receipt_sha256": semantic_sha256(closure_unsigned),
    }
    closure_path = tmp_path / "prequalification-closure.json"
    closure_path.write_bytes(canonical_json_file_bytes(closure_payload))
    panel_unsigned = {
        "schema": M03R_V16_TRAINING_PANEL_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "package_plan_sha256": package.package_plan_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "training_terminal_file_sha256": training_terminal_hashes,
        "training_terminal_receipt_sha256": tuple(training_terminal_receipts),
        "source_tree_root_sha256": source_root,
        "qualification_input_preflight_authorized": True,
        "outer_qualification_authorized": False,
        "setting_fold_adequacy_receipt_sha256": adequacy_matrix,
        "setting_fold_adequacy_status": (("adequate",) * 5,) * 3,
        "terminal_checkpoint_file_sha256": checkpoint_matrix,
        "prequalification_closure_receipt_sha256": closure_payload["receipt_sha256"],
        "prequalification_closure_file_sha256": file_sha256(closure_path),
        "all_setting_folds_adequate": True,
        "outer_qualification_outcomes_accessed": False,
        "next_research_action": "qualification-input-preflight",
        "economic_generation_may_be_minted": False,
        "reinforcement_learning_authorized": False,
        "outer_2026_accessed": False,
    }
    panel_payload = {
        **panel_unsigned,
        "receipt_sha256": semantic_sha256(panel_unsigned),
    }
    training_panel_path = tmp_path / "training-panel-decision.json"
    training_panel_path.write_bytes(canonical_json_file_bytes(panel_payload))
    panel_authority = load_m03r_v16_training_panel_authority(
        training_panel_path=training_panel_path,
        expected_training_panel_file_sha256=file_sha256(training_panel_path),
        prequalification_closure_path=closure_path,
        expected_prequalification_closure_file_sha256=file_sha256(closure_path),
        training_terminal_paths=training_evidence_paths,  # type: ignore[arg-type]
        expected_training_terminal_file_sha256=training_terminal_hashes,  # type: ignore[arg-type]
        package=package,
        authorization=authorization,
    )
    qualification_activation = _issue_m03r_v16_qualification_activation_from_panel_authority(
        package=package,
        authorization=authorization,
        panel=panel_authority,
    )
    activation_path = plans / "qualification-activation.json"
    activation_file_sha = write_m03r_v16_qualification_activation(
        activation_path, qualification_activation
    )
    preflight_root = tmp_path / "qualification-preflight"
    closure_files: list[str] = []
    closure_receipts: list[str] = []
    preflight_terminal_files: list[str] = []
    preflight_terminal_receipts: list[str] = []
    risk_roots: list[str] = []
    for index in range(3):
        fold_rows = tuple(
            {
                "fold_index": fold,
                "checkpoint_file_sha256": checkpoint_matrix[index][fold],
                "checkpoint_selection_receipt_sha256": f"{90 + fold:064x}",
                "risk_state_sha256": f"{100 + index * 5 + fold:064x}",
                "risk_inputs_validated": True,
            }
            for fold in range(5)
        )
        risk_root = semantic_sha256(
            tuple(str(row["risk_state_sha256"]) for row in fold_rows)
        )
        closure_unsigned = {
            "schema": (
                "rl-quant.top2000-dev."
                "m03r-v16-qualification-inputs-complete-v3"
            ),
            "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
            "package_plan_sha256": package.package_plan_sha256,
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "qualification_activation_receipt_sha256": (
                qualification_activation.receipt_sha256
            ),
            "setting_index": index,
            "folds": fold_rows,
            "qualification_risk_input_root_sha256": risk_root,
            "outer_qualification_access_started": False,
            "outer_2026_accessed": False,
        }
        closure_payload = {
            **closure_unsigned,
            "receipt_sha256": semantic_sha256(closure_unsigned),
        }
        closure_path = (
            preflight_root
            / f"completion-{index:02d}-setting-{index:02d}"
            / "qualification-inputs-complete.json"
        )
        closure_files.append(_write_immutable_json(closure_path, closure_payload))
        closure_receipts.append(closure_payload["receipt_sha256"])
        risk_roots.append(risk_root)
        preflight_unsigned = {
            "schema": (
                "rl-quant.top2000-dev."
                "m03r-v16-qualification-preflight-terminal-v2"
            ),
            "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
            "package_plan_sha256": package.package_plan_sha256,
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "qualification_activation_receipt_sha256": (
                qualification_activation.receipt_sha256
            ),
            "setting_index": index,
            "job_uid": "preflight-job-uid",
            "admitted_job_authority_receipt_sha256": "2" * 64,
            "setting_id": package.panel.workers[index].setting_id,
            "qualification_input_closure_file_sha256": closure_files[-1],
            "qualification_input_closure_receipt_sha256": closure_receipts[-1],
            "qualification_risk_input_root_sha256": risk_root,
            "source_tree_root_sha256": qualification_activation.source_tree_root_sha256,
            "launch_authority_receipt_sha256": f"{130 + index:064x}",
            "pod_runtime_attestation_receipt_sha256": f"{140 + index:064x}",
            "storage_semantics_file_sha256": f"{150 + index:064x}",
            "storage_semantics_receipt_sha256": f"{160 + index:064x}",
            "resource_profile_id": (
                "qualification-preflight-cpu12-memory64gi-limit128gi-v1"
            ),
            "measured_peak_rss_bytes": 4 * 1024**3,
            "measured_process_cpu_seconds": 12.5,
            "measured_wall_seconds": 15.0,
            "gpu_requested": False,
            "gpu_visible": False,
            "outer_qualification_access_started": False,
            "outer_2026_accessed": False,
            "economic_optimizer_updates": 0,
            "reinforcement_learning_updates": 0,
        }
        preflight_terminal = {
            **preflight_unsigned,
            "receipt_sha256": semantic_sha256(preflight_unsigned),
        }
        preflight_terminal_path = (
            closure_path.parent / "qualification-preflight-terminal.json"
        )
        preflight_terminal_files.append(
            _write_immutable_json(preflight_terminal_path, preflight_terminal)
        )
        preflight_terminal_receipts.append(
            preflight_terminal["receipt_sha256"]
        )
    outer_access = _issue_m03r_v16_qualification_outer_access_authority(
        package=package,
        authorization=authorization,
        activation=qualification_activation,
        setting_input_closure_file_sha256=tuple(closure_files),  # type: ignore[arg-type]
        setting_input_closure_receipt_sha256=tuple(closure_receipts),  # type: ignore[arg-type]
        setting_preflight_terminal_file_sha256=tuple(preflight_terminal_files),  # type: ignore[arg-type]
        setting_preflight_terminal_receipt_sha256=tuple(preflight_terminal_receipts),  # type: ignore[arg-type]
        qualification_risk_input_root_sha256=semantic_sha256(tuple(risk_roots)),
    )
    outer_access_path = tmp_path / "qualification-outer-access.json"
    outer_access_file_sha = write_m03r_v16_qualification_outer_access_authority(
        outer_access_path, outer_access
    )
    storage, storage_path, storage_file, authority_root, observer_root = (
        _storage_evidence(tmp_path / "aggregate-storage")
    )
    terminal_paths: list[Path] = []
    terminal_hashes: list[str] = []
    for index, worker in enumerate(package.panel.workers):
        worker_root = tmp_path / f"worker-{index}"
        evidence = tuple(
            _synthetic_evidence(
                index,
                fold_index,
                decisions[fold_index],
                package.schedule.receipt_sha256,
            )
            for fold_index in range(5)
        )
        qualification = qualify_m03r_v16_reconciled_evidence(evidence, bootstrap)
        fold_terminal_hashes: list[str] = []
        for fold_index, row in enumerate(evidence):
            artifact = {
                "schema": M03R_V16_QUALIFICATION_ARTIFACT_SCHEMA,
                "terminal_checkpoint_authority_sha256": (
                    row.terminal_checkpoint_authority_sha256
                ),
                "qualified_score_authority_sha256": (
                    row.qualified_score_authority_sha256
                ),
                "trace_unsigned_payload": row.trace.unsigned_payload(),
                "trace_arrays": row.trace.arrays,
                "decision_origin_indices": row.trace.decision_origin_indices,
                "executable_selection_mean": row.executable_selection_mean,
                "selection_target_economic": row.selection_target_economic,
                "selection_valid": row.selection_valid,
                "action_valid": row.selection_valid,
                "outer_2026_accessed": False,
                "economic_optimizer_updates": 0,
                "reinforcement_learning_updates": 0,
            }
            artifact_sha = _write_immutable_torch(
                worker_root
                / "fold-artifacts"
                / f"fold-{fold_index:02d}-qualification.pt",
                artifact,
            )
            unsigned_fold = {
                "schema": M03R_V16_FOLD_TERMINAL_SCHEMA,
                "package_plan_sha256": package.package_plan_sha256,
                "authorization_receipt_sha256": authorization.receipt_sha256,
                "qualification_activation_receipt_sha256": (
                    qualification_activation.receipt_sha256
                    ),
                    "qualification_inputs_complete_file_sha256": (
                        closure_files[index]
                    ),
                    "qualification_outer_access_authority_file_sha256": (
                        outer_access_file_sha
                    ),
                    "qualification_outer_access_authority_receipt_sha256": (
                        outer_access.receipt_sha256
                    ),
                "prequalification_closure_receipt_sha256": (
                    qualification_activation.prequalification_closure_receipt_sha256
                ),
                "training_terminal_file_sha256": training_terminal_hashes[index],
                "worker_plan_sha256": worker.receipt_sha256,
                "setting_index": index,
                "setting_id": worker.setting_id,
                "fold_index": fold_index,
                "terminal_checkpoint_authority_sha256": (
                    row.terminal_checkpoint_authority_sha256
                ),
                "qualified_score_authority_sha256": (
                    row.qualified_score_authority_sha256
                ),
                "panel_schedule_sha256": package.schedule.receipt_sha256,
                "qualification_artifact_file_sha256": artifact_sha,
                "qualification_trace_sha256": row.trace.trace_sha256,
                "qualification_after_strict_terminal_reload": True,
                "economic_optimizer_updates": 0,
                "reinforcement_learning_updates": 0,
                "outer_2026_accessed": False,
                "development_only": True,
                "reportable": False,
                "promotion_eligible": False,
            }
            fold_terminal_hashes.append(
                _write_immutable_json(
                    worker_root
                    / "receipts"
                    / f"fold-{fold_index:02d}-terminal.json",
                    {
                        **unsigned_fold,
                        "receipt_sha256": semantic_sha256(unsigned_fold),
                    },
                )
            )
        unsigned = {
            "schema": M03R_V16_WORKER_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,
            "package_plan_file_sha256": plan_file,
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "qualification_activation_receipt_sha256": (
                qualification_activation.receipt_sha256
                ),
                "qualification_inputs_complete_file_sha256": (
                    closure_files[index]
                ),
                "qualification_outer_access_authority_file_sha256": (
                    outer_access_file_sha
                ),
                "qualification_outer_access_authority_receipt_sha256": (
                    outer_access.receipt_sha256
                ),
            "prequalification_closure_receipt_sha256": (
                qualification_activation.prequalification_closure_receipt_sha256
            ),
            "training_terminal_file_sha256": training_terminal_hashes[index],
            "worker_plan_sha256": worker.receipt_sha256,
            "startup_file_sha256": "e" * 64,
            "setting_index": index,
            "setting_id": worker.setting_id,
            "fold_terminal_file_sha256": tuple(fold_terminal_hashes),
            "bootstrap_plan": asdict(bootstrap),
            "bootstrap_plan_sha256": bootstrap.receipt_sha256,
            "predictive_qualification": asdict(qualification),
            "predictive_qualification_sha256": qualification.receipt_sha256,
            "raw_predictive_gates_passed": (
                qualification.primary_hypothesis_passed
            ),
            "three_seed_confirmation_may_be_minted": False,
            "rendered_manifest_sha256": "b" * 64,
            "pod_template_sha256": "c" * 64,
            "launch_authority_receipt_sha256": "d" * 64,
            "admitted_job_authority_receipt_sha256": "3" * 64,
            "job_uid": "qualification-job-uid",
            "pod_runtime_attestation_receipt_sha256": "f" * 64,
            "storage_semantics_file_sha256": storage_file,
            "storage_semantics_receipt_sha256": storage.receipt_sha256,
            "pod_uid": f"pod-{index}",
            "economic_generation_may_be_minted": False,
            "reinforcement_learning_authorized": False,
            "outer_2026_accessed": False,
            "world_size": 2,
            "gpus_per_worker": 2,
            "development_only": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        payload = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
        path = worker_root / "predictive-terminal.json"
        file_sha = _write_immutable_json(path, payload)
        terminal_paths.append(path)
        terminal_hashes.append(file_sha)
    controller_terminal_paths: list[Path] = []
    controller_terminal_hashes: list[str] = []
    for phase, job_uid, admission_receipt in (
        ("training", "training-job-uid", "1" * 64),
        ("qualification-preflight", "preflight-job-uid", "2" * 64),
        ("qualification", "qualification-job-uid", "3" * 64),
    ):
        pods = [
            {
                "completion_index": index,
                "pod_uid": f"{phase}-pod-{index}",
                "pod_name": f"{phase}-pod-{index}",
                "phase": "Succeeded",
                "node_name": f"node-{index}",
                "main_container_image_id": (
                    "docker-pullable://"
                    f"{package.artifacts.image_reference}"
                ),
                "main_container_image_digest": (
                    package.artifacts.image_digest_sha256
                ),
            }
            for index in range(3)
        ]
        snapshot = {
            "job_name": f"{phase}-job",
            "job_uid": job_uid,
            "state": "present",
            "resource_version": "9",
            "active": 0,
            "ready": 0,
            "succeeded": 3,
            "failed": 0,
            "pods": pods,
        }
        controller_unsigned = {
            "schema": M03R_V16_CONTROLLER_TERMINAL_SCHEMA,
            "phase": phase,
            "package_plan_sha256": package.package_plan_sha256,
            "admitted_job_authority_receipt_sha256": admission_receipt,
            "job_uid": job_uid,
            "job_name": f"{phase}-job",
            "outcome": "complete",
            "scheduling_policy": "independent-per-completion",
            "gang_scheduling_required": False,
            "snapshot": snapshot,
        }
        controller_payload = {
            **controller_unsigned,
            "receipt_sha256": semantic_sha256(controller_unsigned),
        }
        controller_path = tmp_path / f"{phase}-controller-terminal.json"
        controller_terminal_paths.append(controller_path)
        controller_terminal_hashes.append(
            _write_immutable_json(controller_path, controller_payload)
        )
    aggregate = aggregate_m03r_v16_panel(
        package_plan_path=plan_path,
        package_plan_file_sha256=plan_file,
        execution_authorization_path=authorization_path,
        execution_authorization_file_sha256=authorization_file,
        qualification_activation_path=activation_path,
        qualification_activation_file_sha256=activation_file_sha,
        qualification_outer_access_authority_path=outer_access_path,
        qualification_outer_access_authority_file_sha256=(
            outer_access_file_sha
        ),
        qualification_outer_access_authority_receipt_sha256=(
            outer_access.receipt_sha256
        ),
        qualification_preflight_root=preflight_root,
        storage_semantics_path=storage_path,
        storage_semantics_file_sha256=storage_file,
        storage_semantics_receipt_sha256=storage.receipt_sha256,
        authority_root=authority_root,
        authority_observer_root=observer_root,
        training_panel_path=training_panel_path,
        training_terminal_paths=training_evidence_paths,  # type: ignore[arg-type]
        controller_terminal_paths=tuple(controller_terminal_paths),  # type: ignore[arg-type]
        controller_terminal_file_sha256=tuple(controller_terminal_hashes),  # type: ignore[arg-type]
        worker_terminal_paths=(
            terminal_paths[0],
            terminal_paths[1],
            terminal_paths[2],
        ),
        worker_terminal_file_sha256=(
            terminal_hashes[0],
            terminal_hashes[1],
            terminal_hashes[2],
        ),
        output_root=tmp_path / "aggregate",
    )
    assert aggregate["primary_hypothesis_passed"] is True
    assert aggregate["next_research_action"] == "three-seed-predictive-confirmation"
    assert aggregate["economic_generation_may_be_minted"] is False
    assert aggregate["reinforcement_learning_authorized"] is False

    controller_bytes = controller_terminal_paths[2].read_bytes()
    controller_terminal_paths[2].chmod(0o640)
    controller_terminal_paths[2].write_bytes(controller_bytes + b"\n")
    with pytest.raises(M03RV16AggregateError, match="hash drifted"):
        aggregate_m03r_v16_panel(
            package_plan_path=plan_path,
            package_plan_file_sha256=plan_file,
            execution_authorization_path=authorization_path,
            execution_authorization_file_sha256=authorization_file,
            qualification_activation_path=activation_path,
            qualification_activation_file_sha256=activation_file_sha,
            qualification_outer_access_authority_path=outer_access_path,
            qualification_outer_access_authority_file_sha256=(
                outer_access_file_sha
            ),
            qualification_outer_access_authority_receipt_sha256=(
                outer_access.receipt_sha256
            ),
            qualification_preflight_root=preflight_root,
            storage_semantics_path=storage_path,
            storage_semantics_file_sha256=storage_file,
            storage_semantics_receipt_sha256=storage.receipt_sha256,
            authority_root=authority_root,
            authority_observer_root=observer_root,
            training_panel_path=training_panel_path,
            training_terminal_paths=training_evidence_paths,  # type: ignore[arg-type]
            controller_terminal_paths=tuple(controller_terminal_paths),  # type: ignore[arg-type]
            controller_terminal_file_sha256=tuple(controller_terminal_hashes),  # type: ignore[arg-type]
            worker_terminal_paths=(
                terminal_paths[0],
                terminal_paths[1],
                terminal_paths[2],
            ),
            worker_terminal_file_sha256=(
                terminal_hashes[0],
                terminal_hashes[1],
                terminal_hashes[2],
            ),
            output_root=tmp_path / "aggregate-controller-tampered",
        )
    controller_terminal_paths[2].write_bytes(controller_bytes)

    terminal_paths[0].chmod(0o640)
    terminal_paths[0].write_bytes(terminal_paths[0].read_bytes() + b"\n")
    with pytest.raises(M03RV16AggregateError, match="hash drifted"):
        aggregate_m03r_v16_panel(
            package_plan_path=plan_path,
            package_plan_file_sha256=plan_file,
            execution_authorization_path=authorization_path,
            execution_authorization_file_sha256=authorization_file,
            qualification_activation_path=activation_path,
            qualification_activation_file_sha256=activation_file_sha,
            qualification_outer_access_authority_path=outer_access_path,
            qualification_outer_access_authority_file_sha256=(
                outer_access_file_sha
            ),
            qualification_outer_access_authority_receipt_sha256=(
                outer_access.receipt_sha256
            ),
            qualification_preflight_root=preflight_root,
            storage_semantics_path=storage_path,
            storage_semantics_file_sha256=storage_file,
            storage_semantics_receipt_sha256=storage.receipt_sha256,
            authority_root=authority_root,
            authority_observer_root=observer_root,
            training_panel_path=training_panel_path,
            training_terminal_paths=training_evidence_paths,  # type: ignore[arg-type]
            controller_terminal_paths=tuple(controller_terminal_paths),  # type: ignore[arg-type]
            controller_terminal_file_sha256=tuple(controller_terminal_hashes),  # type: ignore[arg-type]
            worker_terminal_paths=(
                terminal_paths[0],
                terminal_paths[1],
                terminal_paths[2],
            ),
            worker_terminal_file_sha256=(
                terminal_hashes[0],
                terminal_hashes[1],
                terminal_hashes[2],
            ),
            output_root=tmp_path / "aggregate-tampered",
        )
