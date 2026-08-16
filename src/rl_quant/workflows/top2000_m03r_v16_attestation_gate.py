"""Init-container verifier for one M03R-v16 Pod runtime attestation."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    load_m03r_v16_admitted_job_authority,
    load_m03r_v16_phase_launch_authority,
    load_m03r_v16_pod_runtime_attestation,
)
from rl_quant.training.top2000_m03r_v16_package import (
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)


class M03RV16AttestationGateError(RuntimeError):
    """The init container could not validate its immutable Pod authority."""


def _read_downward_value(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_marker(path: Path, payload: dict[str, object]) -> None:
    data = canonical_json_file_bytes(payload)
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def validate_m03r_v16_pod_attestation_gate(
    *,
    package_plan_path: str | Path,
    package_plan_file_sha256: str,
    authorization_path: str | Path,
    authorization_file_sha256: str,
    phase: str,
    prerequisite_authority_receipt_sha256: str,
    job_contract_sha256: str,
    pod_contract_sha256: str,
    launch_authority_path: str | Path,
    launch_authority_file_sha256: str,
    launch_authority_receipt_sha256: str,
    admitted_job_authority_path: str | Path,
    admitted_job_authority_file_sha256: str,
    admitted_job_authority_receipt_sha256: str,
    server_side_dry_run_result_path: str | Path,
    admitted_manifest_result_path: str | Path,
    completion_index: int,
    output_root: str | Path | None,
    downward_root: str | Path,
    authority_root: str | Path,
    marker_path: str | Path,
    timeout_seconds: float = 1800.0,
) -> dict[str, object]:
    package = load_m03r_v16_package_plan(
        package_plan_path, expected_file_sha256=package_plan_file_sha256
    )
    authorization = load_m03r_v16_execution_authorization(
        authorization_path,
        expected_file_sha256=authorization_file_sha256,
        package=package,
    )
    admission = load_m03r_v16_admitted_job_authority(
        admitted_job_authority_path,
        expected_file_sha256=admitted_job_authority_file_sha256,
        expected_receipt_sha256=admitted_job_authority_receipt_sha256,
        package=package,
        authorization=authorization,
        expected_phase=phase,
        expected_job_contract_sha256=job_contract_sha256,
        expected_pod_contract_sha256=pod_contract_sha256,
        server_side_dry_run_path=server_side_dry_run_result_path,
        admitted_manifest_path=admitted_manifest_result_path,
    )
    launch = load_m03r_v16_phase_launch_authority(
        launch_authority_path,
        expected_file_sha256=launch_authority_file_sha256,
        expected_receipt_sha256=launch_authority_receipt_sha256,
        package=package,
        authorization=authorization,
        expected_phase=phase,
        expected_prerequisite_receipt_sha256=(
            prerequisite_authority_receipt_sha256
        ),
        expected_job_contract_sha256=job_contract_sha256,
        expected_pod_contract_sha256=pod_contract_sha256,
        admission=admission,
        expected_admission_file_sha256=admitted_job_authority_file_sha256,
    )
    downward = Path(downward_root)
    expected_relative = launch.pod_runtime_attestation_relative_path(
        completion_index
    )
    resolved_output_root = (
        Path(output_root)
        if output_root is not None
        else Path("/mnt/output/capacity-sentinel")
        if phase == "capacity"
        else Path(package.panel.workers[completion_index].output_root)
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        relative_path = _read_downward_value(
            downward / "pod-runtime-attestation-path"
        )
        expected_file_sha256 = _read_downward_value(
            downward / "pod-runtime-attestation-file-sha256"
        )
        expected_receipt_sha256 = _read_downward_value(
            downward / "pod-runtime-attestation-receipt-sha256"
        )
        if relative_path and expected_file_sha256 and expected_receipt_sha256:
            if relative_path != expected_relative:
                raise M03RV16AttestationGateError(
                    "V16 Pod attestation annotation path drifted"
                )
            attestation_path = Path(authority_root) / relative_path
            if attestation_path.is_file():
                break
        if time.monotonic() >= deadline:
            raise M03RV16AttestationGateError(
                "V16 Pod attestation was not atomically published"
            )
        time.sleep(1.0)
    current_pod_uid = os.environ.get("M03R_V16_CURRENT_POD_UID", "")
    current_pod_name = os.environ.get("M03R_V16_CURRENT_POD_NAME", "")
    current_node_name = os.environ.get("M03R_V16_CURRENT_NODE_NAME", "")
    attestation = load_m03r_v16_pod_runtime_attestation(
        attestation_path,
        expected_file_sha256=expected_file_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        package=package,
        authorization=authorization,
        admission=admission,
        launch=launch,
        expected_completion_index=completion_index,
        expected_output_root_sha256=semantic_sha256(
            {"output_root": str(resolved_output_root.resolve())}
        ),
        current_pod_uid=current_pod_uid,
        current_pod_name=current_pod_name,
        current_node_name=current_node_name,
        expected_relative_path=relative_path,
    )
    unsigned: dict[str, object] = {
        "schema": "rl-quant.top2000-dev.m03r-v16-pod-attestation-marker-v1",
        "phase": phase,
        "job_uid": admission.job_uid,
        "completion_index": completion_index,
        "pod_uid": current_pod_uid,
        "pod_name": current_pod_name,
        "node_name": current_node_name,
        "relative_path": relative_path,
        "attestation_file_sha256": expected_file_sha256,
        "attestation_receipt_sha256": attestation.receipt_sha256,
        "launch_authority_receipt_sha256": launch.receipt_sha256,
    }
    marker = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    _write_marker(Path(marker_path), marker)
    return marker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--execution-authorization-file-sha256", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--predecessor-authority-receipt-sha256", required=True)
    parser.add_argument("--job-contract-sha256", required=True)
    parser.add_argument("--pod-contract-sha256", required=True)
    parser.add_argument("--launch-authority", required=True)
    parser.add_argument("--launch-authority-file-sha256", required=True)
    parser.add_argument("--launch-authority-receipt-sha256", required=True)
    parser.add_argument("--admitted-job-authority", required=True)
    parser.add_argument("--admitted-job-authority-file-sha256", required=True)
    parser.add_argument("--admitted-job-authority-receipt-sha256", required=True)
    parser.add_argument("--server-side-dry-run-result", required=True)
    parser.add_argument("--admitted-manifest-result", required=True)
    parser.add_argument("--completion-index", type=int, required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--downward-root", required=True)
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--marker", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_m03r_v16_pod_attestation_gate(
        package_plan_path=args.package_plan,
        package_plan_file_sha256=args.package_plan_file_sha256,
        authorization_path=args.execution_authorization,
        authorization_file_sha256=args.execution_authorization_file_sha256,
        phase=args.phase,
        prerequisite_authority_receipt_sha256=(
            args.predecessor_authority_receipt_sha256
        ),
        job_contract_sha256=args.job_contract_sha256,
        pod_contract_sha256=args.pod_contract_sha256,
        launch_authority_path=args.launch_authority,
        launch_authority_file_sha256=args.launch_authority_file_sha256,
        launch_authority_receipt_sha256=args.launch_authority_receipt_sha256,
        admitted_job_authority_path=args.admitted_job_authority,
        admitted_job_authority_file_sha256=(
            args.admitted_job_authority_file_sha256
        ),
        admitted_job_authority_receipt_sha256=(
            args.admitted_job_authority_receipt_sha256
        ),
        server_side_dry_run_result_path=args.server_side_dry_run_result,
        admitted_manifest_result_path=args.admitted_manifest_result,
        completion_index=args.completion_index,
        output_root=args.output_root,
        downward_root=args.downward_root,
        authority_root=args.authority_root,
        marker_path=args.marker,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
