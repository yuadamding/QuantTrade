"""Pure-file preparation for the M03R-v11 a15 inference-audit lifecycle.

This module performs no Kubernetes calls.  It renders one immutable suspended
Job and creates the exact one-create and attach-only configuration files used
by the audited Seadragon lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from rl_quant.training import top2000_m03r_v7_seadragon_lifecycle as common
from rl_quant.training import (
    top2000_m03r_v11_a15_inference_audit_lifecycle as lifecycle,
)
from rl_quant.training import top2000_m03r_v11_seadragon_operator as operator
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03RV7KubernetesRBACEvidence,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_kubernetes import (
    M03RV11A15AuditLiveEvidence,
    M03RV11A15AuditOneH100Capacity,
    M03RV11A15AuditRenderedJob,
    M03RV11A15AuditTemplateConfig,
    render_m03r_v11_a15_inference_audit_suspended_job,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_package import (
    M03RV11A15InferenceAuditAuthorization,
    M03RV11A15InferenceAuditPackagePlan,
    load_m03r_v11_a15_inference_audit_bundle,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_plan import (
    M03RV11A15InferenceAuditPlan,
)


class M03RV11A15InferenceAuditPrepareError(RuntimeError):
    """A pure-file audit render or lifecycle configuration drifted."""


SEADRAGON_QUANTTRADE_ROOT = Path("/rsrch8/home/bcb/yding4/quant/training")
_OUTPUT_SUBPATH_PREFIX = ("quant", "training")
_DIRECTORY_MODE = 0o750


def _file_sha256(path: Path) -> str:
    source = common._regular_no_symlink(path, label="prepared input")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M03RV11A15InferenceAuditPrepareError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _read_json(path: str | Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    source = common._regular_no_symlink(Path(path), label=label)
    if _file_sha256(source) != expected_sha256:
        raise M03RV11A15InferenceAuditPrepareError(f"{label} file hash drifted")
    try:
        return _mapping(json.loads(source.read_bytes()), label)
    except json.JSONDecodeError as exc:
        raise M03RV11A15InferenceAuditPrepareError(f"{label} is invalid JSON") from exc


def _write(path: str | Path, value: Any) -> str:
    target = Path(path)
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    encoded = _canonical(value)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(encoded)
        sink.flush()
        os.fsync(sink.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _load_bundle(
    args: argparse.Namespace,
) -> tuple[
    M03RV11A15InferenceAuditPlan,
    M03RV11A15InferenceAuditPackagePlan,
    M03RV11A15InferenceAuditAuthorization,
]:
    return load_m03r_v11_a15_inference_audit_bundle(
        audit_plan_path=args.audit_plan,
        audit_plan_file_sha256=args.audit_plan_file_sha256,
        package_plan_path=args.package_plan,
        package_plan_file_sha256=args.package_plan_file_sha256,
        authorization_path=args.authorization,
        authorization_file_sha256=args.authorization_file_sha256,
    )


def _load_live(path: str, expected_sha256: str) -> M03RV11A15AuditLiveEvidence:
    value = dict(_read_json(path, expected_sha256, "audit live evidence"))
    try:
        value["rbac"] = M03RV7KubernetesRBACEvidence(**value["rbac"])
        value["gpu_product_label_values"] = tuple(value["gpu_product_label_values"])
        evidence = M03RV11A15AuditLiveEvidence(**value)
        evidence.validate()
        return evidence
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV11A15InferenceAuditPrepareError(
            "audit live evidence is invalid"
        ) from exc


def _load_template(path: str, expected_sha256: str) -> M03RV11A15AuditTemplateConfig:
    try:
        return M03RV11A15AuditTemplateConfig(
            **dict(_read_json(path, expected_sha256, "audit Kubernetes template"))
        )
    except (TypeError, ValueError) as exc:
        raise M03RV11A15InferenceAuditPrepareError(
            "audit Kubernetes template is invalid"
        ) from exc


def _load_capacity(path: str, expected_sha256: str) -> M03RV11A15AuditOneH100Capacity:
    try:
        capacity = M03RV11A15AuditOneH100Capacity(
            **dict(_read_json(path, expected_sha256, "audit capacity receipt"))
        )
        capacity.validate()
        return capacity
    except (TypeError, ValueError) as exc:
        raise M03RV11A15InferenceAuditPrepareError(
            "audit capacity receipt is invalid"
        ) from exc


def _load_rendered(path: str, expected_sha256: str) -> M03RV11A15AuditRenderedJob:
    try:
        rendered = M03RV11A15AuditRenderedJob(
            **dict(_read_json(path, expected_sha256, "rendered audit Job"))
        )
        rendered.validate()
        return rendered
    except (TypeError, ValueError) as exc:
        raise M03RV11A15InferenceAuditPrepareError(
            "rendered audit Job is invalid"
        ) from exc


def _semantic_receipt(value: Mapping[str, Any], label: str) -> str:
    receipt = value.get("receipt_sha256")
    if not isinstance(receipt, str) or len(receipt) != 64:
        raise M03RV11A15InferenceAuditPrepareError(
            f"{label} omitted its semantic receipt"
        )
    unsigned = {key: row for key, row in value.items() if key != "receipt_sha256"}
    expected = hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    if receipt != expected:
        raise M03RV11A15InferenceAuditPrepareError(f"{label} semantic receipt drifted")
    return receipt


def _worker_output_identity(
    rendered: M03RV11A15AuditRenderedJob,
) -> tuple[Path, int, int]:
    """Resolve the exact host output root and worker identity from one manifest."""

    try:
        pod_spec = _mapping(
            rendered.manifest["spec"]["template"]["spec"],
            "audit Pod spec",
        )
        security = _mapping(pod_spec["securityContext"], "audit Pod security context")
        containers = pod_spec["containers"]
        if not isinstance(containers, list) or len(containers) != 1:
            raise KeyError("containers")
        container = _mapping(containers[0], "audit container")
        mounts = container["volumeMounts"]
        if not isinstance(mounts, list):
            raise KeyError("volumeMounts")
        writable = [
            _mapping(row, "audit writable volume mount")
            for row in mounts
            if isinstance(row, Mapping)
            and row.get("name") == "research-data"
            and row.get("readOnly") is not True
        ]
        if len(writable) != 1:
            raise KeyError("writable research-data mount")
        subpath_value = writable[0]["subPath"]
        run_as_user = security["runAsUser"]
        run_as_group = security["runAsGroup"]
    except (KeyError, TypeError) as exc:
        raise M03RV11A15InferenceAuditPrepareError(
            "rendered audit output mount identity is invalid"
        ) from exc
    if (
        not isinstance(subpath_value, str)
        or not isinstance(run_as_user, int)
        or isinstance(run_as_user, bool)
        or not isinstance(run_as_group, int)
        or isinstance(run_as_group, bool)
    ):
        raise M03RV11A15InferenceAuditPrepareError(
            "rendered audit worker identity is invalid"
        )
    subpath = Path(subpath_value)
    if (
        subpath.is_absolute()
        or subpath.parts[:2] != _OUTPUT_SUBPATH_PREFIX
        or ".." in subpath.parts
        or len(subpath.parts) < 5
        or subpath.parts[2] != "runs"
    ):
        raise M03RV11A15InferenceAuditPrepareError(
            "rendered audit output subpath escaped its frozen run root"
        )
    return (
        SEADRAGON_QUANTTRADE_ROOT.joinpath(*subpath.parts[2:]),
        run_as_user,
        run_as_group,
    )


def _prepare_worker_output_root(
    path: Path,
    *,
    run_as_user: int,
    run_as_group: int,
) -> Path:
    """Create a no-symlink, empty output path owned by the in-Pod worker UID."""

    runs_root = SEADRAGON_QUANTTRADE_ROOT / "runs"
    if (
        not path.is_absolute()
        or not path.is_relative_to(runs_root)
        or path == runs_root
        or len(path.relative_to(runs_root).parts) < 3
    ):
        raise M03RV11A15InferenceAuditPrepareError(
            "audit output root must be one phase below the approved runs root"
        )
    if os.geteuid() != run_as_user or os.getegid() != run_as_group:
        raise M03RV11A15InferenceAuditPrepareError(
            "controller UID/GID do not match the rendered non-root worker"
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(runs_root, flags)
    except OSError as exc:
        raise M03RV11A15InferenceAuditPrepareError(
            "approved audit runs root is unavailable"
        ) from exc
    try:
        for component in path.relative_to(runs_root).parts:
            created = False
            try:
                os.mkdir(component, _DIRECTORY_MODE, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise M03RV11A15InferenceAuditPrepareError(
                    "audit output root contains a non-directory or symlink"
                ) from exc
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            mode = stat.S_IMODE(metadata.st_mode)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != run_as_user
                or metadata.st_gid != run_as_group
                or mode & stat.S_IRWXU != stat.S_IRWXU
                or mode & stat.S_IRWXO
            ):
                raise M03RV11A15InferenceAuditPrepareError(
                    "audit output root is not a private worker-owned directory"
                )
            if created:
                os.fsync(descriptor)
        if os.listdir(descriptor):
            raise M03RV11A15InferenceAuditPrepareError(
                "audit output root must be empty before activation"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def _load_static_gate(
    path: str,
    expected_sha256: str,
    *,
    package: M03RV11A15InferenceAuditPackagePlan,
    authorization: M03RV11A15InferenceAuditAuthorization,
    audit: M03RV11A15InferenceAuditPlan,
) -> str:
    value = _read_json(path, expected_sha256, "audit static gate")
    receipt = _semantic_receipt(value, "audit static gate")
    if (
        value.get("schema") != lifecycle.M03R_V11_A15_AUDIT_STATIC_GATE_SCHEMA
        or value.get("package_plan_sha256") != package.package_plan_sha256
        or value.get("authorization_receipt_sha256") != authorization.receipt_sha256
        or value.get("audit_plan_receipt_sha256") != audit.receipt_sha256
        or value.get("source_archive_sha256") != package.artifacts.source_archive_sha256
        or value.get("gpu_requests") != 0
        or value.get("gpu_limits") != 0
        or value.get("unmasked_visibility_claimed") is not False
        or value.get("h100_capacity_evidence") is not False
        or value.get("passed") is not True
    ):
        raise M03RV11A15InferenceAuditPrepareError(
            "audit static gate semantics drifted"
        )
    return receipt


def _render(args: argparse.Namespace) -> None:
    audit, package, authorization = _load_bundle(args)
    live = _load_live(args.live_evidence, args.live_evidence_file_sha256)
    template = _load_template(args.template, args.template_file_sha256)
    try:
        now = datetime.fromisoformat(args.now_utc)
    except ValueError as exc:
        raise M03RV11A15InferenceAuditPrepareError("now UTC is invalid") from exc
    if now.tzinfo is None:
        raise M03RV11A15InferenceAuditPrepareError("now UTC must be timezone-aware")
    capacity = None
    if args.mode == "audit":
        if args.capacity_receipt is None or args.capacity_receipt_file_sha256 is None:
            raise M03RV11A15InferenceAuditPrepareError(
                "audit render requires exact one-H100 capacity evidence"
            )
        capacity = _load_capacity(
            args.capacity_receipt,
            args.capacity_receipt_file_sha256,
        )
    elif args.capacity_receipt is not None:
        raise M03RV11A15InferenceAuditPrepareError(
            "static/capacity render cannot consume future capacity evidence"
        )
    rendered = render_m03r_v11_a15_inference_audit_suspended_job(
        audit=audit,
        package=package,
        authorization=authorization,
        package_plan_file_sha256=args.package_plan_file_sha256,
        authorization_file_sha256=args.authorization_file_sha256,
        live=live,
        template=template,
        now_utc=now,
        mode=args.mode,
        capacity=capacity,
    )
    if Path(args.manifest_output).exists() or Path(args.rendered_output).exists():
        raise M03RV11A15InferenceAuditPrepareError(
            "audit render outputs must both be absent"
        )
    _write(args.manifest_output, rendered.manifest)
    _write(args.rendered_output, asdict(rendered))


def _build_create(args: argparse.Namespace) -> None:
    rendered = _load_rendered(args.rendered, args.rendered_file_sha256)
    manifest = _read_json(args.manifest, args.manifest_file_sha256, "audit manifest")
    metadata = _mapping(manifest.get("metadata"), "audit manifest metadata")
    annotations = _mapping(metadata.get("annotations"), "audit manifest annotations")
    if rendered.manifest != dict(manifest):
        raise M03RV11A15InferenceAuditPrepareError(
            "rendered audit Job and manifest disagree"
        )
    config = operator.M03RV11CreateOperatorConfig(
        mode=rendered.mode,
        job_name=cast(str, metadata.get("name")),
        run_id=cast(str, annotations.get("rl-quant/run-id")),
        rendered_path=str(Path(args.rendered).resolve()),
        rendered_file_sha256=args.rendered_file_sha256,
        manifest_path=str(Path(args.manifest).resolve()),
        manifest_file_sha256=args.manifest_file_sha256,
        evidence_root=str(Path(args.evidence_root).resolve()),
        binding_output_path=str(Path(args.binding_output).resolve()),
        activation_output_path=str(Path(args.activation_output).resolve()),
        package_plan_sha256=rendered.package_plan_sha256,
        execution_authorization_receipt_sha256=(
            rendered.execution_authorization_receipt_sha256
        ),
        source_archive_sha256=cast(
            str, annotations.get("rl-quant/source-archive-sha256")
        ),
        capacity_receipt_sha256=rendered.capacity_receipt_sha256,
        operator_source_sha256=_file_sha256(Path(operator.__file__)),
        completions=rendered.completions,
        parallelism=rendered.parallelism,
    )
    _write(args.output, asdict(config))


def _build_attach(args: argparse.Namespace) -> None:
    audit, package, authorization = _load_bundle(args)
    rendered = _load_rendered(args.rendered, args.rendered_file_sha256)
    binding = common._binding_from_file(Path(args.binding), args.binding_file_sha256)
    static_receipt = "not-yet-created"
    capacity_receipt = "not-yet-created"
    if rendered.mode in {"capacity", "audit"}:
        if args.static_gate is None or args.static_gate_file_sha256 is None:
            raise M03RV11A15InferenceAuditPrepareError(
                "capacity/audit attach requires the exact static gate"
            )
        static_receipt = _load_static_gate(
            args.static_gate,
            args.static_gate_file_sha256,
            package=package,
            authorization=authorization,
            audit=audit,
        )
    if rendered.mode == "audit":
        if args.capacity_receipt is None or args.capacity_receipt_file_sha256 is None:
            raise M03RV11A15InferenceAuditPrepareError(
                "audit attach requires exact capacity evidence"
            )
        capacity = _load_capacity(
            args.capacity_receipt,
            args.capacity_receipt_file_sha256,
        )
        capacity.validate()
        if (
            capacity.package_plan_sha256 != package.package_plan_sha256
            or capacity.authorization_receipt_sha256 != authorization.receipt_sha256
            or capacity.audit_plan_receipt_sha256 != audit.receipt_sha256
            or capacity.parent_cleanup_receipt_sha256
            != audit.parent_cleanup_receipt_sha256
            or capacity.source_archive_sha256 != package.artifacts.source_archive_sha256
            or capacity.static_gate_receipt_sha256 != static_receipt
        ):
            raise M03RV11A15InferenceAuditPrepareError(
                "capacity evidence does not bind this audit package"
            )
        capacity_receipt = capacity.receipt_sha256
    elif args.capacity_receipt is not None:
        raise M03RV11A15InferenceAuditPrepareError(
            "static/capacity attach cannot consume future capacity evidence"
        )
    if (
        rendered.package_plan_sha256 != package.package_plan_sha256
        or rendered.execution_authorization_receipt_sha256
        != authorization.receipt_sha256
        or rendered.audit_plan_receipt_sha256 != audit.receipt_sha256
        or rendered.capacity_receipt_sha256 != capacity_receipt
        or binding.job_name != package.job_name
        or binding.run_id != package.run_id
        or binding.parallelism != rendered.parallelism
        or binding.desired_manifest_sha256 != rendered.manifest_sha256
    ):
        raise M03RV11A15InferenceAuditPrepareError(
            "audit rendered/binding/package identity drifted"
        )
    config = lifecycle.M03RV11A15AuditAttachConfig(
        mode=rendered.mode,
        job_name=binding.job_name,
        run_id=binding.run_id,
        job_uid=binding.job_uid,
        rendered_path=str(Path(args.rendered).resolve()),
        rendered_file_sha256=args.rendered_file_sha256,
        binding_path=str(Path(args.binding).resolve()),
        binding_file_sha256=args.binding_file_sha256,
        activation_request_path=str(Path(args.activation).resolve()),
        activation_request_file_sha256=args.activation_file_sha256,
        output_root=str(Path(args.output_root).resolve()),
        evidence_root=str(Path(args.evidence_root).resolve()),
        package_plan_sha256=package.package_plan_sha256,
        authorization_receipt_sha256=authorization.receipt_sha256,
        audit_plan_receipt_sha256=audit.receipt_sha256,
        parent_cleanup_receipt_sha256=audit.parent_cleanup_receipt_sha256,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        lifecycle_source_sha256=_file_sha256(Path(lifecycle.__file__)),
        completions=rendered.completions,
        parallelism=rendered.parallelism,
        gpus_per_completion=rendered.gpus_per_completion,
        static_gate_receipt_sha256=static_receipt,
        capacity_receipt_sha256=capacity_receipt,
        phase_receipt_output_path=str(Path(args.phase_receipt_output).resolve()),
        host_python_path=str(Path(args.host_python).resolve()),
        pythonpath=str(Path(args.pythonpath).resolve()),
    )
    expected_output_root, run_as_user, run_as_group = _worker_output_identity(rendered)
    configured_output_root = Path(config.output_root)
    if configured_output_root != expected_output_root:
        raise M03RV11A15InferenceAuditPrepareError(
            "attach output root does not match the rendered PVC subpath"
        )
    _prepare_worker_output_root(
        configured_output_root,
        run_as_user=run_as_user,
        run_as_group=run_as_group,
    )
    _write(args.output, asdict(config))


def _bundle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audit-plan", required=True)
    parser.add_argument("--audit-plan-file-sha256", required=True)
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--authorization-file-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render")
    _bundle_arguments(render)
    render.add_argument(
        "--mode", choices=("static", "capacity", "audit"), required=True
    )
    render.add_argument("--live-evidence", required=True)
    render.add_argument("--live-evidence-file-sha256", required=True)
    render.add_argument("--template", required=True)
    render.add_argument("--template-file-sha256", required=True)
    render.add_argument("--capacity-receipt")
    render.add_argument("--capacity-receipt-file-sha256")
    render.add_argument("--now-utc", required=True)
    render.add_argument("--manifest-output", required=True)
    render.add_argument("--rendered-output", required=True)
    render.set_defaults(handler=_render)

    create = commands.add_parser("build-create-config")
    create.add_argument("--rendered", required=True)
    create.add_argument("--rendered-file-sha256", required=True)
    create.add_argument("--manifest", required=True)
    create.add_argument("--manifest-file-sha256", required=True)
    create.add_argument("--evidence-root", required=True)
    create.add_argument("--binding-output", required=True)
    create.add_argument("--activation-output", required=True)
    create.add_argument("--output", required=True)
    create.set_defaults(handler=_build_create)

    attach = commands.add_parser("build-attach-config")
    _bundle_arguments(attach)
    attach.add_argument("--rendered", required=True)
    attach.add_argument("--rendered-file-sha256", required=True)
    attach.add_argument("--binding", required=True)
    attach.add_argument("--binding-file-sha256", required=True)
    attach.add_argument("--activation", required=True)
    attach.add_argument("--activation-file-sha256", required=True)
    attach.add_argument("--static-gate")
    attach.add_argument("--static-gate-file-sha256")
    attach.add_argument("--capacity-receipt")
    attach.add_argument("--capacity-receipt-file-sha256")
    attach.add_argument("--output-root", required=True)
    attach.add_argument("--evidence-root", required=True)
    attach.add_argument("--phase-receipt-output", required=True)
    attach.add_argument("--host-python", required=True)
    attach.add_argument("--pythonpath", required=True)
    attach.add_argument("--output", required=True)
    attach.set_defaults(handler=_build_attach)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["M03RV11A15InferenceAuditPrepareError", "main"]
