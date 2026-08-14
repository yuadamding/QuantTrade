"""Pure-file preparation for the M03R-v13 Seadragon lifecycle.

This module performs no Kubernetes calls.  It renders one capacity or
predictive suspended Job and builds the exact create/attach configuration
consumed by the mutation-owning operator and attach-only supervisors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from rl_quant.training import top2000_m03r_v7_seadragon_lifecycle as common
from rl_quant.training import top2000_m03r_v13_seadragon_lifecycle as lifecycle
from rl_quant.training import top2000_m03r_v13_seadragon_operator as operator
from rl_quant.training import top2000_m03r_v13_static_gate as static_gate
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
)
from rl_quant.training.top2000_m03r_v13_kubernetes import (
    M03RV13LiveEvidence,
    M03RV13RenderedJob,
    M03RV13TwoH100CapacityQualification,
    render_m03r_v13_suspended_capacity_job,
    render_m03r_v13_suspended_predictive_job,
    render_m03r_v13_suspended_static_job,
)
from rl_quant.training.top2000_m03r_v13_package import (
    load_m03r_v13_execution_authorization,
    load_m03r_v13_package_plan,
)


class M03RV13SeadragonPrepareError(RuntimeError):
    """A pure-file render or configuration boundary drifted."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
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
    ).encode()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M03RV13SeadragonPrepareError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _read_json(path: str | Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    source = common._regular_no_symlink(Path(path), label=label)
    if _file_sha256(source) != expected_sha256:
        raise M03RV13SeadragonPrepareError(f"{label} file hash drifted")
    try:
        return _mapping(json.loads(source.read_bytes()), label)
    except json.JSONDecodeError as exc:
        raise M03RV13SeadragonPrepareError(f"{label} is invalid JSON") from exc


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


def _load_live(path: str, expected_sha256: str) -> M03RV13LiveEvidence:
    payload = dict(_read_json(path, expected_sha256, "live evidence"))
    try:
        payload["rbac"] = M03RV7KubernetesRBACEvidence(**payload["rbac"])
        payload["gpu_product_label_values"] = tuple(payload["gpu_product_label_values"])
        return M03RV13LiveEvidence(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV13SeadragonPrepareError("live evidence is invalid") from exc


def _load_template(path: str, expected_sha256: str) -> M03RV7KubernetesTemplateConfig:
    try:
        return M03RV7KubernetesTemplateConfig(
            **dict(_read_json(path, expected_sha256, "Kubernetes template"))
        )
    except (TypeError, ValueError) as exc:
        raise M03RV13SeadragonPrepareError("Kubernetes template is invalid") from exc


def _load_capacity(
    path: str, expected_sha256: str
) -> M03RV13TwoH100CapacityQualification:
    try:
        return M03RV13TwoH100CapacityQualification(
            **dict(_read_json(path, expected_sha256, "capacity qualification"))
        )
    except (TypeError, ValueError) as exc:
        raise M03RV13SeadragonPrepareError("capacity qualification is invalid") from exc


def _load_rendered(path: str, expected_sha256: str) -> M03RV13RenderedJob:
    try:
        return M03RV13RenderedJob(
            **dict(_read_json(path, expected_sha256, "rendered Job"))
        )
    except (TypeError, ValueError) as exc:
        raise M03RV13SeadragonPrepareError("rendered Job is invalid") from exc


def _render(args: argparse.Namespace) -> None:
    package = load_m03r_v13_package_plan(
        args.package_plan,
        expected_file_sha256=args.package_plan_file_sha256,
    )
    authorization = load_m03r_v13_execution_authorization(
        args.execution_authorization,
        expected_file_sha256=args.execution_authorization_file_sha256,
        package=package,
    )
    live = _load_live(args.live_evidence, args.live_evidence_file_sha256)
    template = _load_template(args.template, args.template_file_sha256)
    try:
        now = datetime.fromisoformat(args.now_utc)
    except ValueError as exc:
        raise M03RV13SeadragonPrepareError("now UTC is invalid") from exc
    if now.tzinfo is None:
        raise M03RV13SeadragonPrepareError("now UTC must be timezone-aware")
    if args.mode == "static":
        if args.capacity_qualification is not None:
            raise M03RV13SeadragonPrepareError(
                "static render cannot consume future qualification"
            )
        rendered = render_m03r_v13_suspended_static_job(
            package=package,
            authorization=authorization,
            package_plan_file_sha256=args.package_plan_file_sha256,
            authorization_file_sha256=args.execution_authorization_file_sha256,
            live=live,
            template=template,
            now_utc=now,
        )
    elif args.mode == "capacity":
        if args.capacity_qualification is not None:
            raise M03RV13SeadragonPrepareError(
                "capacity render cannot consume future qualification"
            )
        rendered = render_m03r_v13_suspended_capacity_job(
            package=package,
            authorization=authorization,
            package_plan_file_sha256=args.package_plan_file_sha256,
            authorization_file_sha256=args.execution_authorization_file_sha256,
            live=live,
            template=template,
            now_utc=now,
        )
    else:
        if (
            args.capacity_qualification is None
            or args.capacity_qualification_file_sha256 is None
        ):
            raise M03RV13SeadragonPrepareError(
                "predictive render requires the exact capacity qualification"
            )
        rendered = render_m03r_v13_suspended_predictive_job(
            package=package,
            authorization=authorization,
            package_plan_file_sha256=args.package_plan_file_sha256,
            authorization_file_sha256=args.execution_authorization_file_sha256,
            capacity=_load_capacity(
                args.capacity_qualification,
                args.capacity_qualification_file_sha256,
            ),
            live=live,
            template=template,
            now_utc=now,
        )
    if Path(args.manifest_output).exists() or Path(args.rendered_output).exists():
        raise M03RV13SeadragonPrepareError("render outputs must both be absent")
    _write(args.manifest_output, rendered.manifest)
    _write(args.rendered_output, asdict(rendered))


def _build_create(args: argparse.Namespace) -> None:
    rendered = _load_rendered(args.rendered, args.rendered_file_sha256)
    manifest = _read_json(args.manifest, args.manifest_file_sha256, "Job manifest")
    metadata = _mapping(manifest.get("metadata"), "manifest metadata")
    annotations = _mapping(metadata.get("annotations"), "manifest annotations")
    if rendered.manifest != dict(manifest):
        raise M03RV13SeadragonPrepareError("rendered and manifest bytes disagree")
    config = operator.M03RV13CreateOperatorConfig(
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
        capacity_receipt_sha256=cast(
            str, annotations.get("rl-quant/capacity-receipt-sha256")
        ),
        operator_source_sha256=_file_sha256(Path(operator.__file__)),
        completions=rendered.completions,
        parallelism=rendered.parallelism,
    )
    _write(args.output, asdict(config))


def _binding(args: argparse.Namespace) -> Any:
    return common._binding_from_file(Path(args.binding), args.binding_file_sha256)


def _build_capacity_attach(args: argparse.Namespace) -> None:
    package = load_m03r_v13_package_plan(
        args.package_plan,
        expected_file_sha256=args.package_plan_file_sha256,
    )
    authorization = load_m03r_v13_execution_authorization(
        args.execution_authorization,
        expected_file_sha256=args.execution_authorization_file_sha256,
        package=package,
    )
    binding = _binding(args)
    config = lifecycle.M03RV13CapacityAttachConfig(
        job_name=binding.job_name,
        run_id=binding.run_id,
        job_uid=binding.job_uid,
        binding_path=str(Path(args.binding).resolve()),
        binding_file_sha256=args.binding_file_sha256,
        activation_request_path=str(Path(args.activation).resolve()),
        activation_request_file_sha256=args.activation_file_sha256,
        package_plan_path=str(Path(args.package_plan).resolve()),
        package_plan_file_sha256=args.package_plan_file_sha256,
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_path=str(Path(args.execution_authorization).resolve()),
        execution_authorization_file_sha256=(args.execution_authorization_file_sha256),
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        static_gate_path=str(Path(args.static_gate).resolve()),
        static_gate_file_sha256=args.static_gate_file_sha256,
        static_gate_receipt_sha256=args.static_gate_receipt_sha256,
        lifecycle_source_sha256=_file_sha256(Path(lifecycle.__file__)),
        output_root=str(Path(args.output_root).resolve()),
        evidence_root=str(Path(args.evidence_root).resolve()),
        host_python_path=str(Path(args.host_python).resolve()),
        pythonpath=str(Path(args.pythonpath).resolve()),
    )
    _write(args.output, asdict(config))


def _build_static_attach(args: argparse.Namespace) -> None:
    package = load_m03r_v13_package_plan(
        args.package_plan,
        expected_file_sha256=args.package_plan_file_sha256,
    )
    authorization = load_m03r_v13_execution_authorization(
        args.execution_authorization,
        expected_file_sha256=args.execution_authorization_file_sha256,
        package=package,
    )
    rendered = _load_rendered(args.rendered, args.rendered_file_sha256)
    binding = _binding(args)
    config = static_gate.M03RV13StaticAttachConfig(
        job_name=binding.job_name,
        run_id=binding.run_id,
        job_uid=binding.job_uid,
        rendered_path=str(Path(args.rendered).resolve()),
        rendered_file_sha256=args.rendered_file_sha256,
        binding_path=str(Path(args.binding).resolve()),
        binding_file_sha256=args.binding_file_sha256,
        activation_request_path=str(Path(args.activation).resolve()),
        activation_request_file_sha256=args.activation_file_sha256,
        package_plan_path=str(Path(args.package_plan).resolve()),
        package_plan_file_sha256=args.package_plan_file_sha256,
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_path=str(Path(args.execution_authorization).resolve()),
        execution_authorization_file_sha256=args.execution_authorization_file_sha256,
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        static_source_sha256=_file_sha256(Path(static_gate.__file__)),
        create_evidence_root=str(Path(args.create_evidence_root).resolve()),
        server_dry_run_file_sha256=args.server_dry_run_file_sha256,
        evidence_root=str(Path(args.evidence_root).resolve()),
    )
    if rendered.mode != "static":
        raise M03RV13SeadragonPrepareError("static attach requires static rendered Job")
    _write(args.output, asdict(config))


def _build_predictive_attach(args: argparse.Namespace) -> None:
    package = load_m03r_v13_package_plan(
        args.package_plan,
        expected_file_sha256=args.package_plan_file_sha256,
    )
    authorization = load_m03r_v13_execution_authorization(
        args.execution_authorization,
        expected_file_sha256=args.execution_authorization_file_sha256,
        package=package,
    )
    capacity = _load_capacity(
        args.capacity_qualification, args.capacity_qualification_file_sha256
    )
    capacity.validate_for(package, authorization)
    binding = _binding(args)
    expected = tuple(
        lifecycle.M03RV13ExpectedCompletion(
            completion_index=index,
            setting_index=worker.setting_index,
            setting_id=worker.setting_id,
            worker_plan_sha256=worker.receipt_sha256,
        )
        for index, worker in enumerate(package.panel.workers)
    )
    config = lifecycle.M03RV13AttachSupervisorConfig(
        job_name=binding.job_name,
        run_id=binding.run_id,
        job_uid=binding.job_uid,
        binding_path=str(Path(args.binding).resolve()),
        binding_file_sha256=args.binding_file_sha256,
        activation_request_path=str(Path(args.activation).resolve()),
        activation_request_file_sha256=args.activation_file_sha256,
        output_root=str(Path(args.output_root).resolve()),
        evidence_root=str(Path(args.evidence_root).resolve()),
        package_plan_path=str(Path(args.package_plan).resolve()),
        package_plan_file_sha256=args.package_plan_file_sha256,
        package_plan_sha256=package.package_plan_sha256,
        execution_authorization_path=str(Path(args.execution_authorization).resolve()),
        execution_authorization_file_sha256=(args.execution_authorization_file_sha256),
        execution_authorization_receipt_sha256=authorization.receipt_sha256,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        capacity_receipt_sha256=capacity.terminal_receipt_sha256,
        lifecycle_source_sha256=_file_sha256(Path(lifecycle.__file__)),
        expected_completions=expected,
        host_python_path=str(Path(args.host_python).resolve()),
        pythonpath=str(Path(args.pythonpath).resolve()),
        parallelism=binding.parallelism,
    )
    _write(args.output, asdict(config))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render")
    render.add_argument(
        "--mode", choices=("static", "capacity", "predictive"), required=True
    )
    render.add_argument("--package-plan", required=True)
    render.add_argument("--package-plan-file-sha256", required=True)
    render.add_argument("--execution-authorization", required=True)
    render.add_argument("--execution-authorization-file-sha256", required=True)
    render.add_argument("--live-evidence", required=True)
    render.add_argument("--live-evidence-file-sha256", required=True)
    render.add_argument("--template", required=True)
    render.add_argument("--template-file-sha256", required=True)
    render.add_argument("--capacity-qualification")
    render.add_argument("--capacity-qualification-file-sha256")
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

    static = commands.add_parser("build-static-attach-config")
    static.add_argument("--package-plan", required=True)
    static.add_argument("--package-plan-file-sha256", required=True)
    static.add_argument("--execution-authorization", required=True)
    static.add_argument("--execution-authorization-file-sha256", required=True)
    static.add_argument("--rendered", required=True)
    static.add_argument("--rendered-file-sha256", required=True)
    static.add_argument("--binding", required=True)
    static.add_argument("--binding-file-sha256", required=True)
    static.add_argument("--activation", required=True)
    static.add_argument("--activation-file-sha256", required=True)
    static.add_argument("--create-evidence-root", required=True)
    static.add_argument("--server-dry-run-file-sha256", required=True)
    static.add_argument("--evidence-root", required=True)
    static.add_argument("--output", required=True)
    static.set_defaults(handler=_build_static_attach)

    for name, handler in (
        ("build-capacity-attach-config", _build_capacity_attach),
        ("build-predictive-attach-config", _build_predictive_attach),
    ):
        command = commands.add_parser(name)
        command.add_argument("--package-plan", required=True)
        command.add_argument("--package-plan-file-sha256", required=True)
        command.add_argument("--execution-authorization", required=True)
        command.add_argument("--execution-authorization-file-sha256", required=True)
        command.add_argument("--binding", required=True)
        command.add_argument("--binding-file-sha256", required=True)
        command.add_argument("--activation", required=True)
        command.add_argument("--activation-file-sha256", required=True)
        command.add_argument("--output-root", required=True)
        command.add_argument("--evidence-root", required=True)
        command.add_argument("--host-python", required=True)
        command.add_argument("--pythonpath", required=True)
        command.add_argument("--output", required=True)
        if name == "build-capacity-attach-config":
            command.add_argument("--static-gate", required=True)
            command.add_argument("--static-gate-file-sha256", required=True)
            command.add_argument("--static-gate-receipt-sha256", required=True)
        if name == "build-predictive-attach-config":
            command.add_argument("--capacity-qualification", required=True)
            command.add_argument("--capacity-qualification-file-sha256", required=True)
        command.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["M03RV13SeadragonPrepareError", "main"]
