"""Local-only operator for seed-17 Kubernetes rendering and binding.

The CLI never invokes ``kubectl``.  It deterministically renders suspended
Jobs, combines independently produced qualification artifacts, binds two
caller-supplied admitted read-backs, and emits the exact activation request.
The external receipt-gated lifecycle remains responsible for every cluster
mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03RV7KubernetesRBACEvidence,
    M03RV7KubernetesTemplateConfig,
    M03RV7LiveAdmissionEvidence,
    M03RV7RenderedSuspendedJob,
    bind_m03r_v7_top2000_admitted_suspended_job,
    build_m03r_v7_exact_job_activation_request,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_kubernetes import (
    M03R_SEED17_TOP2000_GPU_NAME,
    M03RV7Seed17CapacityReceipt,
    M03RV7Seed17ExecutionQualification,
    M03RV7Seed17QualificationArtifactRef,
    M03RV7Seed17QualifiedPackage,
    M03RV7Seed17RenderedQualificationJob,
    build_m03r_v7_seed17_capacity_receipt,
    build_m03r_v7_seed17_execution_qualification,
    build_m03r_v7_seed17_qualification_artifact_ref,
    render_m03r_v7_seed17_top2000_suspended_indexed_job,
    render_m03r_v7_seed17_top2000_suspended_qualification_batch_job,
    render_m03r_v7_seed17_top2000_suspended_validation_sentinel_job,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_package import (
    M03RV7Seed17PackagePlan,
    load_m03r_v7_seed17_top2000_package_plan,
)


class Top2000M03RV7Seed17OperatorError(RuntimeError):
    """An operator input or no-clobber publication invariant failed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV7Seed17OperatorError(
            "operator payload is not canonical-JSON safe"
        ) from exc


def _read_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV7Seed17OperatorError(
            f"operator input cannot be read: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise Top2000M03RV7Seed17OperatorError(
            f"operator input must be a JSON object: {path}"
        )
    return cast(dict[str, Any], value)


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise Top2000M03RV7Seed17OperatorError(
            f"operator evidence must be a regular non-symlink file: {source}"
        )
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(_canonical_json(value) + b"\n")
    except FileExistsError as exc:
        raise Top2000M03RV7Seed17OperatorError(
            f"operator refuses to overwrite {target}"
        ) from exc


def _load_live(path: str | Path) -> M03RV7LiveAdmissionEvidence:
    payload = _read_object(path)
    try:
        payload["rbac"] = M03RV7KubernetesRBACEvidence(**payload["rbac"])
        payload["gpu_product_label_values"] = tuple(
            payload["gpu_product_label_values"]
        )
        return M03RV7LiveAdmissionEvidence(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise Top2000M03RV7Seed17OperatorError(
            "live evidence failed typed validation"
        ) from exc


def _load_template(path: str | Path) -> M03RV7KubernetesTemplateConfig:
    try:
        return M03RV7KubernetesTemplateConfig(**_read_object(path))
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV7Seed17OperatorError(
            "Kubernetes template failed typed validation"
        ) from exc


def _load_artifact(path: str | Path) -> M03RV7Seed17QualificationArtifactRef:
    try:
        payload = _read_object(path)
        payload["rank_gpu_names"] = tuple(payload["rank_gpu_names"])
        return M03RV7Seed17QualificationArtifactRef(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise Top2000M03RV7Seed17OperatorError(
            "qualification artifact failed typed validation"
        ) from exc


def _load_capacity_payload(
    path: str | Path,
) -> tuple[M03RV7Seed17CapacityReceipt, M03RV7Seed17ExecutionQualification]:
    payload = _read_object(path)
    try:
        capacity_payload = dict(payload["capacity_receipt"])
        capacity_payload["sentinel"] = M03RV7Seed17QualificationArtifactRef(
            **{
                **capacity_payload["sentinel"],
                "rank_gpu_names": tuple(
                    capacity_payload["sentinel"]["rank_gpu_names"]
                ),
            }
        )
        capacity_payload["all_setting_qualifications"] = tuple(
            M03RV7Seed17QualificationArtifactRef(
                **{**row, "rank_gpu_names": tuple(row["rank_gpu_names"])}
            )
            for row in capacity_payload["all_setting_qualifications"]
        )
        capacity = M03RV7Seed17CapacityReceipt(**capacity_payload)
        qualification_payload = dict(payload["execution_qualification"])
        qualification_payload["capacity_receipt"] = capacity
        qualification_payload["worker_argv_prefix"] = tuple(
            qualification_payload["worker_argv_prefix"]
        )
        qualification = M03RV7Seed17ExecutionQualification(
            **qualification_payload
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Top2000M03RV7Seed17OperatorError(
            "capacity/qualification package failed typed validation"
        ) from exc
    return capacity, qualification


def _now(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise Top2000M03RV7Seed17OperatorError(
            "--now-utc must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise Top2000M03RV7Seed17OperatorError(
            "--now-utc must be timezone-aware"
        )
    return parsed


def _load_plan(args: argparse.Namespace) -> M03RV7Seed17PackagePlan:
    return load_m03r_v7_seed17_top2000_package_plan(
        args.package_plan,
        expected_package_plan_sha256=args.package_plan_sha256,
        require_file_location_matches_plan=False,
    )


def _publish_rendered(args: argparse.Namespace, rendered: Any) -> None:
    manifest_output = Path(args.manifest_output)
    rendered_output = Path(args.rendered_output)
    if manifest_output.exists() or rendered_output.exists():
        raise Top2000M03RV7Seed17OperatorError(
            "render outputs must both be absent before publication"
        )
    _write_exclusive(manifest_output, rendered.manifest)
    _write_exclusive(rendered_output, asdict(rendered))


def _render_sentinel(args: argparse.Namespace) -> None:
    rendered = render_m03r_v7_seed17_top2000_suspended_validation_sentinel_job(
        plan=_load_plan(args),
        completion_index=args.completion_index,
        live_evidence=_load_live(args.live_evidence),
        template=_load_template(args.template),
        now_utc=_now(args.now_utc),
    )
    _publish_rendered(args, rendered)


def _render_qualification(args: argparse.Namespace) -> None:
    rendered = render_m03r_v7_seed17_top2000_suspended_qualification_batch_job(
        plan=_load_plan(args),
        live_evidence=_load_live(args.live_evidence),
        template=_load_template(args.template),
        now_utc=_now(args.now_utc),
    )
    _publish_rendered(args, rendered)


def _build_capacity(args: argparse.Namespace) -> None:
    plan = _load_plan(args)
    sentinel = _load_artifact(args.sentinel_artifact)
    artifacts = tuple(_load_artifact(path) for path in args.setting_artifact)
    capacity = build_m03r_v7_seed17_capacity_receipt(
        plan=plan,
        worker_entrypoint_sha256=args.worker_entrypoint_sha256,
        runtime_manifest_sha256=args.runtime_manifest_sha256,
        sentinel=sentinel,
        all_setting_qualifications=artifacts,
    )
    qualification = build_m03r_v7_seed17_execution_qualification(
        plan=plan,
        capacity_receipt=capacity,
    )
    _write_exclusive(
        args.output,
        {
            "capacity_receipt": asdict(capacity),
            "execution_qualification": asdict(qualification),
        },
    )


def _build_artifact(args: argparse.Namespace) -> None:
    plan = _load_plan(args)
    if not 0 <= args.completion_index < 12:
        raise Top2000M03RV7Seed17OperatorError(
            "completion index must be in [0, 11]"
        )
    row = plan.indices[args.completion_index]
    qualification = _read_object(args.qualification_receipt)
    validation = _read_object(args.validation_receipt)
    fold_execution = _read_object(args.fold_execution_receipt)
    binding = _read_object(args.execution_plan_binding)
    qualification_sha = _file_sha256(args.qualification_receipt)
    validation_sha = _file_sha256(args.validation_receipt)
    fold_execution_sha = _file_sha256(args.fold_execution_receipt)
    source_archive_sha = _file_sha256(args.source_archive)
    runtime_manifest_sha = _file_sha256(args.runtime_manifest)

    peaks = qualification.get("rank_peak_cuda_memory")
    seed_hashes = qualification.get("seed_validation_receipt_sha256")
    execution_hashes = qualification.get("fold_execution_receipt_sha256")
    if (
        qualification.get("schema")
        != "rl-quant.top2000-dev.m03r-v7-seed17-validation-sentinel-v1"
        or qualification.get("protocol_sha256") != plan.protocol_sha256
        or qualification.get("setting_index") != row.setting_index
        or qualification.get("setting_id") != row.setting_id
        or qualification.get("runtime_setting_id") != row.runtime_setting_id
        or qualification.get("world_size") != 2
        or qualification.get("fold_count") != 1
        or qualification.get("paired_seeds") != [17]
        or qualification.get("completed_cells") != 1
        or qualification.get("complete") is not True
        or qualification.get("development_only") is not True
        or qualification.get("promotion_eligible") is not False
        or not isinstance(peaks, list)
        or len(peaks) != 2
        or not isinstance(seed_hashes, dict)
        or list(seed_hashes.values()) != [validation_sha]
        or not isinstance(execution_hashes, dict)
        or list(execution_hashes.values()) != [fold_execution_sha]
    ):
        raise Top2000M03RV7Seed17OperatorError(
            "qualification receipt does not prove the exact seed-17 boundary"
        )
    for rank, peak in enumerate(peaks):
        if (
            not isinstance(peak, dict)
            or peak.get("rank") != rank
            or peak.get("gpu_name") != M03R_SEED17_TOP2000_GPU_NAME
            or peak.get("compute_capability") != [9, 0]
            or not isinstance(peak.get("gpu_total_memory_bytes"), int)
            or not 79 * 1024**3
            <= peak["gpu_total_memory_bytes"]
            <= 81 * 1024**3
            or peak.get("allocator_oom_count") != 0
            or peak.get("allocator_retry_count") != 0
        ):
            raise Top2000M03RV7Seed17OperatorError(
                "qualification rank evidence is not two healthy H100 80GB ranks"
            )
    metrics = validation.get("metrics")
    if (
        validation.get("schema")
        != "rl-quant.top2000-dev.m03r-v7-seed17-validation-v1"
        or validation.get("protocol_sha256") != plan.protocol_sha256
        or validation.get("setting_index") != row.setting_index
        or validation.get("setting_id") != row.setting_id
        or validation.get("fold_index") != 0
        or validation.get("seed") != 17
        or validation.get("development_only") is not True
        or validation.get("promotion_eligible") is not False
        or not isinstance(metrics, dict)
        or metrics.get("decision_count") != 63
    ):
        raise Top2000M03RV7Seed17OperatorError(
            "seed-validation receipt identity or 63-decision trace drifted"
        )
    if (
        fold_execution.get("schema")
        != "rl-quant.top2000-dev.m03r-v7-seed17-fold-execution-v1"
        or fold_execution.get("protocol_sha256") != plan.protocol_sha256
        or fold_execution.get("setting_index") != row.setting_index
        or fold_execution.get("setting_id") != row.setting_id
        or fold_execution.get("runtime_setting_id") != row.runtime_setting_id
        or fold_execution.get("fold_index") != 0
        or fold_execution.get("ordered_seeds") != [17]
        or fold_execution.get("member_count") != 1
        or fold_execution.get("seed_validation_receipt_sha256s")
        != [validation_sha]
        or fold_execution.get("one_member_fold_execution") is not True
        or fold_execution.get("five_seed_ensemble_eligible") is not False
        or fold_execution.get("development_only") is not True
        or fold_execution.get("promotion_eligible") is not False
    ):
        raise Top2000M03RV7Seed17OperatorError(
            "fold-execution receipt is not the seed-17 one-member path"
        )
    bound_completion = binding.get("completion")
    training_plan = binding.get("training_plan")
    if (
        binding.get("package_plan_sha256") != plan.package_plan_sha256
        or not isinstance(bound_completion, dict)
        or bound_completion.get("completion_index") != row.completion_index
        or bound_completion.get("setting_index") != row.setting_index
        or bound_completion.get("setting_id") != row.setting_id
        or bound_completion.get("runtime_setting_id") != row.runtime_setting_id
        or not isinstance(training_plan, dict)
        or training_plan.get("protocol_sha256") != plan.protocol_sha256
        or training_plan.get("setting_index") != row.setting_index
        or training_plan.get("setting_id") != row.setting_id
        or training_plan.get("runtime_setting_id") != row.runtime_setting_id
        or training_plan.get("paired_seeds") != [17]
        or binding.get("prior_training_evidence_imported") is not False
        or binding.get("one_member_fold_execution") is not True
        or binding.get("promotion_eligible") is not False
        or source_archive_sha != plan.artifacts.source_archive_sha256
    ):
        raise Top2000M03RV7Seed17OperatorError(
            "package/source/execution-plan binding drifted"
        )
    artifact = build_m03r_v7_seed17_qualification_artifact_ref(
        plan=plan,
        completion_index=args.completion_index,
        runtime_manifest_sha256=runtime_manifest_sha,
        qualification_receipt_sha256=qualification_sha,
        validation_receipt_sha256=validation_sha,
        fold_execution_receipt_sha256=fold_execution_sha,
    )
    _write_exclusive(args.output, asdict(artifact))


def _render_final(args: argparse.Namespace) -> None:
    plan = _load_plan(args)
    _capacity, qualification = _load_capacity_payload(args.qualification)
    rendered = render_m03r_v7_seed17_top2000_suspended_indexed_job(
        package=M03RV7Seed17QualifiedPackage(
            plan=plan,
            qualification=qualification,
        ),
        live_evidence=_load_live(args.live_evidence),
        template=_load_template(args.template),
        now_utc=_now(args.now_utc),
    )
    _publish_rendered(args, rendered)


def _bind_activation(args: argparse.Namespace) -> None:
    payload = _read_object(args.rendered_receipt)
    try:
        if "completion_index" in payload:
            rendered: Any = M03RV7Seed17RenderedQualificationJob(**payload)
        else:
            rendered = M03RV7RenderedSuspendedJob(**payload)
        binding = bind_m03r_v7_top2000_admitted_suspended_job(
            rendered=rendered,
            first_read=_read_object(args.first_read),
            second_read=_read_object(args.second_read),
            attached_owned_pod_uids=(),
        )
        activation = build_m03r_v7_exact_job_activation_request(
            binding,
            _read_object(args.fresh_read),
        )
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV7Seed17OperatorError(
            "admitted binding or activation request failed closed"
        ) from exc
    binding_output = Path(args.binding_output)
    activation_output = Path(args.activation_output)
    if binding_output.exists() or activation_output.exists():
        raise Top2000M03RV7Seed17OperatorError(
            "binding and activation outputs must both be absent"
        )
    _write_exclusive(binding_output, asdict(binding))
    _write_exclusive(activation_output, asdict(activation))


def _add_render_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-sha256", required=True)
    parser.add_argument("--live-evidence", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--now-utc", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--rendered-output", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sentinel = commands.add_parser("render-sentinel")
    _add_render_inputs(sentinel)
    sentinel.add_argument("--completion-index", type=int, default=3)
    sentinel.set_defaults(handler=_render_sentinel)

    qualification = commands.add_parser("render-qualification")
    _add_render_inputs(qualification)
    qualification.set_defaults(handler=_render_qualification)

    capacity = commands.add_parser("build-capacity")
    capacity.add_argument("--package-plan", required=True)
    capacity.add_argument("--package-plan-sha256", required=True)
    capacity.add_argument("--sentinel-artifact", required=True)
    capacity.add_argument(
        "--setting-artifact", action="append", required=True
    )
    capacity.add_argument("--worker-entrypoint-sha256", required=True)
    capacity.add_argument("--runtime-manifest-sha256", required=True)
    capacity.add_argument("--output", required=True)
    capacity.set_defaults(handler=_build_capacity)

    artifact = commands.add_parser("build-artifact")
    artifact.add_argument("--package-plan", required=True)
    artifact.add_argument("--package-plan-sha256", required=True)
    artifact.add_argument("--completion-index", type=int, required=True)
    artifact.add_argument("--qualification-receipt", required=True)
    artifact.add_argument("--validation-receipt", required=True)
    artifact.add_argument("--fold-execution-receipt", required=True)
    artifact.add_argument("--execution-plan-binding", required=True)
    artifact.add_argument("--source-archive", required=True)
    artifact.add_argument("--runtime-manifest", required=True)
    artifact.add_argument("--output", required=True)
    artifact.set_defaults(handler=_build_artifact)

    final = commands.add_parser("render-final")
    _add_render_inputs(final)
    final.add_argument("--qualification", required=True)
    final.set_defaults(handler=_render_final)

    bind = commands.add_parser("bind-activation")
    bind.add_argument("--rendered-receipt", required=True)
    bind.add_argument("--first-read", required=True)
    bind.add_argument("--second-read", required=True)
    bind.add_argument("--fresh-read", required=True)
    bind.add_argument("--binding-output", required=True)
    bind.add_argument("--activation-output", required=True)
    bind.set_defaults(handler=_bind_activation)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["Top2000M03RV7Seed17OperatorError", "main"]
