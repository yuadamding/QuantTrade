"""Historical coverage requalification for the completed M03R-v12 h3 run.

This CPU-only continuation never creates, patches, or deletes Kubernetes state.
It loads the corrected lifecycle validator from an exact source file, replays it
against the preserved terminal Job/Pod and worker artifacts, binds the original
supervisor failure and exact cleanup receipt, and publishes new no-clobber
coverage under a distinct continuation root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Mapping, cast

REQUALIFICATION_PLAN_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v12-h3-requalification-plan-v1"
)
REQUALIFICATION_SUCCESS_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v12-h3-historical-requalification-v1"
)
REQUALIFICATION_ERROR_SCHEMA: Final = (
    "rl-quant.top2000-dev.m03r-v12-h3-requalification-error-v1"
)
PARENT_ERROR_SCHEMA: Final = "rl-quant.top2000-dev.m03r-v12-supervisor-error-v1"
TERMINAL_EVIDENCE_SCHEMA: Final = "rl-quant.top2000-m03r-v7-terminal-evidence-v1"
APPROVED_ROOT: Final = Path("/rsrch8/home/bcb/yding4/quant/training")
_SHA256_LENGTH: Final = 64


class M03RV12H3RequalificationError(RuntimeError):
    """The historical h3 continuation failed closed."""


def _canonical(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise M03RV12H3RequalificationError(
            "requalification evidence is not canonical-JSON safe"
        ) from exc


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_sha256(label: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV12H3RequalificationError(f"{label} is not a lowercase SHA-256")
    return value


def _approved_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise M03RV12H3RequalificationError(f"{label} is not a path")
    path = Path(value)
    if not path.is_absolute():
        raise M03RV12H3RequalificationError(f"{label} is not absolute")
    try:
        path.relative_to(APPROVED_ROOT)
    except ValueError as exc:
        raise M03RV12H3RequalificationError(
            f"{label} leaves the approved QuantTrade root"
        ) from exc
    return path


def _read_bound_bytes(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    expected = _require_sha256(f"{label} SHA-256", expected_sha256)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise M03RV12H3RequalificationError(f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise M03RV12H3RequalificationError(f"{label} is not a regular file")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise M03RV12H3RequalificationError(f"{label} exceeds its size bound")
        blocks: list[bytes] = []
        observed_size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed_size += len(block)
            if maximum_bytes is not None and observed_size > maximum_bytes:
                raise M03RV12H3RequalificationError(
                    f"{label} exceeds its size bound"
                )
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or observed_size != before.st_size:
        raise M03RV12H3RequalificationError(f"{label} changed while being read")
    content = b"".join(blocks)
    if hashlib.sha256(content).hexdigest() != expected:
        raise M03RV12H3RequalificationError(f"{label} hash drifted")
    return content


def _read_bound_json(
    path: Path, expected_sha256: str, label: str
) -> Mapping[str, Any]:
    content = _read_bound_bytes(path, expected_sha256, label, maximum_bytes=16 << 20)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise M03RV12H3RequalificationError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise M03RV12H3RequalificationError(f"{label} is not an object")
    return cast(Mapping[str, Any], value)


def _exclusive_json(path: Path, value: Any) -> str:
    content = _canonical(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o440)
    except OSError as exc:
        raise M03RV12H3RequalificationError(
            f"exclusive output already exists or is unavailable: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(content).hexdigest()


def _validate_plan(value: Mapping[str, Any]) -> Mapping[str, Any]:
    expected_keys = {
        "schema",
        "continuation_id",
        "parent_run_id",
        "parent_job_name",
        "parent_job_uid",
        "parent_attach_config_path",
        "parent_attach_config_file_sha256",
        "parent_lifecycle_source_sha256",
        "parent_terminal_evidence_path",
        "parent_terminal_evidence_file_sha256",
        "parent_terminal_job_path",
        "parent_terminal_job_file_sha256",
        "parent_terminal_pods_path",
        "parent_terminal_pods_file_sha256",
        "parent_supervisor_error_path",
        "parent_supervisor_error_file_sha256",
        "parent_cleanup_receipt_path",
        "parent_cleanup_receipt_file_sha256",
        "prior_failed_continuation_id",
        "prior_failure_receipt_path",
        "prior_failure_receipt_file_sha256",
        "parent_output_root",
        "parent_pythonpath",
        "corrected_lifecycle_source_path",
        "corrected_lifecycle_source_file_sha256",
        "requalification_source_path",
        "requalification_source_file_sha256",
        "output_root",
        "selected_horizon_sessions",
        "no_new_kubernetes_job",
        "training_reexecuted",
        "economic_optimizer_updates",
        "outer_2026_accessed",
        "development_only",
        "reportable",
        "promotion_eligible",
        "receipt_sha256",
    }
    if set(value) != expected_keys:
        raise M03RV12H3RequalificationError("requalification plan inventory drifted")
    unsigned = dict(value)
    receipt = unsigned.pop("receipt_sha256")
    if (
        value.get("schema") != REQUALIFICATION_PLAN_SCHEMA
        or not isinstance(value.get("continuation_id"), str)
        or not isinstance(value.get("parent_run_id"), str)
        or not isinstance(value.get("parent_job_name"), str)
        or not isinstance(value.get("parent_job_uid"), str)
        or not isinstance(value.get("prior_failed_continuation_id"), str)
        or value.get("selected_horizon_sessions") != 3
        or value.get("no_new_kubernetes_job") is not True
        or value.get("training_reexecuted") is not False
        or value.get("economic_optimizer_updates") != 0
        or value.get("outer_2026_accessed") is not False
        or value.get("development_only") is not True
        or value.get("reportable") is not False
        or value.get("promotion_eligible") is not False
        or receipt != _content_sha256(unsigned)
    ):
        raise M03RV12H3RequalificationError("requalification plan semantics drifted")
    for key in expected_keys:
        if key.endswith("_path") or key.endswith("_root") or key == "parent_pythonpath":
            _approved_path(value[key], key)
        elif key.endswith("sha256"):
            _require_sha256(key, value[key])
    if (
        value["parent_lifecycle_source_sha256"]
        == value["corrected_lifecycle_source_file_sha256"]
    ):
        raise M03RV12H3RequalificationError(
            "corrected lifecycle source must differ from the failed parent source"
        )
    output_root = _approved_path(value["output_root"], "output_root")
    parent_output_root = _approved_path(
        value["parent_output_root"], "parent_output_root"
    )
    if output_root == parent_output_root or output_root.is_relative_to(
        parent_output_root
    ):
        raise M03RV12H3RequalificationError(
            "requalification output must be disjoint from parent worker artifacts"
        )
    return value


def build_requalification_plan(**fields: Any) -> dict[str, Any]:
    unsigned = {
        "schema": REQUALIFICATION_PLAN_SCHEMA,
        **fields,
        "selected_horizon_sessions": 3,
        "no_new_kubernetes_job": True,
        "training_reexecuted": False,
        "economic_optimizer_updates": 0,
        "outer_2026_accessed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    value = {**unsigned, "receipt_sha256": _content_sha256(unsigned)}
    _validate_plan(value)
    return value


def _load_corrected_lifecycle(plan: Mapping[str, Any]) -> ModuleType:
    pythonpath = _approved_path(plan["parent_pythonpath"], "parent_pythonpath")
    if not pythonpath.is_dir() or pythonpath.is_symlink():
        raise M03RV12H3RequalificationError("parent Python path is invalid")
    source = _approved_path(
        plan["corrected_lifecycle_source_path"], "corrected lifecycle source"
    )
    _read_bound_bytes(
        source,
        cast(str, plan["corrected_lifecycle_source_file_sha256"]),
        "corrected lifecycle source",
        maximum_bytes=4 << 20,
    )
    sys.path.insert(0, str(pythonpath))
    name = "_m03r_v12_h3_corrected_lifecycle"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise M03RV12H3RequalificationError("corrected lifecycle cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise M03RV12H3RequalificationError(
            "corrected lifecycle import failed"
        ) from exc
    return module


def run_historical_requalification(
    plan_path: str | Path, expected_plan_file_sha256: str
) -> dict[str, Any]:
    path = Path(plan_path)
    plan = _validate_plan(
        _read_bound_json(
            path,
            expected_plan_file_sha256,
            "requalification plan",
        )
    )
    runner_path = _approved_path(
        plan["requalification_source_path"], "requalification source"
    )
    _read_bound_bytes(
        runner_path,
        cast(str, plan["requalification_source_file_sha256"]),
        "requalification source",
        maximum_bytes=4 << 20,
    )
    lifecycle = _load_corrected_lifecycle(plan)
    config_path = _approved_path(
        plan["parent_attach_config_path"], "parent attach config"
    )
    config = lifecycle._load_config(
        config_path, cast(str, plan["parent_attach_config_file_sha256"])
    )
    if (
        config.run_id != plan["parent_run_id"]
        or config.job_name != plan["parent_job_name"]
        or config.job_uid != plan["parent_job_uid"]
        or config.lifecycle_source_sha256 != plan["parent_lifecycle_source_sha256"]
        or config.output_root != plan["parent_output_root"]
        or config.pythonpath != plan["parent_pythonpath"]
    ):
        raise M03RV12H3RequalificationError("parent config identity drifted")

    terminal_evidence = _read_bound_json(
        _approved_path(
            plan["parent_terminal_evidence_path"], "parent terminal evidence"
        ),
        cast(str, plan["parent_terminal_evidence_file_sha256"]),
        "parent terminal evidence",
    )
    terminal_job = _read_bound_json(
        _approved_path(plan["parent_terminal_job_path"], "parent terminal Job"),
        cast(str, plan["parent_terminal_job_file_sha256"]),
        "parent terminal Job",
    )
    terminal_pods = _read_bound_json(
        _approved_path(plan["parent_terminal_pods_path"], "parent terminal Pods"),
        cast(str, plan["parent_terminal_pods_file_sha256"]),
        "parent terminal Pods",
    )
    supervisor_error = _read_bound_json(
        _approved_path(
            plan["parent_supervisor_error_path"], "parent supervisor error"
        ),
        cast(str, plan["parent_supervisor_error_file_sha256"]),
        "parent supervisor error",
    )
    cleanup = _read_bound_json(
        _approved_path(plan["parent_cleanup_receipt_path"], "parent cleanup receipt"),
        cast(str, plan["parent_cleanup_receipt_file_sha256"]),
        "parent cleanup receipt",
    )
    prior_failure = _read_bound_json(
        _approved_path(plan["prior_failure_receipt_path"], "prior failure receipt"),
        cast(str, plan["prior_failure_receipt_file_sha256"]),
        "prior failure receipt",
    )
    prior_failure_unsigned = dict(prior_failure)
    prior_failure_receipt_sha256 = prior_failure_unsigned.pop(
        "receipt_sha256", None
    )
    if (
        prior_failure.get("schema") != REQUALIFICATION_ERROR_SCHEMA
        or prior_failure.get("continuation_id")
        != plan["prior_failed_continuation_id"]
        or prior_failure.get("error_type") != "M03RV12H3RequalificationError"
        or prior_failure.get("error") != "parent cleanup receipt drifted"
        or prior_failure.get("no_new_kubernetes_job") is not True
        or prior_failure.get("training_reexecuted") is not False
        or prior_failure.get("economic_optimizer_updates") != 0
        or prior_failure.get("outer_2026_accessed") is not False
        or prior_failure.get("development_only") is not True
        or prior_failure.get("reportable") is not False
        or prior_failure.get("promotion_eligible") is not False
        or prior_failure_receipt_sha256
        != _content_sha256(prior_failure_unsigned)
    ):
        raise M03RV12H3RequalificationError(
            "prior requalification failure lineage drifted"
        )
    if (
        terminal_evidence.get("schema") != TERMINAL_EVIDENCE_SCHEMA
        or terminal_evidence.get("reason") != "complete"
        or terminal_evidence.get("job_sha256")
        != plan["parent_terminal_job_file_sha256"]
        or terminal_evidence.get("pods_sha256")
        != plan["parent_terminal_pods_file_sha256"]
        or supervisor_error.get("schema") != PARENT_ERROR_SCHEMA
        or supervisor_error.get("error_type") != "M03RV12SeadragonLifecycleError"
        or supervisor_error.get("error") != "fold horizon inventory drifted"
        or supervisor_error.get("attach_required") is not False
    ):
        raise M03RV12H3RequalificationError(
            "parent terminal or exact historical failure semantics drifted"
        )
    lifecycle.common._job_identity(
        terminal_job,
        job_name=config.job_name,
        run_id=config.run_id,
        job_uid=config.job_uid,
    )
    lifecycle._job_artifact_identity(terminal_job, config)
    if lifecycle.common._true_condition(terminal_job) != "Complete":
        raise M03RV12H3RequalificationError("parent Job was not complete")
    if terminal_pods.get("apiVersion") != "v1" or terminal_pods.get("kind") != "PodList":
        raise M03RV12H3RequalificationError("parent terminal PodList drifted")
    items = terminal_pods.get("items")
    if not isinstance(items, list):
        raise M03RV12H3RequalificationError("parent terminal Pods are invalid")

    cleanup_request = cleanup.get("request")
    if not isinstance(cleanup_request, Mapping):
        raise M03RV12H3RequalificationError(
            "parent cleanup request is not an object"
        )
    request = lifecycle.M03RV7ExactJobCleanupRequest(**dict(cleanup_request))
    rebuilt_cleanup = lifecycle.build_m03r_v7_exact_cleanup_receipt(
        request=request,
        first_job_absent=cleanup.get("first_job_absent"),
        second_job_absent=cleanup.get("second_job_absent"),
        first_owned_pod_uids=tuple(cleanup.get("first_owned_pod_uids", ())),
        second_owned_pod_uids=tuple(cleanup.get("second_owned_pod_uids", ())),
        verification_evidence_sha256=cleanup.get("verification_evidence_sha256"),
    )
    if _canonical(asdict(rebuilt_cleanup)) != _canonical(cleanup):
        raise M03RV12H3RequalificationError("parent cleanup receipt drifted")
    if request.job_uid != config.job_uid or request.run_id != config.run_id:
        raise M03RV12H3RequalificationError("parent cleanup identity drifted")

    coverage = lifecycle.validate_m03r_v12_predictive_coverage(
        config, owned_pods=items
    )
    output_root = _approved_path(plan["output_root"], "output root")
    try:
        output_root.mkdir(mode=0o750, parents=False, exist_ok=False)
    except OSError as exc:
        raise M03RV12H3RequalificationError(
            "fresh requalification output root is unavailable"
        ) from exc
    coverage_file_sha256 = _exclusive_json(
        output_root / "completion-coverage.json", coverage
    )
    unsigned = {
        "schema": REQUALIFICATION_SUCCESS_SCHEMA,
        "continuation_id": plan["continuation_id"],
        "plan_receipt_sha256": plan["receipt_sha256"],
        "plan_file_sha256": expected_plan_file_sha256,
        "parent_run_id": config.run_id,
        "parent_job_name": config.job_name,
        "parent_job_uid": config.job_uid,
        "parent_attach_config_file_sha256": plan[
            "parent_attach_config_file_sha256"
        ],
        "parent_lifecycle_source_sha256": config.lifecycle_source_sha256,
        "corrected_lifecycle_source_file_sha256": plan[
            "corrected_lifecycle_source_file_sha256"
        ],
        "requalification_source_file_sha256": plan[
            "requalification_source_file_sha256"
        ],
        "parent_terminal_evidence_file_sha256": plan[
            "parent_terminal_evidence_file_sha256"
        ],
        "parent_terminal_job_file_sha256": plan["parent_terminal_job_file_sha256"],
        "parent_terminal_pods_file_sha256": plan[
            "parent_terminal_pods_file_sha256"
        ],
        "parent_supervisor_error_file_sha256": plan[
            "parent_supervisor_error_file_sha256"
        ],
        "parent_cleanup_receipt_file_sha256": plan[
            "parent_cleanup_receipt_file_sha256"
        ],
        "prior_failed_continuation_id": plan["prior_failed_continuation_id"],
        "prior_failure_receipt_file_sha256": plan[
            "prior_failure_receipt_file_sha256"
        ],
        "prior_failure_receipt_sha256": prior_failure_receipt_sha256,
        "completion_coverage_file_sha256": coverage_file_sha256,
        "completion_coverage_sha256": coverage["coverage_sha256"],
        "selected_horizon_sessions": 3,
        "no_new_kubernetes_job": True,
        "training_reexecuted": False,
        "economic_optimizer_updates": 0,
        "outer_2026_accessed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    result = {**unsigned, "receipt_sha256": _content_sha256(unsigned)}
    _exclusive_json(output_root / "historical-requalification.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    for name in (
        "continuation-id",
        "parent-run-id",
        "parent-job-name",
        "parent-job-uid",
        "parent-attach-config-path",
        "parent-attach-config-file-sha256",
        "parent-lifecycle-source-sha256",
        "parent-terminal-evidence-path",
        "parent-terminal-evidence-file-sha256",
        "parent-terminal-job-path",
        "parent-terminal-job-file-sha256",
        "parent-terminal-pods-path",
        "parent-terminal-pods-file-sha256",
        "parent-supervisor-error-path",
        "parent-supervisor-error-file-sha256",
        "parent-cleanup-receipt-path",
        "parent-cleanup-receipt-file-sha256",
        "prior-failed-continuation-id",
        "prior-failure-receipt-path",
        "prior-failure-receipt-file-sha256",
        "parent-output-root",
        "parent-pythonpath",
        "corrected-lifecycle-source-path",
        "corrected-lifecycle-source-file-sha256",
        "requalification-source-path",
        "requalification-source-file-sha256",
        "output-root",
        "output",
    ):
        build.add_argument(f"--{name}", required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--plan-file-sha256", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "build-plan":
        values = vars(arguments)
        values.pop("command")
        output = Path(values.pop("output"))
        plan = build_requalification_plan(
            **{name.replace("-", "_"): value for name, value in values.items()}
        )
        print(_exclusive_json(output, plan))
        return
    result = run_historical_requalification(
        arguments.plan, arguments.plan_file_sha256
    )
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "M03RV12H3RequalificationError",
    "REQUALIFICATION_PLAN_SCHEMA",
    "REQUALIFICATION_SUCCESS_SCHEMA",
    "build_requalification_plan",
    "run_historical_requalification",
]
