"""Aggregate three immutable M03R-v16 worker terminals into one panel decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Sequence

from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackagePlan,
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_selection import (
    M03RV16BootstrapPlan,
    M03RV16PredictiveQualification,
    build_m03r_v16_panel_decision,
    write_m03r_v16_panel_decision,
)
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03R_V16_WORKER_TERMINAL_SCHEMA,
)

M03R_V16_PANEL_AGGREGATE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-panel-aggregate-v1"
)
_MAX_TERMINAL_BYTES = 16 * 1024**2


class M03RV16AggregateError(ValueError):
    """A worker terminal or panel-decision authority drifted."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)[:-1]).hexdigest()


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV16AggregateError(f"{name} must be a lowercase SHA-256")


def _read_exact(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    _digest("expected_file_sha256", expected_file_sha256)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16AggregateError("V16 worker terminal is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_TERMINAL_BYTES
        ):
            raise M03RV16AggregateError("V16 worker terminal type or size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV16AggregateError("V16 worker terminal changed while read")
    finally:
        os.close(descriptor)
    if (
        len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
    ):
        raise M03RV16AggregateError("V16 worker terminal hash drifted")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16AggregateError("V16 worker terminal is malformed") from exc
    if not isinstance(payload, dict):
        raise M03RV16AggregateError("V16 worker terminal is not an object")
    return payload


def _bootstrap(row: dict[str, Any]) -> M03RV16BootstrapPlan:
    converted = dict(row)
    for name in (
        "decision_fold_lengths",
        "execution_fold_lengths",
        "diagnostic_draw_sha256_by_block",
        "economic_draw_sha256_by_block",
        "block_sessions",
    ):
        converted[name] = tuple(converted[name])
    try:
        result = M03RV16BootstrapPlan(**converted)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16AggregateError("V16 bootstrap payload is malformed") from exc
    result.validate()
    return result


def _qualification(row: dict[str, Any]) -> M03RV16PredictiveQualification:
    converted = dict(row)
    for name in (
        "fold_trace_sha256",
        "terminal_checkpoint_authority_sha256",
        "qualified_score_authority_sha256",
        "gross_active_lcb_by_block",
        "net_10bp_active_lcb_by_block",
        "spread_lcb_by_block",
    ):
        converted[name] = tuple(converted[name])
    try:
        result = M03RV16PredictiveQualification(**converted)
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16AggregateError(
            "V16 predictive qualification is malformed"
        ) from exc
    result.validate()
    return result


def _worker_terminal(
    payload: dict[str, Any],
    *,
    setting_index: int,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
) -> tuple[M03RV16BootstrapPlan, M03RV16PredictiveQualification, str]:
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    try:
        bootstrap = _bootstrap(dict(payload["bootstrap_plan"]))
        qualification = _qualification(dict(payload["predictive_qualification"]))
        receipt_sha256 = str(payload["receipt_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16AggregateError("V16 worker terminal is incomplete") from exc
    worker = package.panel.workers[setting_index]
    if (
        payload.get("schema") != M03R_V16_WORKER_TERMINAL_SCHEMA
        or receipt_sha256 != _sha256(unsigned)
        or payload.get("package_plan_sha256") != package.package_plan_sha256
        or payload.get("authorization_receipt_sha256")
        != authorization.receipt_sha256
        or payload.get("worker_plan_sha256") != worker.receipt_sha256
        or payload.get("setting_index") != setting_index
        or payload.get("setting_id") != worker.setting_id
        or payload.get("bootstrap_plan_sha256") != bootstrap.receipt_sha256
        or payload.get("predictive_qualification_sha256")
        != qualification.receipt_sha256
        or qualification.setting_index != setting_index
        or payload.get("primary_hypothesis_passed")
        != qualification.primary_hypothesis_passed
        or payload.get("three_seed_confirmation_may_be_minted")
        != qualification.three_seed_confirmation_may_be_minted
        or payload.get("economic_generation_may_be_minted") is not False
        or payload.get("reinforcement_learning_authorized") is not False
        or payload.get("outer_2026_accessed") is not False
        or payload.get("development_only") is not True
        or payload.get("reportable") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise M03RV16AggregateError("V16 worker terminal authority drifted")
    return bootstrap, qualification, receipt_sha256


def _write_exclusive(path: Path, value: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise M03RV16AggregateError("V16 aggregate target already exists")
    payload = _canonical(value)
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def aggregate_m03r_v16_panel(
    *,
    package_plan_path: str | Path,
    package_plan_file_sha256: str,
    execution_authorization_path: str | Path,
    execution_authorization_file_sha256: str,
    worker_terminal_paths: tuple[Path, Path, Path],
    worker_terminal_file_sha256: tuple[str, str, str],
    output_root: str | Path,
) -> dict[str, Any]:
    package = load_m03r_v16_package_plan(
        package_plan_path,
        expected_file_sha256=package_plan_file_sha256,
    )
    authorization = load_m03r_v16_execution_authorization(
        execution_authorization_path,
        expected_file_sha256=execution_authorization_file_sha256,
        package=package,
    )
    rows = tuple(
        _worker_terminal(
            _read_exact(path, expected_sha),
            setting_index=index,
            package=package,
            authorization=authorization,
        )
        for index, (path, expected_sha) in enumerate(
            zip(worker_terminal_paths, worker_terminal_file_sha256, strict=True)
        )
    )
    bootstraps = tuple(row[0] for row in rows)
    if len({row.receipt_sha256 for row in bootstraps}) != 1:
        raise M03RV16AggregateError("V16 workers used different bootstrap plans")
    bootstrap = bootstraps[0]
    qualifications = (rows[0][1], rows[1][1], rows[2][1])
    decision = build_m03r_v16_panel_decision(qualifications, bootstrap)
    output = Path(output_root)
    output.mkdir(mode=0o750, parents=True, exist_ok=False)
    decision_file_sha256 = write_m03r_v16_panel_decision(
        output / "panel-decision.json",
        decision,
        qualifications,
        bootstrap,
    )
    unsigned = {
        "schema": M03R_V16_PANEL_AGGREGATE_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_file_sha256": package_plan_file_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "execution_authorization_file_sha256": (
            execution_authorization_file_sha256
        ),
        "worker_terminal_file_sha256": worker_terminal_file_sha256,
        "worker_terminal_receipt_sha256": tuple(row[2] for row in rows),
        "bootstrap_plan_sha256": bootstrap.receipt_sha256,
        "setting_qualification_sha256": tuple(
            row.receipt_sha256 for row in qualifications
        ),
        "panel_decision_file_sha256": decision_file_sha256,
        "panel_decision_receipt_sha256": decision.receipt_sha256,
        "primary_hypothesis_passed": decision.primary_hypothesis_passed,
        "next_research_action": decision.next_research_action,
        "economic_generation_may_be_minted": False,
        "reinforcement_learning_authorized": False,
        "outer_2026_access_authorized": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    aggregate = {**unsigned, "receipt_sha256": _sha256(unsigned)}
    aggregate_file_sha256 = _write_exclusive(
        output / "panel-aggregate.json",
        aggregate,
    )
    return {
        **aggregate,
        "aggregate_file_sha256": aggregate_file_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--execution-authorization-file-sha256", required=True)
    parser.add_argument("--worker-terminal", action="append", required=True)
    parser.add_argument(
        "--worker-terminal-file-sha256", action="append", required=True
    )
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.worker_terminal) != 3 or len(args.worker_terminal_file_sha256) != 3:
        raise M03RV16AggregateError("V16 aggregate requires three worker terminals")
    aggregate_m03r_v16_panel(
        package_plan_path=args.package_plan,
        package_plan_file_sha256=args.package_plan_file_sha256,
        execution_authorization_path=args.execution_authorization,
        execution_authorization_file_sha256=(
            args.execution_authorization_file_sha256
        ),
        worker_terminal_paths=(
            Path(args.worker_terminal[0]),
            Path(args.worker_terminal[1]),
            Path(args.worker_terminal[2]),
        ),
        worker_terminal_file_sha256=(
            args.worker_terminal_file_sha256[0],
            args.worker_terminal_file_sha256[1],
            args.worker_terminal_file_sha256[2],
        ),
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V16_PANEL_AGGREGATE_SCHEMA",
    "M03RV16AggregateError",
    "aggregate_m03r_v16_panel",
    "main",
]
