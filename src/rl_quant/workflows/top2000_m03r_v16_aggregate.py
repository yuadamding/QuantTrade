"""Aggregate three immutable M03R-v16 worker terminals into one panel decision."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
from pathlib import Path
from typing import Any, Sequence

import torch

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes as _canonical,
    semantic_sha256 as _sha256,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackagePlan,
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    M03RV16QualificationActivation,
    load_m03r_v16_qualification_activation,
)
from rl_quant.training.top2000_m03r_v16_selection import (
    M03RV16BootstrapPlan,
    M03RV16PredictiveQualification,
    M03RV16ReconciledFoldEvidence,
    build_m03r_v16_panel_decision,
    build_m03r_v16_bootstrap_plan,
    qualify_m03r_v16_reconciled_evidence,
    write_m03r_v16_panel_decision,
)
from rl_quant.training.top2000_m03r_v16_cohort_runtime import (
    M03RV16CohortTrace,
)
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
)
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03R_V16_FOLD_TERMINAL_SCHEMA,
    M03R_V16_QUALIFICATION_ARTIFACT_SCHEMA,
    M03R_V16_WORKER_TERMINAL_SCHEMA,
)

M03R_V16_PANEL_AGGREGATE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-panel-aggregate-v3"
)
_MAX_TERMINAL_BYTES = 16 * 1024**2
_MAX_ARTIFACT_BYTES = 256 * 1024**2


class M03RV16AggregateError(ValueError):
    """A worker terminal or panel-decision authority drifted."""


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV16AggregateError(f"{name} must be a lowercase SHA-256")


def _read_exact(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    raw = _read_exact_bytes(
        path,
        expected_file_sha256,
        maximum_bytes=_MAX_TERMINAL_BYTES,
        label="worker terminal",
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16AggregateError("V16 worker terminal is malformed") from exc
    if not isinstance(payload, dict):
        raise M03RV16AggregateError("V16 worker terminal is not an object")
    return payload


def _read_exact_bytes(
    path: Path,
    expected_file_sha256: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    _digest("expected_file_sha256", expected_file_sha256)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16AggregateError(f"V16 {label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise M03RV16AggregateError(f"V16 {label} type or size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise M03RV16AggregateError(f"V16 {label} changed while read")
    finally:
        os.close(descriptor)
    if (
        len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
    ):
        raise M03RV16AggregateError(f"V16 {label} hash drifted")
    return raw


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_exact_torch(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    raw = _read_exact_bytes(
        path,
        expected_file_sha256,
        maximum_bytes=_MAX_ARTIFACT_BYTES,
        label="qualification artifact",
    )
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise M03RV16AggregateError(
            "V16 qualification artifact is malformed"
        ) from exc
    if not isinstance(payload, dict):
        raise M03RV16AggregateError("V16 qualification artifact is not an object")
    return payload


def _trace_from_artifact(payload: dict[str, Any]) -> M03RV16CohortTrace:
    try:
        unsigned = dict(payload["trace_unsigned_payload"])
        arrays = tuple(payload["trace_arrays"])
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16AggregateError("V16 trace artifact is incomplete") from exc
    costs = len(M03R_V16_PREDICTIVE_SPEC.evaluation_cost_basis_points)
    expected = 14 + 6 * costs
    if len(arrays) != expected or any(
        not isinstance(value, torch.Tensor) for value in arrays
    ):
        raise M03RV16AggregateError("V16 trace array inventory drifted")
    scalar = {
        key: value
        for key, value in unsigned.items()
        if key not in {"development_only", "reportable", "promotion_eligible"}
    }
    scalar["array_sha256"] = tuple(scalar["array_sha256"])
    try:
        trace = M03RV16CohortTrace(
            **scalar,
            decision_origin_indices=arrays[0],
            execution_origin_indices=arrays[1],
            policy_gross_returns=arrays[2],
            benchmark_gross_returns=arrays[3],
            policy_one_way_turnover=arrays[4],
            benchmark_one_way_turnover=arrays[5],
            active_one_way_mass=arrays[6],
            cohort_entry_one_way_mass=arrays[7],
            signal_cohort_mass_reduction_after_execution=arrays[8],
            weighted_mean_cohort_age=arrays[9],
            requested_to_executed_retention=arrays[10],
            risk_repair_active_one_way_mass=arrays[11],
            prior_risk_repair_unwind_one_way_mass=arrays[12],
            risk_projection_request_to_execution_one_way_distance=arrays[13],
            absolute_policy_cost_by_cost=arrays[14 : 14 + costs],
            benchmark_cost_by_cost=arrays[14 + costs : 14 + 2 * costs],
            incremental_active_cost_by_cost=arrays[14 + 2 * costs : 14 + 3 * costs],
            net_policy_return_by_cost=arrays[14 + 3 * costs : 14 + 4 * costs],
            net_benchmark_return_by_cost=arrays[14 + 4 * costs : 14 + 5 * costs],
            net_active_return_by_cost=arrays[14 + 5 * costs : 14 + 6 * costs],
            trace_sha256=_sha256(unsigned),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16AggregateError("V16 trace payload is malformed") from exc
    trace.validate()
    if trace.unsigned_payload() != unsigned:
        raise M03RV16AggregateError("V16 trace payload reconstruction drifted")
    return trace


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
    qualification_activation: M03RV16QualificationActivation,
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
        or payload.get("qualification_activation_receipt_sha256")
        != qualification_activation.receipt_sha256
        or payload.get("training_terminal_file_sha256")
        != qualification_activation.training_terminal_file_sha256[setting_index]
        or payload.get("worker_plan_sha256") != worker.receipt_sha256
        or payload.get("setting_index") != setting_index
        or payload.get("setting_id") != worker.setting_id
        or payload.get("bootstrap_plan_sha256") != bootstrap.receipt_sha256
        or payload.get("predictive_qualification_sha256")
        != qualification.receipt_sha256
        or qualification.setting_index != setting_index
        or payload.get("raw_predictive_gates_passed")
        != qualification.primary_hypothesis_passed
        or payload.get("three_seed_confirmation_may_be_minted") is not False
        or payload.get("economic_generation_may_be_minted") is not False
        or payload.get("reinforcement_learning_authorized") is not False
        or payload.get("outer_2026_accessed") is not False
        or payload.get("development_only") is not True
        or payload.get("reportable") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise M03RV16AggregateError("V16 worker terminal authority drifted")
    return bootstrap, qualification, receipt_sha256


def _fold_evidence(
    *,
    worker_root: Path,
    fold_terminal_file_sha256: str,
    setting_index: int,
    fold_index: int,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    qualification_activation: M03RV16QualificationActivation,
) -> M03RV16ReconciledFoldEvidence:
    fold_terminal = _read_exact(
        worker_root / "receipts" / f"fold-{fold_index:02d}-terminal.json",
        fold_terminal_file_sha256,
    )
    unsigned = {
        key: value for key, value in fold_terminal.items() if key != "receipt_sha256"
    }
    worker = package.panel.workers[setting_index]
    try:
        artifact_sha256 = str(
            fold_terminal["qualification_artifact_file_sha256"]
        )
        trace_sha256 = str(fold_terminal["qualification_trace_sha256"])
        terminal_authority = str(
            fold_terminal["terminal_checkpoint_authority_sha256"]
        )
        score_authority = str(
            fold_terminal["qualified_score_authority_sha256"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16AggregateError("V16 fold terminal is incomplete") from exc
    for name, value in (
        ("qualification_artifact_file_sha256", artifact_sha256),
        ("qualification_trace_sha256", trace_sha256),
        ("terminal_checkpoint_authority_sha256", terminal_authority),
        ("qualified_score_authority_sha256", score_authority),
    ):
        _digest(name, value)
    if (
        fold_terminal.get("schema") != M03R_V16_FOLD_TERMINAL_SCHEMA
        or fold_terminal.get("receipt_sha256") != _sha256(unsigned)
        or fold_terminal.get("package_plan_sha256") != package.package_plan_sha256
        or fold_terminal.get("authorization_receipt_sha256")
        != authorization.receipt_sha256
        or fold_terminal.get("qualification_activation_receipt_sha256")
        != qualification_activation.receipt_sha256
        or fold_terminal.get("worker_plan_sha256") != worker.receipt_sha256
        or fold_terminal.get("setting_index") != setting_index
        or fold_terminal.get("setting_id") != worker.setting_id
        or fold_terminal.get("fold_index") != fold_index
        or fold_terminal.get("panel_schedule_sha256")
        != package.schedule.receipt_sha256
        or fold_terminal.get("qualification_after_strict_terminal_reload")
        is not True
        or fold_terminal.get("economic_optimizer_updates") != 0
        or fold_terminal.get("reinforcement_learning_updates") != 0
        or fold_terminal.get("outer_2026_accessed") is not False
        or fold_terminal.get("development_only") is not True
        or fold_terminal.get("reportable") is not False
        or fold_terminal.get("promotion_eligible") is not False
    ):
        raise M03RV16AggregateError("V16 fold terminal authority drifted")

    artifact = _load_exact_torch(
        worker_root
        / "fold-artifacts"
        / f"fold-{fold_index:02d}-qualification.pt",
        artifact_sha256,
    )
    try:
        trace = _trace_from_artifact(artifact)
        score = artifact["executable_selection_mean"]
        target = artifact["selection_target_economic"]
        valid = artifact["selection_valid"]
        action_valid = artifact["action_valid"]
        decision_origins = artifact["decision_origin_indices"]
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16AggregateError(
            "V16 qualification artifact is incomplete"
        ) from exc
    if (
        artifact.get("schema") != M03R_V16_QUALIFICATION_ARTIFACT_SCHEMA
        or artifact.get("terminal_checkpoint_authority_sha256")
        != terminal_authority
        or artifact.get("qualified_score_authority_sha256") != score_authority
        or trace.trace_sha256 != trace_sha256
        or trace.fold_index != fold_index
        or trace.setting_index != setting_index
        or not isinstance(action_valid, torch.Tensor)
        or not isinstance(decision_origins, torch.Tensor)
        or trace.action_valid_sha256 != _tensor_sha256(action_valid)
        or not torch.equal(trace.decision_origin_indices, decision_origins)
        or artifact.get("outer_2026_accessed") is not False
        or artifact.get("economic_optimizer_updates") != 0
        or artifact.get("reinforcement_learning_updates") != 0
    ):
        raise M03RV16AggregateError("V16 qualification artifact authority drifted")
    evidence = M03RV16ReconciledFoldEvidence(
        trace=trace,
        executable_selection_mean=score,
        selection_target_economic=target,
        selection_valid=valid,
        terminal_checkpoint_authority_sha256=terminal_authority,
        qualified_score_authority_sha256=score_authority,
        panel_schedule_sha256=package.schedule.receipt_sha256,
    )
    evidence.validate()
    return evidence


def _reconcile_worker_evidence(
    *,
    terminal_path: Path,
    terminal_payload: dict[str, Any],
    setting_index: int,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    qualification_activation: M03RV16QualificationActivation,
) -> tuple[M03RV16ReconciledFoldEvidence, ...]:
    try:
        fold_hashes = tuple(terminal_payload["fold_terminal_file_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise M03RV16AggregateError(
            "V16 worker terminal fold inventory is malformed"
        ) from exc
    folds = M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
    if len(fold_hashes) != folds:
        raise M03RV16AggregateError("V16 worker terminal fold inventory drifted")
    rows = tuple(
        _fold_evidence(
            worker_root=terminal_path.parent,
            fold_terminal_file_sha256=fold_hashes[index],
            setting_index=setting_index,
            fold_index=index,
            package=package,
            authorization=authorization,
            qualification_activation=qualification_activation,
        )
        for index in range(folds)
    )
    return rows


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
    qualification_activation_path: str | Path,
    qualification_activation_file_sha256: str,
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
    qualification_activation: M03RV16QualificationActivation = (
        load_m03r_v16_qualification_activation(
            qualification_activation_path,
            expected_file_sha256=qualification_activation_file_sha256,
            package=package,
            authorization=authorization,
        )
    )
    terminal_payloads = tuple(
        _read_exact(path, expected_sha)
        for path, expected_sha in zip(
            worker_terminal_paths, worker_terminal_file_sha256, strict=True
        )
    )
    rows = tuple(
        _worker_terminal(
            terminal_payloads[index],
            setting_index=index,
            package=package,
            authorization=authorization,
            qualification_activation=qualification_activation,
        )
        for index in range(len(terminal_payloads))
    )
    bootstraps = tuple(row[0] for row in rows)
    if len({row.receipt_sha256 for row in bootstraps}) != 1:
        raise M03RV16AggregateError("V16 workers used different bootstrap plans")
    bootstrap = bootstraps[0]
    reconciled_rows = tuple(
        _reconcile_worker_evidence(
            terminal_path=worker_terminal_paths[index],
            terminal_payload=terminal_payloads[index],
            setting_index=index,
            package=package,
            authorization=authorization,
            qualification_activation=qualification_activation,
        )
        for index in range(3)
    )
    reconciled_evidence = reconciled_rows
    reconciled_bootstrap = build_m03r_v16_bootstrap_plan(
        tuple(row.trace.decision_origin_indices for row in reconciled_evidence[0]),
        tuple(row.trace.execution_origin_indices for row in reconciled_evidence[0]),
    )
    if reconciled_bootstrap != bootstrap:
        raise M03RV16AggregateError(
            "V16 bootstrap could not be reproduced from fold evidence"
        )
    qualifications = tuple(
        qualify_m03r_v16_reconciled_evidence(evidence, bootstrap)
        for evidence in reconciled_evidence
    )
    worker_qualifications = (rows[0][1], rows[1][1], rows[2][1])
    if qualifications != worker_qualifications:
        raise M03RV16AggregateError(
            "V16 worker qualification could not be reproduced from fold evidence"
        )
    decision = build_m03r_v16_panel_decision(
        qualifications,
        bootstrap,
        primary_training_adequacy="adequate",
        primary_training_adequacy_receipt_sha256=(
            qualification_activation.primary_training_adequacy_receipt_sha256
        ),
    )
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
        "qualification_activation_file_sha256": (
            qualification_activation_file_sha256
        ),
        "qualification_activation_receipt_sha256": (
            qualification_activation.receipt_sha256
        ),
        "worker_terminal_file_sha256": worker_terminal_file_sha256,
        "worker_terminal_receipt_sha256": tuple(row[2] for row in rows),
        "bootstrap_plan_sha256": bootstrap.receipt_sha256,
        "setting_qualification_sha256": tuple(
            row.receipt_sha256 for row in qualifications
        ),
        "panel_decision_file_sha256": decision_file_sha256,
        "panel_decision_receipt_sha256": decision.receipt_sha256,
        "primary_training_adequacy": "adequate",
        "primary_training_adequacy_receipt_sha256": (
            qualification_activation.primary_training_adequacy_receipt_sha256
        ),
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
    parser.add_argument("--qualification-activation", required=True)
    parser.add_argument("--qualification-activation-file-sha256", required=True)
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
        qualification_activation_path=args.qualification_activation,
        qualification_activation_file_sha256=(
            args.qualification_activation_file_sha256
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
