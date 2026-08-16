"""Recompute V16 fit adequacy before any outer qualification is authorized."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Sequence

from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes, semantic_sha256
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    issue_m03r_v16_qualification_activation,
    write_m03r_v16_qualification_activation,
)
from rl_quant.training.top2000_m03r_v16_fit import (
    M03R_V16_EPOCH_FIT_SCHEMA,
    M03RV16TrainingAdequacy,
    classify_m03r_v16_training_adequacy,
)
from rl_quant.training.top2000_m03r_v16_package import (
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_validation_runtime import (
    M03RV16InnerValidationReceipt,
)
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03R_V16_TRAINING_FOLD_TERMINAL_SCHEMA,
    M03R_V16_TRAINING_TERMINAL_SCHEMA,
)

M03R_V16_TRAINING_PANEL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-training-adequacy-panel-v1"
)
_MAX_BYTES = 64 * 1024**2


class M03RV16TrainingAggregateError(ValueError):
    """Training evidence was incomplete, outcome-contaminated, or inconsistent."""


def _read(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16TrainingAggregateError("V16 training evidence is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_BYTES:
            raise M03RV16TrainingAggregateError("V16 training evidence size drifted")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise M03RV16TrainingAggregateError("V16 training evidence changed while read")
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise M03RV16TrainingAggregateError("V16 training evidence hash drifted")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16TrainingAggregateError("V16 training evidence is malformed") from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise M03RV16TrainingAggregateError("V16 training evidence is not canonical")
    return payload


def _write(path: Path, payload: dict[str, Any], mode: int = 0o440) -> str:
    data = canonical_json_file_bytes(payload)
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def _recompute_fold(
    root: Path,
    fold_terminal_sha256: str,
    *,
    setting_index: int,
    fold_index: int,
) -> M03RV16TrainingAdequacy:
    fold = _read(
        root / "receipts" / f"fold-{fold_index:02d}-training-terminal.json",
        fold_terminal_sha256,
    )
    unsigned = {key: value for key, value in fold.items() if key != "receipt_sha256"}
    if (
        fold.get("schema") != M03R_V16_TRAINING_FOLD_TERMINAL_SCHEMA
        or fold.get("receipt_sha256") != semantic_sha256(unsigned)
        or fold.get("setting_index") != setting_index
        or fold.get("fold_index") != fold_index
        or fold.get("qualification_tail_accessed") is not False
        or fold.get("outer_2026_accessed") is not False
    ):
        raise M03RV16TrainingAggregateError("V16 training fold terminal drifted")
    fit_files = tuple(fold.get("epoch_fit_file_sha256", ()))
    fit_receipts = tuple(fold.get("epoch_fit_receipt_sha256", ()))
    epochs = M03R_V16_PREDICTIVE_SPEC.score_training_epochs
    if len(fit_files) != epochs or len(fit_receipts) != epochs:
        raise M03RV16TrainingAggregateError("V16 training epoch inventory drifted")
    fits: list[dict[str, Any]] = []
    validations: list[M03RV16InnerValidationReceipt] = []
    for epoch in range(epochs):
        fit = _read(
            root
            / "receipts"
            / f"fold-{fold_index:02d}-epoch-{epoch + 1:02d}-fit.json",
            str(fit_files[epoch]),
        )
        fit_unsigned = {key: value for key, value in fit.items() if key != "receipt_sha256"}
        if (
            fit.get("schema") != M03R_V16_EPOCH_FIT_SCHEMA
            or fit.get("protocol_sha256") != M03R_V16_PROTOCOL_SHA256
            or fit.get("receipt_sha256") != semantic_sha256(fit_unsigned)
            or fit.get("receipt_sha256") != fit_receipts[epoch]
            or fit.get("setting_index") != setting_index
            or fit.get("fold_index") != fold_index
            or fit.get("epoch_index") != epoch
            or fit.get("qualification_tail_accessed") is not False
        ):
            raise M03RV16TrainingAggregateError("V16 epoch fit evidence drifted")
        validation = M03RV16InnerValidationReceipt(**dict(fit["inner_validation"]))
        validation.validate()
        fits.append(fit)
        validations.append(validation)
    recomputed = classify_m03r_v16_training_adequacy(
        tuple(validations), tuple(fits)
    )
    adequacy_file_sha = str(fold["training_adequacy_file_sha256"])
    adequacy_payload = _read(
        root / "receipts" / f"fold-{fold_index:02d}-training-adequacy.json",
        adequacy_file_sha,
    )
    row = {
        key: value
        for key, value in adequacy_payload.items()
        if key not in {"receipt_sha256", "epoch_fit_file_sha256"}
    }
    row["epoch_fit_receipt_sha256"] = tuple(row["epoch_fit_receipt_sha256"])
    published = M03RV16TrainingAdequacy(**row)
    published.validate()
    if (
        recomputed != published
        or adequacy_payload.get("receipt_sha256") != recomputed.receipt_sha256
        or fold.get("training_adequacy_receipt_sha256") != recomputed.receipt_sha256
        or fold.get("training_adequacy_status") != recomputed.status
    ):
        raise M03RV16TrainingAggregateError(
            "V16 training adequacy could not be independently reproduced"
        )
    return recomputed


def aggregate_m03r_v16_training_panel(
    *,
    package_plan_path: str | Path,
    package_plan_file_sha256: str,
    execution_authorization_path: str | Path,
    execution_authorization_file_sha256: str,
    training_terminal_paths: tuple[Path, Path, Path],
    training_terminal_file_sha256: tuple[str, str, str],
    output_root: str | Path,
) -> dict[str, Any]:
    package = load_m03r_v16_package_plan(
        package_plan_path, expected_file_sha256=package_plan_file_sha256
    )
    authorization = load_m03r_v16_execution_authorization(
        execution_authorization_path,
        expected_file_sha256=execution_authorization_file_sha256,
        package=package,
    )
    all_adequacy: list[tuple[M03RV16TrainingAdequacy, ...]] = []
    terminal_receipts: list[str] = []
    source_roots: set[str] = set()
    for setting, (path, expected_sha) in enumerate(
        zip(training_terminal_paths, training_terminal_file_sha256, strict=True)
    ):
        payload = _read(path, expected_sha)
        unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        fold_hashes = tuple(payload.get("fold_terminal_file_sha256", ()))
        if (
            payload.get("schema") != M03R_V16_TRAINING_TERMINAL_SCHEMA
            or payload.get("receipt_sha256") != semantic_sha256(unsigned)
            or payload.get("package_plan_sha256") != package.package_plan_sha256
            or payload.get("authorization_receipt_sha256") != authorization.receipt_sha256
            or payload.get("worker_plan_sha256") != package.panel.workers[setting].receipt_sha256
            or payload.get("setting_index") != setting
            or payload.get("qualification_tail_accessed") is not False
            or payload.get("outer_qualification_authorized") is not False
            or payload.get("three_seed_confirmation_may_be_minted") is not False
            or len(fold_hashes) != M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
        ):
            raise M03RV16TrainingAggregateError("V16 training terminal drifted")
        root = path.parent
        if (root / "fold-artifacts").exists():
            raise M03RV16TrainingAggregateError(
                "V16 training worker opened outer qualification artifacts"
            )
        source_roots.add(str(payload["source_tree_root_sha256"]))
        terminal_receipts.append(str(payload["receipt_sha256"]))
        all_adequacy.append(
            tuple(
                _recompute_fold(
                    root,
                    str(fold_hashes[fold]),
                    setting_index=setting,
                    fold_index=fold,
                )
                for fold in range(M03R_V16_PREDICTIVE_SPEC.chronological_fold_count)
            )
        )
    if len(source_roots) != 1:
        raise M03RV16TrainingAggregateError("V16 training source roots diverged")
    primary_rows = all_adequacy[M03R_V16_PREDICTIVE_SPEC.primary_setting_index]
    adequate = all(row.status == "adequate" for row in primary_rows)
    output = Path(output_root)
    output.mkdir(mode=0o750, parents=True, exist_ok=False)
    unsigned = {
        "schema": M03R_V16_TRAINING_PANEL_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "package_plan_sha256": package.package_plan_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "training_terminal_file_sha256": training_terminal_file_sha256,
        "training_terminal_receipt_sha256": tuple(terminal_receipts),
        "setting_fold_adequacy_receipt_sha256": tuple(
            tuple(row.receipt_sha256 for row in setting_rows)
            for setting_rows in all_adequacy
        ),
        "setting_fold_adequacy_status": tuple(
            tuple(row.status for row in setting_rows) for setting_rows in all_adequacy
        ),
        "primary_training_adequacy": (
            "adequate" if adequate else "inconclusive-undertrained"
        ),
        "outer_qualification_authorized": adequate,
        "next_research_action": (
            "qualification-only-execution" if adequate else "longer-training-protocol"
        ),
        "source_tree_root_sha256": next(iter(source_roots)),
        "outer_qualification_outcomes_accessed": False,
        "economic_generation_may_be_minted": False,
        "reinforcement_learning_authorized": False,
        "outer_2026_accessed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    panel = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    panel_file_sha = _write(output / "training-panel-decision.json", panel)
    result = {**panel, "panel_file_sha256": panel_file_sha}
    if adequate:
        activation = issue_m03r_v16_qualification_activation(
            package=package,
            authorization=authorization,
            training_panel_receipt_sha256=str(panel["receipt_sha256"]),
            training_terminal_file_sha256=training_terminal_file_sha256,
            primary_training_adequacy_receipt_sha256=tuple(
                row.receipt_sha256 for row in primary_rows
            ),
            source_tree_root_sha256=next(iter(source_roots)),
        )
        result["qualification_activation_file_sha256"] = (
            write_m03r_v16_qualification_activation(
                output / "qualification-activation.json", activation
            )
        )
        result["qualification_activation_receipt_sha256"] = activation.receipt_sha256
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--execution-authorization-file-sha256", required=True)
    parser.add_argument("--training-terminal", action="append", required=True)
    parser.add_argument("--training-terminal-file-sha256", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.training_terminal) != 3 or len(args.training_terminal_file_sha256) != 3:
        raise M03RV16TrainingAggregateError("V16 training panel requires three terminals")
    aggregate_m03r_v16_training_panel(
        package_plan_path=args.package_plan,
        package_plan_file_sha256=args.package_plan_file_sha256,
        execution_authorization_path=args.execution_authorization,
        execution_authorization_file_sha256=args.execution_authorization_file_sha256,
        training_terminal_paths=(
            Path(args.training_terminal[0]),
            Path(args.training_terminal[1]),
            Path(args.training_terminal[2]),
        ),
        training_terminal_file_sha256=(
            args.training_terminal_file_sha256[0],
            args.training_terminal_file_sha256[1],
            args.training_terminal_file_sha256[2],
        ),
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V16_TRAINING_PANEL_SCHEMA",
    "M03RV16TrainingAggregateError",
    "aggregate_m03r_v16_training_panel",
    "main",
]
