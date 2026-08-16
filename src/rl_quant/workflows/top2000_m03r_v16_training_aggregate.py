"""Recompute V16 fit adequacy before any outer qualification is authorized."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    M03R_V16_PREQUALIFICATION_CLOSURE_SCHEMA,
    M03R_V16_TRAINING_PANEL_SCHEMA,
    _issue_m03r_v16_qualification_activation_from_panel_authority,
    load_m03r_v16_training_panel_authority,
    write_m03r_v16_qualification_activation,
)
from rl_quant.training.top2000_m03r_v16_checkpoint import (
    load_m03r_v16_epoch_checkpoint_for_evaluation,
)
from rl_quant.training.top2000_m03r_v16_fit import (
    M03R_V16_EPOCH_FIT_SCHEMA,
    M03R_V16_NUMERICAL_TRAINING_FAILURE_SCHEMA,
    M03RV16NumericalTrainingFailure,
    M03RV16TrainingAdequacy,
    classify_m03r_v16_training_adequacy,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_package import (
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_validation_runtime import (
    M03RV16InnerValidationReceipt,
)
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03R_V16_TRAINING_FOLD_TERMINAL_SCHEMA,
    M03R_V16_TRAINING_TERMINAL_SCHEMA,
)

_MAX_BYTES = 64 * 1024**2


class M03RV16TrainingAggregateError(ValueError):
    """Training evidence was incomplete, outcome-contaminated, or inconsistent."""


def _digest(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV16TrainingAggregateError(f"{name} must be a SHA-256")


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
    package_plan_sha256: str,
    authorization_receipt_sha256: str,
    worker_plan_sha256: str,
    training_activation_receipt_sha256: str,
    panel_schedule_sha256: str,
    structural_slab_receipt_sha256: str,
    action_operator_root_sha256: str,
    target_operator_root_sha256: str,
) -> tuple[M03RV16TrainingAdequacy, dict[str, Any]]:
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
        or fold.get("package_plan_sha256") != package_plan_sha256
        or fold.get("authorization_receipt_sha256")
        != authorization_receipt_sha256
        or fold.get("worker_plan_sha256") != worker_plan_sha256
        or fold.get("training_activation_receipt_sha256")
        != training_activation_receipt_sha256
        or fold.get("panel_schedule_sha256") != panel_schedule_sha256
        or fold.get("structural_slab_receipt_sha256")
        != structural_slab_receipt_sha256
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
            or fit.get("package_plan_sha256") != package_plan_sha256
            or fit.get("worker_plan_sha256") != worker_plan_sha256
            or fit.get("training_activation_receipt_sha256")
            != training_activation_receipt_sha256
            or fit.get("panel_schedule_sha256") != panel_schedule_sha256
            or fit.get("structural_slab_receipt_sha256")
            != structural_slab_receipt_sha256
            or fit.get("qualification_tail_accessed") is not False
        ):
            raise M03RV16TrainingAggregateError("V16 epoch fit evidence drifted")
        validation = M03RV16InnerValidationReceipt(**dict(fit["inner_validation"]))
        validation.validate()
        update_pairs = tuple(fit.get("update_rows", ()))
        expected_updates = render_m03r_v16_fold_geometries(1001)[
            fold_index
        ].training_block_count
        if len(update_pairs) != expected_updates:
            raise M03RV16TrainingAggregateError(
                "V16 epoch update inventory drifted"
            )
        for local_update, pair in enumerate(update_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise M03RV16TrainingAggregateError(
                    "V16 rank-pair update evidence is incomplete"
                )
            left, right = pair
            if (
                {left.get("distributed_rank"), right.get("distributed_rank")}
                != {0, 1}
                or any(
                    row.get("setting_index") != setting_index
                    or row.get("fold_index") != fold_index
                    or row.get("selection_target_operator_root_sha256")
                    != target_operator_root_sha256
                    or row.get("action_operator_root_sha256")
                    != action_operator_root_sha256
                    for row in (left, right)
                )
                or left.get("update_plan_sha256")
                != right.get("update_plan_sha256")
                or left.get("source_array_sha256")
                != right.get("source_array_sha256")
                or left.get("completed_updates_after")
                != right.get("completed_updates_after")
                or left.get("completed_updates_after")
                != epoch * expected_updates + local_update + 1
                or left.get("global_origin_count")
                != right.get("global_origin_count")
                or int(left.get("local_origin_count", -1))
                + int(right.get("local_origin_count", -1))
                != int(left.get("global_origin_count", -2))
            ):
                raise M03RV16TrainingAggregateError(
                    "V16 rank-pair update evidence drifted"
                )
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
    return recomputed, fold


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
    all_folds: list[tuple[dict[str, Any], ...]] = []
    terminal_receipts: list[str] = []
    source_roots: set[str] = set()
    numerical_failures: dict[int, M03RV16NumericalTrainingFailure] = {}
    outcome_kinds: list[str] = []
    for setting, (path, expected_sha) in enumerate(
        zip(training_terminal_paths, training_terminal_file_sha256, strict=True)
    ):
        payload = _read(path, expected_sha)
        if payload.get("schema") == M03R_V16_NUMERICAL_TRAINING_FAILURE_SCHEMA:
            if (path.parent / "training-terminal.json").exists():
                raise M03RV16TrainingAggregateError(
                    "V16 setting published both normal and numerical outcomes"
                )
            try:
                row = {
                    key: value
                    for key, value in payload.items()
                    if key != "receipt_sha256"
                }
                row["completed_fold_terminal_file_sha256"] = tuple(
                    row.get("completed_fold_terminal_file_sha256", ())
                )
                failure = M03RV16NumericalTrainingFailure(**row)
                failure.validate()
            except (KeyError, TypeError, ValueError) as exc:
                raise M03RV16TrainingAggregateError(
                    "V16 numerical training failure is malformed"
                ) from exc
            if (
                payload.get("receipt_sha256") != failure.receipt_sha256
                or failure.package_plan_sha256 != package.package_plan_sha256
                or failure.authorization_receipt_sha256
                != authorization.receipt_sha256
                or failure.worker_plan_sha256
                != package.panel.workers[setting].receipt_sha256
                or failure.setting_index != setting
                or failure.setting_id != package.panel.workers[setting].setting_id
                or failure.outer_qualification_access_started
                or failure.outer_2026_accessed
            ):
                raise M03RV16TrainingAggregateError(
                    "V16 numerical training failure identity drifted"
                )
            if (path.parent / "fold-artifacts").exists():
                raise M03RV16TrainingAggregateError(
                    "V16 numerical worker opened outer qualification artifacts"
                )
            numerical_failures[setting] = failure
            all_adequacy.append(())
            all_folds.append(())
            source_roots.add(failure.source_tree_root_sha256)
            terminal_receipts.append(failure.receipt_sha256)
            outcome_kinds.append("numerical-failure")
            continue
        if (path.parent / "training-numerical-failure.json").exists():
            raise M03RV16TrainingAggregateError(
                "V16 setting published both normal and numerical outcomes"
            )
        outcome_kinds.append("training-terminal")
        unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        fold_hashes = tuple(payload.get("fold_terminal_file_sha256", ()))
        for name in (
            "rendered_manifest_sha256",
            "pod_template_sha256",
            "launch_authority_receipt_sha256",
            "admitted_job_authority_receipt_sha256",
        ):
            _digest(name, str(payload.get(name)))
        if (
            payload.get("schema") != M03R_V16_TRAINING_TERMINAL_SCHEMA
            or payload.get("receipt_sha256") != semantic_sha256(unsigned)
            or payload.get("package_plan_sha256") != package.package_plan_sha256
            or payload.get("authorization_receipt_sha256") != authorization.receipt_sha256
            or payload.get("worker_plan_sha256") != package.panel.workers[setting].receipt_sha256
            or not isinstance(payload.get("training_activation_receipt_sha256"), str)
            or payload.get("setting_index") != setting
            or payload.get("qualification_tail_accessed") is not False
            or payload.get("outer_qualification_authorized") is not False
            or payload.get("three_seed_confirmation_may_be_minted") is not False
            or not payload.get("job_uid")
            or not payload.get("pod_uid")
            or not isinstance(
                payload.get("pod_runtime_attestation_receipt_sha256"), str
            )
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
        recomputed_folds = tuple(
            _recompute_fold(
                    root,
                    str(fold_hashes[fold]),
                    setting_index=setting,
                    fold_index=fold,
                    package_plan_sha256=package.package_plan_sha256,
                    authorization_receipt_sha256=authorization.receipt_sha256,
                    worker_plan_sha256=package.panel.workers[setting].receipt_sha256,
                    training_activation_receipt_sha256=str(
                        payload["training_activation_receipt_sha256"]
                    ),
                    panel_schedule_sha256=package.schedule.receipt_sha256,
                    structural_slab_receipt_sha256=(
                        package.artifacts.structural_slab_receipt_sha256
                    ),
                    action_operator_root_sha256=(
                        package.artifacts.structural_action_operator_root_sha256
                    ),
                    target_operator_root_sha256=(
                        package.artifacts.structural_target_operator_root_sha256
                    ),
                )
            for fold in range(M03R_V16_PREDICTIVE_SPEC.chronological_fold_count)
        )
        all_adequacy.append(tuple(row[0] for row in recomputed_folds))
        all_folds.append(tuple(row[1] for row in recomputed_folds))
    if len(source_roots) != 1:
        raise M03RV16TrainingAggregateError("V16 training source roots diverged")
    # A paired target comparison is only valid when every control and primary
    # fold has an adequate terminal fit.
    adequate = not numerical_failures and all(
        row.status == "adequate"
        for setting_rows in all_adequacy
        for row in setting_rows
    )
    status_inventory = tuple(
        row.status for setting_rows in all_adequacy for row in setting_rows
    )
    if numerical_failures:
        next_action = "numerical-investigation"
    elif adequate:
        next_action = "qualification-only-execution"
    elif set(status_inventory).issubset({"adequate", "still-improving"}):
        next_action = "fresh-longer-training-protocol"
    else:
        next_action = "fit-pathology-investigation"

    # Close every terminal checkpoint before issuing any authority that may
    # open an outer origin.  A failure here leaves qualification unauthorized.
    checkpoint_matrix: list[tuple[str, str, str, str, str]] = []
    geometries = render_m03r_v16_fold_geometries(1001)
    if adequate:
        for setting, (terminal_path, folds) in enumerate(
            zip(training_terminal_paths, all_folds, strict=True)
        ):
            root = terminal_path.parent
            checkpoint_rows: list[str] = []
            for geometry, fold in zip(geometries, folds, strict=True):
                checkpoint_sha = str(fold["checkpoint_file_sha256"])
                policy = Top2000M03RV16PredictivePolicy(setting)
                load_m03r_v16_epoch_checkpoint_for_evaluation(
                    root
                    / "checkpoints"
                    / (
                        f"fold-{geometry.fold_index:02d}-epoch-"
                        f"{M03R_V16_PREDICTIVE_SPEC.score_training_epochs:02d}.pt"
                    ),
                    expected_file_sha256=checkpoint_sha,
                    expected_setting_index=setting,
                    expected_fold_index=geometry.fold_index,
                    expected_epoch_index=(
                        M03R_V16_PREDICTIVE_SPEC.score_training_epochs - 1
                    ),
                    expected_completed_score_updates=(
                        geometry.maximum_optimizer_updates
                    ),
                    expected_panel_schedule_sha256=package.schedule.receipt_sha256,
                    expected_selection_target_operator_root_sha256=(
                        package.artifacts.structural_target_operator_root_sha256
                    ),
                    expected_action_operator_root_sha256=(
                        package.artifacts.structural_action_operator_root_sha256
                    ),
                    expected_source_array_sha256=str(
                        fold["checkpoint_source_array_sha256"]
                    ),
                    expected_asset_axis_sha256=package.artifacts.asset_axis_sha256,
                    policy=policy,
                )
                checkpoint_rows.append(checkpoint_sha)
            checkpoint_matrix.append(tuple(checkpoint_rows))  # type: ignore[arg-type]
    closure_unsigned = {
        "schema": M03R_V16_PREQUALIFICATION_CLOSURE_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "package_plan_sha256": package.package_plan_sha256,
        "training_terminal_file_sha256": training_terminal_file_sha256,
        "training_outcome_kind": tuple(outcome_kinds),
        "numerical_failure_receipt_sha256": tuple(
            numerical_failures[index].receipt_sha256
            for index in sorted(numerical_failures)
        ),
        "terminal_checkpoint_file_sha256": tuple(checkpoint_matrix),
        "all_setting_folds_adequate": adequate,
        "outer_qualification_outcomes_accessed": False,
    }
    closure_receipt = semantic_sha256(closure_unsigned)
    output = Path(output_root)
    output.mkdir(mode=0o750, parents=True, exist_ok=False)
    closure = {**closure_unsigned, "receipt_sha256": closure_receipt}
    closure_file_sha = _write(
        output / "prequalification-closure.json", closure
    )
    primary_index = M03R_V16_PREDICTIVE_SPEC.primary_setting_index
    primary_status = (
        ("numerically-invalid",)
        if primary_index in numerical_failures
        else tuple(row.status for row in all_adequacy[primary_index])
    )
    if primary_status and all(status == "adequate" for status in primary_status):
        primary_aggregate = "all-adequate"
    elif "numerically-invalid" in primary_status:
        primary_aggregate = "numerically-invalid"
    elif set(primary_status).issubset({"adequate", "still-improving"}):
        primary_aggregate = "still-improving"
    else:
        primary_aggregate = "fit-pathology"
    unsigned = {
        "schema": M03R_V16_TRAINING_PANEL_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "package_plan_sha256": package.package_plan_sha256,
        "execution_authorization_receipt_sha256": authorization.receipt_sha256,
        "training_terminal_file_sha256": training_terminal_file_sha256,
        "training_outcome_kind": tuple(outcome_kinds),
        "numerical_failure_receipt_sha256": tuple(
            numerical_failures[index].receipt_sha256
            for index in sorted(numerical_failures)
        ),
        "training_terminal_receipt_sha256": tuple(terminal_receipts),
        "setting_fold_adequacy_receipt_sha256": tuple(
            tuple(row.receipt_sha256 for row in setting_rows)
            for setting_rows in all_adequacy
        ),
        "setting_fold_adequacy_status": tuple(
            tuple(row.status for row in setting_rows) for setting_rows in all_adequacy
        ),
        "primary_fold_adequacy_status": primary_status,
        "primary_aggregate_adequacy": primary_aggregate,
        "all_setting_folds_adequate": adequate,
        "terminal_checkpoint_file_sha256": tuple(checkpoint_matrix),
        "prequalification_closure_receipt_sha256": closure_receipt,
        "prequalification_closure_file_sha256": closure_file_sha,
        "outer_qualification_authorized": adequate,
        "next_research_action": next_action,
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
        panel_authority = load_m03r_v16_training_panel_authority(
            training_panel_path=output / "training-panel-decision.json",
            expected_training_panel_file_sha256=panel_file_sha,
            prequalification_closure_path=(
                output / "prequalification-closure.json"
            ),
            expected_prequalification_closure_file_sha256=closure_file_sha,
            training_terminal_paths=training_terminal_paths,
            expected_training_terminal_file_sha256=(
                training_terminal_file_sha256
            ),
            package=package,
            authorization=authorization,
        )
        activation = _issue_m03r_v16_qualification_activation_from_panel_authority(
            package=package,
            authorization=authorization,
            panel=panel_authority,
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
