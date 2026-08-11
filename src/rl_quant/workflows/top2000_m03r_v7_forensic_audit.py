"""Replay one frozen seed-17 M03R-v7 fold and publish Phase-0 audit evidence.

This is an inference-only development workflow.  It must run from an immutable
evaluation bundle against the original source-homogeneous training tree.  It
does not train, select checkpoints, mutate the training output, or authorize a
new experiment.  One invocation owns one setting/fold output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from rl_quant.evaluation.m03r_alpha_head_diagnostics import (
    M03RAlphaHeadDiagnosticInput,
    build_unavailable_m03r_alpha_head_diagnostics,
    evaluate_m03r_alpha_head_diagnostics,
)
from rl_quant.evaluation.m03r_cost_ladder_evaluator import (
    M03RCostLadderInput,
    evaluate_m03r_cost_ladder,
)
from rl_quant.evaluation.m03r_projection_attribution import (
    M03RProjectionAttributionInput,
    evaluate_m03r_projection_attribution,
)
from rl_quant.evaluation.m03r_setting9_risk_audit import (
    M03R_SETTING9_INDEX,
    M03RSetting9RiskAuditInput,
    evaluate_m03r_setting9_risk_audit,
)
from rl_quant.evaluation.m03r_v7_trace_audit import (
    M03RV7FrozenCheckpointIdentity,
    build_m03r_v7_forensic_trace,
)
from rl_quant.evaluation.top2000_m03r_v7_dev import (
    build_top2000_m03r_v7_validation_runtime,
    tensor_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03RTop2000DevSetting,
    resolve_m03r_top2000_dev_setting,
)
from rl_quant.training.hold30 import Hold30ReplayGeometry
from rl_quant.training.hold30_runtime import Hold30CanonicalTrace
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    Top2000VerifiedDevelopmentCache,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_ALPHA_HORIZONS,
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    bind_top2000_m03r_v7_runtime_sequence,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.workflows import top2000_m03r_v7_dev as base_worker
from rl_quant.workflows.top2000_m03r_v7_seed17_dev import (
    Top2000M03RV7Seed17TrainingPlan,
)

M03R_V7_FORENSIC_WORKFLOW_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-forensic-workflow-v1"
)
M03R_V7_FORENSIC_BUNDLE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-forensic-bundle-v1"
)


class M03RV7ForensicWorkflowError(RuntimeError):
    """A frozen input, replay, or no-clobber boundary failed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV7ForensicWorkflowError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, *, expected_file_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise M03RV7ForensicWorkflowError(f"required JSON is not a regular file: {path}")
    if expected_file_sha256 is not None and _file_sha256(path) != expected_file_sha256:
        raise M03RV7ForensicWorkflowError(f"JSON file hash drifted: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise M03RV7ForensicWorkflowError(f"cannot decode JSON: {path}") from exc
    if not isinstance(value, dict):
        raise M03RV7ForensicWorkflowError(f"JSON must contain an object: {path}")
    return value


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_json(dict(payload)) + b"\n"
    expected = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and not path.is_symlink() and _file_sha256(path) == expected:
            return expected
        raise M03RV7ForensicWorkflowError(f"immutable receipt collision: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return expected


def _write_immutable_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise M03RV7ForensicWorkflowError(f"audit array artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(output, **dict(arrays))  # type: ignore[arg-type]
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise M03RV7ForensicWorkflowError(
                f"audit array artifact appeared during publication: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _load_plan(setting_root: Path, expected_file_sha256: str) -> Top2000M03RV7Seed17TrainingPlan:
    payload = _read_json(
        setting_root / "training-plan.json",
        expected_file_sha256=expected_file_sha256,
    )
    expected = {field.name for field in fields(Top2000M03RV7Seed17TrainingPlan)}
    if set(payload) != expected:
        raise M03RV7ForensicWorkflowError("training-plan fields drifted")
    typed = dict(payload)
    for name in ("fold_indices", "paired_seeds"):
        if isinstance(typed.get(name), list):
            typed[name] = tuple(typed[name])
    try:
        return Top2000M03RV7Seed17TrainingPlan(**typed)
    except (TypeError, ValueError) as exc:
        raise M03RV7ForensicWorkflowError("training plan failed typed validation") from exc


def _resolve_plan_development_setting(
    plan: Top2000M03RV7Seed17TrainingPlan,
) -> M03RTop2000DevSetting:
    """Resolve seed-17 semantics through its bound numerical-route identity."""

    resolved = resolve_m03r_top2000_dev_setting(plan.runtime_setting_id)
    if (
        resolved.setting_index != plan.setting_index
        or resolved.setting_id != plan.runtime_setting_id
    ):
        raise M03RV7ForensicWorkflowError(
            "training plan and development setting identity disagree"
        )
    return resolved


def _alpha_arrays(
    trace: Hold30CanonicalTrace,
    sequence: Any,
    *,
    start: int,
    stop: int,
    expected_available: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rows = stop - start
    assets = sequence.num_assets
    horizons = len(TOP2000_M03R_V7_DEV_ALPHA_HORIZONS)
    observed_predictions = tuple(
        trace.transitions[origin].raw_intent.auxiliary_alpha_mean
        for origin in range(start, stop)
    )
    if not expected_available:
        if any(prediction is not None for prediction in observed_predictions):
            raise M03RV7ForensicWorkflowError(
                "no-alpha-head setting unexpectedly emitted auxiliary predictions"
            )
        return None
    if any(
        prediction is None or tuple(prediction.shape) != (1, assets, horizons)
        for prediction in observed_predictions
    ):
        raise M03RV7ForensicWorkflowError(
            "frozen replay omitted auxiliary alpha heads"
        )
    predictions = np.zeros((rows, assets, horizons), dtype=np.float64)
    targets = np.zeros_like(predictions)
    valid = np.zeros_like(predictions, dtype=np.bool_)
    transition_count = int(sequence.asset_returns.shape[0])
    for local, (origin, prediction) in enumerate(
        zip(range(start, stop), observed_predictions, strict=True)
    ):
        assert prediction is not None
        predictions[local] = prediction[0].detach().to(torch.float64).cpu().numpy()
        for horizon_index, horizon in enumerate(TOP2000_M03R_V7_DEV_ALPHA_HORIZONS):
            first = origin + 1
            horizon_stop = first + horizon
            if horizon_stop > transition_count:
                continue
            stock = torch.log1p(
                sequence.asset_returns[first:horizon_stop, 0].clamp_min(-0.999999)
            ).sum(0)
            benchmark = torch.log1p(
                sequence.benchmark_net_returns[first:horizon_stop, 0].clamp_min(-0.999999)
            ).sum()
            target = stock - benchmark
            mask = sequence.decision_available[first : horizon_stop + 1, 0].all(0)
            mask = mask.detach().cpu().clone()
            mask[0] = False
            targets[local, :, horizon_index] = target.detach().to(torch.float64).cpu().numpy()
            valid[local, :, horizon_index] = mask.numpy()
    return predictions, targets, valid


def run_m03r_v7_seed17_forensic_fold(
    *,
    setting_root: str | Path,
    cache_path: str | Path,
    expected_cache_sha256: str,
    expected_training_plan_file_sha256: str,
    evaluation_source_inventory_sha256: str,
    source_training_archive_sha256: str,
    fold_index: int,
    output_root: str | Path,
    device: str | torch.device,
    _prepared_plan: Top2000M03RV7Seed17TrainingPlan | None = None,
    _verified_cache: Top2000VerifiedDevelopmentCache | None = None,
) -> dict[str, Any]:
    """Replay and publish one exact setting/fold audit bundle."""

    for name, digest in (
        ("expected_cache_sha256", expected_cache_sha256),
        ("expected_training_plan_file_sha256", expected_training_plan_file_sha256),
        ("evaluation_source_inventory_sha256", evaluation_source_inventory_sha256),
        ("source_training_archive_sha256", source_training_archive_sha256),
    ):
        _require_sha256(name, digest)
    if isinstance(fold_index, bool) or not isinstance(fold_index, int) or not 0 <= fold_index < 6:
        raise M03RV7ForensicWorkflowError("fold_index must lie in [0,5]")
    setting_path = Path(setting_root)
    if not setting_path.is_dir() or setting_path.is_symlink():
        raise M03RV7ForensicWorkflowError("setting_root must be a regular directory")
    output_path = Path(output_root)
    if output_path.exists() and (not output_path.is_dir() or output_path.is_symlink()):
        raise M03RV7ForensicWorkflowError("output_root must be a regular directory")
    output_path.mkdir(parents=True, exist_ok=True)
    plan = (
        _load_plan(setting_path, expected_training_plan_file_sha256)
        if _prepared_plan is None
        else _prepared_plan
    )
    if (
        plan.setting_index < 0
        or plan.output_root != str(setting_path)
        or plan.receipt_sha256
        != Top2000M03RV7Seed17TrainingPlan(
            **{
                field.name: getattr(plan, field.name)
                for field in fields(Top2000M03RV7Seed17TrainingPlan)
            }
        ).receipt_sha256
    ):
        raise M03RV7ForensicWorkflowError("prepared training plan identity drifted")
    if plan.cache_sha256 != expected_cache_sha256:
        raise M03RV7ForensicWorkflowError("training plan and requested cache hash disagree")
    completed_receipt_path = output_path / "forensic-audit-receipt.json"
    if completed_receipt_path.exists():
        prior = _read_json(completed_receipt_path)
        trace_receipt = prior.get("forensic_trace")
        checkpoint_receipt = (
            trace_receipt.get("checkpoint") if isinstance(trace_receipt, Mapping) else None
        )
        artifact_name = prior.get("array_artifact_path")
        artifact_hash = prior.get("array_artifact_file_sha256")
        artifact_path = (
            output_path / artifact_name if isinstance(artifact_name, str) else output_path
        )
        if (
            prior.get("schema") != M03R_V7_FORENSIC_BUNDLE_SCHEMA
            or prior.get("evaluation_source_inventory_sha256")
            != evaluation_source_inventory_sha256
            or prior.get("source_training_archive_sha256")
            != source_training_archive_sha256
            or prior.get("training_plan_file_sha256")
            != expected_training_plan_file_sha256
            or not isinstance(checkpoint_receipt, Mapping)
            or checkpoint_receipt.get("setting_index") != plan.setting_index
            or checkpoint_receipt.get("fold_index") != fold_index
            or not isinstance(artifact_hash, str)
            or not artifact_path.is_file()
            or artifact_path.is_symlink()
            or _file_sha256(artifact_path) != artifact_hash
        ):
            raise M03RV7ForensicWorkflowError("completed forensic receipt or artifact drifted")
        return {
            "receipt_path": str(completed_receipt_path),
            "receipt_file_sha256": _file_sha256(completed_receipt_path),
            "receipt_sha256": _sha256(prior),
            "setting_index": plan.setting_index,
            "fold_index": fold_index,
            "retraining_performed": False,
            "validation_only_retry": True,
        }
    training_root = setting_path / "training"
    seed_receipt_path = (
        training_root
        / "receipts"
        / "seed-validation"
        / f"fold-{fold_index:02d}-seed-17.json"
    )
    seed_receipt = _read_json(seed_receipt_path)
    fold = render_top2000_m03r_v7_development_folds(
        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
    )[fold_index]
    expected_seed_fields = {
        "protocol_sha256": plan.protocol_sha256,
        "setting_index": plan.setting_index,
        "setting_id": plan.setting_id,
        "fold_index": fold_index,
        "seed": 17,
        "fold_receipt_sha256": fold.receipt_sha256,
        "checkpoint_selection_rule": "frozen-final-optimizer-update-no-validation-selection-v1",
        "development_only": True,
        "promotion_eligible": False,
    }
    if any(seed_receipt.get(name) != expected for name, expected in expected_seed_fields.items()):
        raise M03RV7ForensicWorkflowError("seed validation receipt identity drifted")
    model_file_sha256 = _require_sha256(
        "checkpoint_file_sha256", seed_receipt.get("checkpoint_file_sha256")
    )
    model_path = training_root / "cells" / f"fold-{fold_index:02d}-seed-17" / "model.rank-00.pt"
    if not model_path.is_file() or model_path.is_symlink() or _file_sha256(model_path) != model_file_sha256:
        raise M03RV7ForensicWorkflowError("frozen checkpoint file drifted")
    try:
        model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise M03RV7ForensicWorkflowError("cannot inspect frozen checkpoint") from exc
    if not isinstance(model_payload, Mapping):
        raise M03RV7ForensicWorkflowError("frozen checkpoint must be a mapping")
    alpha_optimizer_sha256 = _require_sha256(
        "alpha_core_optimizer_state_sha256",
        model_payload.get("alpha_core_optimizer_state_sha256"),
    )
    overlay_raw = model_payload.get("overlay_optimizer_state_sha256")
    overlay_sha256 = None if overlay_raw is None else _require_sha256(
        "overlay_optimizer_state_sha256", overlay_raw
    )
    device_value = torch.device(device)
    # The original worker disables TF32 and fixes CUDA/cuDNN determinism before
    # constructing every seed-17 cell.  Re-establish that exact process-local
    # state before the detached replay; otherwise a fresh image may use a
    # different matmul mode even though the checkpoint bytes are identical.
    base_worker._seed_everything(17)
    policy, observed_model_sha256 = base_worker._load_saved_seed_policy(
        model_path,
        expected_file_sha256=model_file_sha256,
        plan=cast(Any, plan),
        fold=fold,
        seed=17,
        device=device_value,
    )
    expected_model_sha256 = _require_sha256(
        "model_state_sha256", seed_receipt.get("model_state_sha256")
    )
    if observed_model_sha256 != expected_model_sha256:
        raise M03RV7ForensicWorkflowError("loaded model state drifted")
    cache = (
        load_verified_top2000_hold30_development_cache(
            cache_path,
            expected_cache_sha256=expected_cache_sha256,
            acknowledgement=DEVELOPMENT_ACK,
        )
        if _verified_cache is None
        else _verified_cache
    )
    if cache.cache_sha256 != expected_cache_sha256:
        raise M03RV7ForensicWorkflowError("prepared cache identity drifted")
    cache.validate_unmodified()
    built, sequence, calibration, score_start, score_stop = (
        base_worker._build_validation_episode(cache, fold, device=device_value)
    )
    if seed_receipt.get("sequence_receipt_sha256") != built.identity.receipt_sha256:
        raise M03RV7ForensicWorkflowError("replayed validation sequence identity drifted")
    policy.bind_episode_factor_loadings(calibration.loadings)
    bound, provider = bind_top2000_m03r_v7_runtime_sequence(sequence, policy)
    runtime = build_top2000_m03r_v7_validation_runtime(policy, state_provider=provider)
    roles = Hold30ReplayGeometry(
        warmup_decisions=63,
        label_support_decisions=63,
        max_origin_batch=1,
    ).roles(sequence.n_positions)
    with torch.no_grad():
        canonical, _ = runtime.canonical_pass(policy, bound, roles)
    if not isinstance(canonical, Hold30CanonicalTrace):
        raise M03RV7ForensicWorkflowError("runtime did not return a canonical trace")
    benchmark = built.benchmark
    score_dates = tuple(built.exchange_dates[score_start:score_stop])
    checkpoint = M03RV7FrozenCheckpointIdentity(
        setting_index=plan.setting_index,
        setting_id=plan.runtime_setting_id,
        fold_index=fold_index,
        seed=17,
        checkpoint_file_sha256=model_file_sha256,
        model_state_sha256=expected_model_sha256,
        alpha_core_optimizer_state_sha256=alpha_optimizer_sha256,
        overlay_optimizer_state_sha256=overlay_sha256,
        factor_calibration_receipt_sha256=calibration.receipt_sha256,
    )
    forensic = build_m03r_v7_forensic_trace(
        canonical,
        checkpoint=checkpoint,
        score_dates=score_dates,
        score_transition_start=score_start,
        score_transition_stop_exclusive=score_stop,
        benchmark_weights=benchmark.weights[score_start:score_stop],
        benchmark_gross_returns=benchmark.gross_returns[score_start:score_stop],
        benchmark_net_returns_20bp=benchmark.net_returns[score_start:score_stop],
        benchmark_total_one_way_turnover=benchmark.total_one_way_turnover[
            score_start:score_stop
        ],
    )
    original_arrays = seed_receipt.get("array_sha256")
    if not isinstance(original_arrays, Mapping):
        raise M03RV7ForensicWorkflowError("seed receipt omitted compact array hashes")
    compact_replay = {
        "policy_net_returns": forensic.policy_net_returns_20bp,
        "benchmark_net_returns": forensic.benchmark_net_returns_20bp,
        "total_one_way_turnover": forensic.policy_total_one_way_turnover,
    }
    for name, array in compact_replay.items():
        observed = tensor_sha256(torch.from_numpy(np.asarray(array)))
        if original_arrays.get(name) != observed:
            raise M03RV7ForensicWorkflowError(
                f"forensic replay does not match original compact {name} evidence"
            )
    cost_ladder = evaluate_m03r_cost_ladder(
        M03RCostLadderInput(
            setting_index=plan.setting_index,
            setting_id=plan.setting_id,
            fold_index=fold_index,
            score_dates=score_dates,
            policy_net_returns_20bp=forensic.policy_net_returns_20bp,
            benchmark_net_returns_20bp=forensic.benchmark_net_returns_20bp,
            policy_total_one_way_turnover=forensic.policy_total_one_way_turnover,
            benchmark_total_one_way_turnover=forensic.benchmark_total_one_way_turnover,
        )
    )
    resolved_setting = _resolve_plan_development_setting(plan)
    residual_alpha_head_mode = resolved_setting.residual_alpha_head_mode
    alpha_arrays = _alpha_arrays(
        canonical,
        sequence,
        start=score_start,
        stop=score_stop,
        expected_available=residual_alpha_head_mode != "none",
    )
    alpha_artifact_arrays: dict[str, np.ndarray]
    alpha_array_status: dict[str, Any]
    if alpha_arrays is None:
        alpha_diagnostics = build_unavailable_m03r_alpha_head_diagnostics(
            setting_index=plan.setting_index,
            setting_id=plan.setting_id,
            fold_index=fold_index,
            score_dates=score_dates,
            action_ids=built.action_ids,
            residual_alpha_head_mode=residual_alpha_head_mode,
        )
        alpha_scores = None
        alpha_artifact_arrays = {}
        alpha_array_status = {
            "status": "unavailable",
            "reason": "setting-intentionally-disables-residual-alpha-heads",
            "residual_alpha_head_mode": residual_alpha_head_mode,
            "array_names": [],
        }
    else:
        predictions, targets, valid = alpha_arrays
        alpha_diagnostics = evaluate_m03r_alpha_head_diagnostics(
            M03RAlphaHeadDiagnosticInput(
                setting_index=plan.setting_index,
                setting_id=plan.setting_id,
                fold_index=fold_index,
                score_dates=score_dates,
                action_ids=built.action_ids,
                predictions=predictions,
                targets=targets,
                valid=valid,
            )
        )
        alpha_scores = predictions[:, :, 2]
        alpha_artifact_arrays = {
            "alpha_predictions": predictions,
            "alpha_targets": targets,
            "alpha_valid": valid,
        }
        alpha_array_status = {
            "status": "available",
            "reason": None,
            "residual_alpha_head_mode": residual_alpha_head_mode,
            "array_names": sorted(alpha_artifact_arrays),
        }
    projection = evaluate_m03r_projection_attribution(
        M03RProjectionAttributionInput(
            setting_index=plan.setting_index,
            setting_id=plan.setting_id,
            fold_index=fold_index,
            score_dates=score_dates,
            benchmark_weights=forensic.benchmark_weights,
            requested_weights=forensic.requested_weights,
            post_hazard_weights=forensic.post_hazard_weights,
            post_projection_weights=forensic.post_projection_weights,
            executed_weights=forensic.executed_weights,
            alpha_scores=alpha_scores,
            covariance=None,
        )
    )
    setting9 = None
    if plan.setting_index == M03R_SETTING9_INDEX:
        active = forensic.policy_net_returns_20bp - forensic.benchmark_net_returns_20bp
        setting9 = evaluate_m03r_setting9_risk_audit(
            M03RSetting9RiskAuditInput(
                fold_index=fold_index,
                initial_policy_weights=canonical.boundary_states[0].ledger.weights[0]
                .detach()
                .to(torch.float64)
                .cpu()
                .numpy(),
                initial_benchmark_weights=benchmark.weights[0]
                .detach()
                .to(torch.float64)
                .cpu()
                .numpy(),
                requested_annual_tracking_error=None,
                post_projection_annual_tracking_error=None,
                realized_active_returns=active,
                reported_total_one_way_turnover=forensic.policy_total_one_way_turnover,
                startup_turnover=0.0,
                startup_turnover_in_reported_mean=False,
                benchmark_anchoring_enabled=True,
                tracking_error_control_enabled=False,
                active_beta_control_enabled=False,
            )
        )
    arrays_path = output_path / "forensic-arrays.npz"
    array_file_sha256 = _write_immutable_npz(
        arrays_path,
        {
            **{
                name: getattr(forensic, name)
                for name in forensic.array_sha256s()
            },
            **alpha_artifact_arrays,
        },
    )
    payload: dict[str, Any] = {
        "schema": M03R_V7_FORENSIC_BUNDLE_SCHEMA,
        "workflow_schema": M03R_V7_FORENSIC_WORKFLOW_SCHEMA,
        "evaluation_source_inventory_sha256": evaluation_source_inventory_sha256,
        "source_training_archive_sha256": source_training_archive_sha256,
        "training_plan_file_sha256": expected_training_plan_file_sha256,
        "training_plan_receipt_sha256": plan.receipt_sha256,
        "original_seed_validation_receipt_file_sha256": _file_sha256(seed_receipt_path),
        "original_seed_validation_receipt_sha256": _sha256(seed_receipt),
        "forensic_trace": forensic.receipt,
        "cost_ladder": cost_ladder,
        "alpha_head_diagnostics": alpha_diagnostics,
        "alpha_head_array_status": alpha_array_status,
        "projection_attribution": projection,
        "setting9_risk_audit": setting9,
        "array_artifact_path": arrays_path.name,
        "array_artifact_file_sha256": array_file_sha256,
        "retraining_performed": False,
        "checkpoint_selection_performed": False,
        "development_only": True,
        "future_selected_universe": True,
        "reportable": False,
        "promotable": False,
    }
    receipt_path = completed_receipt_path
    receipt_file_sha256 = _write_immutable_json(receipt_path, payload)
    return {
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": _sha256(payload),
        "setting_index": plan.setting_index,
        "fold_index": fold_index,
        "retraining_performed": False,
    }


def run_m03r_v7_seed17_forensic_setting(
    *,
    setting_root: str | Path,
    cache_path: str | Path,
    expected_cache_sha256: str,
    expected_training_plan_file_sha256: str,
    evaluation_source_inventory_sha256: str,
    source_training_archive_sha256: str,
    output_root: str | Path,
    device: str | torch.device,
) -> dict[str, Any]:
    """Replay six folds serially while loading and verifying the cache once."""

    setting_path = Path(setting_root)
    output_path = Path(output_root)
    plan = _load_plan(setting_path, expected_training_plan_file_sha256)
    cache = load_verified_top2000_hold30_development_cache(
        cache_path,
        expected_cache_sha256=expected_cache_sha256,
        acknowledgement=DEVELOPMENT_ACK,
    )
    results = tuple(
        run_m03r_v7_seed17_forensic_fold(
            setting_root=setting_path,
            cache_path=cache_path,
            expected_cache_sha256=expected_cache_sha256,
            expected_training_plan_file_sha256=expected_training_plan_file_sha256,
            evaluation_source_inventory_sha256=evaluation_source_inventory_sha256,
            source_training_archive_sha256=source_training_archive_sha256,
            fold_index=fold_index,
            output_root=output_path / f"fold-{fold_index:02d}",
            device=device,
            _prepared_plan=plan,
            _verified_cache=cache,
        )
        for fold_index in range(6)
    )
    fold_receipt_hashes = {
        f"fold-{fold_index:02d}": result["receipt_file_sha256"]
        for fold_index, result in enumerate(results)
    }
    payload: dict[str, Any] = {
        "schema": "rl-quant.top2000-dev.m03r-v7-seed17-forensic-setting-v1",
        "setting_index": plan.setting_index,
        "setting_id": plan.setting_id,
        "runtime_setting_id": plan.runtime_setting_id,
        "training_plan_file_sha256": expected_training_plan_file_sha256,
        "training_plan_receipt_sha256": plan.receipt_sha256,
        "cache_sha256": expected_cache_sha256,
        "evaluation_source_inventory_sha256": evaluation_source_inventory_sha256,
        "source_training_archive_sha256": source_training_archive_sha256,
        "fold_receipt_file_sha256": fold_receipt_hashes,
        "fold_count": 6,
        "cache_load_count": 1,
        "retraining_performed": False,
        "checkpoint_selection_performed": False,
        "development_only": True,
        "future_selected_universe": True,
        "reportable": False,
        "promotable": False,
    }
    receipt_path = output_path / "forensic-setting-receipt.json"
    receipt_file_sha256 = _write_immutable_json(receipt_path, payload)
    return {
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": _sha256(payload),
        "setting_index": plan.setting_index,
        "fold_count": 6,
        "cache_load_count": 1,
        "retraining_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting-root", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--training-plan-file-sha256", required=True)
    parser.add_argument("--evaluation-source-inventory-sha256", required=True)
    parser.add_argument("--source-training-archive-sha256", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold-index", type=int)
    group.add_argument("--all-folds", action="store_true")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "setting_root": args.setting_root,
        "cache_path": args.cache_path,
        "expected_cache_sha256": args.cache_sha256,
        "expected_training_plan_file_sha256": args.training_plan_file_sha256,
        "evaluation_source_inventory_sha256": args.evaluation_source_inventory_sha256,
        "source_training_archive_sha256": args.source_training_archive_sha256,
        "output_root": args.output_root,
        "device": args.device,
    }
    result = (
        run_m03r_v7_seed17_forensic_setting(**common)
        if args.all_folds
        else run_m03r_v7_seed17_forensic_fold(
            **common,
            fold_index=args.fold_index,
        )
    )
    print(_canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V7_FORENSIC_BUNDLE_SCHEMA",
    "M03R_V7_FORENSIC_WORKFLOW_SCHEMA",
    "M03RV7ForensicWorkflowError",
    "main",
    "run_m03r_v7_seed17_forensic_fold",
    "run_m03r_v7_seed17_forensic_setting",
]
