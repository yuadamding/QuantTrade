#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a contextual second-context action scorer; this is not a full sequential RL policy."
    )
    parser.add_argument("--dataset", type=Path, default=DATA_ROOT / "rl_decision_datasets" / "stock_second_context_15m_v001" / "dataset.pt")
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "second_context_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--train-end", default="2026-06-12T16:00:00+00:00")
    parser.add_argument("--val-end", default="2026-06-12T18:00:00+00:00")
    parser.add_argument("--test-start", default="2026-06-12T18:00:00+00:00")
    parser.add_argument("--test-end")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reward-scale", type=float, default=10_000.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-action-confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confidence-hurdle-bps", type=float, default=2.0)
    parser.add_argument("--confidence-interval-alpha", type=float, default=0.05)
    parser.add_argument("--confidence-min-calibration-rows", type=int, default=1_000)
    parser.add_argument("--confidence-beta-best", type=float, default=0.5)
    parser.add_argument("--confidence-beta-positive", type=float, default=0.5)
    parser.add_argument("--confidence-ood-lambda", type=float, default=1.0)
    return parser.parse_args(argv)


def split_manifest_for(*splits) -> dict[str, object]:
    def summarize(split) -> dict[str, object]:
        return {
            "start": split.decision_timestamps[0] if split.decision_timestamps else None,
            "end": split.decision_timestamps[-1] if split.decision_timestamps else None,
            "rows": len(split.decision_timestamps),
            "valid_rows": int(split.valid_start_indices.numel()),
            "reward_end_max": max(split.next_timestamps) if split.next_timestamps else None,
            "segment_count": int(split.segment_ids.unique().numel()) if split.segment_ids.numel() else 0,
        }

    return {
        "schema_version": "split_manifest_v1",
        "rule": "decision_ts in split and next_ts <= split_end",
        "embargo": None,
        **{("validation" if split.name == "val" else split.name): summarize(split) for split in splits},
    }


def main() -> int:
    import torch

    from rl_quant.confidence import ActionConfidenceCalibrator, ActionConfidenceConfig, save_action_confidence_npz
    from rl_quant.core import configure_torch_runtime, resolve_torch_device
    from rl_quant.research_protocol import stable_json_hash
    from rl_quant.second_context_transformer import (
        SecondContextTransformerQNetwork,
        build_second_context_splits,
        evaluate_second_context_action_scorer,
        evaluate_second_context_baselines,
        evaluate_second_context_trading_policy,
        fixed_rollout_cost_stress,
        masked_contextual_q_loss,
    )

    args = parse_args()
    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train, val, test = build_second_context_splits(
        dataset_path=args.dataset,
        train_end=args.train_end,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    model = SecondContextTransformerQNetwork(
        market_feature_dim=train.market_context.shape[-1],
        action_feature_dim=train.action_features.shape[-1],
        portfolio_state_dim=train.portfolio_state.shape[-1],
        constraint_state_dim=train.constraint_state.shape[-1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        temporal_layers=args.temporal_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_lookback_blocks=train.market_context.shape[1],
        action_count=len(train.action_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    train_dev = train.to(device)
    rows = train_dev.market_context.shape[0]

    @torch.no_grad()
    def predict_q_values(split) -> torch.Tensor:
        data = split.to(device)
        model.eval()
        return model(
            data.market_context,
            data.market_context_mask,
            data.action_features,
            data.portfolio_state,
            data.constraint_state,
        ).detach().cpu()

    def maybe_float(value: torch.Tensor) -> float | None:
        number = float(value.item())
        return number if math.isfinite(number) else None

    def write_selected_action_confidence(
        *,
        path: Path,
        split,
        confidence,
        selected_rows: torch.Tensor,
        selected_actions: torch.Tensor,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as sink:
            for row_value, action_value in zip(selected_rows.tolist(), selected_actions.tolist()):
                row = int(row_value)
                executed_action = int(action_value)
                valid = confidence.valid_actions[row]
                q_row = confidence.q_mean[row].clone()
                q_row = q_row.masked_fill(~valid, -float("inf"))
                raw_policy_action = int(q_row.argmax().item()) if bool(valid.any().item()) else 0
                second_q = q_row.clone()
                if 0 <= executed_action < second_q.numel():
                    second_q[executed_action] = -float("inf")
                second_best_action = int(second_q.argmax().item()) if bool(torch.isfinite(second_q).any().item()) else 0
                confidence_row = confidence.confidence[row].clone()
                confidence_row = confidence_row.masked_fill(~valid, -float("inf"))
                confidence_other = confidence_row.clone()
                if 0 <= executed_action < confidence_other.numel():
                    confidence_other[executed_action] = -float("inf")
                best_other_confidence = (
                    float(confidence_other.max().item()) if bool(torch.isfinite(confidence_other).any().item()) else float("nan")
                )
                selected_confidence = confidence.confidence[row, executed_action]
                q_margin = confidence.q_mean[row, executed_action] - confidence.q_mean[row, second_best_action]
                confidence_margin = float(selected_confidence.item()) - best_other_confidence
                record = {
                    "decision_timestamp": split.decision_timestamps[row],
                    "row_index": row,
                    "raw_policy_action": split.action_names[raw_policy_action],
                    "executed_action": split.action_names[executed_action],
                    "selection_reason": (
                        "selected_by_policy" if raw_policy_action == executed_action else "constraint_or_evaluator_adjusted"
                    ),
                    "selected_q_mean": maybe_float(confidence.q_mean[row, executed_action]),
                    "selected_q_std": maybe_float(confidence.q_std_total[row, executed_action]),
                    "selected_q_lcb_05": maybe_float(confidence.q_lcb[row, executed_action]),
                    "selected_q_ucb_95": maybe_float(confidence.q_ucb[row, executed_action]),
                    "selected_p_positive": maybe_float(confidence.p_positive[row, executed_action]),
                    "selected_p_beats_cash": maybe_float(confidence.p_beats_cash[row, executed_action]),
                    "selected_p_best": maybe_float(confidence.p_best[row, executed_action]),
                    "selected_advantage_mean": maybe_float(confidence.advantage_mean[row, executed_action]),
                    "selected_advantage_lcb": maybe_float(confidence.advantage_lcb[row, executed_action]),
                    "selected_confidence": maybe_float(selected_confidence),
                    "cash_q_mean": maybe_float(confidence.q_mean[row, 0]),
                    "cash_p_best": maybe_float(confidence.p_best[row, 0]),
                    "cash_p_positive": maybe_float(confidence.p_positive[row, 0]),
                    "second_best_action": split.action_names[second_best_action],
                    "second_best_q_mean": maybe_float(confidence.q_mean[row, second_best_action]),
                    "q_margin_vs_second_best": maybe_float(q_margin),
                    "confidence_margin_vs_second_best": confidence_margin if math.isfinite(confidence_margin) else None,
                    "ood_score": maybe_float(confidence.ood_score[row]),
                    "valid_action_count": int(valid.sum().item()),
                    "forced_action_flag": raw_policy_action != executed_action,
                }
                sink.write(json.dumps(record, sort_keys=True) + "\n")

    best_state: dict[str, object] | None = None
    best_score = float("-inf")
    best_epoch = 0
    best_val_policy: dict[str, object] = {}
    selection_protocol = {
        "schema_version": "checkpoint_selection_v1",
        "model_kind": "contextual_action_scorer",
        "primary_selection_metric": "validation_sequential_total_return_after_costs",
        "secondary_terms": [
            "validation_active_net_return_diagnostic",
            "minus_0.001_times_validation_switch_rate",
        ],
        "formula": "validation_total_return + validation_active_net_return - 0.001 * validation_switch_rate",
        "tie_breaker": "lower_validation_switch_rate_then_earlier_epoch",
        "num_trials": int(args.epochs),
        "search_space_hash": stable_json_hash(
            {
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "temporal_layers": args.temporal_layers,
                "feedforward_dim": args.feedforward_dim,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "reward_scale": args.reward_scale,
                "amp": bool(args.amp),
                "seed": args.seed,
            }
        ),
    }
    candidate_validation_scores: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(rows, device=device)
        losses: list[float] = []
        for start in range(0, rows, args.batch_size):
            index = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(args.amp and device.type == "cuda")):
                q_values = model(
                    train_dev.market_context[index],
                    train_dev.market_context_mask[index],
                    train_dev.action_features[index],
                    train_dev.portfolio_state[index],
                    train_dev.constraint_state[index],
                )
                loss = masked_contextual_q_loss(
                    q_values,
                    train_dev.action_returns[index],
                    train_dev.action_valid_mask[index],
                    action_cost_bps=train_dev.action_cost_bps[index],
                    action_target_weights=train_dev.action_target_weights[index],
                    reward_scale=args.reward_scale,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
        val_policy = evaluate_second_context_trading_policy(val, model, device=device, reward_scale=args.reward_scale)
        active = val_policy.get("active_window_diagnostics", {})
        active_return = float(active.get("active_net_return", 0.0)) if isinstance(active, dict) else 0.0
        switch_penalty = 0.001 * float(val_policy.get("switch_rate", 0.0) or 0.0)
        checkpoint_score = float(val_policy.get("total_return", 0.0) or 0.0) + active_return - switch_penalty
        candidate_validation_scores.append(
            {
                "epoch": epoch + 1,
                "validation_total_return": float(val_policy.get("total_return", 0.0) or 0.0),
                "validation_active_net_return": active_return,
                "validation_switch_rate": float(val_policy.get("switch_rate", 0.0) or 0.0),
                "score": checkpoint_score,
            }
        )
        is_better = checkpoint_score > best_score
        if checkpoint_score == best_score and candidate_validation_scores:
            best_switch_rate = float(best_val_policy.get("switch_rate", float("inf")) or float("inf"))
            is_better = float(val_policy.get("switch_rate", 0.0) or 0.0) < best_switch_rate
        if is_better:
            best_score = checkpoint_score
            best_epoch = epoch + 1
            best_val_policy = copy.deepcopy(val_policy)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            val_metrics = evaluate_second_context_action_scorer(val, model, device=device, reward_scale=args.reward_scale)
            avg_loss = sum(losses) / max(len(losses), 1)
            print(
                "epoch="
                f"{epoch + 1} loss={avg_loss:.4f} "
                f"val_rowwise_total={val_metrics['total_return']:.6f} "
                f"val_seq_total={val_policy['total_return']:.6f}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate_second_context_action_scorer(train, model, device=device, reward_scale=args.reward_scale)
    val_metrics = evaluate_second_context_action_scorer(val, model, device=device, reward_scale=args.reward_scale)
    test_metrics = evaluate_second_context_action_scorer(test, model, device=device, reward_scale=args.reward_scale)
    train_policy_metrics = evaluate_second_context_trading_policy(
        train,
        model,
        device=device,
        reward_scale=args.reward_scale,
        return_selected_actions=True,
    )
    val_policy_metrics = evaluate_second_context_trading_policy(
        val,
        model,
        device=device,
        reward_scale=args.reward_scale,
        return_selected_actions=True,
    )
    test_policy_metrics = evaluate_second_context_trading_policy(
        test,
        model,
        device=device,
        reward_scale=args.reward_scale,
        return_decision_logs=True,
        return_selected_actions=True,
    )
    run_name = args.run_name or f"second_context_action_scorer_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_logs = list(test_policy_metrics.pop("decision_logs", []))
    train_selected_actions = torch.tensor(list(train_policy_metrics.pop("selected_actions", [])), dtype=torch.long)
    val_selected_actions = torch.tensor(list(val_policy_metrics.pop("selected_actions", [])), dtype=torch.long)
    test_selected_actions = torch.tensor(list(test_policy_metrics.pop("selected_actions", [])), dtype=torch.long)
    train_selected_rows = torch.tensor(list(train_policy_metrics.pop("selected_rows", [])), dtype=torch.long)
    val_selected_rows = torch.tensor(list(val_policy_metrics.pop("selected_rows", [])), dtype=torch.long)
    test_selected_rows = torch.tensor(list(test_policy_metrics.pop("selected_rows", [])), dtype=torch.long)
    if decision_logs:
        with (run_dir / "decision_logs.jsonl").open("w") as sink:
            for row in decision_logs:
                sink.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    torch.save(
        {
            "train_actions": train_selected_actions,
            "train_rows": train_selected_rows,
            "val_actions": val_selected_actions,
            "val_rows": val_selected_rows,
            "test_actions": test_selected_actions,
            "test_rows": test_selected_rows,
        },
        run_dir / "selected_action_paths.pt",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "market_mean": train.market_mean.detach().cpu(),
            "market_std": train.market_std.detach().cpu(),
            "action_feature_mean": train.action_feature_mean.detach().cpu(),
            "action_feature_std": train.action_feature_std.detach().cpu(),
            "action_names": train.action_names,
            "feature_names": train.feature_names,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "model_kind": "contextual_action_scorer",
        },
        run_dir / "model.pt",
    )
    baselines = evaluate_second_context_baselines(
        train,
        reference_actions=train_selected_actions if train_selected_actions.numel() else None,
        seed=args.seed,
    )
    val_baselines = evaluate_second_context_baselines(
        val,
        reference_actions=val_selected_actions if val_selected_actions.numel() else None,
        seed=args.seed,
    )
    test_baselines = evaluate_second_context_baselines(
        test,
        reference_actions=test_selected_actions if test_selected_actions.numel() else None,
        seed=args.seed,
    )
    cost_stress = (
        fixed_rollout_cost_stress(test, test_selected_actions, row_indices=test_selected_rows)
        if test_selected_actions.numel()
        else {}
    )
    confidence_artifacts: dict[str, str] = {}
    confidence_summary: dict[str, object] = {}
    action_confidence_manifest: dict[str, object] | None = None
    if args.save_action_confidence:
        confidence_config = ActionConfidenceConfig(
            method="single_model_residual",
            hurdle_bps=args.confidence_hurdle_bps,
            interval_alpha=args.confidence_interval_alpha,
            min_calibration_rows=args.confidence_min_calibration_rows,
            confidence_beta_best=args.confidence_beta_best,
            confidence_beta_positive=args.confidence_beta_positive,
            ood_lambda=args.confidence_ood_lambda,
            q_value_scale=args.reward_scale,
        )
        q_by_split = {
            "train": predict_q_values(train),
            "val": predict_q_values(val),
            "test": predict_q_values(test),
        }
        calibrator = ActionConfidenceCalibrator(confidence_config).fit(
            q_by_split["val"],
            val.action_returns,
            val.action_valid_mask,
            action_target_weights=val.action_target_weights,
            action_cost_bps=val.action_cost_bps,
        )
        confidence_by_split = {
            "train": calibrator.predict(q_by_split["train"], train.action_valid_mask),
            "val": calibrator.predict(q_by_split["val"], val.action_valid_mask),
            "test": calibrator.predict(q_by_split["test"], test.action_valid_mask),
        }
        action_confidence_manifest = calibrator.manifest(
            split_name="all",
            ensemble_size=1,
            calibration_split="val",
            uses_test_for_calibration=False,
            uses_checkpoint_selection_for_calibration=True,
        )
        (run_dir / "action_confidence_manifest.json").write_text(
            json.dumps(action_confidence_manifest, indent=2, sort_keys=True) + "\n"
        )
        confidence_artifacts["action_confidence_manifest_json"] = str(run_dir / "action_confidence_manifest.json")
        for split_name, split_obj in (("train", train), ("val", val), ("test", test)):
            path = run_dir / f"action_confidence_{split_name}.npz"
            save_action_confidence_npz(
                path,
                confidence_by_split[split_name],
                row_indices=torch.arange(len(split_obj.decision_timestamps), dtype=torch.long),
                decision_timestamps=split_obj.decision_timestamps,
                action_names=split_obj.action_names,
                manifest={
                    **action_confidence_manifest,
                    "split": split_name,
                    "rows": len(split_obj.decision_timestamps),
                    "valid_rows": int(split_obj.valid_start_indices.numel()),
                },
            )
            confidence_artifacts[f"action_confidence_{split_name}_npz"] = str(path)
        selected_confidence_path = run_dir / "selected_action_confidence_test.jsonl"
        if test_selected_actions.numel():
            write_selected_action_confidence(
                path=selected_confidence_path,
                split=test,
                confidence=confidence_by_split["test"],
                selected_rows=test_selected_rows,
                selected_actions=test_selected_actions,
            )
            confidence_artifacts["selected_action_confidence_test_jsonl"] = str(selected_confidence_path)
        confidence_summary = {
            "enabled": True,
            "method": confidence_config.method,
            "calibration_split": "val",
            "uses_checkpoint_selection_for_calibration": True,
            "metrics": dict(calibrator.metrics),
            "warnings": list(calibrator.warnings),
        }
    payload = torch.load(args.dataset, map_location="cpu", weights_only=True)
    dataset_manifest = dict(payload.get("dataset_manifest", {}))
    data_quality_report = dict(payload.get("data_quality_report", {}))
    split_manifest = split_manifest_for(train, val, test)
    feature_manifest = {
        "feature_names": payload.get("feature_names", {}),
        "schema_version": payload.get("schema_version"),
        "protocol_version": payload.get("protocol_version", "legacy_second_context"),
        "payload_hash": payload.get("payload_hash"),
    }
    model_manifest = {
        "model_kind": "contextual_action_scorer",
        "selected_checkpoint_epoch": best_epoch,
        "selected_checkpoint_score": best_score,
        "selection_metric": selection_protocol["formula"],
        "selection_protocol": selection_protocol,
        "candidate_validation_scores": candidate_validation_scores,
        "selected_val_policy_metrics": best_val_policy,
        "action_confidence": action_confidence_manifest,
        "split_manifest": split_manifest,
        "action_names": train.action_names,
        "feature_names": train.feature_names,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    reportability_reasons: list[str] = []
    if test_policy_metrics.get("final_position_open"):
        reportability_reasons.append("test_final_position_open")
    if "RandomSameTurnoverSameTiming" not in test_baselines:
        reportability_reasons.append("missing_random_same_turnover_same_timing_baseline")
    if not dataset_manifest.get("reportable", False):
        reportability_reasons.append("dataset_non_reportable")
    reportability = {
        "reportable": not reportability_reasons,
        "reasons": reportability_reasons,
        "model_kind": "contextual_action_scorer",
    }
    summary = {
        "dataset": str(args.dataset),
        "device": str(device),
        "rows": {"train": len(train.decision_timestamps), "val": len(val.decision_timestamps), "test": len(test.decision_timestamps)},
        "model_kind": "contextual_action_scorer",
        "checkpoint_selection": {
            "selected_epoch": best_epoch,
            "selected_score": best_score,
            "metric": selection_protocol["formula"],
            "protocol": selection_protocol,
            "candidate_validation_scores": candidate_validation_scores,
        },
        "artifacts": {
            "decision_logs_jsonl": str(run_dir / "decision_logs.jsonl") if decision_logs else None,
            "selected_action_paths_pt": str(run_dir / "selected_action_paths.pt"),
            "split_manifest_json": str(run_dir / "split_manifest.json"),
            **confidence_artifacts,
        },
        "metrics": {
            "action_scorer_rowwise": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
            "sequential_policy_switch_cost": {
                "train": train_policy_metrics,
                "val": val_policy_metrics,
                "test": test_policy_metrics,
            },
            "baselines": {"train": baselines, "val": val_baselines, "test": test_baselines},
            "fixed_rollout_cost_stress": cost_stress,
            "action_confidence": confidence_summary,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "data_quality_report.json").write_text(json.dumps(data_quality_report, indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "reportability.json").write_text(json.dumps(reportability, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True, default=str))
    print(f"Run -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
