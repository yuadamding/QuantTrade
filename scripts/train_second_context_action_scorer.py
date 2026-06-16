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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Effective optimizer batch size. Use --micro-batch-size to control the actual CUDA batch.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="Rows placed through the transformer at once before gradient accumulation; defaults to --batch-size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Rows scored at once for validation, testing, and confidence artifacts; defaults to the micro-batch size.",
    )
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-epochs", type=int)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reward-scale", type=float, default=10_000.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="AMP autocast precision when --amp is set. bf16 (wider exponent range, no GradScaler) "
        "is preferred on Ampere/Hopper GPUs; fp16 (default) preserves prior behavior.",
    )
    parser.add_argument(
        "--target-vram-gb",
        type=float,
        help="OPT-IN CUDA ballast: reserve byte tensors after warmup toward this total used VRAM. "
        "This INCREASES memory (it does not cap/shard/offload) -- leave unset for large models; use "
        "--min-free-vram-gb to guard headroom instead.",
    )
    parser.add_argument("--vram-safety-gb", type=float, default=0.12)
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=0.0,
        help="Fail fast before training if free CUDA memory is below this many GiB (0 disables).",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--empty-cache-every-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-action-confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confidence-hurdle-bps", type=float, default=2.0)
    parser.add_argument("--confidence-interval-alpha", type=float, default=0.05)
    parser.add_argument("--confidence-min-calibration-rows", type=int, default=1_000)
    parser.add_argument("--confidence-beta-best", type=float, default=0.5)
    parser.add_argument("--confidence-beta-positive", type=float, default=0.5)
    parser.add_argument("--confidence-ood-lambda", type=float, default=1.0)
    parser.add_argument("--confidence-p-best-draws", type=int, default=512)
    parser.add_argument("--confidence-p-best-draw-batch-rows", type=int, default=512)
    parser.add_argument("--confidence-p-best-draw-batch-size", type=int, default=64)
    parser.add_argument("--confidence-p-best-draw-seed", type=int, default=17)
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


def resolve_batch_plan(
    *,
    batch_size: int,
    micro_batch_size: int | None,
    eval_batch_size: int | None,
    checkpoint_every_epochs: int,
    log_every_epochs: int | None,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    resolved_micro_batch = batch_size if micro_batch_size is None else int(micro_batch_size)
    if resolved_micro_batch <= 0:
        raise ValueError("micro_batch_size must be positive.")
    if resolved_micro_batch > batch_size:
        raise ValueError("micro_batch_size cannot exceed batch_size.")
    resolved_eval_batch = resolved_micro_batch if eval_batch_size is None else int(eval_batch_size)
    if resolved_eval_batch <= 0:
        raise ValueError("eval_batch_size must be positive.")
    if checkpoint_every_epochs <= 0:
        raise ValueError("checkpoint_every_epochs must be positive.")
    resolved_log_every = log_every_epochs if log_every_epochs is not None else 0
    if resolved_log_every < 0:
        raise ValueError("log_every_epochs cannot be negative.")
    return {
        "batch_size": int(batch_size),
        "micro_batch_size": int(resolved_micro_batch),
        "gradient_accumulation_steps": int(math.ceil(batch_size / resolved_micro_batch)),
        "eval_batch_size": int(resolved_eval_batch),
        "checkpoint_every_epochs": int(checkpoint_every_epochs),
        "log_every_epochs": int(resolved_log_every),
    }


def _slice_to_device(tensor, indices, device, *, pin_memory: bool):
    index = indices.to(tensor.device) if tensor.device.type != "cpu" else indices
    values = tensor[index]
    if values.device.type == "cpu" and device.type == "cuda" and pin_memory:
        values = values.pin_memory()
    return values.to(device, non_blocking=device.type == "cuda")


def train_second_context_epoch(
    train_split,
    model,
    optimizer,
    scaler,
    *,
    device,
    batch_size: int,
    micro_batch_size: int,
    reward_scale: float,
    use_amp: bool,
    grad_clip: float,
    amp_dtype: str = "fp16",
    pin_memory: bool = True,
    vram_reservation=None,
) -> dict[str, float | int]:
    import torch

    from rl_quant.core import autocast_context
    from rl_quant.second_context_transformer import masked_contextual_q_loss

    if batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError("batch_size and micro_batch_size must be positive.")
    if micro_batch_size > batch_size:
        raise ValueError("micro_batch_size cannot exceed batch_size.")
    rows = int(train_split.market_context.shape[0])
    if rows <= 0:
        raise ValueError("Cannot train on an empty split.")

    model.train()
    order = torch.randperm(rows)
    losses: list[float] = []
    optimizer_steps = 0
    micro_batches = 0
    rows_seen = 0
    valid_targets_seen = 0
    for start in range(0, rows, batch_size):
        batch_indices = order[start : start + batch_size]
        train_loss_mask = train_split.supervised_action_valid_mask
        total_valid = int(train_loss_mask[batch_indices].sum().item())
        if total_valid <= 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        batch_micro_batches = 0
        for micro_start in range(0, int(batch_indices.numel()), micro_batch_size):
            micro_indices = batch_indices[micro_start : micro_start + micro_batch_size]
            micro_valid = int(train_loss_mask[micro_indices].sum().item())
            if micro_valid <= 0:
                continue
            market_context = _slice_to_device(train_split.market_context, micro_indices, device, pin_memory=pin_memory)
            market_mask = _slice_to_device(train_split.market_context_mask, micro_indices, device, pin_memory=pin_memory)
            action_features = _slice_to_device(train_split.action_features, micro_indices, device, pin_memory=pin_memory)
            portfolio_state = _slice_to_device(train_split.portfolio_state, micro_indices, device, pin_memory=pin_memory)
            constraint_state = _slice_to_device(train_split.constraint_state, micro_indices, device, pin_memory=pin_memory)
            action_returns = _slice_to_device(train_split.action_returns, micro_indices, device, pin_memory=pin_memory)
            action_valid_mask = _slice_to_device(train_loss_mask, micro_indices, device, pin_memory=pin_memory)
            action_cost_bps = _slice_to_device(train_split.action_cost_bps, micro_indices, device, pin_memory=pin_memory)
            action_target_weights = _slice_to_device(
                train_split.action_target_weights,
                micro_indices,
                device,
                pin_memory=pin_memory,
            )
            with autocast_context(device, use_amp, amp_dtype):
                q_values = model(market_context, market_mask, action_features, portfolio_state, constraint_state)
                loss = masked_contextual_q_loss(
                    q_values,
                    action_returns,
                    action_valid_mask,
                    action_cost_bps=action_cost_bps,
                    action_target_weights=action_target_weights,
                    reward_scale=reward_scale,
                )
                scaled_loss = loss * (micro_valid / total_valid)
            scaler.scale(scaled_loss).backward()
            batch_loss += float(loss.detach().cpu().item()) * micro_valid / total_valid
            batch_micro_batches += 1
            micro_batches += 1
            rows_seen += int(micro_indices.numel())
            valid_targets_seen += micro_valid
        if batch_micro_batches <= 0:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        optimizer_steps += 1
        losses.append(batch_loss)
        if vram_reservation is not None:
            vram_reservation.maybe_reserve(device)

    return {
        "average_loss": sum(losses) / max(len(losses), 1),
        "optimizer_steps": optimizer_steps,
        "micro_batches": micro_batches,
        "rows_seen": rows_seen,
        "valid_targets_seen": valid_targets_seen,
    }


def main() -> int:
    import torch

    from rl_quant.confidence import ActionConfidenceCalibrator, ActionConfidenceConfig, save_action_confidence_npz
    from rl_quant.core import (
        CudaVramReservation,
        configure_torch_runtime,
        make_grad_scaler,
        require_min_free_vram,
        resolve_torch_device,
    )
    from rl_quant.research_protocol import stable_json_hash
    from rl_quant.second_context_transformer import (
        SecondContextTransformerQNetwork,
        build_second_context_splits,
        evaluate_second_context_action_scorer,
        evaluate_second_context_baselines,
        evaluate_second_context_trading_policy,
        fixed_rollout_cost_stress,
        predict_second_context_q_values,
        second_context_missing_label_report,
    )

    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    batch_plan = resolve_batch_plan(
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        eval_batch_size=args.eval_batch_size,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        log_every_epochs=args.log_every_epochs,
    )
    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    require_min_free_vram(device, args.min_free_vram_gb)
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
    scaler = make_grad_scaler(device, args.amp, args.amp_dtype)
    reservation = CudaVramReservation(target_gb=args.target_vram_gb, safety_gb=args.vram_safety_gb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    @torch.no_grad()
    def predict_q_values(split) -> torch.Tensor:
        return predict_second_context_q_values(
            split,
            model,
            device=device,
            batch_size=batch_plan["eval_batch_size"],
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            pin_memory=args.pin_memory,
        )

    def maybe_float(value: torch.Tensor) -> float | None:
        number = float(value.item())
        return number if math.isfinite(number) else None

    def write_selected_action_confidence(
        *,
        path: Path,
        split,
        confidence,
        selected_rows: torch.Tensor,
        executed_actions: torch.Tensor,
        requested_actions: torch.Tensor | None = None,
        raw_policy_actions: torch.Tensor | None = None,
        constraint_adjusted_actions: torch.Tensor | None = None,
        selection_reasons: list[str] | None = None,
    ) -> None:
        def action_name(action: int) -> str:
            return split.action_names[action] if 0 <= action < len(split.action_names) else str(action)

        def maybe_action_float(matrix: torch.Tensor, row: int, action: int) -> float | None:
            if action < 0 or action >= matrix.shape[1]:
                return None
            return maybe_float(matrix[row, action])

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as sink:
            for position, (row_value, action_value) in enumerate(zip(selected_rows.tolist(), executed_actions.tolist())):
                row = int(row_value)
                executed_action = int(action_value)
                valid = confidence.valid_actions[row]
                q_row = confidence.q_mean[row].clone()
                q_row = q_row.masked_fill(~valid, -float("inf"))
                inferred_raw_policy_action = int(q_row.argmax().item()) if bool(valid.any().item()) else 0
                raw_policy_action = (
                    int(raw_policy_actions[position].item())
                    if raw_policy_actions is not None and position < raw_policy_actions.numel()
                    else inferred_raw_policy_action
                )
                requested_action = (
                    int(requested_actions[position].item())
                    if requested_actions is not None and position < requested_actions.numel()
                    else executed_action
                )
                constraint_adjusted_action = (
                    int(constraint_adjusted_actions[position].item())
                    if constraint_adjusted_actions is not None and position < constraint_adjusted_actions.numel()
                    else requested_action
                )
                selection_reason = (
                    str(selection_reasons[position])
                    if selection_reasons is not None and position < len(selection_reasons)
                    else (
                        "selected_by_policy"
                        if raw_policy_action == requested_action == executed_action
                        else "constraint_or_evaluator_adjusted"
                    )
                )
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
                    "raw_policy_action": action_name(raw_policy_action),
                    "constraint_adjusted_action": action_name(constraint_adjusted_action),
                    "requested_action": action_name(requested_action),
                    "executed_action": action_name(executed_action),
                    "selection_reason": selection_reason,
                    "selected_q_mean": maybe_float(confidence.q_mean[row, executed_action]),
                    "selected_q_std": maybe_float(confidence.q_std_total[row, executed_action]),
                    "selected_q_lcb_05": maybe_float(confidence.q_lcb[row, executed_action]),
                    "selected_q_ucb_95": maybe_float(confidence.q_ucb[row, executed_action]),
                    "selected_p_positive": maybe_float(confidence.p_positive[row, executed_action]),
                    "selected_profit_confidence": maybe_float(confidence.profit_confidence[row, executed_action]),
                    "selected_p_beats_cash": maybe_float(confidence.p_beats_cash[row, executed_action]),
                    "selected_p_best": maybe_float(confidence.p_best[row, executed_action]),
                    "selected_p_best_member_vote": maybe_float(confidence.p_best_member_vote[row, executed_action]),
                    "selected_p_best_draw": maybe_float(confidence.p_best_draw[row, executed_action]),
                    "selected_selection_confidence": maybe_float(confidence.selection_confidence[row, executed_action]),
                    "selected_advantage_mean": maybe_float(confidence.advantage_mean[row, executed_action]),
                    "selected_advantage_lcb": maybe_float(confidence.advantage_lcb[row, executed_action]),
                    "selected_confidence": maybe_float(selected_confidence),
                    "requested_q_mean": maybe_action_float(confidence.q_mean, row, requested_action),
                    "requested_q_std": maybe_action_float(confidence.q_std_total, row, requested_action),
                    "requested_profit_confidence": maybe_action_float(confidence.profit_confidence, row, requested_action),
                    "requested_selection_confidence": maybe_action_float(confidence.selection_confidence, row, requested_action),
                    "requested_confidence": maybe_action_float(confidence.confidence, row, requested_action),
                    "cash_q_mean": maybe_float(confidence.q_mean[row, 0]),
                    "cash_p_best": maybe_float(confidence.p_best[row, 0]),
                    "cash_p_best_member_vote": maybe_float(confidence.p_best_member_vote[row, 0]),
                    "cash_p_best_draw": maybe_float(confidence.p_best_draw[row, 0]),
                    "cash_p_positive": maybe_float(confidence.p_positive[row, 0]),
                    "second_best_action": action_name(second_best_action),
                    "second_best_q_mean": maybe_float(confidence.q_mean[row, second_best_action]),
                    "q_margin_vs_second_best": maybe_float(q_margin),
                    "confidence_margin_vs_second_best": confidence_margin if math.isfinite(confidence_margin) else None,
                    "ood_score": maybe_float(confidence.ood_score[row]),
                    "valid_action_count": int(valid.sum().item()),
                    "forced_action_flag": raw_policy_action != requested_action or requested_action != executed_action,
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
        "num_training_epochs": int(args.epochs),
        "checkpoint_every_epochs": batch_plan["checkpoint_every_epochs"],
        "num_checkpoint_evaluations": int(math.ceil(args.epochs / batch_plan["checkpoint_every_epochs"])),
        "search_space_hash": stable_json_hash(
            {
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "temporal_layers": args.temporal_layers,
                "feedforward_dim": args.feedforward_dim,
                "dropout": args.dropout,
                "batch_size": batch_plan["batch_size"],
                "micro_batch_size": batch_plan["micro_batch_size"],
                "gradient_accumulation_steps": batch_plan["gradient_accumulation_steps"],
                "eval_batch_size": batch_plan["eval_batch_size"],
                "checkpoint_every_epochs": batch_plan["checkpoint_every_epochs"],
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "reward_scale": args.reward_scale,
                "amp": bool(args.amp),
                "target_vram_gb": args.target_vram_gb,
                "seed": args.seed,
            }
        ),
    }
    candidate_validation_scores: list[dict[str, object]] = []
    training_trace: list[dict[str, float | int | None]] = []
    for epoch in range(args.epochs):
        epoch_stats = train_second_context_epoch(
            train,
            model,
            optimizer,
            scaler,
            device=device,
            batch_size=batch_plan["batch_size"],
            micro_batch_size=batch_plan["micro_batch_size"],
            reward_scale=args.reward_scale,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            grad_clip=args.grad_clip,
            pin_memory=args.pin_memory,
            vram_reservation=reservation,
        )
        epoch_record: dict[str, float | int | None] = {
            "epoch": epoch + 1,
            "average_loss": float(epoch_stats["average_loss"]),
            "optimizer_steps": int(epoch_stats["optimizer_steps"]),
            "micro_batches": int(epoch_stats["micro_batches"]),
            "rows_seen": int(epoch_stats["rows_seen"]),
            "valid_targets_seen": int(epoch_stats["valid_targets_seen"]),
            "validation_checkpoint_score": None,
        }
        should_checkpoint = (epoch + 1) % batch_plan["checkpoint_every_epochs"] == 0 or epoch + 1 == args.epochs
        if should_checkpoint:
            val_policy = evaluate_second_context_trading_policy(
                val,
                model,
                device=device,
                reward_scale=args.reward_scale,
                batch_size=batch_plan["eval_batch_size"],
                use_amp=args.amp,
            amp_dtype=args.amp_dtype,
                pin_memory=args.pin_memory,
            )
            active = val_policy.get("active_window_diagnostics", {})
            active_return = float(active.get("active_net_return", 0.0)) if isinstance(active, dict) else 0.0
            switch_penalty = 0.001 * float(val_policy.get("switch_rate", 0.0) or 0.0)
            checkpoint_score = float(val_policy.get("total_return", 0.0) or 0.0) + active_return - switch_penalty
            epoch_record["validation_checkpoint_score"] = checkpoint_score
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
        training_trace.append(epoch_record)
        log_every = batch_plan["log_every_epochs"] or max(1, args.epochs // 5)
        if (epoch + 1) % log_every == 0 or epoch + 1 == args.epochs:
            val_metrics = evaluate_second_context_action_scorer(
                val,
                model,
                device=device,
                reward_scale=args.reward_scale,
                batch_size=batch_plan["eval_batch_size"],
                use_amp=args.amp,
            amp_dtype=args.amp_dtype,
                pin_memory=args.pin_memory,
            )
            print(
                "epoch="
                f"{epoch + 1} loss={float(epoch_stats['average_loss']):.4f} "
                f"micro_batches={int(epoch_stats['micro_batches'])} "
                f"val_rowwise_total={val_metrics['total_return']:.6f} "
                f"val_checkpoint_score={epoch_record['validation_checkpoint_score']}"
            )
        if args.empty_cache_every_epochs > 0 and device.type == "cuda" and (epoch + 1) % args.empty_cache_every_epochs == 0:
            torch.cuda.empty_cache()
    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate_second_context_action_scorer(
        train,
        model,
        device=device,
        reward_scale=args.reward_scale,
        batch_size=batch_plan["eval_batch_size"],
        use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        pin_memory=args.pin_memory,
    )
    val_metrics = evaluate_second_context_action_scorer(
        val,
        model,
        device=device,
        reward_scale=args.reward_scale,
        batch_size=batch_plan["eval_batch_size"],
        use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        pin_memory=args.pin_memory,
    )
    test_metrics = evaluate_second_context_action_scorer(
        test,
        model,
        device=device,
        reward_scale=args.reward_scale,
        batch_size=batch_plan["eval_batch_size"],
        use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        pin_memory=args.pin_memory,
    )
    train_policy_metrics = evaluate_second_context_trading_policy(
        train,
        model,
        device=device,
        reward_scale=args.reward_scale,
        return_selected_actions=True,
        batch_size=batch_plan["eval_batch_size"],
        use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        pin_memory=args.pin_memory,
    )
    val_policy_metrics = evaluate_second_context_trading_policy(
        val,
        model,
        device=device,
        reward_scale=args.reward_scale,
        return_selected_actions=True,
        batch_size=batch_plan["eval_batch_size"],
        use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        pin_memory=args.pin_memory,
    )
    test_policy_metrics = evaluate_second_context_trading_policy(
        test,
        model,
        device=device,
        reward_scale=args.reward_scale,
        return_decision_logs=True,
        return_selected_actions=True,
        batch_size=batch_plan["eval_batch_size"],
        use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        pin_memory=args.pin_memory,
    )
    run_name = args.run_name or f"second_context_action_scorer_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_logs = list(test_policy_metrics.pop("decision_logs", []))

    def pop_policy_path(metrics: dict[str, object]) -> dict[str, object]:
        legacy_selected = list(metrics.pop("selected_actions", []))
        executed = list(metrics.pop("executed_actions", legacy_selected))
        requested = list(metrics.pop("requested_actions", legacy_selected))
        raw_policy = list(metrics.pop("raw_policy_actions", requested))
        constraint_adjusted = list(metrics.pop("constraint_adjusted_actions", requested))
        reasons = [str(reason) for reason in list(metrics.pop("selection_reasons", []))]
        if len(reasons) != len(executed):
            reasons = [
                "selected_by_policy" if raw == req == exe else "constraint_or_evaluator_adjusted"
                for raw, req, exe in zip(raw_policy, requested, executed)
            ]
        return {
            "executed_actions": torch.tensor(executed, dtype=torch.long),
            "requested_actions": torch.tensor(requested, dtype=torch.long),
            "raw_policy_actions": torch.tensor(raw_policy, dtype=torch.long),
            "constraint_adjusted_actions": torch.tensor(constraint_adjusted, dtype=torch.long),
            "selection_reasons": reasons,
            "rows": torch.tensor(list(metrics.pop("selected_rows", [])), dtype=torch.long),
        }

    train_path = pop_policy_path(train_policy_metrics)
    val_path = pop_policy_path(val_policy_metrics)
    test_path = pop_policy_path(test_policy_metrics)
    train_selected_actions = train_path["executed_actions"]
    val_selected_actions = val_path["executed_actions"]
    test_selected_actions = test_path["executed_actions"]
    train_requested_actions = train_path["requested_actions"]
    val_requested_actions = val_path["requested_actions"]
    test_requested_actions = test_path["requested_actions"]
    train_selected_rows = train_path["rows"]
    val_selected_rows = val_path["rows"]
    test_selected_rows = test_path["rows"]
    if decision_logs:
        with (run_dir / "decision_logs.jsonl").open("w") as sink:
            for row in decision_logs:
                sink.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    torch.save(
        {
            "action_path_schema_version": "selected_action_paths_v2",
            "selected_actions_semantics": "executed_actions_after_missing_label_fallback",
            "requested_actions_semantics": "policy_actions_after_min_hold_and_cooldown_constraints",
            "same_action_weight_policy": "freeze_executed_weight_until_action_change",
            "train_actions": train_selected_actions,
            "train_executed_actions": train_selected_actions,
            "train_requested_actions": train_requested_actions,
            "train_raw_policy_actions": train_path["raw_policy_actions"],
            "train_constraint_adjusted_actions": train_path["constraint_adjusted_actions"],
            "train_selection_reasons": train_path["selection_reasons"],
            "train_rows": train_selected_rows,
            "val_actions": val_selected_actions,
            "val_executed_actions": val_selected_actions,
            "val_requested_actions": val_requested_actions,
            "val_raw_policy_actions": val_path["raw_policy_actions"],
            "val_constraint_adjusted_actions": val_path["constraint_adjusted_actions"],
            "val_selection_reasons": val_path["selection_reasons"],
            "val_rows": val_selected_rows,
            "test_actions": test_selected_actions,
            "test_executed_actions": test_selected_actions,
            "test_requested_actions": test_requested_actions,
            "test_raw_policy_actions": test_path["raw_policy_actions"],
            "test_constraint_adjusted_actions": test_path["constraint_adjusted_actions"],
            "test_selection_reasons": test_path["selection_reasons"],
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
            "training_memory_plan": batch_plan,
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
            p_best_draws=args.confidence_p_best_draws,
            p_best_draw_batch_rows=args.confidence_p_best_draw_batch_rows,
            p_best_draw_batch_size=args.confidence_p_best_draw_batch_size,
            p_best_draw_seed=args.confidence_p_best_draw_seed,
        )
        q_by_split = {
            "train": predict_q_values(train),
            "val": predict_q_values(val),
            "test": predict_q_values(test),
        }
        # Fit the residual calibrator over the SAME mask family used at predict time
        # (decision-valid actions). fit() already filters non-finite (label-missing) returns, so
        # using action_valid_mask here keeps the fitted residual-sigma action set identical to the
        # set the calibrator is applied to, rather than the smaller supervised (decision&label) set.
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
        # When confidence is non-reportable (e.g. the calibration split was reused for checkpoint
        # selection), write the per-row artifacts under non_reportable/ so a downstream consumer
        # cannot silently ingest selection-contaminated confidence as if it were validated.
        confidence_reportable = bool(action_confidence_manifest.get("confidence_reportable", False))
        confidence_dir = run_dir if confidence_reportable else (run_dir / "non_reportable")
        confidence_dir.mkdir(parents=True, exist_ok=True)
        confidence_artifacts["confidence_reportable"] = confidence_reportable
        confidence_artifacts["confidence_artifact_dir"] = str(confidence_dir)
        for split_name, split_obj in (("train", train), ("val", val), ("test", test)):
            path = confidence_dir / f"action_confidence_{split_name}.npz"
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
        selected_by_split = {
            "train": (train, train_path),
            "val": (val, val_path),
            "test": (test, test_path),
        }
        for split_name, (split_obj, selected_path) in selected_by_split.items():
            selected_rows = selected_path["rows"]
            selected_actions = selected_path["executed_actions"]
            selected_confidence_path = confidence_dir / f"selected_action_confidence_{split_name}.jsonl"
            if selected_actions.numel():
                write_selected_action_confidence(
                    path=selected_confidence_path,
                    split=split_obj,
                    confidence=confidence_by_split[split_name],
                    selected_rows=selected_rows,
                    executed_actions=selected_actions,
                    requested_actions=selected_path["requested_actions"],
                    raw_policy_actions=selected_path["raw_policy_actions"],
                    constraint_adjusted_actions=selected_path["constraint_adjusted_actions"],
                    selection_reasons=selected_path["selection_reasons"],
                )
                confidence_artifacts[f"selected_action_confidence_{split_name}_jsonl"] = str(selected_confidence_path)
        confidence_summary = {
            "enabled": True,
            "method": confidence_config.method,
            "calibration_split": "val",
            "uses_checkpoint_selection_for_calibration": True,
            "confidence_reportable": bool(action_confidence_manifest.get("confidence_reportable", False)),
            "confidence_reportability_errors": list(
                action_confidence_manifest.get("confidence_reportability_errors", [])
            ),
            "metrics": dict(calibrator.metrics),
            "warnings": list(action_confidence_manifest.get("warnings", [])),
        }
    payload = torch.load(args.dataset, map_location="cpu", weights_only=True)
    dataset_manifest = dict(payload.get("dataset_manifest", {}))
    data_quality_report = dict(payload.get("data_quality_report", {}))
    split_missing_label_reportability = {
        "train": second_context_missing_label_report(
            train,
            row_indices=train_selected_rows,
            selected_actions=train_requested_actions,
        ),
        "val": second_context_missing_label_report(
            val,
            row_indices=val_selected_rows,
            selected_actions=val_requested_actions,
        ),
        "test": second_context_missing_label_report(
            test,
            row_indices=test_selected_rows,
            selected_actions=test_requested_actions,
        ),
    }
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
        "training_trace": training_trace,
        "training_memory_plan": batch_plan,
        "selected_val_policy_metrics": best_val_policy,
        "action_confidence": action_confidence_manifest,
        "split_manifest": split_manifest,
        "action_names": train.action_names,
        "feature_names": train.feature_names,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    def _as_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        return [str(value)]

    conversion_errors: list[str] = []
    conversion_errors.extend(_as_list(dataset_manifest.get("conversion_reportability_errors")))
    conversion_errors.extend(_as_list(data_quality_report.get("conversion_reportability_errors")))
    if dataset_manifest.get("conversion_reportable") is False:
        conversion_errors.append("conversion_manifest_non_reportable")
    conversion_reportable = not conversion_errors

    dataset_errors: list[str] = []
    dataset_errors.extend(_as_list(dataset_manifest.get("reportability_errors")))
    dataset_errors.extend(_as_list(data_quality_report.get("reportability_errors")))
    if not dataset_manifest.get("reportable", False):
        dataset_errors.append("dataset_non_reportable")
    dataset_reportable = not dataset_errors

    evaluation_errors: list[str] = []
    if test_policy_metrics.get("final_position_open"):
        evaluation_errors.append("test_final_position_open")
    if "RandomSameTurnoverSameTiming" not in test_baselines:
        evaluation_errors.append("missing_random_same_turnover_same_timing_baseline")
    test_missing_label_report = split_missing_label_reportability["test"]
    if not bool(test_missing_label_report.get("evaluation_reportable", True)):
        evaluation_errors.extend(_as_list(test_missing_label_report.get("reportability_errors")))
    evaluation_reportable = not evaluation_errors

    confidence_errors: list[str] = []
    confidence_reportable: bool | None = None
    if args.save_action_confidence:
        confidence_reportable = bool(confidence_summary.get("confidence_reportable", False))
        if not confidence_reportable:
            confidence_errors.extend(_as_list(confidence_summary.get("confidence_reportability_errors")))
            if not confidence_errors:
                confidence_errors.append("confidence_non_reportable")
    reportability_errors = {
        "conversion": list(dict.fromkeys(conversion_errors)),
        "dataset": list(dict.fromkeys(dataset_errors)),
        "evaluation": list(dict.fromkeys(evaluation_errors)),
        "confidence": list(dict.fromkeys(confidence_errors)),
    }
    reportability_reasons = [
        *reportability_errors["conversion"],
        *reportability_errors["dataset"],
        *reportability_errors["evaluation"],
        *reportability_errors["confidence"],
    ]
    reportability = {
        "reportable": conversion_reportable
        and dataset_reportable
        and evaluation_reportable
        and confidence_reportable is not False,
        "conversion_reportable": conversion_reportable,
        "dataset_reportable": dataset_reportable,
        "evaluation_reportable": evaluation_reportable,
        "confidence_reportable": confidence_reportable,
        "reportability_errors": reportability_errors,
        "reasons": reportability_reasons,
        "model_kind": "contextual_action_scorer",
    }
    summary = {
        "dataset": str(args.dataset),
        "device": str(device),
        "rows": {"train": len(train.decision_timestamps), "val": len(val.decision_timestamps), "test": len(test.decision_timestamps)},
        "model_kind": "contextual_action_scorer",
        "training": {
            "batch_plan": batch_plan,
            "epoch_trace": training_trace,
            "amp_enabled": scaler.is_enabled(),
            "pin_memory": bool(args.pin_memory),
            "target_vram_gb": args.target_vram_gb,
            "vram_reservation": reservation.report,
        },
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
            "missing_label_reportability": split_missing_label_reportability,
            "action_confidence": confidence_summary,
        },
        "reportability": reportability,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        cuda_report = {
            "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 4),
            "cuda_peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1024**3, 4),
            "cuda_device_used_end_gb": round((total - free) / 1024**3, 4),
            "cuda_device_free_end_gb": round(free / 1024**3, 4),
        }
        summary["training"].update(cuda_report)
        model_manifest["training_memory_plan"] = {**batch_plan, **cuda_report, "vram_reservation": reservation.report}
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
