#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    return parser.parse_args(argv)


def main() -> int:
    import torch

    from rl_quant.core import configure_torch_runtime, resolve_torch_device
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
                    reward_scale=args.reward_scale,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            val_metrics = evaluate_second_context_action_scorer(val, model, device=device, reward_scale=args.reward_scale)
            avg_loss = sum(losses) / max(len(losses), 1)
            print(f"epoch={epoch + 1} loss={avg_loss:.4f} val_total={val_metrics['total_return']:.6f}")
    train_metrics = evaluate_second_context_action_scorer(train, model, device=device, reward_scale=args.reward_scale)
    val_metrics = evaluate_second_context_action_scorer(val, model, device=device, reward_scale=args.reward_scale)
    test_metrics = evaluate_second_context_action_scorer(test, model, device=device, reward_scale=args.reward_scale)
    train_policy_metrics = evaluate_second_context_trading_policy(train, model, device=device, reward_scale=args.reward_scale)
    val_policy_metrics = evaluate_second_context_trading_policy(val, model, device=device, reward_scale=args.reward_scale)
    test_policy_metrics = evaluate_second_context_trading_policy(
        test,
        model,
        device=device,
        reward_scale=args.reward_scale,
        return_decision_logs=True,
    )
    run_name = args.run_name or f"second_context_action_scorer_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_logs = list(test_policy_metrics.pop("decision_logs", []))
    selected_actions = torch.tensor(
        [test.action_names.index(str(row["selected_action"])) for row in decision_logs],
        dtype=torch.long,
    )
    if decision_logs:
        with (run_dir / "decision_logs.jsonl").open("w") as sink:
            for row in decision_logs:
                sink.write(json.dumps(row, sort_keys=True, default=str) + "\n")
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
        test,
        reference_actions=selected_actions if selected_actions.numel() else None,
        seed=args.seed,
    )
    cost_stress = fixed_rollout_cost_stress(test, selected_actions) if selected_actions.numel() else {}
    summary = {
        "dataset": str(args.dataset),
        "device": str(device),
        "rows": {"train": len(train.decision_timestamps), "val": len(val.decision_timestamps), "test": len(test.decision_timestamps)},
        "model_kind": "contextual_action_scorer",
        "artifacts": {"decision_logs_jsonl": str(run_dir / "decision_logs.jsonl") if decision_logs else None},
        "metrics": {
            "action_scorer_rowwise": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
            "sequential_policy_switch_cost": {
                "train": train_policy_metrics,
                "val": val_policy_metrics,
                "test": test_policy_metrics,
            },
            "baselines": baselines,
            "fixed_rollout_cost_stress": cost_stress,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True, default=str))
    print(f"Run -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
