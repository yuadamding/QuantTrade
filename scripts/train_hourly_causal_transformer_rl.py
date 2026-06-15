#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal-transformer DQN allocator on top-volume bar data.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "derived" / "rl_hourly" / "top_volume_2026" / "hourly_transformer_dataset.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "derived" / "rl_hourly_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--train-start")
    parser.add_argument("--train-end", default="2026-04-30T23:59:59+00:00")
    parser.add_argument("--val-end", default="2026-05-29T23:59:59+00:00")
    parser.add_argument("--test-start", default="2026-06-01T00:00:00+00:00")
    parser.add_argument("--test-end")
    parser.add_argument("--initial-action", default="CASH")
    parser.add_argument("--switch-cost-bps", type=float, default=1.0)
    parser.add_argument("--min-hold-bars", type=int, default=1)
    parser.add_argument("--cooldown-bars", type=int, default=0)
    parser.add_argument("--max-switches-per-day", type=int)
    parser.add_argument("--max-switches-per-episode", type=int)
    parser.add_argument("--max-order-legs-per-day", type=float)
    parser.add_argument("--max-order-legs-per-episode", type=float)
    parser.add_argument("--q-switch-margin-bps", type=float, default=0.0)
    parser.add_argument("--extra-switch-penalty-bps", type=float, default=0.0)
    parser.add_argument(
        "--cost-stress-bps",
        type=float,
        nargs="*",
        default=[0.0, 1.0, 2.0, 5.0, 10.0],
        help="One-way per-leg cost levels for post-training evaluation.",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episode-length", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.04)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--action-embedding-dim", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--target-vram-gb", type=float, help="Reserve CUDA VRAM after warmup toward this total used amount.")
    parser.add_argument("--vram-safety-gb", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args(argv)


def build_constraints_from_args(args: argparse.Namespace, *, cash_index: int = 0):
    from rl_quant.trading_constraints import TradingConstraintConfig

    return TradingConstraintConfig(
        max_switches_per_day=args.max_switches_per_day,
        max_switches_per_episode=args.max_switches_per_episode,
        max_order_legs_per_day=args.max_order_legs_per_day,
        max_order_legs_per_episode=args.max_order_legs_per_episode,
        min_hold_bars=args.min_hold_bars,
        cooldown_bars=args.cooldown_bars,
        q_switch_margin_bps=args.q_switch_margin_bps,
        extra_switch_penalty_bps=args.extra_switch_penalty_bps,
        one_way_cost_bps=args.switch_cost_bps,
        cash_index=cash_index,
    )


def unique_cost_stresses(values: list[float]) -> list[float]:
    return sorted({float(value) for value in values})


def metric_sharpe(values: list[float], *, periods_per_year: float) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    if variance <= 0:
        return None
    return avg / (variance**0.5) * (periods_per_year**0.5)


def metric_max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 1.0
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def fixed_rollout_cost_stress(
    rollout_records: list[dict[str, object]],
    *,
    cost_bps_values: list[float],
    extra_switch_penalty_bps: float,
    periods_per_year: float,
) -> dict[str, dict[str, float | int | None]]:
    results: dict[str, dict[str, float | int | None]] = {}
    for cost_bps in unique_cost_stresses(cost_bps_values):
        equity = 1.0
        equity_curve = [equity]
        returns: list[float] = []
        switches = 0
        order_legs = 0.0
        for record in rollout_records:
            action = int(record["action"])
            previous_action = int(record["previous_action"])
            legs = float(record["market_order_legs"])
            is_switch = action != previous_action
            gross_return = float(record["gross_return"] if "gross_return" in record else record["bar_return"])
            realized_cost_bps = legs * float(cost_bps) + float(is_switch) * float(extra_switch_penalty_bps)
            net_return = gross_return - realized_cost_bps / 10_000.0
            equity *= 1.0 + net_return
            equity_curve.append(equity)
            returns.append(net_return)
            switches += int(is_switch)
            order_legs += legs
        results[f"{cost_bps:g}bps"] = {
            "total_return": equity - 1.0,
            "total_reward_bps": sum(returns) * 10_000.0,
            "total_switches": switches,
            "market_order_legs": order_legs,
            "max_drawdown": metric_max_drawdown(equity_curve),
            "annualized_sharpe": metric_sharpe(returns, periods_per_year=periods_per_year),
        }
    return results


def equal_weight_metrics(split, *, cash_index: int) -> dict[str, float | int | None]:
    action_indices = [idx for idx in range(len(split.action_names)) if idx != cash_index]
    if not action_indices:
        action_indices = [cash_index]
    equity = 1.0
    equity_curve = [equity]
    returns: list[float] = []
    for index in split.valid_start_indices.detach().cpu().tolist():
        simple_return = float(split.action_returns[index, action_indices].mean().item())
        equity *= 1.0 + simple_return
        equity_curve.append(equity)
        returns.append(simple_return)
    return {
        "total_return": equity - 1.0,
        "max_drawdown": metric_max_drawdown(equity_curve),
        "annualized_sharpe": metric_sharpe(returns, periods_per_year=split.periods_per_year),
        "total_switches": 0,
        "market_order_legs": 0.0,
    }


def write_rollout(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    try:
        import torch

        from rl_quant.hourly_transformer import (
            HourlyEnvConfig,
            HourlyTransformerTrainingConfig,
            action_index,
            assert_matching_hourly_schema,
            build_hourly_splits,
            evaluate_hourly_policy,
            train_hourly_transformer_dqn,
        )
        from rl_quant.core import (
            DQNLearningConfig,
            configure_torch_runtime,
            resolve_torch_device,
            torch_runtime_summary,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Torch is required. Use: conda run -n ml1 python scripts/train_hourly_causal_transformer_rl.py"
            ) from exc
        raise

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_split, val_split, test_split = build_hourly_splits(
        dataset_path=args.dataset,
        lookback=args.lookback,
        train_start=args.train_start,
        train_end=args.train_end,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    assert_matching_hourly_schema(train_split, val_split, test_split)
    initial_action = action_index(train_split.action_names, args.initial_action)
    cash_index = train_split.action_names.index("CASH") if "CASH" in train_split.action_names else initial_action
    constraints = build_constraints_from_args(args, cash_index=cash_index)
    runtime = torch_runtime_summary(device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {runtime['cuda_device_name']} | CUDA: {runtime['cuda_version']}")
        print(f"Total memory: {runtime['cuda_total_memory_gb']} GiB | AMP: {args.amp}")
    print(
        f"Rows train/val/test: {len(train_split.timestamps)}/"
        f"{len(val_split.timestamps)}/{len(test_split.timestamps)}"
    )
    print(f"Bar interval: {train_split.bar_interval} | periods/year: {train_split.periods_per_year:.1f}")
    print(f"Features: {len(train_split.feature_names)} | Actions: {len(train_split.action_names)}")

    env_config = HourlyEnvConfig(
        lookback=args.lookback,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        switch_cost_bps=args.switch_cost_bps,
        initial_action=initial_action,
        constraints=constraints,
    )
    learning_config = DQNLearningConfig(
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        warmup_steps=args.warmup_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        eval_interval=args.eval_interval,
        grad_clip=args.grad_clip,
        use_amp=args.amp,
    )
    config = HourlyTransformerTrainingConfig(
        env=env_config,
        learning=learning_config,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        action_embedding_dim=args.action_embedding_dim,
        target_vram_gb=args.target_vram_gb,
        vram_safety_gb=args.vram_safety_gb,
    )
    model, artifacts = train_hourly_transformer_dqn(
        train_split,
        val_split,
        device=device,
        config=config,
    )
    train_result = evaluate_hourly_policy(
        train_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        switch_cost_bps=args.switch_cost_bps,
        constraints=constraints,
        episode_length=args.episode_length,
    )
    val_result = evaluate_hourly_policy(
        val_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        switch_cost_bps=args.switch_cost_bps,
        constraints=constraints,
        episode_length=args.episode_length,
    )
    test_result = evaluate_hourly_policy(
        test_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        switch_cost_bps=args.switch_cost_bps,
        constraints=constraints,
        episode_length=args.episode_length,
        capture_rollout=True,
    )
    adaptive_cost_stress = {
        f"{cost_bps:g}bps": evaluate_hourly_policy(
            test_split.to(device),
            model,
            device=device,
            initial_action=initial_action,
            switch_cost_bps=cost_bps,
            constraints=replace(constraints, one_way_cost_bps=cost_bps),
            episode_length=args.episode_length,
        ).to_dict()
        for cost_bps in unique_cost_stresses(args.cost_stress_bps)
    }
    fixed_cost_stress = fixed_rollout_cost_stress(
        test_result.rollout_records,
        cost_bps_values=args.cost_stress_bps,
        extra_switch_penalty_bps=args.extra_switch_penalty_bps,
        periods_per_year=test_split.periods_per_year,
    )

    class FixedActionPolicy(torch.nn.Module):
        def __init__(self, *, action_count: int, action: int) -> None:
            super().__init__()
            self.action_count = int(action_count)
            self.action = int(action)

        def forward(self, state_windows, previous_actions, constraint_features=None):
            q_values = torch.zeros((state_windows.shape[0], self.action_count), device=state_windows.device)
            q_values[:, self.action] = 100.0
            return q_values

    def fixed_action_baseline(action_name: str) -> dict[str, object]:
        action = action_index(train_split.action_names, action_name)
        policy = FixedActionPolicy(action_count=len(train_split.action_names), action=action)
        return {
            split.name: evaluate_hourly_policy(
                split,
                policy,
                device=torch.device("cpu"),
                initial_action=initial_action,
                switch_cost_bps=args.switch_cost_bps,
                constraints=constraints,
                episode_length=args.episode_length,
            ).to_dict()
            for split in (train_split, val_split, test_split)
        }

    baselines: dict[str, object] = {"CASH": fixed_action_baseline(train_split.action_names[cash_index])}
    for action_name in ("QQQ", "SPY"):
        if action_name in train_split.action_names:
            baselines[f"BuyAndHold_{action_name}"] = fixed_action_baseline(action_name)
    baselines["EqualWeight_ETFs"] = {
        split.name: equal_weight_metrics(split, cash_index=cash_index)
        for split in (train_split, val_split, test_split)
    }

    run_name = args.run_name or f"{train_split.bar_interval}_causal_transformer_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": train_split.feature_mean.detach().cpu(),
            "feature_std": train_split.feature_std.detach().cpu(),
            "feature_names": train_split.feature_names,
            "action_names": train_split.action_names,
            "bar_interval": train_split.bar_interval,
            "periods_per_year": train_split.periods_per_year,
            "config": serializable_args,
            "constraints": asdict(constraints),
        },
        run_dir / "model.pt",
    )
    write_rollout(run_dir / "test_rollout.csv", test_result.rollout_records)
    summary = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_runtime": runtime,
        "config": serializable_args,
        "constraints": asdict(constraints),
        "bar_interval": train_split.bar_interval,
        "periods_per_year": train_split.periods_per_year,
        "feature_names": train_split.feature_names,
        "action_names": train_split.action_names,
        "training": artifacts,
        "train_metrics": train_result.to_dict(),
        "val_metrics": val_result.to_dict(),
        "test_metrics": test_result.to_dict(),
        "baselines": baselines,
        "cost_stress": {
            "adaptive": adaptive_cost_stress,
            "fixed_rollout": fixed_cost_stress,
        },
        "adaptive_cost_stress": adaptive_cost_stress,
        "fixed_rollout_cost_stress": fixed_cost_stress,
    }
    with (run_dir / "summary.json").open("w") as sink:
        json.dump(summary, sink, indent=2)

    print(
        f"Train TR: {train_result.total_return:.2%} | "
        f"Val TR: {val_result.total_return:.2%} | "
        f"Test TR: {test_result.total_return:.2%}"
    )
    print(
        f"Test switches: {test_result.total_switches} | "
        f"order legs: {test_result.market_order_legs:.1f}"
    )
    if artifacts.get("vram_reservation"):
        print(f"VRAM reservation: {artifacts['vram_reservation']}")
    if "cuda_device_used_end_gb" in artifacts:
        print(
            f"CUDA used end: {artifacts['cuda_device_used_end_gb']} GiB | "
            f"peak reserved: {artifacts['cuda_peak_reserved_gb']} GiB"
        )
    print(f"Artifacts written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
