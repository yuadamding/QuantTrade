#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hierarchical minute-to-hour causal-transformer DQN allocator.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "rl_hour_from_minute" / "top_volume_1m_recent" / "hour_from_minute_dataset.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "rl_hour_from_minute_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--train-start")
    parser.add_argument("--train-end", default="2026-06-05T23:59:59+00:00")
    parser.add_argument("--val-end", default="2026-06-10T23:59:59+00:00")
    parser.add_argument("--test-start", default="2026-06-11T00:00:00+00:00")
    parser.add_argument("--test-end", default="2026-06-12T23:59:59+00:00")
    parser.add_argument("--initial-action", default="CASH")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-update-interval", type=int, default=50)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.04)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--minute-layers", type=int, default=2)
    parser.add_argument("--hour-layers", type=int, default=3)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--action-embedding-dim", type=int, default=32)
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    parser.add_argument("--extra-switch-penalty-bps", type=float, default=1.0)
    parser.add_argument("--q-switch-margin-bps", type=float, default=3.0)
    parser.add_argument("--max-switches-per-day", type=int, default=2)
    parser.add_argument("--max-switches-per-episode", type=int)
    parser.add_argument("--max-order-legs-per-day", type=float)
    parser.add_argument("--max-order-legs-per-episode", type=float)
    parser.add_argument("--min-hold-bars", type=int, default=1)
    parser.add_argument("--cooldown-bars", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--target-vram-gb", type=float, help="Reserve CUDA VRAM after warmup toward this total used amount.")
    parser.add_argument("--vram-safety-gb", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def build_constraints_from_args(args: argparse.Namespace):
    from rl_quant.minute_to_hour_transformer import TradingConstraintConfig

    return TradingConstraintConfig(
        max_switches_per_day=args.max_switches_per_day,
        max_switches_per_episode=args.max_switches_per_episode,
        max_order_legs_per_day=args.max_order_legs_per_day,
        max_order_legs_per_episode=args.max_order_legs_per_episode,
        min_hold_bars=args.min_hold_bars,
        cooldown_bars=args.cooldown_bars,
        q_switch_margin_bps=args.q_switch_margin_bps,
        extra_switch_penalty_bps=args.extra_switch_penalty_bps,
        one_way_cost_bps=args.one_way_cost_bps,
    )


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

        from rl_quant.core import (
            DQNLearningConfig,
            configure_torch_runtime,
            resolve_torch_device,
            torch_runtime_summary,
        )
        from rl_quant.minute_to_hour_transformer import (
            MinuteToHourEnvConfig,
            MinuteToHourTrainingConfig,
            action_index,
            build_hour_from_minute_splits,
            evaluate_minute_to_hour_policy,
            train_minute_to_hour_dqn,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Torch is required. Use: conda run -n ml1 python scripts/train_hourly_from_minute_context_rl.py"
            ) from exc
        raise

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_split, val_split, test_split = build_hour_from_minute_splits(
        dataset_path=args.dataset,
        train_start=args.train_start,
        train_end=args.train_end,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    initial_action = action_index(train_split.action_names, args.initial_action)
    runtime = torch_runtime_summary(device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {runtime['cuda_device_name']} | CUDA: {runtime['cuda_version']}")
        print(f"Total memory: {runtime['cuda_total_memory_gb']} GiB | AMP: {args.amp}")
    print(
        f"Rows train/val/test: {len(train_split.decision_timestamps)}/"
        f"{len(val_split.decision_timestamps)}/{len(test_split.decision_timestamps)}"
    )
    print(
        f"Minute tensor: {tuple(train_split.minute_features.shape[1:])} | "
        f"Hour tensor: {tuple(train_split.hour_features.shape[1:])} | "
        f"Actions: {len(train_split.action_names)}"
    )

    constraints = build_constraints_from_args(args)
    env_config = MinuteToHourEnvConfig(
        num_envs=args.num_envs,
        episode_length=args.episode_length,
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
    config = MinuteToHourTrainingConfig(
        env=env_config,
        learning=learning_config,
        d_model=args.d_model,
        n_heads=args.n_heads,
        minute_layers=args.minute_layers,
        hour_layers=args.hour_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        action_embedding_dim=args.action_embedding_dim,
        target_vram_gb=args.target_vram_gb,
        vram_safety_gb=args.vram_safety_gb,
    )
    model, artifacts = train_minute_to_hour_dqn(train_split, val_split, device=device, config=config)
    train_result = evaluate_minute_to_hour_policy(
        train_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        constraints=constraints,
        episode_length=args.episode_length,
    )
    val_result = evaluate_minute_to_hour_policy(
        val_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        constraints=constraints,
        episode_length=args.episode_length,
    )
    test_result = evaluate_minute_to_hour_policy(
        test_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        constraints=constraints,
        episode_length=args.episode_length,
        capture_rollout=True,
    )

    run_name = args.run_name or f"minute_to_hour_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "minute_feature_mean": train_split.minute_feature_mean.detach().cpu(),
            "minute_feature_std": train_split.minute_feature_std.detach().cpu(),
            "hour_feature_mean": train_split.hour_feature_mean.detach().cpu(),
            "hour_feature_std": train_split.hour_feature_std.detach().cpu(),
            "minute_feature_names": train_split.minute_feature_names,
            "hour_feature_names": train_split.hour_feature_names,
            "action_names": train_split.action_names,
            "constraints": asdict(constraints),
            "config": serializable_args,
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
        "minute_feature_names": train_split.minute_feature_names,
        "hour_feature_names": train_split.hour_feature_names,
        "action_names": train_split.action_names,
        "training": artifacts,
        "train_metrics": train_result.to_dict(),
        "val_metrics": val_result.to_dict(),
        "test_metrics": test_result.to_dict(),
    }
    with (run_dir / "summary.json").open("w") as sink:
        json.dump(summary, sink, indent=2)

    print(
        f"Train TR: {train_result.total_return:.2%} | "
        f"Val TR: {val_result.total_return:.2%} | "
        f"Test TR: {test_result.total_return:.2%}"
    )
    print(
        f"Test switches: {test_result.allocation_switches} | "
        f"market order legs: {test_result.market_order_legs:.0f}"
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
