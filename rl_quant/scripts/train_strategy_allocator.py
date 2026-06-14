#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def default_state_features_path() -> Path:
    ecology = PROJECT_ROOT / "derived" / "rl_daily" / "stock_top1000_2026" / "state_features_with_market_ecology.csv"
    if ecology.exists():
        return ecology
    return PROJECT_ROOT / "derived" / "rl_daily" / "stock_top1000_2026" / "state_features.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN strategy allocator on daily strategy returns.",
    )
    parser.add_argument("--state-features", type=Path, default=default_state_features_path())
    parser.add_argument(
        "--action-returns",
        type=Path,
        default=PROJECT_ROOT / "derived" / "rl_daily" / "stock_top1000_2026" / "action_returns.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "derived" / "rl_daily_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--train-start")
    parser.add_argument("--train-end", default="2026-04-30")
    parser.add_argument("--val-end", default="2026-05-29")
    parser.add_argument("--test-start", default="2026-06-01")
    parser.add_argument("--test-end")
    parser.add_argument("--initial-action", default="BH_QQQ")
    parser.add_argument("--switch-cost-bps", type=float, default=5.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--action-embedding-dim", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision during training")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


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
        from rl_quant.strategy_data import action_index, build_strategy_splits
        from rl_quant.strategy_dqn import (
            StrategyEnvConfig,
            StrategyTrainingConfig,
            evaluate_strategy_policy,
            train_strategy_dqn_agent,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Torch is required to train the strategy allocator. "
                "Use the ml1 conda environment, for example: conda run -n ml1 python scripts/train_strategy_allocator.py"
            ) from exc
        raise

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_split, val_split, test_split = build_strategy_splits(
        state_features_path=args.state_features,
        action_returns_path=args.action_returns,
        lookback=args.lookback,
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
        print(f"TF32 matmul: {runtime['cuda_tf32_matmul']} | AMP: {args.amp}")
    print(f"Actions: {len(train_split.action_names)} | Features: {len(train_split.feature_names)}")

    env_config = StrategyEnvConfig(
        lookback=args.lookback,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        switch_cost_bps=args.switch_cost_bps,
        initial_action=initial_action,
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
    config = StrategyTrainingConfig(
        env=env_config,
        learning=learning_config,
        hidden_size=args.hidden_size,
        action_embedding_dim=args.action_embedding_dim,
    )
    model, artifacts = train_strategy_dqn_agent(
        train_split,
        val_split,
        device=device,
        config=config,
    )
    train_result = evaluate_strategy_policy(
        train_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        switch_cost_bps=args.switch_cost_bps,
    )
    val_result = evaluate_strategy_policy(
        val_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        switch_cost_bps=args.switch_cost_bps,
    )
    test_result = evaluate_strategy_policy(
        test_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        switch_cost_bps=args.switch_cost_bps,
        capture_rollout=True,
    )

    run_name = args.run_name or datetime.now().strftime("strategy_dqn_%Y%m%d_%H%M%S")
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
        "feature_names": train_split.feature_names,
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
    print(f"Artifacts written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
