#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.quote_utils import (  # noqa: E402
    NANOS_PER_MILLISECOND,
    NANOS_PER_SECOND,
    format_bucket_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CUDA-first DQN trading agent on QQQ NBBO features.",
    )
    parser.add_argument("--feature-dir", type=Path, default=PROJECT_ROOT / "derived" / "nbbo_features")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "QQQ_2025")
    parser.add_argument("--bucket-seconds", type=int)
    parser.add_argument("--bucket-ms", type=int)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--step-horizon", type=int, default=5)
    parser.add_argument("--latency-steps", type=int, default=0)
    parser.add_argument("--action-threshold", type=float, default=0.0)
    parser.add_argument("--threshold-grid", default="0,0.25,0.5,1,2,3,5")
    parser.add_argument("--train-dates", required=True)
    parser.add_argument("--val-dates", required=True)
    parser.add_argument("--test-dates", required=True)
    parser.add_argument("--auto-extract", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "derived" / "rl_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=4_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--trade-lot-size", type=int, default=1)
    parser.add_argument("--commission-per-share", type=float, default=0.0)
    parser.add_argument("--extra-cost-per-share", type=float, default=0.0025)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--pretrain-batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision during training")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_bucket_ns(args: argparse.Namespace) -> int:
    if args.bucket_seconds is not None and args.bucket_ms is not None:
        raise SystemExit("Specify only one of --bucket-seconds or --bucket-ms.")
    if args.bucket_seconds is None and args.bucket_ms is None:
        return NANOS_PER_SECOND
    if args.bucket_seconds is not None:
        if args.bucket_seconds <= 0:
            raise SystemExit("--bucket-seconds must be positive")
        return args.bucket_seconds * NANOS_PER_SECOND
    if args.bucket_ms <= 0:
        raise SystemExit("--bucket-ms must be positive")
    return args.bucket_ms * NANOS_PER_MILLISECOND


def ensure_features(
    *,
    dates: list[str],
    raw_dir: Path,
    feature_dir: Path,
    bucket_ns: int,
) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    extractor = PACKAGE_ROOT / "scripts" / "extract_nbbo_features.py"
    bucket_label = format_bucket_label(bucket_ns)

    for date in dates:
        feature_path = feature_dir / f"{date}_nbbo_{bucket_label}.csv"
        if feature_path.exists():
            continue

        raw_file = raw_dir / f"{date}.csv"
        if not raw_file.exists():
            raise FileNotFoundError(f"Missing raw file for requested date: {raw_file}")

        print(f"Extracting features for {date} ...")
        if bucket_ns % NANOS_PER_SECOND == 0:
            bucket_args = ["--bucket-seconds", str(bucket_ns // NANOS_PER_SECOND)]
        elif bucket_ns % NANOS_PER_MILLISECOND == 0:
            bucket_args = ["--bucket-ms", str(bucket_ns // NANOS_PER_MILLISECOND)]
        else:
            raise ValueError(f"Unsupported bucket size for extractor: {bucket_ns} ns")
        subprocess.run(
            [
                sys.executable,
                str(extractor),
                "--input-file",
                str(raw_file),
                "--output-dir",
                str(feature_dir),
                *bucket_args,
            ],
            check=True,
        )


def write_rollout_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    try:
        import torch

        from rl_quant.core import (
            configure_torch_runtime,
            resolve_torch_device,
            torch_runtime_summary,
        )
        from rl_quant.intraday_data import build_splits, parse_date_list
        from rl_quant.intraday_dqn import (
            TrainingConfig,
            evaluate_policy,
            select_action_threshold,
            train_dqn_agent,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Torch is required to train the DQN agent. "
                "Use the ml1 conda environment, for example: conda run -n ml1 python scripts/train_dqn_agent.py"
            ) from exc
        raise

    bucket_ns = resolve_bucket_ns(args)
    if args.lookback <= 0:
        raise SystemExit("--lookback must be positive")
    if args.step_horizon <= 0:
        raise SystemExit("--step-horizon must be positive")
    if args.latency_steps < 0:
        raise SystemExit("--latency-steps must be non-negative")
    if args.latency_steps > args.step_horizon:
        raise SystemExit("--latency-steps cannot exceed --step-horizon in this simulator.")
    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dates = parse_date_list(args.train_dates)
    val_dates = parse_date_list(args.val_dates)
    test_dates = parse_date_list(args.test_dates)
    all_dates = train_dates + val_dates + test_dates

    if args.auto_extract:
        ensure_features(
            dates=all_dates,
            raw_dir=args.raw_dir,
            feature_dir=args.feature_dir,
            bucket_ns=bucket_ns,
        )

    train_split, val_split, test_split = build_splits(
        feature_dir=args.feature_dir,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        lookback=args.lookback,
        bucket_ns=bucket_ns,
    )

    runtime = torch_runtime_summary(device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {runtime['cuda_device_name']}")
        print(f"CUDA version: {runtime['cuda_version']}")
        print(f"TF32 matmul: {runtime['cuda_tf32_matmul']} | AMP: {args.amp}")

    config = TrainingConfig(
        lookback=args.lookback,
        step_horizon=args.step_horizon,
        latency_steps=args.latency_steps,
        action_threshold=args.action_threshold,
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
        trade_lot_size=args.trade_lot_size,
        commission_per_share=args.commission_per_share,
        extra_cost_per_share=args.extra_cost_per_share,
        grad_clip=args.grad_clip,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        use_amp=args.amp,
    )

    train_split = train_split.to(device)
    val_split = val_split.to(device)
    test_split = test_split.to(device)

    model, training_artifacts = train_dqn_agent(
        train_split,
        val_split,
        device=device,
        config=config,
    )

    threshold_candidates = [
        float(item.strip())
        for item in args.threshold_grid.split(",")
        if item.strip()
    ]
    selected_threshold, threshold_search = select_action_threshold(
        val_split,
        model,
        device=device,
        step_horizon=config.step_horizon,
        latency_steps=config.latency_steps,
        trade_lot_size=config.trade_lot_size,
        commission_per_share=config.commission_per_share,
        extra_cost_per_share=config.extra_cost_per_share,
        candidate_thresholds=threshold_candidates,
    )
    print(f"Selected action threshold on validation: {selected_threshold:.4f}")

    train_result = evaluate_policy(
        train_split,
        model,
        device=device,
        step_horizon=config.step_horizon,
        latency_steps=config.latency_steps,
        trade_lot_size=config.trade_lot_size,
        commission_per_share=config.commission_per_share,
        extra_cost_per_share=config.extra_cost_per_share,
        capture_rollout=False,
        action_threshold=selected_threshold,
    )
    val_result = evaluate_policy(
        val_split,
        model,
        device=device,
        step_horizon=config.step_horizon,
        latency_steps=config.latency_steps,
        trade_lot_size=config.trade_lot_size,
        commission_per_share=config.commission_per_share,
        extra_cost_per_share=config.extra_cost_per_share,
        capture_rollout=False,
        action_threshold=selected_threshold,
    )
    test_result = evaluate_policy(
        test_split,
        model,
        device=device,
        step_horizon=config.step_horizon,
        latency_steps=config.latency_steps,
        trade_lot_size=config.trade_lot_size,
        commission_per_share=config.commission_per_share,
        extra_cost_per_share=config.extra_cost_per_share,
        capture_rollout=True,
        action_threshold=selected_threshold,
    )

    run_name = args.run_name or datetime.now().strftime("dqn_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    serializable_args["resolved_bucket_ns"] = bucket_ns
    serializable_args["resolved_bucket_label"] = format_bucket_label(bucket_ns)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": train_split.feature_mean.detach().cpu(),
            "feature_std": train_split.feature_std.detach().cpu(),
            "feature_names": train_split.feature_names,
            "config": serializable_args,
            "selected_action_threshold": selected_threshold,
        },
        run_dir / "model.pt",
    )

    write_rollout_csv(run_dir / "test_rollout.csv", test_result.rollout_records)

    summary = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "torch_runtime": runtime,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "config": serializable_args,
        "threshold_search": threshold_search,
        "selected_action_threshold": selected_threshold,
        "feature_names": train_split.feature_names,
        "training": training_artifacts,
        "train_metrics": train_result.to_dict(),
        "val_metrics": val_result.to_dict(),
        "test_metrics": test_result.to_dict(),
    }
    with (run_dir / "summary.json").open("w") as sink:
        json.dump(summary, sink, indent=2)

    print(
        f"Train PnL: {train_result.total_pnl:.2f} | "
        f"Val PnL: {val_result.total_pnl:.2f} | "
        f"Test PnL: {test_result.total_pnl:.2f}"
    )
    print(
        f"Test trades: {test_result.total_trades} | "
        f"Test win rate: {test_result.win_rate:.2%} | "
        f"Test max drawdown: {test_result.max_drawdown:.2f}"
    )
    print(f"Artifacts written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
