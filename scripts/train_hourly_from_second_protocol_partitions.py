#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
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


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_PARTITIONS_ROOT = (
    DATA_ROOT
    / "protocol"
    / "polygon_second_top500_2025_to_2026-06-15"
    / "hour_from_second_1s"
    / "partitions"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the hour-from-second transformer across protocol partitions with rolling warm-start fine-tuning."
        )
    )
    parser.add_argument("--partitions-root", type=Path, default=DEFAULT_PARTITIONS_ROOT)
    parser.add_argument("--dataset-file-name", default="hour_from_second_dataset.pt")
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "rl_hour_from_second_runs")
    parser.add_argument("--run-name", default=f"whole_protocol_1s_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--start-partition", help="First partition label to include, inclusive.")
    parser.add_argument("--end-partition", help="Last partition label to include, inclusive.")
    parser.add_argument("--max-partitions", type=int, default=0, help="0 means all matching partitions.")
    parser.add_argument("--initial-action", default="CASH")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=150)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-update-interval", type=int, default=50)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.04)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--minute-layers", type=int, default=2)
    parser.add_argument("--hour-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--action-embedding-dim", type=int, default=32)
    parser.add_argument("--max-subhour-tokens", type=int, default=256)
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    parser.add_argument("--extra-switch-penalty-bps", type=float, default=1.0)
    parser.add_argument("--q-switch-margin-bps", type=float, default=3.0)
    parser.add_argument("--max-switches-per-day", type=int, default=2)
    parser.add_argument("--max-switches-per-episode", type=int, default=3)
    parser.add_argument("--max-order-legs-per-day", type=float)
    parser.add_argument("--max-order-legs-per-episode", type=float, default=6.0)
    parser.add_argument("--min-hold-bars", type=int, default=1)
    parser.add_argument("--cooldown-bars", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--target-vram-gb", type=float)
    parser.add_argument("--vram-safety-gb", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def partition_paths(args: argparse.Namespace) -> list[Path]:
    if not args.partitions_root.exists():
        raise FileNotFoundError(f"Partitions root does not exist: {args.partitions_root}")
    paths = sorted(args.partitions_root.glob(f"*/{args.dataset_file_name}"))
    if args.start_partition:
        paths = [path for path in paths if path.parent.name >= args.start_partition]
    if args.end_partition:
        paths = [path for path in paths if path.parent.name <= args.end_partition]
    if args.max_partitions > 0:
        paths = paths[: args.max_partitions]
    if not paths:
        raise ValueError("No partition datasets matched the requested filters.")
    return paths


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


def split_window(split) -> dict[str, str | int]:
    valid = [int(item) for item in split.valid_start_indices.detach().cpu().tolist()]
    first_valid = valid[0]
    last_valid = valid[-1]
    return {
        "selected_rows": len(split.decision_timestamps),
        "valid_starts": len(valid),
        "first_valid_decision": split.decision_timestamps[first_valid],
        "first_valid_reward_end": split.next_timestamps[first_valid],
        "last_valid_decision": split.decision_timestamps[last_valid],
        "last_valid_reward_end": split.next_timestamps[last_valid],
    }


def iso_timestamp_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(parsed.timestamp() * 1000)


def build_rolling_partition_splits(dataset_path: Path):
    from rl_quant.minute_to_hour_transformer import _build_split, _load_payload

    payload = _load_payload(dataset_path)
    decisions = list(payload["decision_timestamps"])
    next_timestamps = list(payload["next_timestamps"])
    if len(decisions) < 4:
        raise ValueError(
            f"Partition {dataset_path} has too few decision rows for independent train/validation/test splits."
        )
    train_end = decisions[-3]
    train_reward_end = next_timestamps[-3]
    validation_start = decisions[-2]
    validation_reward_end = next_timestamps[-2]
    test_start = decisions[-1]
    test_reward_end = next_timestamps[-1]
    if iso_timestamp_ms(train_reward_end) > iso_timestamp_ms(validation_start):
        raise ValueError("Training reward window overlaps the validation decision window.")
    if iso_timestamp_ms(validation_reward_end) > iso_timestamp_ms(test_start):
        raise ValueError("Validation reward window overlaps the test decision window.")
    train = _build_split(
        name="train",
        payload=payload,
        end_ts=train_end,
        reward_end_ts=train_reward_end,
    )
    val = _build_split(
        name="val",
        payload=payload,
        start_ts=validation_start,
        end_ts=validation_start,
        reward_start_ts=validation_start,
        reward_end_ts=validation_reward_end,
        minute_feature_mean=train.minute_feature_mean,
        minute_feature_std=train.minute_feature_std,
        hour_feature_mean=train.hour_feature_mean,
        hour_feature_std=train.hour_feature_std,
    )
    test = _build_split(
        name="test",
        payload=payload,
        start_ts=test_start,
        end_ts=test_start,
        reward_start_ts=test_start,
        reward_end_ts=test_reward_end,
        minute_feature_mean=train.minute_feature_mean,
        minute_feature_std=train.minute_feature_std,
        hour_feature_mean=train.hour_feature_mean,
        hour_feature_std=train.hour_feature_std,
    )
    return train, val, test


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
            evaluate_minute_to_hour_policy,
            train_minute_to_hour_dqn,
        )
        from rl_quant.trading_constraints import (
            CONSTRAINED_POLICY_MODEL_VERSION,
            CONSTRAINT_FEATURE_NAMES,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("Torch is required. Use: conda run -n ml1 python ...") from exc
        raise

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    paths = partition_paths(args)
    run_dir = args.output_dir / args.run_name
    partition_dir = run_dir / "partitions"
    partition_dir.mkdir(parents=True, exist_ok=True)
    constraints = build_constraints_from_args(args)
    runtime = torch_runtime_summary(device)
    config_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    manifest_path = run_dir / "rolling_summary.json"
    records: list[dict[str, object]] = []
    previous_checkpoint: Path | None = None

    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(
            f"CUDA device: {runtime['cuda_device_name']} | total memory: {runtime['cuda_total_memory_gb']} GiB | AMP: {args.amp}",
            flush=True,
        )
    print(f"Training {len(paths)} partitions from {paths[0].parent.name} to {paths[-1].parent.name}", flush=True)

    for ordinal, dataset_path in enumerate(paths, start=1):
        label = dataset_path.parent.name
        started = datetime.now()
        print(f"[{ordinal}/{len(paths)}] loading {label}", flush=True)
        try:
            train_split, val_split, test_split = build_rolling_partition_splits(dataset_path)
            initial_action = action_index(train_split.action_names, args.initial_action)
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
            train_config = MinuteToHourTrainingConfig(
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
                warm_start_model=previous_checkpoint,
                max_subhour_tokens=args.max_subhour_tokens,
            )
            model, artifacts = train_minute_to_hour_dqn(train_split, val_split, device=device, config=train_config)
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
            current_dir = partition_dir / f"{ordinal:03d}_{label}"
            current_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = current_dir / "model.pt"
            torch.save(
                {
                    "model_version": CONSTRAINED_POLICY_MODEL_VERSION,
                    "uses_constraint_features": True,
                    "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
                    "model_state_dict": model.state_dict(),
                    "minute_feature_mean": train_split.minute_feature_mean.detach().cpu(),
                    "minute_feature_std": train_split.minute_feature_std.detach().cpu(),
                    "hour_feature_mean": train_split.hour_feature_mean.detach().cpu(),
                    "hour_feature_std": train_split.hour_feature_std.detach().cpu(),
                    "minute_feature_names": train_split.minute_feature_names,
                    "hour_feature_names": train_split.hour_feature_names,
                    "action_names": train_split.action_names,
                    "source_bar_interval": train_split.source_bar_interval,
                    "context_bars_per_hour": train_split.effective_context_bars_per_hour,
                    "max_subhour_tokens": args.max_subhour_tokens,
                    "constraints": asdict(constraints),
                    "config": config_payload,
                    "warm_start": artifacts.get("warm_start"),
                    "partition_label": label,
                    "partition_dataset": str(dataset_path),
                },
                checkpoint_path,
            )
            write_rollout(current_dir / "test_rollout.csv", test_result.rollout_records)
            train_window = split_window(train_split)
            val_window = split_window(val_split)
            test_window = split_window(test_split)
            summary = {
                "partition": label,
                "ordinal": ordinal,
                "dataset": str(dataset_path),
                "checkpoint": str(checkpoint_path),
                "rows": {
                    "train": len(train_split.decision_timestamps),
                    "val": len(val_split.decision_timestamps),
                    "test": len(test_split.decision_timestamps),
                    "train_valid_starts": int(train_split.valid_start_indices.numel()),
                    "val_valid_starts": int(val_split.valid_start_indices.numel()),
                    "test_valid_starts": int(test_split.valid_start_indices.numel()),
                },
                "split_windows": {
                    "train": train_window,
                    "val": val_window,
                    "test": test_window,
                    "validation_and_test_are_distinct": (
                        val_window["first_valid_decision"] != test_window["first_valid_decision"]
                    ),
                },
                "training": artifacts,
                "train_metrics": train_result.to_dict(),
                "val_metrics": val_result.to_dict(),
                "test_metrics": test_result.to_dict(),
                "elapsed_seconds": round((datetime.now() - started).total_seconds(), 3),
            }
            with (current_dir / "summary.json").open("w") as sink:
                json.dump(summary, sink, indent=2)
            record = {
                "partition": label,
                "ordinal": ordinal,
                "status": "ok",
                "checkpoint": str(checkpoint_path),
                "train_total_return": train_result.total_return,
                "val_total_return": val_result.total_return,
                "test_total_return": test_result.total_return,
                "test_switches": test_result.allocation_switches,
                "test_order_legs": test_result.market_order_legs,
                "evaluation_reportable": test_result.evaluation_reportable,
                "reportability_errors": test_result.reportability_errors,
                "elapsed_seconds": summary["elapsed_seconds"],
                "cuda_peak_reserved_gb": artifacts.get("cuda_peak_reserved_gb"),
                "cuda_device_used_end_gb": artifacts.get("cuda_device_used_end_gb"),
            }
            records.append(record)
            previous_checkpoint = checkpoint_path
            print(
                f"[{ordinal}/{len(paths)}] ok {label} "
                f"train={train_result.total_return:.2%} val={val_result.total_return:.2%} "
                f"test={test_result.total_return:.2%} elapsed={record['elapsed_seconds']}s",
                flush=True,
            )
        except Exception as exc:
            record = {
                "partition": label,
                "ordinal": ordinal,
                "status": "failed",
                "error": repr(exc),
                "elapsed_seconds": round((datetime.now() - started).total_seconds(), 3),
            }
            records.append(record)
            print(f"[{ordinal}/{len(paths)}] failed {label}: {exc!r}", flush=True)
            raise
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        aggregate = {
            "run_name": args.run_name,
            "created_at": datetime.now().isoformat(),
            "partitions_root": str(args.partitions_root),
            "output_dir": str(run_dir),
            "device": str(device),
            "torch_runtime": runtime,
            "config": config_payload,
            "constraints": asdict(constraints),
            "partition_count": len(paths),
            "completed_count": sum(1 for item in records if item["status"] == "ok"),
            "failed_count": sum(1 for item in records if item["status"] != "ok"),
            "latest_checkpoint": str(previous_checkpoint) if previous_checkpoint is not None else None,
            "records": records,
        }
        with manifest_path.open("w") as sink:
            json.dump(aggregate, sink, indent=2)

    print(f"Rolling training complete. Summary -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
