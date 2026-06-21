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

# Shared latest-period partition protocol (stdlib-only module, safe to import at module scope once SRC
# is on sys.path). Kept under the script's historical private names so existing callers/tests are stable.
from rl_quant.partition_protocol import (  # noqa: E402
    chronological_latest_label as _chronological_latest_label,
    strict_latest_partition_violations,
)


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
    parser.add_argument(
        "--partition-selection",
        choices=["latest", "earliest"],
        default="latest",
        help="When --max-partitions is set, choose latest partitions by default; earliest is diagnostic only.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["latest_holdout", "latest_rows_smoke"],
        default="latest_holdout",
        help="latest_holdout uses latest complete sessions inside each partition; latest_rows_smoke is diagnostic.",
    )
    parser.add_argument("--test-sessions", type=int, default=1)
    parser.add_argument("--val-sessions", type=int, default=1)
    parser.add_argument("--embargo-sessions", type=int, default=0)
    parser.add_argument("--min-train-sessions", type=int, default=1)
    parser.add_argument("--test-rows", type=int, default=1)
    parser.add_argument("--val-rows", type=int, default=1)
    parser.add_argument("--min-train-rows", type=int, default=2)
    parser.add_argument(
        "--insufficient-split-policy",
        choices=["smoke_fallback", "fail"],
        default="fail",
        help=(
            "For short partitions, either fail (default, research-safe) or fall back to explicit "
            "non-reportable row splits. Use --smoke (or set smoke_fallback) for diagnostic runs."
        ),
    )
    parser.add_argument(
        "--reportability-policy",
        choices=["strict", "diagnostic"],
        default="strict",
        help="strict fails a partition whose splits/evaluation are non-reportable; diagnostic records and continues.",
    )
    parser.add_argument(
        "--allow-truncated-training-history",
        action="store_true",
        help=(
            "Permit a strict latest-period run whose earliest selected partition is not the earliest "
            "available one (i.e. prior history is excluded). Off by default: the stated protocol is "
            "latest periods for test, ALL earlier periods for train/validation."
        ),
    )
    parser.add_argument(
        "--skip-failed-partitions",
        action="store_true",
        help=(
            "Record a partition that fails to build/train/evaluate as status='failed' and CONTINUE to the next "
            "partition (warm-starting from the last GOOD checkpoint), instead of aborting the whole run. Off by "
            "default (fail-loud): a single malformed partition raises. Opt in for a long full-history run so a "
            "few bad partitions (e.g. a gold partition violating a mask contract) do not derail the protocol -- "
            "the failures stay visible in failed_count/records, and corrupt data is skipped, never trained on."
        ),
    )
    parser.add_argument(
        "--recency-weighting",
        choices=["none", "exponential"],
        default="none",
        help=(
            "Down-weight older TRAINING rows toward the recent pre-validation regime. Default none "
            "(uniform). Weights are anchored to the validation start; the held-out test block is "
            "never weighted or referenced by training."
        ),
    )
    parser.add_argument(
        "--recency-half-life-days",
        type=float,
        default=120.0,
        help="Half-life (calendar days) for exponential recency weighting; smaller = stronger recency focus.",
    )
    parser.add_argument(
        "--recency-min-weight",
        type=float,
        default=0.05,
        help="Lower bound on the recency weight so old regimes are down-weighted but never fully ignored.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Diagnostic convenience: enable smoke_fallback splits and diagnostic reportability policy.",
    )
    parser.add_argument(
        "--action-covariate-sidecar",
        choices=["auto", "required", "none"],
        default="auto",
        help="Control whether neighboring action_covariates.pt sidecars are loaded.",
    )
    parser.add_argument(
        "--news-llm-sidecar",
        choices=["auto", "required", "none"],
        default="none",
        help="Opt into neighboring action_news_llm_covariates.pt sidecars.",
    )
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
    parser.add_argument(
        "--cash-idle-penalty-bps",
        type=float,
        default=0.0,
        help="Training-only reward penalty for choosing CASH; evaluation P&L remains unpenalized.",
    )
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
    parser.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="AMP autocast precision when --amp is set. bf16 has a wider exponent range and is "
        "preferred on Ampere/Hopper GPUs; fp16 (default) preserves prior behavior.",
    )
    parser.add_argument(
        "--target-vram-gb",
        type=float,
        help="OPT-IN VRAM ballast: reserve byte tensors to raise used VRAM toward this target after "
        "warmup. This INCREASES memory (it does not cap/shard/offload) -- leave unset for large "
        "models; use --min-free-vram-gb to guard headroom instead.",
    )
    parser.add_argument("--vram-safety-gb", type=float, default=0.50)
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=0.0,
        help="Fail fast before training if free CUDA memory is below this many GiB (0 disables).",
    )
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
        if args.partition_selection == "latest":
            paths = paths[-args.max_partitions :]
        elif args.partition_selection == "earliest":
            paths = paths[: args.max_partitions]
        else:
            raise ValueError(f"Unsupported partition selection: {args.partition_selection!r}")
    if not paths:
        raise ValueError("No partition datasets matched the requested filters.")
    return paths


def partition_selection_reportability_errors(
    args: argparse.Namespace,
    *,
    selected_labels: list[str] | None = None,
    all_available_labels: list[str] | None = None,
) -> list[str]:
    if all_available_labels is None:
        all_available_labels = [
            path.parent.name for path in sorted(args.partitions_root.glob(f"*/{args.dataset_file_name}"))
        ]
    if selected_labels is None:
        selected_labels = [path.parent.name for path in partition_paths(args)]
    return strict_latest_partition_violations(
        selected_labels=selected_labels,
        all_available_labels=all_available_labels,
        allow_truncated_training_history=bool(getattr(args, "allow_truncated_training_history", False)),
    )


def latest_available_partition_label(args: argparse.Namespace) -> str | None:
    """Newest partition present on disk, ignoring --start/--end/--max selection filters."""
    labels = [path.parent.name for path in args.partitions_root.glob(f"*/{args.dataset_file_name}")]
    return _chronological_latest_label(labels)


def build_training_time_policy(args: argparse.Namespace, final_test_is_latest_available: bool) -> dict[str, object]:
    """Declared recency / latest-period training-time policy recorded for reproducibility.

    The critical attestation is ``test_used_for_recency_selection: False``: recency weighting is
    anchored to the validation start and the test block is never passed to the trainer, so it cannot
    influence training or model selection. Recency hyperparameters here are fixed CLI arguments; a
    future validation-based recency-profile sweep would set ``recency_hyperparameters_selected_on``
    to ``validation``.
    """
    return {
        "evaluation_design": "per_partition_walkforward_latest_holdout",
        "test_is_latest_period": bool(final_test_is_latest_available),
        "recency_weighting": args.recency_weighting,
        "recency_half_life_days": float(args.recency_half_life_days),
        "recency_min_weight": float(args.recency_min_weight),
        "checkpoint_selection": "best_validation_return_then_fewer_order_legs",
        "cash_hurdle_bps": float(args.q_switch_margin_bps),
        "cash_hurdle_selected_on": "fixed_cli_argument",
        "recency_hyperparameters_selected_on": "fixed_cli_argument",
        "test_used_for_recency_selection": False,
    }


def official_test_block(records: list[dict[str, object]], final_test_is_latest_available: bool) -> dict[str, object] | None:
    """First-class official latest-period test result so consumers need not scan partition records."""
    official = next((item for item in records if item.get("is_official_latest_test")), None)
    if official is None:
        return None
    return {
        "partition": official.get("partition"),
        "ordinal": official.get("ordinal"),
        "is_latest_available": bool(final_test_is_latest_available),
        "status": official.get("status"),
        "reportable": bool(official.get("evaluation_reportable", False)),
        "reportability_errors": list(official.get("reportability_errors", [])),
        "test_total_return": official.get("test_total_return"),
        "test_total_reward_bps": official.get("test_total_reward_bps"),
        "test_max_drawdown": official.get("test_max_drawdown"),
        "test_annualized_sharpe": official.get("test_annualized_sharpe"),
        "test_switches": official.get("test_switches"),
        "test_order_legs": official.get("test_order_legs"),
        "val_total_return": official.get("val_total_return"),
        "checkpoint_selected_on": "validation",
        "test_used_for_selection": False,
    }


def split_policy_with_partition_selection(
    split_policy: dict[str, object],
    args: argparse.Namespace,
    *,
    selection_errors: list[str] | None = None,
) -> dict[str, object]:
    if selection_errors is None:
        selection_errors = partition_selection_reportability_errors(args)
    errors = [*list(split_policy.get("reportability_errors", [])), *selection_errors]
    out = dict(split_policy)
    out["partition_selection"] = args.partition_selection
    out["partition_selection_reportability_errors"] = selection_errors
    out["reportability_errors"] = list(dict.fromkeys(errors))
    out["reportable"] = bool(split_policy.get("reportable", True)) and not out["reportability_errors"]
    return out


def combined_evaluation_reportability(
    *,
    evaluator_reportable: bool,
    evaluator_errors: list[str],
    split_policy: dict[str, object],
    args: argparse.Namespace,
    selection_errors: list[str] | None = None,
) -> tuple[bool, list[str]]:
    partition_errors = (
        partition_selection_reportability_errors(args)
        if selection_errors is None
        else list(selection_errors)
    )
    split_errors = [str(error) for error in split_policy.get("reportability_errors", [])]
    errors = list(dict.fromkeys([*evaluator_errors, *split_errors, *partition_errors]))
    reportable = bool(evaluator_reportable) and bool(split_policy.get("reportable", True)) and not errors
    return reportable, errors


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


def build_rolling_partition_splits(
    dataset_path: Path,
    *,
    split_mode: str = "latest_holdout",
    val_sessions: int = 1,
    test_sessions: int = 1,
    embargo_sessions: int = 0,
    min_train_sessions: int = 1,
    val_rows: int = 1,
    test_rows: int = 1,
    min_train_rows: int = 2,
    insufficient_split_policy: str = "smoke_fallback",
    action_covariate_sidecar: str = "auto",
    news_llm_sidecar: str = "none",
):
    from rl_quant.minute_to_hour_transformer import (
        _build_split,
        _load_payload,
        infer_latest_holdout_split_policy,
        infer_latest_rows_smoke_split_policy,
    )

    payload = _load_payload(
        dataset_path,
        action_covariate_sidecar=action_covariate_sidecar,
        news_llm_sidecar=news_llm_sidecar,
    )
    decisions = list(payload["decision_timestamps"])
    if len(decisions) < 4:
        raise ValueError(
            f"Partition {dataset_path} has too few decision rows for independent train/validation/test splits."
        )
    if split_mode == "latest_holdout":
        try:
            split_policy = infer_latest_holdout_split_policy(
                payload,
                val_sessions=val_sessions,
                test_sessions=test_sessions,
                embargo_sessions=embargo_sessions,
                min_train_sessions=min_train_sessions,
            )
        except ValueError as exc:
            if insufficient_split_policy != "smoke_fallback":
                raise
            split_policy = infer_latest_rows_smoke_split_policy(
                payload,
                val_rows=val_rows,
                test_rows=test_rows,
                min_train_rows=min_train_rows,
            )
            fallback_errors = list(split_policy.get("reportability_errors", []))
            fallback_errors.extend(
                [
                    "latest_holdout_insufficient_complete_sessions",
                    "fallback_to_latest_rows_smoke_split",
                ]
            )
            split_policy = dict(split_policy)
            split_policy["requested_split_mode"] = "latest_holdout"
            split_policy["split_mode"] = "latest_rows_smoke_fallback"
            split_policy["fallback_from_split_mode"] = "latest_holdout"
            split_policy["fallback_reason"] = str(exc)
            split_policy["reportable"] = False
            split_policy["reportability_errors"] = list(dict.fromkeys(fallback_errors))
    elif split_mode == "latest_rows_smoke":
        split_policy = infer_latest_rows_smoke_split_policy(
            payload,
            val_rows=val_rows,
            test_rows=test_rows,
            min_train_rows=min_train_rows,
        )
    else:
        raise ValueError(f"Unsupported split_mode {split_mode!r}.")
    blocks = split_policy["blocks"]
    train_block = dict(blocks["train"])
    val_block = dict(blocks["val"])
    test_block = dict(blocks["test"])
    train = _build_split(
        name="train",
        payload=payload,
        start_ts=str(train_block["start"]),
        end_ts=str(train_block["end"]),
        reward_end_ts=str(train_block["reward_end"]),
        split_policy=split_policy,
    )
    val = _build_split(
        name="val",
        payload=payload,
        start_ts=str(val_block["start"]),
        end_ts=str(val_block["end"]),
        reward_start_ts=str(val_block["reward_start"]),
        reward_end_ts=str(val_block["reward_end"]),
        minute_feature_mean=train.minute_feature_mean,
        minute_feature_std=train.minute_feature_std,
        hour_feature_mean=train.hour_feature_mean,
        hour_feature_std=train.hour_feature_std,
        action_feature_mean=train.action_feature_mean,
        action_feature_std=train.action_feature_std,
        split_policy=split_policy,
    )
    test = _build_split(
        name="test",
        payload=payload,
        start_ts=str(test_block["start"]),
        end_ts=str(test_block["end"]),
        reward_start_ts=str(test_block["reward_start"]),
        reward_end_ts=str(test_block["reward_end"]),
        minute_feature_mean=train.minute_feature_mean,
        minute_feature_std=train.minute_feature_std,
        hour_feature_mean=train.hour_feature_mean,
        hour_feature_std=train.hour_feature_std,
        action_feature_mean=train.action_feature_mean,
        action_feature_std=train.action_feature_std,
        split_policy=split_policy,
    )
    return train, val, test


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        # --smoke is an explicit diagnostic switch: permit short-partition fallback splits and
        # record (rather than fail on) non-reportable partitions.
        args.insufficient_split_policy = "smoke_fallback"
        args.reportability_policy = "diagnostic"
    try:
        import torch

        from rl_quant.core import (
            DQNLearningConfig,
            configure_torch_runtime,
            require_min_free_vram,
            resolve_torch_device,
            torch_runtime_summary,
        )
        from rl_quant.minute_to_hour_transformer import (
            MinuteToHourEnvConfig,
            MinuteToHourTrainingConfig,
            RecencyWeightConfig,
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
    require_min_free_vram(device, args.min_free_vram_gb)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    paths = partition_paths(args)
    all_available_labels = [p.parent.name for p in sorted(args.partitions_root.glob(f"*/{args.dataset_file_name}"))]
    selected_labels = [p.parent.name for p in paths]
    partition_selection_errors = partition_selection_reportability_errors(
        args,
        selected_labels=selected_labels,
        all_available_labels=all_available_labels,
    )
    # The headline (official) test is the LATEST selected partition; earlier partitions are
    # walk-forward diagnostics, not the latest-period KPI. Under strict reportability the final
    # selected partition must be the latest available one, so --end-partition / earliest selection
    # cannot silently make an OLD period the reported test.
    latest_available_label = _chronological_latest_label(all_available_labels)
    official_test_label = selected_labels[-1]
    final_test_is_latest_available = official_test_label == latest_available_label
    # How many earlier available partitions were excluded from the selected train/validation history.
    excluded_prior_partition_count = (
        all_available_labels.index(selected_labels[0]) if selected_labels[0] in all_available_labels else 0
    )
    training_history_truncated = excluded_prior_partition_count > 0
    all_prior_partitions_included = not training_history_truncated
    if args.reportability_policy == "strict":
        if partition_selection_errors:
            raise SystemExit(
                "strict latest-period evaluation refused the partition selection:\n  - "
                + "\n  - ".join(partition_selection_errors)
                + "\nAdjust --start-partition/--end-partition/--partition-selection/--max-partitions, "
                "pass --allow-truncated-training-history where appropriate, or use "
                "--reportability-policy diagnostic for an explicitly non-latest/non-exhaustive run."
            )
    training_time_policy = build_training_time_policy(args, final_test_is_latest_available)
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
            train_split, val_split, test_split = build_rolling_partition_splits(
                dataset_path,
                split_mode=args.split_mode,
                val_sessions=args.val_sessions,
                test_sessions=args.test_sessions,
                embargo_sessions=args.embargo_sessions,
                min_train_sessions=args.min_train_sessions,
                val_rows=args.val_rows,
                test_rows=args.test_rows,
                min_train_rows=args.min_train_rows,
                insufficient_split_policy=args.insufficient_split_policy,
                action_covariate_sidecar=args.action_covariate_sidecar,
                news_llm_sidecar=args.news_llm_sidecar,
            )
            run_split_policy = split_policy_with_partition_selection(
                train_split.split_policy,
                args,
                selection_errors=partition_selection_errors,
            )
            initial_action = action_index(train_split.action_names, args.initial_action)
            env_config = MinuteToHourEnvConfig(
                num_envs=args.num_envs,
                episode_length=args.episode_length,
                initial_action=initial_action,
                cash_idle_penalty_bps=args.cash_idle_penalty_bps,
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
                amp_dtype=args.amp_dtype,
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
                recency=RecencyWeightConfig(
                    mode=args.recency_weighting,
                    half_life_days=args.recency_half_life_days,
                    min_weight=args.recency_min_weight,
                ),
            )
            model, artifacts = train_minute_to_hour_dqn(train_split, val_split, device=device, config=train_config)
            train_result = evaluate_minute_to_hour_policy(
                train_split.to(device),
                model,
                device=device,
                initial_action=initial_action,
                constraints=constraints,
                episode_length=args.episode_length,
                cash_idle_penalty_bps=args.cash_idle_penalty_bps,
            )
            val_result = evaluate_minute_to_hour_policy(
                val_split.to(device),
                model,
                device=device,
                initial_action=initial_action,
                constraints=constraints,
                episode_length=args.episode_length,
                cash_idle_penalty_bps=args.cash_idle_penalty_bps,
            )
            test_result = evaluate_minute_to_hour_policy(
                test_split.to(device),
                model,
                device=device,
                initial_action=initial_action,
                constraints=constraints,
                episode_length=args.episode_length,
                cash_idle_penalty_bps=args.cash_idle_penalty_bps,
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
                    "action_feature_mean": (
                        train_split.action_feature_mean.detach().cpu()
                        if train_split.action_feature_mean is not None
                        else None
                    ),
                    "action_feature_std": (
                        train_split.action_feature_std.detach().cpu()
                        if train_split.action_feature_std is not None
                        else None
                    ),
                    "minute_feature_names": train_split.minute_feature_names,
                    "hour_feature_names": train_split.hour_feature_names,
                    "action_feature_names": train_split.action_feature_names,
                    "action_feature_groups": train_split.action_feature_groups,
                    "action_names": train_split.action_names,
                    "source_bar_interval": train_split.source_bar_interval,
                    "context_bars_per_hour": train_split.effective_context_bars_per_hour,
                    "max_subhour_tokens": args.max_subhour_tokens,
                    "split_policy": run_split_policy,
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
                "split_policy": run_split_policy,
                "training_time_policy": training_time_policy,
                "training": artifacts,
                "train_metrics": train_result.to_dict(),
                "val_metrics": val_result.to_dict(),
                "test_metrics": test_result.to_dict(),
                "elapsed_seconds": round((datetime.now() - started).total_seconds(), 3),
            }
            with (current_dir / "summary.json").open("w") as sink:
                json.dump(summary, sink, indent=2)
            evaluation_reportable, reportability_errors = combined_evaluation_reportability(
                evaluator_reportable=bool(test_result.evaluation_reportable),
                evaluator_errors=list(test_result.reportability_errors),
                split_policy=run_split_policy,
                args=args,
                selection_errors=partition_selection_errors,
            )
            split_reportable = bool(run_split_policy.get("reportable", True))
            if args.reportability_policy == "strict" and not (evaluation_reportable and split_reportable):
                # Fail-fast under strict: do not warm-start the next partition from, or report, a
                # non-reportable result. (--smoke / --reportability-policy diagnostic records instead.)
                strict_errors = list(reportability_errors) + list(run_split_policy.get("reportability_errors", []))
                raise RuntimeError(
                    f"partition {label} is non-reportable under strict reportability policy: "
                    + "; ".join(str(error) for error in strict_errors[:20])
                )
            record = {
                "partition": label,
                "ordinal": ordinal,
                "status": "ok",
                # Only the latest selected partition's test is the official latest-period KPI;
                # earlier partitions are walk-forward diagnostics.
                "is_official_latest_test": bool(label == official_test_label),
                "checkpoint": str(checkpoint_path),
                "train_total_return": train_result.total_return,
                "val_total_return": val_result.total_return,
                "test_total_return": test_result.total_return,
                "test_total_reward_bps": test_result.total_reward_bps,
                "test_max_drawdown": test_result.max_drawdown,
                "test_annualized_sharpe": test_result.annualized_sharpe,
                "test_switches": test_result.allocation_switches,
                "test_order_legs": test_result.market_order_legs,
                "evaluation_reportable": evaluation_reportable,
                "reportability_errors": reportability_errors,
                "split_mode": run_split_policy.get("split_mode"),
                "split_reportable": run_split_policy.get("reportable"),
                "split_reportability_errors": run_split_policy.get("reportability_errors", []),
                "partition_selection": args.partition_selection,
                "partition_selection_reportability_errors": partition_selection_errors,
                "action_features_present": train_split.action_features is not None,
                "action_feature_dim": artifacts.get("action_feature_dim"),
                "action_feature_names": artifacts.get("action_feature_names", []),
                "action_feature_groups": artifacts.get("action_feature_groups", {}),
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
            # Default: fail loud (re-raise). With --skip-failed-partitions, record the failure and continue to
            # the next partition -- previous_checkpoint is NOT updated, so the next partition warm-starts from the
            # last GOOD model and the corrupt partition is skipped (never trained on), staying visible in records.
            if not args.skip_failed_partitions:
                raise
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Persist the aggregate manifest in `finally` (runs before a re-raised partition
            # failure propagates), so the failed record and incremented failed_count survive for
            # post-mortem rather than being lost when the loop aborts.
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
                "partition_selection": args.partition_selection,
                "partition_selection_reportability_errors": partition_selection_errors,
                "selection_reportable": not partition_selection_errors,
                "evaluation_design": "per_partition_walkforward_latest_holdout",
                "training_time_policy": training_time_policy,
                "official_test_partition": official_test_label,
                "final_test_is_latest_available": bool(final_test_is_latest_available),
                "all_prior_partitions_included": bool(all_prior_partitions_included),
                "training_history_truncated": bool(training_history_truncated),
                "excluded_prior_partition_count": int(excluded_prior_partition_count),
                "allow_truncated_training_history": bool(args.allow_truncated_training_history),
                # First-class official result so dashboards/papers do not have to scan `records`
                # for is_official_latest_test (None until the official/latest partition completes).
                "official_test": official_test_block(records, final_test_is_latest_available),
                "official_partition_count": sum(1 for item in records if item.get("is_official_latest_test")),
                "diagnostic_partition_count": sum(
                    1 for item in records if item["status"] == "ok" and not item.get("is_official_latest_test")
                ),
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
