#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Shared latest-period partition protocol -- the SAME gate the protocol-partition trainer uses, so the
# two paths cannot drift. (stdlib-only module, safe to import at module scope once SRC is on sys.path.)
from rl_quant.partition_protocol import (  # noqa: E402
    chronological_latest_label,
    label_span,
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
    / "polygon_second_top500_2023_to_2026-06-15"
    / "hour_from_second_1s"
    / "partitions"
)


ROW_LIST_KEYS = {
    "decision_timestamps",
    "next_timestamps",
    "minute_timestamp_grid",
    "subhour_timestamp_grid",
    "decision_timestamps_ms",
    "session_ids",
    "session_dates",
}
ROW_TENSOR_KEYS = {
    "minute_features",
    "subhour_features",
    "minute_mask",
    "subhour_mask",
    "hour_features",
    "action_returns",
    "decision_action_valid_mask",
    "action_valid_mask",
    "action_label_valid_mask",
    "label_valid_mask",
    "action_features",
    "action_feature_available_timestamps_ms",
    "action_features_available_timestamps_ms",
    "action_features_any_available_timestamps_ms",
}
SCHEMA_KEYS = {
    "minute_feature_names",
    "subhour_feature_names",
    "hour_feature_names",
    "action_names",
    "action_feature_names",
    "feature_names",
    "feature_names_by_tensor",
    "action_feature_groups",
    "hours_lookback",
    "minutes_per_hour",
    "context_bars_per_hour",
    "source_bar_interval",
    "decision_grid_minutes",
    "decision_stride_minutes",
    "periods_per_year",
    "bar_latency_ms",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one hour-from-second model with a calendar holdout: latest N months are test, "
            "the preceding validation window selects the checkpoint, and all earlier data trains."
        )
    )
    parser.add_argument("--partitions-root", type=Path, default=DEFAULT_PARTITIONS_ROOT)
    parser.add_argument("--dataset-file-name", default="hour_from_second_dataset.pt")
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "rl_hour_from_second_calendar_holdout_runs")
    parser.add_argument("--run-name", default=f"calendar_holdout_1s_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--test-start-date", help="Optional YYYY-MM-DD override for the first held-out test date.")
    parser.add_argument("--val-start-date", help="Optional YYYY-MM-DD override for the first validation date.")
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD exclusive end date; defaults to latest partition end.")
    parser.add_argument("--start-partition", help="First partition label to include, inclusive.")
    parser.add_argument("--end-partition", help="Last partition label to include, inclusive.")
    parser.add_argument("--max-partitions", type=int, default=0, help="0 means all matching partitions.")
    parser.add_argument(
        "--partition-selection",
        choices=["latest", "earliest"],
        default="latest",
        help="When --max-partitions is set, choose latest partitions by default; earliest is diagnostic only.",
    )
    parser.add_argument("--action-covariate-sidecar", choices=["auto", "required", "none"], default="auto")
    parser.add_argument("--news-llm-sidecar", choices=["auto", "required", "none"], default="none")
    parser.add_argument("--smoke", action="store_true", help="Compatibility flag for launcher smoke profiles.")
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
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--target-vram-gb", type=float)
    parser.add_argument("--vram-safety-gb", type=float, default=0.50)
    parser.add_argument("--min-free-vram-gb", type=float, default=0.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-state-file", default="training_state.pt")
    parser.add_argument("--checkpoint-every-steps", type=int, default=250)
    parser.add_argument("--payload-progress-every", type=int, default=5)
    parser.add_argument(
        "--strict-protocol-mask-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize loaded decision/label mask aliases, intersect label_valid_mask with "
            "decision_action_valid_mask, and set now-invalid action_returns to NaN before split construction."
        ),
    )
    parser.add_argument("--recency-weighting", choices=["none", "exponential"], default="none")
    parser.add_argument("--recency-half-life-days", type=float, default=120.0)
    parser.add_argument("--recency-min-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def parse_partition_end(label: str) -> date:
    # Use the shared, fully-anchored parser so a malformed/suffixed label (e.g. ..._to_..._garbage) is
    # rejected rather than silently truncated to its first 10 chars. The span end is exclusive.
    span = label_span(label)
    if span is None:
        raise ValueError(f"Partition label must be YYYY-MM-DD or YYYY-MM-DD_to_YYYY-MM-DD, got {label!r}")
    return span[1].date()


def calendar_selection_reportability(args: argparse.Namespace, selected_paths: list[Path]) -> tuple[list[str], bool]:
    """Strict latest-period reportability for the calendar-holdout partition selection.

    Mirrors the protocol-partition gate: the selected partitions must be valid, non-overlapping, and end
    at the latest AVAILABLE partition (not merely the latest among the selected paths). Any manual
    override -- a date boundary (``--end-date`` / ``--test-start-date`` / ``--val-start-date``) or a
    partition restriction (``--start/--end-partition`` / ``--max-partitions`` / ``earliest``) -- marks the
    split as manual and non-reportable: an OFFICIAL latest-period claim must be the auto-derived split
    over all available partitions. Returns ``(reportability_errors, manual_split_used)``."""
    all_available = [p.parent.name for p in sorted(args.partitions_root.glob(f"*/{args.dataset_file_name}"))]
    selected = [p.parent.name for p in selected_paths]
    # allow_truncated_training_history=True: a calendar holdout may legitimately train on a recent window,
    # so missing EARLIER history is not itself an error -- but the test must still be the latest available.
    errors = strict_latest_partition_violations(
        selected_labels=selected,
        all_available_labels=all_available,
        allow_truncated_training_history=True,
    )
    manual_split_used = bool(
        args.test_start_date
        or args.val_start_date
        or args.end_date
        or args.start_partition
        or args.end_partition
        or int(args.max_partitions) > 0
        or args.partition_selection != "latest"
    )
    if manual_split_used:
        errors = [*errors, "manual_or_restricted_calendar_split_used"]
    # --end-date is a calendar boundary the partition-label gate cannot see; if it excludes the latest
    # available data, the test is not the latest complete period.
    if args.end_date and all_available:
        latest_label = chronological_latest_label(all_available)
        latest = label_span(latest_label) if latest_label else None
        if latest is not None and datetime.strptime(args.end_date, "%Y-%m-%d") < latest[1]:
            errors.append("end_date_excludes_latest_available_data")
    return list(dict.fromkeys(errors)), manual_split_used


def subtract_months(value: date, months: int) -> date:
    if months <= 0:
        raise ValueError("months must be positive.")
    month_index = value.year * 12 + value.month - 1 - months
    year = month_index // 12
    month = month_index % 12 + 1
    month_lengths = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(year, month, min(value.day, month_lengths[month - 1]))


def midnight_utc(value: date) -> str:
    return f"{value.isoformat()}T00:00:00+00:00"


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
            paths = paths[-int(args.max_partitions) :]
        elif args.partition_selection == "earliest":
            paths = paths[: int(args.max_partitions)]
        else:
            raise ValueError(f"Unsupported partition selection: {args.partition_selection!r}")
    if not paths:
        raise FileNotFoundError(f"No {args.dataset_file_name} files under {args.partitions_root}")
    return paths


def calendar_boundaries(args: argparse.Namespace, paths: list[Path]) -> dict[str, str | int]:
    latest_end = (
        datetime.strptime(args.end_date, "%Y-%m-%d").date()
        if args.end_date
        else max(parse_partition_end(path.parent.name) for path in paths)
    )
    test_start = (
        datetime.strptime(args.test_start_date, "%Y-%m-%d").date()
        if args.test_start_date
        else subtract_months(latest_end, args.test_months)
    )
    val_start = (
        datetime.strptime(args.val_start_date, "%Y-%m-%d").date()
        if args.val_start_date
        else subtract_months(test_start, args.val_months)
    )
    if not val_start < test_start < latest_end:
        raise ValueError(f"Need val_start < test_start < end; got {val_start}, {test_start}, {latest_end}.")
    return {
        "val_start_date": val_start.isoformat(),
        "test_start_date": test_start.isoformat(),
        "end_date_exclusive": latest_end.isoformat(),
        "train_end_ts": midnight_utc(val_start),
        "val_start_ts": midnight_utc(val_start),
        "val_end_ts": midnight_utc(test_start),
        "test_start_ts": midnight_utc(test_start),
        "test_end_ts": midnight_utc(latest_end),
        "test_months": int(args.test_months),
        "val_months": int(args.val_months),
    }


def _same_schema(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return False
    return left == right


def _strict_protocol_repair_masks(payload: dict[str, Any]) -> list[str]:
    repairs: list[str] = []
    decision = payload.get("decision_action_valid_mask", payload.get("action_valid_mask"))
    label = payload.get("label_valid_mask", payload.get("action_label_valid_mask"))
    returns = payload.get("action_returns")

    if torch.is_tensor(decision):
        decision = decision.bool()
        payload["decision_action_valid_mask"] = decision
        payload["action_valid_mask"] = decision.clone()

    if torch.is_tensor(label):
        label = label.bool()
        if torch.is_tensor(decision) and tuple(label.shape) == tuple(decision.shape):
            outside_decision = label & ~decision
            outside_count = int(outside_decision.sum().item())
            if outside_count:
                repairs.append(f"label_valid_mask_intersected_with_decision_action_valid_mask:{outside_count}")
                label = label & decision
        payload["label_valid_mask"] = label
        payload["action_label_valid_mask"] = label.clone()

        if torch.is_tensor(returns) and tuple(returns.shape) == tuple(label.shape):
            invalid_finite = ~label & torch.isfinite(returns)
            invalid_finite_count = int(invalid_finite.sum().item())
            if invalid_finite_count:
                repairs.append(f"non_label_valid_action_returns_set_to_nan:{invalid_finite_count}")
                repaired_returns = returns.clone()
                repaired_returns[invalid_finite] = float("nan")
                payload["action_returns"] = repaired_returns

    if repairs:
        payload["strict_protocol_repairs"] = list(
            dict.fromkeys([*list(payload.get("strict_protocol_repairs", [])), *repairs])
        )
    return repairs


def concatenate_payloads(
    paths: list[Path],
    *,
    action_covariate_sidecar: str,
    news_llm_sidecar: str,
    progress_every: int = 5,
    strict_protocol_mask_repair: bool = True,
) -> dict[str, Any]:
    from rl_quant.minute_to_hour_transformer import _load_payload

    started = time.monotonic()
    progress_every = max(1, int(progress_every))
    first: dict[str, Any] | None = None
    out: dict[str, Any] = {}
    row_lists: dict[str, list[Any]] = {key: [] for key in ROW_LIST_KEYS}
    row_tensors: dict[str, list[torch.Tensor]] = {key: [] for key in ROW_TENSOR_KEYS}
    errors: list[str] = []
    strict_protocol_repairs: list[str] = []
    repair_counts: dict[str, int] = {}
    dataset_reportable = True
    loaded_count = 0
    row_count = 0

    print(f"Loading {len(paths)} partition payloads with sidecars...", flush=True)
    for index, path in enumerate(paths, start=1):
        payload = _load_payload(path, action_covariate_sidecar=action_covariate_sidecar, news_llm_sidecar=news_llm_sidecar)
        if strict_protocol_mask_repair:
            for repair in _strict_protocol_repair_masks(payload):
                strict_protocol_repairs.append(f"{path.parent.name}:{repair}")
                repair_name, _, raw_count = repair.partition(":")
                try:
                    repair_counts[repair_name] = repair_counts.get(repair_name, 0) + int(raw_count)
                except ValueError:
                    repair_counts[repair_name] = repair_counts.get(repair_name, 0) + 1
        if first is None:
            first = payload
            for key in SCHEMA_KEYS:
                if key in first:
                    out[key] = first[key]
        else:
            for key in SCHEMA_KEYS:
                if key in out and key in payload and not _same_schema(out[key], payload[key]):
                    raise ValueError(f"Schema key {key!r} differs across partitions.")
        rows_this = len(payload["decision_timestamps"])
        row_count += rows_this
        loaded_count += 1
        for key in ROW_LIST_KEYS:
            if first is not None and key in first:
                row_lists[key].extend(list(payload.get(key, [])))
        for key in ROW_TENSOR_KEYS:
            if key in payload:
                row_tensors[key].append(payload[key])
        errors.extend(str(item) for item in payload.get("dataset_reportability_errors", []))
        dataset_reportable = dataset_reportable and bool(payload.get("dataset_reportable", payload.get("reportable", True)))
        if index == 1 or index == len(paths) or index % progress_every == 0:
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                "payload_load_progress "
                f"{index}/{len(paths)} rows={row_count} elapsed_s={elapsed:.1f} "
                f"rows_per_s={row_count / elapsed:.2f} last={path.parent.name}",
                flush=True,
            )

    if first is None:
        raise ValueError("No payloads loaded.")
    for key in SCHEMA_KEYS:
        if key in first:
            out[key] = first[key]
    for key in ROW_LIST_KEYS:
        if key in first:
            out[key] = row_lists[key]
    print(f"Concatenating tensor fields for {row_count} rows...", flush=True)
    for key in ROW_TENSOR_KEYS:
        tensors = row_tensors[key]
        if tensors:
            if len(tensors) != loaded_count:
                raise ValueError(f"Row tensor key {key!r} is missing from some partitions.")
            print(f"  cat {key}: {len(tensors)} chunks", flush=True)
            out[key] = torch.cat(tensors, dim=0)
    out["dataset_reportability_errors"] = list(dict.fromkeys(errors))
    out["dataset_reportable"] = dataset_reportable and not errors
    if strict_protocol_repairs:
        out["strict_protocol_repairs"] = strict_protocol_repairs
        print(
            "Applied strict protocol mask repairs before split construction: "
            + json.dumps(
                {
                    "partition_count": len({item.split(":", 1)[0] for item in strict_protocol_repairs}),
                    "repair_counts": repair_counts,
                    "examples": strict_protocol_repairs[:10],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    out["composite_partition_count"] = len(paths)
    out["composite_partition_start"] = paths[0].parent.name
    out["composite_partition_end"] = paths[-1].parent.name
    out["composite_row_count"] = row_count
    print(
        f"Composite payload ready: partitions={loaded_count} rows={row_count} elapsed_s={time.monotonic() - started:.1f}",
        flush=True,
    )
    return out


def build_calendar_splits(
    payload: dict[str, Any],
    boundaries: dict[str, str | int],
    *,
    selection_errors: list[str] | None = None,
    manual_split_used: bool = False,
):
    from rl_quant.minute_to_hour_transformer import _build_split

    # Reportability comes from the actual partition selection, not an unconditional True: a restricted
    # or manually-bounded run cannot claim the latest complete period (see calendar_selection_reportability).
    selection_errors = list(selection_errors or [])
    split_policy = {
        "split_mode": "calendar_holdout",
        "evaluation_design": "calendar_holdout_latest_months",
        "test_months": boundaries["test_months"],
        "val_months": boundaries["val_months"],
        "train_end": boundaries["train_end_ts"],
        "val_start": boundaries["val_start_ts"],
        "val_end": boundaries["val_end_ts"],
        "test_start": boundaries["test_start_ts"],
        "test_end": boundaries["test_end_ts"],
        "test_uses_latest_complete_period": not selection_errors,
        "manual_split_used": bool(manual_split_used),
        "reportable": not selection_errors,
        "reportability_errors": selection_errors,
        # Machine-explicit attestation of the reportable split protocol: test = latest complete
        # period(s); validation = the immediately preceding block; train = strictly earlier; the test
        # block never drives model/recency selection. The first is gated on the partition selection
        # (selection_errors); the rest are asserted from the calendar walk-forward boundaries.
        "test_is_latest_available_suffix": not selection_errors,
        "validation_immediately_precedes_test": boundaries["val_end_ts"] == boundaries["test_start_ts"],
        "train_ends_before_validation": boundaries["train_end_ts"] <= boundaries["val_start_ts"],
        "test_used_for_model_selection": False,
        "test_used_for_recency_selection": False,
        "blocks": {
            "train": {"start": None, "end": boundaries["train_end_ts"], "reward_end": boundaries["train_end_ts"]},
            "val": {
                "start": boundaries["val_start_ts"],
                "end": boundaries["val_end_ts"],
                "reward_start": boundaries["val_start_ts"],
                "reward_end": boundaries["val_end_ts"],
            },
            "test": {
                "start": boundaries["test_start_ts"],
                "end": boundaries["test_end_ts"],
                "reward_start": boundaries["test_start_ts"],
                "reward_end": boundaries["test_end_ts"],
            },
        },
    }
    train = _build_split(
        name="train",
        payload=payload,
        end_ts=str(boundaries["train_end_ts"]),
        reward_end_ts=str(boundaries["train_end_ts"]),
        split_policy=split_policy,
    )
    val = _build_split(
        name="val",
        payload=payload,
        start_ts=str(boundaries["val_start_ts"]),
        end_ts=str(boundaries["val_end_ts"]),
        reward_start_ts=str(boundaries["val_start_ts"]),
        reward_end_ts=str(boundaries["val_end_ts"]),
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
        start_ts=str(boundaries["test_start_ts"]),
        end_ts=str(boundaries["test_end_ts"]),
        reward_start_ts=str(boundaries["test_start_ts"]),
        reward_end_ts=str(boundaries["test_end_ts"]),
        minute_feature_mean=train.minute_feature_mean,
        minute_feature_std=train.minute_feature_std,
        hour_feature_mean=train.hour_feature_mean,
        hour_feature_std=train.hour_feature_std,
        action_feature_mean=train.action_feature_mean,
        action_feature_std=train.action_feature_std,
        split_policy=split_policy,
    )
    return train, val, test


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


def split_window(split) -> dict[str, str | int | None]:
    valid = [int(item) for item in split.valid_start_indices.detach().cpu().tolist()]
    if not valid:
        return {
            "selected_rows": len(split.decision_timestamps),
            "valid_starts": 0,
            "first_valid_decision": None,
            "first_valid_reward_end": None,
            "last_valid_decision": None,
            "last_valid_reward_end": None,
        }
    return {
        "selected_rows": len(split.decision_timestamps),
        "valid_starts": len(valid),
        "first_valid_decision": split.decision_timestamps[valid[0]],
        "first_valid_reward_end": split.next_timestamps[valid[0]],
        "last_valid_decision": split.decision_timestamps[valid[-1]],
        "last_valid_reward_end": split.next_timestamps[valid[-1]],
    }


def write_rollout(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    from rl_quant.trading_constraints import CONSTRAINED_POLICY_MODEL_VERSION, CONSTRAINT_FEATURE_NAMES

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    require_min_free_vram(device, args.min_free_vram_gb)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    paths = partition_paths(args)
    boundaries = calendar_boundaries(args, paths)
    print(
        json.dumps(
            {
                "evaluation_design": "calendar_holdout_latest_months",
                "partitions": len(paths),
                "first_partition": paths[0].parent.name,
                "last_partition": paths[-1].parent.name,
                "partition_selection": args.partition_selection,
                "max_partitions": args.max_partitions,
                "start_partition": args.start_partition,
                "end_partition": args.end_partition,
                **boundaries,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    payload = concatenate_payloads(
        paths,
        action_covariate_sidecar=args.action_covariate_sidecar,
        news_llm_sidecar=args.news_llm_sidecar,
        progress_every=args.payload_progress_every,
        strict_protocol_mask_repair=args.strict_protocol_mask_repair,
    )
    print("Building calendar train/val/test splits...", flush=True)
    selection_errors, manual_split_used = calendar_selection_reportability(args, paths)
    if selection_errors:
        print(
            "calendar-holdout partition selection is NOT latest-period reportable:\n  - "
            + "\n  - ".join(selection_errors),
            flush=True,
        )
    train_split, val_split, test_split = build_calendar_splits(
        payload, boundaries, selection_errors=selection_errors, manual_split_used=manual_split_used
    )
    constraints = build_constraints_from_args(args)
    initial_action = action_index(train_split.action_names, args.initial_action)
    runtime = torch_runtime_summary(device)
    config_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_state_file = Path(args.resume_state_file)
    if resume_state_file.is_absolute():
        raise ValueError("--resume-state-file must be relative to --output-dir/--run-name.")
    training_state_path = run_dir / resume_state_file

    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(
            f"CUDA device: {runtime['cuda_device_name']} | total memory: {runtime['cuda_total_memory_gb']} GiB | AMP: {args.amp}",
            flush=True,
        )
    print(
        "Calendar split rows: "
        f"train={train_split.valid_start_indices.numel()} "
        f"val={val_split.valid_start_indices.numel()} "
        f"test={test_split.valid_start_indices.numel()}",
        flush=True,
    )
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
        max_subhour_tokens=args.max_subhour_tokens,
        resume_training_state=training_state_path if args.resume else None,
        checkpoint_training_state=training_state_path if args.checkpoint_every_steps > 0 else None,
        checkpoint_every_steps=args.checkpoint_every_steps,
        recency=RecencyWeightConfig(
            mode=args.recency_weighting,
            half_life_days=args.recency_half_life_days,
            min_weight=args.recency_min_weight,
        ),
    )
    started = datetime.now(timezone.utc)
    try:
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
        checkpoint_path = run_dir / "model.pt"
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
                "split_policy": train_split.split_policy,
                "constraints": asdict(constraints),
                "config": config_payload,
                "partition_dataset_count": len(paths),
            },
            checkpoint_path,
        )
        write_rollout(run_dir / "test_rollout.csv", test_result.rollout_records)
        summary = {
            "run_name": args.run_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
            "evaluation_design": "calendar_holdout_latest_months",
            "partitions_root": str(args.partitions_root),
            "partition_count": len(paths),
            "first_partition": paths[0].parent.name,
            "last_partition": paths[-1].parent.name,
            "boundaries": boundaries,
            "device": str(device),
            "torch_runtime": runtime,
            "config": config_payload,
            "constraints": asdict(constraints),
            "checkpoint": str(checkpoint_path),
            "training_state_checkpoint": str(training_state_path),
            "split_windows": {
                "train": split_window(train_split),
                "val": split_window(val_split),
                "test": split_window(test_split),
            },
            "rows": {
                "train": len(train_split.decision_timestamps),
                "val": len(val_split.decision_timestamps),
                "test": len(test_split.decision_timestamps),
                "train_valid_starts": int(train_split.valid_start_indices.numel()),
                "val_valid_starts": int(val_split.valid_start_indices.numel()),
                "test_valid_starts": int(test_split.valid_start_indices.numel()),
            },
            "training": artifacts,
            "train_metrics": train_result.to_dict(),
            "val_metrics": val_result.to_dict(),
            "test_metrics": test_result.to_dict(),
            "official_test": {
                "period_start": boundaries["test_start_date"],
                "period_end_exclusive": boundaries["end_date_exclusive"],
                "test_total_return": test_result.total_return,
                "test_total_reward_bps": test_result.total_reward_bps,
                "test_max_drawdown": test_result.max_drawdown,
                "test_annualized_sharpe": test_result.annualized_sharpe,
                "test_switches": test_result.allocation_switches,
                "test_order_legs": test_result.market_order_legs,
                "checkpoint_selected_on": "validation",
                "test_used_for_selection": False,
                "reportable": bool(test_result.evaluation_reportable),
                "reportability_errors": list(test_result.reportability_errors),
            },
        }
        with (run_dir / "calendar_holdout_summary.json").open("w") as sink:
            json.dump(summary, sink, indent=2)
        print(f"Calendar holdout training complete. Summary -> {run_dir / 'calendar_holdout_summary.json'}", flush=True)
        return 0
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
