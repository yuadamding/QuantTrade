#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.research_protocol import hash_string_sequence, stable_json_hash, utc_now_iso  # noqa: E402


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


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
    parser.add_argument("--max-effective-leverage", type=float, default=1.0)
    parser.add_argument("--allow-leveraged-actions", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--allow-inverse-actions", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--max-leveraged-bars-per-day", type=int, default=30)
    parser.add_argument("--max-consecutive-leveraged-bars", type=int, default=15)
    parser.add_argument("--max-same-group-share-per-day", type=float, default=0.50)
    parser.add_argument("--min-group-share-observations", type=int, default=20)
    parser.add_argument("--reportable-max-group-share", type=float, default=0.75)
    parser.add_argument("--reportable-max-leveraged-share", type=float, default=0.50)
    parser.add_argument("--random-baseline-paths", type=int, default=256)
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
    parser.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="AMP autocast precision when --amp is set. bf16 (wider exponent range) is preferred on "
        "Ampere/Hopper GPUs; fp16 (default) preserves prior behavior.",
    )
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=0.0,
        help="Fail fast before training if free CUDA memory is below this many GiB (0 disables).",
    )
    parser.add_argument(
        "--target-vram-gb",
        type=float,
        help="OPT-IN CUDA ballast that INCREASES used VRAM toward this amount (not a cap); prefer "
        "--min-free-vram-gb to guard headroom.",
    )
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


def build_exposure_constraints_from_args(args: argparse.Namespace):
    from rl_quant.action_risk import ExposureConstraintConfig

    return ExposureConstraintConfig(
        max_effective_leverage=args.max_effective_leverage,
        allow_leveraged_actions=args.allow_leveraged_actions,
        allow_inverse_actions=args.allow_inverse_actions,
        max_leveraged_bars_per_day=args.max_leveraged_bars_per_day,
        max_same_group_share_per_day=args.max_same_group_share_per_day,
        max_consecutive_leveraged_bars=args.max_consecutive_leveraged_bars,
        min_group_share_observations=args.min_group_share_observations,
    )


def unique_cost_stresses(values: list[float]) -> list[float]:
    return sorted({float(value) for value in values})


def read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


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
            traded_notional = record.get("traded_notional")
            is_switch = action != previous_action
            if "gross_return" not in record:
                raise ValueError(
                    "fixed_rollout_cost_stress requires gross_return; "
                    "re-run evaluation with the current rollout schema."
            )
            gross_return = float(record["gross_return"])
            if traded_notional is None:
                realized_cost_bps = legs * float(cost_bps) + float(is_switch) * float(extra_switch_penalty_bps)
            else:
                realized_cost_bps = float(traded_notional) * (
                    float(cost_bps) + float(is_switch) * float(extra_switch_penalty_bps)
                )
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
            "total_traded_notional": sum(
                float(record.get("traded_notional", record["market_order_legs"])) for record in rollout_records
            ),
            "max_drawdown": metric_max_drawdown(equity_curve),
            "annualized_sharpe": metric_sharpe(returns, periods_per_year=periods_per_year),
        }
    return results


def equal_weight_metrics(split, *, cash_index: int, action_weights=None) -> dict[str, float | int | None]:
    action_indices = [idx for idx in range(len(split.action_names)) if idx != cash_index]
    if not action_indices:
        action_indices = [cash_index]
    weights = None if action_weights is None else action_weights.detach().to(device=split.action_returns.device)
    valid_mask = getattr(split, "action_valid_mask", None)
    equity = 1.0
    equity_curve = [equity]
    returns: list[float] = []
    for index in split.valid_start_indices.detach().cpu().tolist():
        action_returns = split.action_returns[index, action_indices]
        if weights is not None:
            action_returns = action_returns * weights[action_indices]
        # Average only over FINITE and (when available) decision-valid actions. A valid_start row
        # does not guarantee every action is valid that bar; invalid action returns are NaN by
        # contract, and an unmasked mean would make equity NaN from that bar onward and silently
        # NaN the whole EqualWeight baseline. With no valid non-cash action, hold cash (0 return).
        finite = action_returns.isfinite()
        if valid_mask is not None:
            finite = finite & valid_mask[index, action_indices].to(device=finite.device)
        if bool(finite.any().item()):
            simple_return = float(action_returns[finite].mean().item())
        else:
            simple_return = 0.0
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


def _trade_legs_scalar(
    previous_action: int,
    action: int,
    *,
    cash_index: int,
    count_etf_to_etf_as_two_legs: bool = True,
) -> float:
    if action == previous_action:
        return 0.0
    if not count_etf_to_etf_as_two_legs:
        return 1.0
    legs = 0.0
    if previous_action != cash_index:
        legs += 1.0
    if action != cash_index:
        legs += 1.0
    return legs


def _valid_action_indices(split, index: int, *, cash_index: int) -> list[int]:
    if getattr(split, "action_valid_mask", None) is None:
        valid = list(range(len(split.action_names)))
    else:
        row = split.action_valid_mask[index].detach().cpu().tolist()
        valid = [action for action, is_valid in enumerate(row) if bool(is_valid)]
    if cash_index not in valid:
        valid.append(cash_index)
    return valid


def _split_timestamps(split) -> list[str]:
    return list(getattr(split, "timestamps", getattr(split, "decision_timestamps", [])))


def _split_row_date(split, index: int) -> str:
    session_dates = getattr(split, "session_dates", None)
    if session_dates is not None:
        return str(session_dates[index])
    timestamps = _split_timestamps(split)
    return timestamps[index][:10] if timestamps else ""


def _new_baseline_state(*, initial_action: int, constraints, group_count: int) -> dict[str, object]:
    import torch

    return {
        "previous_action": int(initial_action),
        "previous_index": None,
        "previous_date": None,
        "bars_held": int(getattr(constraints, "min_hold_bars", 1)) if constraints is not None else 1,
        "cooldown_remaining": 0,
        "switches_today": 0,
        "switches_episode": 0,
        "order_legs_today": 0.0,
        "order_legs_episode": 0.0,
        "steps_today": 0,
        "leveraged_bars_today": 0,
        "consecutive_leveraged_bars": 0,
        "episode_steps": 0,
        "group_counts_today": torch.zeros((1, max(int(group_count), 1)), dtype=torch.long),
    }


def _reset_baseline_daily_state(state: dict[str, object]) -> None:
    state["switches_today"] = 0
    state["order_legs_today"] = 0.0
    state["steps_today"] = 0
    state["leveraged_bars_today"] = 0
    state["consecutive_leveraged_bars"] = 0
    state["group_counts_today"].zero_()


def _prepare_baseline_state_for_row(
    state: dict[str, object],
    split,
    row_index: int,
    *,
    initial_action: int,
    constraints,
    episode_length: int | None,
) -> None:
    current_date = _split_row_date(split, row_index)
    previous_index = state["previous_index"]
    segment_reset = previous_index is None or row_index != int(previous_index) + 1
    if segment_reset:
        state["previous_action"] = int(initial_action)
        state["bars_held"] = int(getattr(constraints, "min_hold_bars", 1)) if constraints is not None else 1
        state["cooldown_remaining"] = 0
        state["switches_episode"] = 0
        state["order_legs_episode"] = 0.0
        state["episode_steps"] = 0
        _reset_baseline_daily_state(state)
    elif state["previous_date"] is not None and current_date != state["previous_date"]:
        _reset_baseline_daily_state(state)
    if episode_length is not None and int(state["episode_steps"]) >= int(episode_length):
        state["switches_episode"] = 0
        state["order_legs_episode"] = 0.0
        state["episode_steps"] = 0


def _baseline_mask_context(
    split,
    *,
    action_weights,
    constraints,
    exposure_constraints,
    action_meta,
    cash_index: int,
) -> dict[str, object] | None:
    if constraints is None and exposure_constraints is None:
        return None
    import torch

    from rl_quant.action_risk import (
        action_is_inverse_tensor,
        action_is_leveraged_tensor,
        action_leverage_tensor,
        build_action_metadata,
        group_ids_for_actions,
    )

    action_meta = action_meta or build_action_metadata(split.action_names)
    action_weights = action_weights.detach().cpu()
    action_group_ids, action_groups = group_ids_for_actions(action_meta, device=torch.device("cpu"))
    return {
        "constraints": constraints,
        "exposure_constraints": exposure_constraints,
        "cash_index": int(cash_index),
        "action_weights": action_weights,
        "action_leverage": action_leverage_tensor(action_meta, device=torch.device("cpu")),
        "action_is_leveraged": action_is_leveraged_tensor(action_meta, device=torch.device("cpu")),
        "action_is_inverse": action_is_inverse_tensor(action_meta, device=torch.device("cpu")),
        "action_group_ids": action_group_ids,
        "group_count": len(action_groups),
    }


def _full_valid_action_indices(
    split,
    row_index: int,
    *,
    state: dict[str, object],
    context: dict[str, object] | None,
    cash_index: int,
    episode_length: int | None,
) -> list[int]:
    if context is None:
        return _valid_action_indices(split, row_index, cash_index=cash_index)

    import torch

    from rl_quant.action_risk import apply_exposure_masks
    from rl_quant.trading_constraints import TradingConstraintConfig, build_action_mask

    constraints = context["constraints"] or TradingConstraintConfig(cash_index=cash_index)
    previous = torch.tensor([int(state["previous_action"])], dtype=torch.long)
    mask = build_action_mask(
        current_action=previous,
        bars_held=torch.tensor([int(state["bars_held"])], dtype=torch.long),
        cooldown_remaining=torch.tensor([int(state["cooldown_remaining"])], dtype=torch.long),
        switches_today=torch.tensor([int(state["switches_today"])], dtype=torch.long),
        min_hold_bars=constraints.min_hold_bars,
        action_count=len(split.action_names),
        max_switches_per_day=constraints.max_switches_per_day,
        switches_episode=torch.tensor([int(state["switches_episode"])], dtype=torch.long),
        max_switches_per_episode=constraints.max_switches_per_episode,
        order_legs_today=torch.tensor([float(state["order_legs_today"])], dtype=torch.float32),
        max_order_legs_per_day=constraints.max_order_legs_per_day,
        order_legs_episode=torch.tensor([float(state["order_legs_episode"])], dtype=torch.float32),
        max_order_legs_per_episode=constraints.max_order_legs_per_episode,
        cash_index=cash_index,
        count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
    )
    availability = torch.ones((1, len(split.action_names)), dtype=torch.bool)
    if getattr(split, "action_valid_mask", None) is not None:
        availability = split.action_valid_mask[row_index].detach().cpu().bool().unsqueeze(0)
    availability[:, cash_index] = True
    mask = mask & availability
    exposure_constraints = context["exposure_constraints"]
    if exposure_constraints is not None:
        mask = apply_exposure_masks(
            mask,
            current_action=previous,
            action_leverage=context["action_leverage"],
            action_weights=context["action_weights"],
            action_is_leveraged=context["action_is_leveraged"],
            action_is_inverse=context["action_is_inverse"],
            action_group_ids=context["action_group_ids"],
            group_counts_today=state["group_counts_today"],
            steps_today=torch.tensor([int(state["steps_today"])], dtype=torch.long),
            leveraged_bars_today=torch.tensor([int(state["leveraged_bars_today"])], dtype=torch.long),
            consecutive_leveraged_bars=torch.tensor([int(state["consecutive_leveraged_bars"])], dtype=torch.long),
            constraints=exposure_constraints,
            cash_index=cash_index,
        )
    if not bool(mask.any().item()):
        mask[:, cash_index] = True
    return [action for action, valid in enumerate(mask[0].tolist()) if bool(valid)]


def _advance_baseline_state(
    state: dict[str, object],
    split,
    row_index: int,
    *,
    action: int,
    legs: float,
    constraints,
    context: dict[str, object] | None,
) -> None:
    previous_action = int(state["previous_action"])
    is_switch = int(action) != previous_action
    if is_switch:
        state["bars_held"] = 1
        state["cooldown_remaining"] = int(getattr(constraints, "cooldown_bars", 0)) if constraints is not None else 0
        state["switches_today"] = int(state["switches_today"]) + 1
        state["switches_episode"] = int(state["switches_episode"]) + 1
    else:
        state["bars_held"] = int(state["bars_held"]) + 1
        state["cooldown_remaining"] = max(int(state["cooldown_remaining"]) - 1, 0)
    state["order_legs_today"] = float(state["order_legs_today"]) + float(legs)
    state["order_legs_episode"] = float(state["order_legs_episode"]) + float(legs)
    if context is not None:
        group_id = int(context["action_group_ids"][int(action)].item())
        state["group_counts_today"][0, group_id] += 1
        selected_leveraged = bool(context["action_is_leveraged"][int(action)].item())
        state["leveraged_bars_today"] = int(state["leveraged_bars_today"]) + int(selected_leveraged)
        state["consecutive_leveraged_bars"] = (
            int(state["consecutive_leveraged_bars"]) + 1 if selected_leveraged else 0
        )
    state["steps_today"] = int(state["steps_today"]) + 1
    state["previous_action"] = int(action)
    state["previous_index"] = int(row_index)
    state["previous_date"] = _split_row_date(split, row_index)
    state["episode_steps"] = int(state["episode_steps"]) + 1


def _evaluate_action_sequence(
    split,
    actions: list[int],
    *,
    initial_action: int,
    cash_index: int,
    action_weights,
    switch_cost_bps: float,
    extra_switch_penalty_bps: float,
    count_etf_to_etf_as_two_legs: bool = True,
    constraints=None,
    exposure_constraints=None,
    action_meta=None,
    episode_length: int | None = None,
) -> dict[str, float | int | None]:
    valid_indices = split.valid_start_indices.detach().cpu().tolist()
    if len(actions) != len(valid_indices):
        raise ValueError("actions length must match split.valid_start_indices length.")
    weights = [float(value) for value in action_weights.detach().cpu().tolist()]
    equity = 1.0
    equity_curve = [equity]
    returns: list[float] = []
    switches = 0
    order_legs = 0.0
    total_traded_notional = 0.0
    context = _baseline_mask_context(
        split,
        action_weights=action_weights,
        constraints=constraints,
        exposure_constraints=exposure_constraints,
        action_meta=action_meta,
        cash_index=cash_index,
    )
    state = _new_baseline_state(
        initial_action=initial_action,
        constraints=constraints,
        group_count=int(context["group_count"]) if context is not None else 1,
    )
    for sequence_index, row_index in enumerate(valid_indices):
        _prepare_baseline_state_for_row(
            state,
            split,
            row_index,
            initial_action=initial_action,
            constraints=constraints,
            episode_length=episode_length,
        )
        previous_action = int(state["previous_action"])
        valid_actions = _full_valid_action_indices(
            split,
            row_index,
            state=state,
            context=context,
            cash_index=cash_index,
            episode_length=episode_length,
        )
        action = int(actions[sequence_index])
        if action not in valid_actions:
            action = int(cash_index) if cash_index in valid_actions else int(valid_actions[0])
        raw_return = float(split.action_returns[row_index, action].item())
        position_weight = weights[action]
        gross_return = position_weight * raw_return
        is_switch = action != previous_action
        previous_weight = 0.0 if previous_action == cash_index else weights[previous_action]
        next_weight = 0.0 if action == cash_index else weights[action]
        traded_notional = previous_weight + next_weight if is_switch else 0.0
        cost_bps = traded_notional * (float(switch_cost_bps) + float(is_switch) * float(extra_switch_penalty_bps))
        net_return = gross_return - cost_bps / 10_000.0
        equity *= 1.0 + net_return
        equity_curve.append(equity)
        returns.append(net_return)
        switches += int(is_switch)
        order_legs += _trade_legs_scalar(
            previous_action,
            action,
            cash_index=cash_index,
            count_etf_to_etf_as_two_legs=count_etf_to_etf_as_two_legs,
        )
        total_traded_notional += traded_notional
        _advance_baseline_state(
            state,
            split,
            row_index,
            action=action,
            legs=_trade_legs_scalar(
                previous_action,
                action,
                cash_index=cash_index,
                count_etf_to_etf_as_two_legs=count_etf_to_etf_as_two_legs,
            ),
            constraints=constraints,
            context=context,
        )
    return {
        "total_return": equity - 1.0,
        "max_drawdown": metric_max_drawdown(equity_curve),
        "annualized_sharpe": metric_sharpe(returns, periods_per_year=split.periods_per_year),
        "total_switches": switches,
        "market_order_legs": order_legs,
        "total_traded_notional": total_traded_notional,
    }


def _aggregate_random_path_metrics(paths: list[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    if not paths:
        return {"paths": 0, "total_return": 0.0}
    result: dict[str, float | int | None] = {"paths": len(paths)}
    numeric_keys = [
        "total_return",
        "max_drawdown",
        "annualized_sharpe",
        "total_switches",
        "market_order_legs",
        "total_traded_notional",
    ]
    for key in numeric_keys:
        values = [float(path[key]) for path in paths if path.get(key) is not None]
        if not values:
            result[key] = None
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result[key] = mean
        result[f"{key}_std"] = variance**0.5
    return result


def random_same_action_distribution_baseline(
    split,
    rollout_records: list[dict[str, object]],
    *,
    seed: int,
    n_paths: int,
    initial_action: int,
    cash_index: int,
    action_weights,
    switch_cost_bps: float,
    extra_switch_penalty_bps: float,
    count_etf_to_etf_as_two_legs: bool = True,
    constraints=None,
    exposure_constraints=None,
    action_meta=None,
    episode_length: int | None = None,
) -> dict[str, float | int | None]:
    valid_indices = split.valid_start_indices.detach().cpu().tolist()
    counts = Counter(int(record["action"]) for record in rollout_records if "action" in record)
    if not counts:
        counts = Counter({cash_index: 1})
    action_ids = sorted(counts)
    action_counts = [counts[action] for action in action_ids]
    rng = random.Random(seed)
    paths = []
    for _ in range(max(int(n_paths), 1)):
        sampled = rng.choices(action_ids, weights=action_counts, k=len(valid_indices))
        paths.append(
            _evaluate_action_sequence(
                split,
                sampled,
                initial_action=initial_action,
                cash_index=cash_index,
                action_weights=action_weights,
                switch_cost_bps=switch_cost_bps,
                extra_switch_penalty_bps=extra_switch_penalty_bps,
                count_etf_to_etf_as_two_legs=count_etf_to_etf_as_two_legs,
                constraints=constraints,
                exposure_constraints=exposure_constraints,
                action_meta=action_meta,
                episode_length=episode_length,
            )
        )
    out = _aggregate_random_path_metrics(paths)
    out["seed"] = seed
    return out


def random_same_turnover_baseline(
    split,
    rollout_records: list[dict[str, object]],
    *,
    seed: int,
    n_paths: int,
    initial_action: int,
    cash_index: int,
    action_weights,
    switch_cost_bps: float,
    extra_switch_penalty_bps: float,
    count_etf_to_etf_as_two_legs: bool = True,
    constraints=None,
    exposure_constraints=None,
    action_meta=None,
    episode_length: int | None = None,
) -> dict[str, float | int | None]:
    valid_indices = split.valid_start_indices.detach().cpu().tolist()
    switch_flags = [
        int(record.get("action", cash_index)) != int(record.get("previous_action", cash_index))
        for record in rollout_records
    ]
    if len(switch_flags) < len(valid_indices):
        switch_flags.extend([False] * (len(valid_indices) - len(switch_flags)))
    switch_flags = switch_flags[: len(valid_indices)]
    rng = random.Random(seed)
    paths = []
    context = _baseline_mask_context(
        split,
        action_weights=action_weights,
        constraints=constraints,
        exposure_constraints=exposure_constraints,
        action_meta=action_meta,
        cash_index=cash_index,
    )
    for _ in range(max(int(n_paths), 1)):
        sampled: list[int] = []
        state = _new_baseline_state(
            initial_action=initial_action,
            constraints=constraints,
            group_count=int(context["group_count"]) if context is not None else 1,
        )
        for sequence_index, row_index in enumerate(valid_indices):
            _prepare_baseline_state_for_row(
                state,
                split,
                row_index,
                initial_action=initial_action,
                constraints=constraints,
                episode_length=episode_length,
            )
            previous_action = int(state["previous_action"])
            valid_actions = _full_valid_action_indices(
                split,
                row_index,
                state=state,
                context=context,
                cash_index=cash_index,
                episode_length=episode_length,
            )
            if switch_flags[sequence_index]:
                choices = [action for action in valid_actions if action != previous_action]
                action = rng.choice(choices or [cash_index])
            else:
                action = previous_action if previous_action in valid_actions else cash_index
            if action not in valid_actions:
                action = cash_index if cash_index in valid_actions else int(valid_actions[0])
            sampled.append(action)
            legs = _trade_legs_scalar(
                previous_action,
                action,
                cash_index=cash_index,
                count_etf_to_etf_as_two_legs=count_etf_to_etf_as_two_legs,
            )
            _advance_baseline_state(
                state,
                split,
                row_index,
                action=action,
                legs=legs,
                constraints=constraints,
                context=context,
            )
        paths.append(
            _evaluate_action_sequence(
                split,
                sampled,
                initial_action=initial_action,
                cash_index=cash_index,
                action_weights=action_weights,
                switch_cost_bps=switch_cost_bps,
                extra_switch_penalty_bps=extra_switch_penalty_bps,
                count_etf_to_etf_as_two_legs=count_etf_to_etf_as_two_legs,
                constraints=constraints,
                exposure_constraints=exposure_constraints,
                action_meta=action_meta,
                episode_length=episode_length,
            )
        )
    out = _aggregate_random_path_metrics(paths)
    out["seed"] = seed
    out["target_switches"] = sum(int(flag) for flag in switch_flags)
    return out


def write_rollout(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in record.items()
                }
            )


def write_decision_logs(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    from rl_quant.decision_framework import DecisionLog

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as sink:
        for record in records:
            log = DecisionLog(
                decision_id=str(record["decision_id"]),
                decision_ts=str(record.get("decision_ts", record["timestamp"])),
                model_id=str(record["model_id"]),
                selected_action=str(record["selected_action"]),
                previous_action=str(record["previous_asset"]),
                action_mask_reasons=dict(record.get("action_mask_reasons", {})),
                q_values={str(key): float(value) for key, value in dict(record.get("q_values", {})).items()},
                risk_checks={str(key): bool(value) for key, value in dict(record.get("risk_checks", {})).items()},
                expected_cost_bps=float(record["expected_cost_bps"]),
                data_quality_score=float(record.get("data_quality_score", 1.0)),
                readiness_score=float(record.get("readiness_score", 1.0)),
                readiness_config_hash=str(record["readiness_config_hash"]),
                candidates=dict(record.get("candidates", {})),
            )
            log.validate()
            sink.write(json.dumps(log.to_dict(), sort_keys=True) + "\n")


def _baseline_metric(payload: object, *, split_name: str = "test") -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    if split_name in payload and isinstance(payload[split_name], dict):
        return payload[split_name]
    if "total_return" in payload:
        return payload
    return None


def _tensor_hash(tensor) -> str:
    values = [round(float(value), 10) for value in tensor.detach().cpu().flatten().tolist()]
    return stable_json_hash(values)


def _split_quality(split) -> dict[str, object]:
    import torch

    action_mask = getattr(split, "action_valid_mask", None)
    returns = split.action_returns.detach().cpu()
    if action_mask is None:
        valid_return_count = int(returns.numel())
        invalid_return_count = 0
        valid_returns_finite = bool(torch.isfinite(returns).all().item())
        invalid_returns_nan = True
    else:
        mask = action_mask.detach().cpu().bool()
        valid_returns = returns[mask]
        invalid_returns = returns[~mask]
        valid_return_count = int(valid_returns.numel())
        invalid_return_count = int(invalid_returns.numel())
        valid_returns_finite = bool(torch.isfinite(valid_returns).all().item()) if valid_returns.numel() else True
        invalid_returns_nan = bool(torch.isnan(invalid_returns).all().item()) if invalid_returns.numel() else True
    return {
        "rows": len(_split_timestamps(split)),
        "valid_decision_rows": int(split.valid_start_indices.numel()),
        "feature_count": len(split.feature_names),
        "action_count": len(split.action_names),
        "features_finite": bool(torch.isfinite(split.features.detach().cpu()).all().item()),
        "valid_return_count": valid_return_count,
        "invalid_return_count": invalid_return_count,
        "valid_returns_finite": valid_returns_finite,
        "invalid_returns_nan": invalid_returns_nan,
        "has_action_valid_mask": action_mask is not None,
    }


def build_reportability_artifacts(
    *,
    args: argparse.Namespace,
    run_name: str,
    train_split,
    val_split,
    test_split,
    action_metadata: list[dict[str, object]],
    action_metadata_hash: str,
    action_risk_config_hash: str,
    constraints: object,
    exposure_constraints: object,
    baselines: dict[str, object],
    fixed_cost_stress: dict[str, object],
    adaptive_cost_stress: dict[str, object],
    test_metrics: dict[str, object],
    artifacts: dict[str, object],
    model_version: int,
    constraint_feature_names: list[str],
) -> dict[str, object]:
    created_at = utc_now_iso()
    dataset_manifest_path = args.dataset.parent / "dataset_manifest.json"
    dataset_manifest = read_json_if_exists(dataset_manifest_path) or {
        "dataset": str(args.dataset),
        "manifest_path": str(dataset_manifest_path),
        "manifest_available": False,
    }
    dataset_manifest_hash = stable_json_hash(dataset_manifest)
    feature_manifest = {
        "feature_set_id": f"hourly_features_{stable_json_hash(train_split.feature_names)[:12]}",
        "created_at_utc": created_at,
        "input_dataset": str(args.dataset),
        "input_dataset_manifest_hash": dataset_manifest_hash,
        "feature_names": train_split.feature_names,
        "feature_count": len(train_split.feature_names),
        "normalizer": {
            "fit_split": "train",
            "mean_hash": _tensor_hash(train_split.feature_mean),
            "std_hash": _tensor_hash(train_split.feature_std),
        },
        "constraint_feature_names": constraint_feature_names,
        "code_version": model_version,
    }
    data_quality_report = {
        "report_id": f"{run_name}_data_quality",
        "created_at_utc": created_at,
        "dataset": str(args.dataset),
        "splits": {
            "train": _split_quality(train_split),
            "val": _split_quality(val_split),
            "test": _split_quality(test_split),
        },
        "quality_score": 1.0
        if all(
            bool(_split_quality(split)["features_finite"])
            and bool(_split_quality(split)["valid_returns_finite"])
            and bool(_split_quality(split)["invalid_returns_nan"])
            for split in (train_split, val_split, test_split)
        )
        else 0.0,
    }
    action_mask = getattr(test_split, "action_valid_mask", None)
    action_eligibility = []
    for action_id, action_name in enumerate(test_split.action_names):
        valid_bar_count = (
            len(test_split.timestamps)
            if action_mask is None
            else int(action_mask[:, action_id].detach().cpu().bool().sum().item())
        )
        tradable = action_name == "CASH" or valid_bar_count > 0
        metadata = action_metadata[action_id]
        action_eligibility.append(
            {
                "symbol_id": action_name,
                "tradable": tradable,
                "reason_if_excluded": None if tradable else "no_valid_test_bars",
                "valid_bar_count": valid_bar_count,
                "asset_class": metadata["asset_class"],
                "risk_bucket": metadata["group"],
                "leverage_factor": metadata["leverage"],
                "inverse": metadata["inverse"],
            }
        )
    baseline_results = []
    for name, payload in sorted(baselines.items()):
        metric = _baseline_metric(payload)
        if metric is None:
            continue
        baseline_results.append(
            {
                "name": name,
                "total_return": metric.get("total_return"),
                "annualized_sharpe": metric.get("annualized_sharpe"),
                "max_drawdown": metric.get("max_drawdown"),
                "total_switches": metric.get("total_switches"),
            }
        )
    cost_stress_results = [
        {"name": name, "kind": "fixed_rollout", **dict(metric)}
        for name, metric in sorted(fixed_cost_stress.items())
        if isinstance(metric, dict)
    ] + [
        {"name": name, "kind": "adaptive_policy", **dict(metric)}
        for name, metric in sorted(adaptive_cost_stress.items())
        if isinstance(metric, dict)
    ]
    model_manifest = {
        "model_id": run_name,
        "created_at_utc": created_at,
        "model_version": model_version,
        "algorithm": "DoubleDQN",
        "encoder": "CausalTransformer",
        "training_dataset": str(args.dataset),
        "training_dataset_manifest_hash": dataset_manifest_hash,
        "validation_protocol": {
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_end": args.val_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "purge_rule": "chronological_no_overlap",
        },
        # Hash the TRAINING search space only. Excluding evaluation-window, reportability-threshold,
        # and infrastructure args means two runs that share a training configuration but evaluate on
        # different test windows / cost-stress grids get the SAME hyperparameters_hash, instead of
        # silently tying model identity to evaluation-only choices.
        "hyperparameters_hash": stable_json_hash(
            {
                key: value
                for key, value in vars(args).items()
                if key
                not in {
                    "dataset",
                    "output_dir",
                    "val_end",
                    "test_start",
                    "test_end",
                    "cost_stress_bps",
                    "reportable_max_group_share",
                    "reportable_max_leveraged_share",
                    "device",
                    "amp",
                    "target_vram_gb",
                }
            }
        ),
        "selected_by": "best_validation_total_return_then_switch_count",
        "feature_names_hash": hash_string_sequence(train_split.feature_names),
        "action_names_hash": hash_string_sequence(train_split.action_names),
        "action_metadata_hash": action_metadata_hash,
        "action_risk_config_hash": action_risk_config_hash,
        "constraints": asdict(constraints),
        "exposure_constraints": asdict(exposure_constraints),
        "training_artifacts": artifacts,
        "test_metrics": test_metrics,
        "baseline_results": baseline_results,
        "cost_stress_results": cost_stress_results,
        "frequency_stress_results": [
            {
                "name": f"base_{train_split.bar_interval}",
                "kind": "frequency",
                "bar_interval": train_split.bar_interval,
                "total_return": test_metrics.get("total_return"),
                "max_drawdown": test_metrics.get("max_drawdown"),
                "annualized_sharpe": test_metrics.get("annualized_sharpe"),
            }
        ],
    }
    return {
        "dataset_manifest": dataset_manifest,
        "feature_manifest": feature_manifest,
        "model_manifest": model_manifest,
        "data_quality_report": data_quality_report,
        "action_eligibility": action_eligibility,
    }


def main() -> int:
    args = parse_args()
    try:
        import torch

        from rl_quant.hourly_transformer import (
            HOURLY_CONSTRAINT_FEATURE_NAMES,
            HourlyEnvConfig,
            HourlyTransformerTrainingConfig,
            RISK_AWARE_POLICY_MODEL_VERSION,
            action_index,
            assert_matching_hourly_schema,
            build_hourly_splits,
            evaluate_hourly_policy,
            train_hourly_transformer_dqn,
        )
        from rl_quant.core import (
            DQNLearningConfig,
            configure_torch_runtime,
            require_min_free_vram,
            resolve_torch_device,
            torch_runtime_summary,
        )
        from rl_quant.decision_framework import validate_reportable_summary
        from rl_quant.action_risk import (
            action_concentration,
            action_metadata_to_dicts,
            action_weight_tensor,
            build_action_metadata,
            reportability_flags,
            rollout_return_diagnostics,
            stable_action_metadata_hash,
            stable_action_risk_config_hash,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Torch is required. Use: conda run -n ml1 python scripts/train_hourly_causal_transformer_rl.py"
            ) from exc
        raise

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    require_min_free_vram(device, args.min_free_vram_gb)
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
    exposure_constraints = build_exposure_constraints_from_args(args)
    action_meta = build_action_metadata(train_split.action_names)
    action_metadata = action_metadata_to_dicts(action_meta)
    action_metadata_hash = stable_action_metadata_hash(action_meta)
    action_risk_config_hash = stable_action_risk_config_hash(exposure_constraints)
    cpu_action_weights = action_weight_tensor(
        action_meta,
        device=torch.device("cpu"),
        max_effective_leverage=exposure_constraints.max_effective_leverage,
    )
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
        exposure_constraints=exposure_constraints,
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
        exposure_constraints=exposure_constraints,
        action_meta=action_meta,
        episode_length=args.episode_length,
    )
    val_result = evaluate_hourly_policy(
        val_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        switch_cost_bps=args.switch_cost_bps,
        constraints=constraints,
        exposure_constraints=exposure_constraints,
        action_meta=action_meta,
        episode_length=args.episode_length,
    )
    test_result = evaluate_hourly_policy(
        test_split.to(device),
        model,
        device=device,
        initial_action=initial_action,
        switch_cost_bps=args.switch_cost_bps,
        constraints=constraints,
        exposure_constraints=exposure_constraints,
        action_meta=action_meta,
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
            exposure_constraints=exposure_constraints,
            action_meta=action_meta,
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
                exposure_constraints=exposure_constraints,
                action_meta=action_meta,
                episode_length=args.episode_length,
            ).to_dict()
            for split in (train_split, val_split, test_split)
        }

    baselines: dict[str, object] = {"CASH": fixed_action_baseline(train_split.action_names[cash_index])}
    for action_name in ("QQQ", "SPY", "SOXL", "SOXS", "TQQQ", "SQQQ"):
        if action_name in train_split.action_names:
            baseline = fixed_action_baseline(action_name)
            action_id = action_index(train_split.action_names, action_name)
            suffix = "risk_scaled_constrained" if float(cpu_action_weights[action_id].item()) < 0.999 else "constrained"
            baselines[f"BuyAndHold_{action_name}_{suffix}"] = baseline
            baselines[f"BuyAndHold_{action_name}"] = baseline
    equal_weight_risk_scaled = {
        split.name: equal_weight_metrics(split, cash_index=cash_index, action_weights=cpu_action_weights)
        for split in (train_split, val_split, test_split)
    }
    baselines["EqualWeight_ETFs_risk_scaled_frictionless"] = equal_weight_risk_scaled
    baselines["EqualWeight_ETFs"] = equal_weight_risk_scaled
    baselines["RandomSameTurnover"] = {
        "test": random_same_turnover_baseline(
            test_split,
            test_result.rollout_records,
            seed=args.seed + 10_001,
            n_paths=args.random_baseline_paths,
            initial_action=initial_action,
            cash_index=cash_index,
            action_weights=cpu_action_weights,
            switch_cost_bps=args.switch_cost_bps,
            extra_switch_penalty_bps=args.extra_switch_penalty_bps,
            count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
            constraints=constraints,
            exposure_constraints=exposure_constraints,
            action_meta=action_meta,
            episode_length=args.episode_length,
        )
    }
    baselines["RandomSameActionDistribution"] = {
        "test": random_same_action_distribution_baseline(
            test_split,
            test_result.rollout_records,
            seed=args.seed + 20_001,
            n_paths=args.random_baseline_paths,
            initial_action=initial_action,
            cash_index=cash_index,
            action_weights=cpu_action_weights,
            switch_cost_bps=args.switch_cost_bps,
            extra_switch_penalty_bps=args.extra_switch_penalty_bps,
            count_etf_to_etf_as_two_legs=constraints.count_etf_to_etf_as_two_legs,
            constraints=constraints,
            exposure_constraints=exposure_constraints,
            action_meta=action_meta,
            episode_length=args.episode_length,
        )
    }
    concentration = action_concentration(test_result.rollout_records, action_meta=action_meta)
    return_diagnostics = rollout_return_diagnostics(test_result.rollout_records)
    reportability = reportability_flags(
        test_metrics=test_result.to_dict(),
        baselines=baselines,
        concentration=concentration,
        max_group_share=args.reportable_max_group_share,
        max_leveraged_share=args.reportable_max_leveraged_share,
    )

    run_name = args.run_name or f"{train_split.bar_interval}_causal_transformer_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model_version": RISK_AWARE_POLICY_MODEL_VERSION,
            "uses_constraint_features": True,
            "constraint_feature_names": HOURLY_CONSTRAINT_FEATURE_NAMES,
            "model_state_dict": model.state_dict(),
            "feature_mean": train_split.feature_mean.detach().cpu(),
            "feature_std": train_split.feature_std.detach().cpu(),
            "feature_names": train_split.feature_names,
            "action_names": train_split.action_names,
            "bar_interval": train_split.bar_interval,
            "periods_per_year": train_split.periods_per_year,
            "config": serializable_args,
            "constraints": asdict(constraints),
            "exposure_constraints": asdict(exposure_constraints),
            "action_metadata": action_metadata,
            "action_metadata_hash": action_metadata_hash,
            "action_risk_config_hash": action_risk_config_hash,
        },
        run_dir / "model.pt",
    )
    write_rollout(run_dir / "test_rollout.csv", test_result.rollout_records)
    write_decision_logs(run_dir / "decision_logs.jsonl", test_result.rollout_records)
    report_artifacts = build_reportability_artifacts(
        args=args,
        run_name=run_name,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        action_metadata=action_metadata,
        action_metadata_hash=action_metadata_hash,
        action_risk_config_hash=action_risk_config_hash,
        constraints=constraints,
        exposure_constraints=exposure_constraints,
        baselines=baselines,
        fixed_cost_stress=fixed_cost_stress,
        adaptive_cost_stress=adaptive_cost_stress,
        test_metrics=test_result.to_dict(),
        artifacts=artifacts,
        model_version=RISK_AWARE_POLICY_MODEL_VERSION,
        constraint_feature_names=HOURLY_CONSTRAINT_FEATURE_NAMES,
    )
    artifact_files = {
        "feature_manifest": run_dir / "feature_manifest.json",
        "model_manifest": run_dir / "model_manifest.json",
        "data_quality_report": run_dir / "data_quality_report.json",
        "action_eligibility": run_dir / "action_eligibility.json",
    }
    for artifact_name, artifact_path in artifact_files.items():
        with artifact_path.open("w") as sink:
            json.dump(report_artifacts[artifact_name], sink, indent=2)
    summary = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_runtime": runtime,
        "config": serializable_args,
        "constraints": asdict(constraints),
        "exposure_constraints": asdict(exposure_constraints),
        "bar_interval": train_split.bar_interval,
        "periods_per_year": train_split.periods_per_year,
        "feature_names": train_split.feature_names,
        "action_names": train_split.action_names,
        "action_metadata": action_metadata,
        "action_metadata_hash": action_metadata_hash,
        "action_risk_config_hash": action_risk_config_hash,
        **report_artifacts,
        "artifact_files": {name: str(path) for name, path in artifact_files.items()},
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
        "action_concentration": concentration,
        "return_diagnostics": return_diagnostics,
        "reportability": reportability,
    }
    reportability_errors = validate_reportable_summary(summary)
    summary["reportability"] = {
        "reportable": bool(reportability["reportable"]) and not reportability_errors,
        "reasons": list(dict.fromkeys([*reportability.get("reasons", []), *reportability_errors])),
    }
    with (run_dir / "summary.json").open("w") as sink:
        json.dump(summary, sink, indent=2)
    with (run_dir / "reportability.json").open("w") as sink:
        json.dump(summary["reportability"], sink, indent=2)

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
