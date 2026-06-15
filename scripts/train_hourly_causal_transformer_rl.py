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
    equity = 1.0
    equity_curve = [equity]
    returns: list[float] = []
    for index in split.valid_start_indices.detach().cpu().tolist():
        action_returns = split.action_returns[index, action_indices]
        if weights is not None:
            action_returns = action_returns * weights[action_indices]
        simple_return = float(action_returns.mean().item())
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
    previous_action = int(initial_action)
    previous_index: int | None = None
    for sequence_index, row_index in enumerate(valid_indices):
        if previous_index is None or row_index != previous_index + 1:
            previous_action = int(initial_action)
        valid_actions = _valid_action_indices(split, row_index, cash_index=cash_index)
        action = int(actions[sequence_index])
        if action not in valid_actions:
            action = int(cash_index)
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
        previous_action = action
        previous_index = row_index
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
    for _ in range(max(int(n_paths), 1)):
        sampled: list[int] = []
        previous_action = int(initial_action)
        previous_index: int | None = None
        for sequence_index, row_index in enumerate(valid_indices):
            if previous_index is None or row_index != previous_index + 1:
                previous_action = int(initial_action)
            valid_actions = _valid_action_indices(split, row_index, cash_index=cash_index)
            if switch_flags[sequence_index]:
                choices = [action for action in valid_actions if action != previous_action]
                action = rng.choice(choices or [cash_index])
            else:
                action = previous_action if previous_action in valid_actions else cash_index
            sampled.append(action)
            previous_action = action
            previous_index = row_index
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
        writer.writerows(records)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as sink:
        for record in records:
            sink.write(json.dumps(record, sort_keys=True) + "\n")


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
    write_jsonl(run_dir / "decision_logs.jsonl", test_result.rollout_records)
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
