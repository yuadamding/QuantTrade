"""Evaluation layer: second-context action-scorer / trading-policy evaluation + baselines + scoring utils (extracted from rl_quant.second_context_transformer, protocol-first reorg Phase 4; verbatim/byte-identical, see architecture_migration_plan.md)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn.functional as F

from rl_quant.models.second_context import SecondContextTransformerQNetwork
from rl_quant.core import autocast_context
from rl_quant.reportability import evaluate_decision_log_reportability
from rl_quant.datasets.second_context import SecondContextDataSplit, _parse_utc_timestamp


def _slice_to_device(tensor: torch.Tensor, indices: torch.Tensor, device: torch.device, *, pin_memory: bool) -> torch.Tensor:
    index = indices.to(tensor.device) if tensor.device.type != "cpu" else indices
    values = tensor[index]
    if values.device.type == "cpu" and device.type == "cuda" and pin_memory:
        values = values.pin_memory()
    return values.to(device, non_blocking=device.type == "cuda")


@torch.no_grad()
def predict_second_context_q_values(
    split: SecondContextDataSplit,
    model: SecondContextTransformerQNetwork,
    *,
    device: torch.device,
    batch_size: int | None = None,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    pin_memory: bool = True,
) -> torch.Tensor:
    rows = split.market_context.shape[0]
    if rows == 0:
        return torch.empty((0, len(split.action_names)), dtype=torch.float32)
    if batch_size is None:
        batch_size = rows
    if batch_size <= 0:
        raise ValueError("batch_size must be positive when supplied.")
    model.eval()
    outputs: list[torch.Tensor] = []
    for start in range(0, rows, int(batch_size)):
        indices = torch.arange(start, min(start + int(batch_size), rows), dtype=torch.long)
        market_context = _slice_to_device(split.market_context, indices, device, pin_memory=pin_memory)
        market_mask = _slice_to_device(split.market_context_mask, indices, device, pin_memory=pin_memory)
        action_features = _slice_to_device(split.action_features, indices, device, pin_memory=pin_memory)
        portfolio_state = _slice_to_device(split.portfolio_state, indices, device, pin_memory=pin_memory)
        constraint_state = _slice_to_device(split.constraint_state, indices, device, pin_memory=pin_memory)
        with autocast_context(device, use_amp, amp_dtype):
            q_values = model(market_context, market_mask, action_features, portfolio_state, constraint_state)
        outputs.append(q_values.detach().float().cpu())
    return torch.cat(outputs, dim=0)


def masked_contextual_q_loss(
    q_values: torch.Tensor,
    target_returns: torch.Tensor,
    action_valid_mask: torch.Tensor,
    *,
    action_cost_bps: torch.Tensor | None = None,
    action_target_weights: torch.Tensor | None = None,
    reward_scale: float = 10_000.0,
) -> torch.Tensor:
    targets = target_returns.float()
    if action_target_weights is not None:
        targets = targets * action_target_weights.float()
    if action_cost_bps is not None:
        costs = action_cost_bps.float() / 10_000.0
        if action_target_weights is not None:
            costs = costs * action_target_weights.float().abs()
        targets = targets - costs
    targets = targets * float(reward_scale)
    valid = action_valid_mask.bool()
    if not bool(valid.any().item()):
        raise ValueError("At least one valid action is required to compute loss.")
    return F.smooth_l1_loss(q_values[valid], targets[valid])


def _timestamp_ms_to_iso(timestamp_ms: int) -> str:
    if timestamp_ms < 0:
        return ""
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc).replace(microsecond=0).isoformat()


def _summarize_returns(returns: torch.Tensor, *, periods_per_year: float) -> dict[str, float | int | None]:
    finite = returns[torch.isfinite(returns)]
    total_return = float(torch.prod(1.0 + finite).item() - 1.0) if finite.numel() else 0.0
    avg = float(finite.mean().item()) if finite.numel() else 0.0
    std = float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0
    sharpe = None if std <= 0 else avg / std * (periods_per_year ** 0.5)
    return {
        "rows": int(finite.numel()),
        "total_return": total_return,
        "mean_return": avg,
        "sharpe": sharpe,
    }


def _trade_legs(previous_action: int, action: int) -> float:
    if action == previous_action:
        return 0.0
    return 1.0 if previous_action == 0 or action == 0 else 2.0


def _trade_notional(previous_executed_weight: float, target_weight: float) -> float:
    return abs(float(previous_executed_weight)) + abs(float(target_weight))


def _row_segment_id(split: SecondContextDataSplit, row: int) -> int:
    return int(split.segment_ids[row].item()) if 0 <= row < split.segment_ids.numel() else 0


def _path_reset_reason(split: SecondContextDataSplit, previous_row: int | None, row: int) -> str | None:
    if previous_row is None:
        return None
    if _row_segment_id(split, row) != _row_segment_id(split, previous_row):
        return "segment_change"
    if row != previous_row + 1:
        return "non_contiguous_row_index"
    if _parse_utc_timestamp(split.decision_timestamps[row]) != _parse_utc_timestamp(split.next_timestamps[previous_row]):
        return "timestamp_gap"
    return None


def _normalise_path_rows(
    split: SecondContextDataSplit,
    actions: torch.Tensor,
    row_indices: torch.Tensor | None,
    *,
    assume_prefix_rows: bool = False,
) -> torch.Tensor:
    if actions.ndim != 1:
        raise ValueError("actions must be a 1D tensor.")
    if row_indices is None:
        if not assume_prefix_rows:
            raise ValueError("row_indices are required unless assume_prefix_rows=True.")
        rows = torch.arange(actions.numel(), dtype=torch.long)
    else:
        rows = row_indices.detach().cpu().long().flatten()
    if rows.numel() != actions.numel():
        raise ValueError("row_indices length must match actions length.")
    if rows.numel() and bool(((rows < 0) | (rows >= len(split.decision_timestamps))).any().item()):
        raise ValueError("row_indices contains out-of-range rows.")
    return rows


def second_context_missing_label_report(
    split: SecondContextDataSplit,
    *,
    row_indices: torch.Tensor | None = None,
    selected_actions: torch.Tensor | None = None,
    cash_action_id: int = 0,
) -> dict[str, Any]:
    if row_indices is None:
        rows = split.valid_start_indices.detach().cpu().long().flatten()
    elif selected_actions is not None:
        rows = _normalise_path_rows(split, selected_actions.detach().cpu().long().flatten(), row_indices)
    else:
        rows = row_indices.detach().cpu().long().flatten()
        if rows.numel() and bool(((rows < 0) | (rows >= len(split.decision_timestamps))).any().item()):
            raise ValueError("row_indices contains out-of-range rows.")
    if rows.numel() == 0:
        return {
            "evaluated_rows": 0,
            "selectable_action_count": 0,
            "non_cash_selectable_action_count": 0,
            "selectable_missing_label_count": 0,
            "selectable_missing_label_fraction": 0.0,
            "non_cash_selectable_missing_label_fraction": 0.0,
            "rows_with_any_selectable_missing_label": 0,
            "selected_action_missing_label_count": 0 if selected_actions is not None else None,
            "policy_unscorable_rows": 0,
            "evaluation_reportable": True,
            "reportability_errors": [],
        }
    decision_valid = split.action_valid_mask[rows].detach().cpu().bool()
    if split.label_valid_mask is None:
        label_valid = decision_valid.clone()
    else:
        label_valid = split.label_valid_mask[rows].detach().cpu().bool()
    finite_returns = torch.isfinite(split.action_returns[rows].detach().cpu())
    label_evaluable = label_valid & finite_returns
    selectable_missing = decision_valid & ~label_evaluable
    if 0 <= cash_action_id < selectable_missing.shape[1]:
        selectable_missing[:, cash_action_id] = False
    selectable_count = int(decision_valid.sum().item())
    if decision_valid.shape[1] > 1:
        non_cash_selectable_count = int(decision_valid[:, 1:].sum().item())
    else:
        non_cash_selectable_count = 0
    missing_count = int(selectable_missing.sum().item())
    rows_with_missing = int(selectable_missing.any(dim=1).sum().item())
    selected_missing_count: int | None = None
    if selected_actions is not None:
        actions = selected_actions.detach().cpu().long().flatten()
        if actions.numel() != rows.numel():
            raise ValueError("selected_actions length must match row_indices length.")
        selected_missing_count = 0
        for position, action_value in enumerate(actions.tolist()):
            action = int(action_value)
            if action == cash_action_id or action < 0 or action >= decision_valid.shape[1]:
                continue
            if bool(decision_valid[position, action].item()) and not bool(label_evaluable[position, action].item()):
                selected_missing_count += 1
    policy_unscorable_rows = selected_missing_count if selected_missing_count is not None else rows_with_missing
    errors = ["selectable_actions_with_missing_reward_labels"] if missing_count > 0 else []
    return {
        "evaluated_rows": int(rows.numel()),
        "selectable_action_count": selectable_count,
        "non_cash_selectable_action_count": non_cash_selectable_count,
        "selectable_missing_label_count": missing_count,
        "selectable_missing_label_fraction": missing_count / float(max(selectable_count, 1)),
        "non_cash_selectable_missing_label_fraction": missing_count / float(max(non_cash_selectable_count, 1)),
        "rows_with_any_selectable_missing_label": rows_with_missing,
        "selected_action_missing_label_count": selected_missing_count,
        "policy_unscorable_rows": int(policy_unscorable_rows or 0),
        "evaluation_reportable": missing_count == 0,
        "reportability_errors": errors,
    }


def _evaluate_action_path(
    split: SecondContextDataSplit,
    actions: torch.Tensor,
    *,
    row_indices: torch.Tensor | None = None,
    q_values: torch.Tensor | None = None,
    cost_bps_override: float | None = None,
    initial_action: int = 0,
    liquidate_at_end: bool = False,
    return_decision_logs: bool = False,
    assume_prefix_rows: bool = False,
) -> dict[str, Any]:
    actions = actions.detach().cpu().long().flatten()
    rows = _normalise_path_rows(split, actions, row_indices, assume_prefix_rows=assume_prefix_rows)
    if q_values is not None and q_values.shape[0] != actions.numel():
        raise ValueError("q_values first dimension must match actions length.")
    previous_action = int(initial_action)
    previous_executed_weight = 0.0
    net_returns: list[float] = []
    active_net_returns: list[float] = []
    active_gross_returns: list[float] = []
    decision_logs: list[dict[str, Any]] = []
    switches = 0
    cash_actions = 0
    equity = 1.0
    previous_row: int | None = None
    segment_resets = 0
    gap_resets = 0
    invalid_action_attempts = 0
    fallback_to_cash_count = 0
    fallback_due_to_missing_label_count = 0
    rows_with_no_valid_action = 0
    for position, row in enumerate(rows.tolist()):
        reset_reason = _path_reset_reason(split, previous_row, row)
        if reset_reason is not None:
            previous_action = int(initial_action)
            previous_executed_weight = 0.0
            if reset_reason == "segment_change":
                segment_resets += 1
            else:
                gap_resets += 1
        requested_action = int(actions[position].item())
        action = requested_action
        row_valid = split.action_valid_mask[row].bool()
        row_label_valid = (
            split.label_valid_mask[row].bool()
            if split.label_valid_mask is not None
            else torch.isfinite(split.action_returns[row])
        )
        requested_missing_label = (
            0 < requested_action < len(split.action_names)
            and bool(row_valid[requested_action].item())
            and (
                not bool(row_label_valid[requested_action].item())
                or not bool(torch.isfinite(split.action_returns[row, requested_action]).item())
            )
        )
        fallback_due_to_missing_label = False
        if not bool(row_valid.any().item()):
            rows_with_no_valid_action += 1
            invalid_action_attempts += 1
            fallback_to_cash_count += 1
            action = 0
        elif action < 0 or action >= len(split.action_names) or not bool(row_valid[action].item()):
            invalid_action_attempts += 1
            fallback_to_cash_count += 1
            action = 0
        gross_raw = float(split.action_returns[row, action].item())
        action_missing_label = (
            action != 0
            and (
                not bool(row_label_valid[action].item())
                or not bool(torch.isfinite(split.action_returns[row, action]).item())
            )
        )
        if action_missing_label or not torch.isfinite(split.action_returns[row, action]).item():
            invalid_action_attempts += int(action == requested_action)
            fallback_to_cash_count += 1
            fallback_due_to_missing_label = bool(action_missing_label)
            fallback_due_to_missing_label_count += int(fallback_due_to_missing_label)
            action = 0
            gross_raw = 0.0
        row_weights = split.action_target_weights[row].detach().cpu()
        target_weight = float(row_weights[action].item())
        previous_weight_before = float(previous_executed_weight)
        if action != previous_action:
            executed_weight = target_weight
        elif previous_row is None and previous_action != 0 and abs(previous_executed_weight) <= 1e-12:
            executed_weight = target_weight
        elif action == 0:
            executed_weight = 0.0
        else:
            executed_weight = previous_executed_weight
        gross = gross_raw * executed_weight
        legs = _trade_legs(previous_action, action)
        traded_notional = _trade_notional(previous_weight_before, target_weight) if legs > 0 else 0.0
        if legs > 0:
            previous_cost_bps = (
                float(split.action_cost_bps[row, previous_action].item())
                if 0 <= previous_action < split.action_cost_bps.shape[1]
                else 0.0
            )
            selected_cost_bps = float(split.action_cost_bps[row, action].item())
            if cost_bps_override is not None:
                cost_bps = float(cost_bps_override)
                cost = traded_notional * cost_bps / 10_000.0
            elif previous_action == 0:
                cost_bps = selected_cost_bps
                cost = abs(target_weight) * selected_cost_bps / 10_000.0
            elif action == 0:
                cost_bps = previous_cost_bps
                cost = abs(previous_weight_before) * previous_cost_bps / 10_000.0
            else:
                total_cost_bps_notional = (
                    abs(previous_weight_before) * previous_cost_bps + abs(target_weight) * selected_cost_bps
                )
                cost_bps = total_cost_bps_notional / traded_notional if traded_notional > 0 else 0.0
                cost = total_cost_bps_notional / 10_000.0
            switches += 1
        else:
            cost_bps = 0.0
            cost = 0.0
        net = gross - cost
        equity *= 1.0 + net
        net_returns.append(net)
        if action != 0 and abs(executed_weight) > 1e-12:
            active_gross_returns.append(gross)
            active_net_returns.append(net)
        cash_actions += int(action == 0)
        if return_decision_logs:
            q_row = None if q_values is None else q_values[position].detach().cpu()
            q_map = None
            q_edge_vs_cash = None
            q_edge_vs_current = None
            if q_row is not None:
                q_map = {name: float(q_row[index].item()) for index, name in enumerate(split.action_names)}
                q_edge_vs_cash = float(q_row[action].item() - q_row[0].item())
                q_edge_vs_current = float(q_row[action].item() - q_row[previous_action].item())
            valid_mask = split.action_valid_mask[row].detach().cpu()
            label_mask = (
                split.label_valid_mask[row].detach().cpu()
                if split.label_valid_mask is not None
                else valid_mask
            )
            context_available_until = int(split.market_context_available_timestamps_ms[row].max().item())
            decision_logs.append(
                {
                    "decision_ts": split.decision_timestamps[row],
                    "source_row": row,
                    "context_available_until": _timestamp_ms_to_iso(context_available_until),
                    "entry_execution_ts": _timestamp_ms_to_iso(int(split.entry_execution_timestamps_ms[row, action].item())),
                    "reward_end_ts": split.next_timestamps[row],
                    "exit_execution_ts": _timestamp_ms_to_iso(int(split.exit_execution_timestamps_ms[row, action].item())),
                    "entry_price_source": split.entry_price_source,
                    "exit_price_source": split.exit_price_source,
                    "execution_model": split.execution_model,
                    "same_action_weight_policy": "freeze_executed_weight_until_action_change",
                    "previous_action": split.action_names[previous_action],
                    "selected_action": split.action_names[action],
                    "requested_action": (
                        split.action_names[requested_action]
                        if 0 <= requested_action < len(split.action_names)
                        else str(requested_action)
                    ),
                    "path_reset_reason": reset_reason,
                    "selected_action_missing_label": requested_missing_label,
                    "fallback_due_to_missing_label": fallback_due_to_missing_label,
                    "target_weight": target_weight,
                    "previous_executed_weight": previous_weight_before,
                    "executed_weight": executed_weight,
                    "order_legs": legs,
                    "traded_notional": traded_notional,
                    "q_values": q_map,
                    "q_edge_vs_cash": q_edge_vs_cash,
                    "q_edge_vs_current": q_edge_vs_current,
                    "action_mask": {name: bool(valid_mask[index].item()) for index, name in enumerate(split.action_names)},
                    "label_mask": {name: bool(label_mask[index].item()) for index, name in enumerate(split.action_names)},
                    "mask_reasons": {
                        name: (
                            None
                            if bool(valid_mask[index].item()) and bool(label_mask[index].item())
                            else (
                                "decision_action_invalid"
                                if not bool(valid_mask[index].item())
                                else "missing_realized_return_label"
                            )
                        )
                        for index, name in enumerate(split.action_names)
                    },
                    "data_quality_score": float(split.constraint_state[row, 0].item()) if split.constraint_state.shape[1] else None,
                    "readiness_score": float(split.constraint_state[row, 0].item()) if split.constraint_state.shape[1] else None,
                    "entry_price": None,
                    "exit_price": None,
                    "gross_return": gross,
                    "cost_bps": cost_bps,
                    "net_return": net,
                    "equity_after": equity,
                }
            )
        previous_action = action
        previous_executed_weight = float(executed_weight)
        previous_row = row
    final_action = previous_action
    final_executed_weight = float(previous_executed_weight)
    terminal_liquidation_cost = 0.0
    terminal_liquidation_cost_bps = 0.0
    terminal_liquidation_order_legs = 0.0
    if liquidate_at_end and final_action != 0 and abs(final_executed_weight) > 1e-12 and net_returns:
        last_row = int(rows[-1].item()) if rows.numel() else min(actions.numel() - 1, split.action_cost_bps.shape[0] - 1)
        terminal_liquidation_cost_bps = (
            float(cost_bps_override)
            if cost_bps_override is not None
            else float(split.action_cost_bps[last_row, final_action].item())
        )
        terminal_liquidation_cost = abs(final_executed_weight) * terminal_liquidation_cost_bps / 10_000.0
        terminal_liquidation_order_legs = 1.0
        equity *= 1.0 - terminal_liquidation_cost
        net_returns.append(-terminal_liquidation_cost)
        final_executed_weight = 0.0
    returns = torch.tensor(net_returns, dtype=torch.float32)
    metrics = _summarize_returns(returns, periods_per_year=split.periods_per_year)
    active_returns = torch.tensor(active_net_returns, dtype=torch.float32)
    active_gross = torch.tensor(active_gross_returns, dtype=torch.float32)
    active_metrics = _summarize_returns(active_returns, periods_per_year=split.periods_per_year)
    metrics.update(second_context_missing_label_report(split, row_indices=rows, selected_actions=actions))
    metrics.update(
        {
            "cash_action_share": cash_actions / len(net_returns) if net_returns else 1.0,
            "switches": switches,
            "switch_rate": switches / len(net_returns) if net_returns else 0.0,
            "final_action": split.action_names[final_action] if 0 <= final_action < len(split.action_names) else None,
            "final_position_open": bool(final_action != 0 and abs(final_executed_weight) > 1e-12 and not liquidate_at_end),
            "final_executed_weight": final_executed_weight,
            "liquidate_at_end": bool(liquidate_at_end),
            "terminal_liquidation_cost": terminal_liquidation_cost,
            "terminal_liquidation_cost_bps": terminal_liquidation_cost_bps,
            "terminal_liquidation_order_legs": terminal_liquidation_order_legs,
            "same_action_weight_policy": "freeze_executed_weight_until_action_change",
            "selected_action_semantics": "executed_actions_after_missing_label_fallback",
            "requested_action_semantics": "policy_actions_after_min_hold_and_cooldown_constraints",
            "segment_resets": segment_resets,
            "gap_resets": gap_resets,
            "path_state_resets": segment_resets + gap_resets,
            "invalid_action_attempts": invalid_action_attempts,
            "fallback_to_cash_count": fallback_to_cash_count,
            "fallback_due_to_missing_label_count": fallback_due_to_missing_label_count,
            "rows_with_no_valid_action": rows_with_no_valid_action,
            "evaluated_rows": int(actions.numel()),
            "warnings": (
                ["final_position_open"] if final_action != 0 and abs(final_executed_weight) > 1e-12 and not liquidate_at_end else []
            ),
            "active_window_diagnostics": {
                "active_bars": int(active_returns.numel()),
                "cash_bars": int(len(net_returns) - active_returns.numel()),
                "active_gross_return": float(torch.prod(1.0 + active_gross).item() - 1.0) if active_gross.numel() else 0.0,
                "active_net_return": active_metrics["total_return"],
                "active_mean_return": active_metrics["mean_return"],
            },
        }
    )
    if return_decision_logs:
        metrics["decision_logs"] = decision_logs
    return metrics


@torch.no_grad()
def evaluate_second_context_action_scorer(
    split: SecondContextDataSplit,
    model: SecondContextTransformerQNetwork,
    *,
    device: torch.device,
    reward_scale: float = 10_000.0,
    batch_size: int | None = None,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    pin_memory: bool = True,
    evaluate_all_rows: bool = False,
) -> dict[str, float | int | None]:
    q_values = predict_second_context_q_values(
        split,
        model,
        device=device,
        batch_size=batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        pin_memory=pin_memory,
    )
    rows = (
        torch.arange(split.action_returns.shape[0], dtype=torch.long)
        if evaluate_all_rows
        else split.valid_start_indices.detach().cpu().long()
    )
    if rows.numel() == 0:
        metrics = _summarize_returns(torch.empty(0, dtype=torch.float32), periods_per_year=split.periods_per_year)
        metrics.update(second_context_missing_label_report(split, row_indices=rows))
        metrics.update(
            {
                "cash_action_share": 1.0,
                "diagnostic_only": True,
                "cost_model": "rowwise_from_cash_cost_each_decision",
                "diagnostic_rows": "all_rows" if evaluate_all_rows else "valid_start_indices",
                "evaluated_rows": 0,
            }
        )
        return metrics
    q_rows = q_values[rows]
    valid_rows = split.action_valid_mask[rows]
    masked_q = q_rows.masked_fill(~valid_rows, torch.finfo(q_rows.dtype).min)
    actions = masked_q.argmax(dim=1)
    rewards = split.action_returns[rows, actions]
    weights = split.action_target_weights[rows, actions]
    costs = split.action_cost_bps[rows, actions] / 10_000.0 * weights.abs()
    net_returns = (rewards * weights - costs).detach().cpu()
    if split.label_valid_mask is not None:
        selected_label_valid = split.label_valid_mask[rows, actions].detach().cpu().bool()
        cash_selected = actions.detach().cpu() == 0
        net_returns = net_returns.masked_fill(~(selected_label_valid | cash_selected), float("nan"))
    finite = torch.isfinite(net_returns)
    net_returns = net_returns[finite]
    metrics = _summarize_returns(net_returns, periods_per_year=split.periods_per_year)
    metrics.update(second_context_missing_label_report(split, row_indices=rows, selected_actions=actions))
    cash_share = float((actions.detach().cpu() == 0).float().mean().item()) if actions.numel() else 1.0
    metrics.update(
        {
            "cash_action_share": cash_share,
            "diagnostic_only": True,
            "cost_model": "rowwise_from_cash_cost_each_decision",
            "diagnostic_rows": "all_rows" if evaluate_all_rows else "valid_start_indices",
            "evaluated_rows": int(rows.numel()),
        }
    )
    return metrics


@torch.no_grad()
def evaluate_second_context_trading_policy(
    split: SecondContextDataSplit,
    model: SecondContextTransformerQNetwork,
    *,
    device: torch.device,
    reward_scale: float = 10_000.0,
    initial_action: int = 0,
    min_hold_bars: int = 1,
    cooldown_bars: int = 0,
    liquidate_at_end: bool = False,
    return_decision_logs: bool = False,
    return_selected_actions: bool = False,
    batch_size: int | None = None,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    pin_memory: bool = True,
) -> dict[str, float | int | None]:
    q_values = predict_second_context_q_values(
        split,
        model,
        device=device,
        batch_size=batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        pin_memory=pin_memory,
    )
    previous_action = int(initial_action)
    bars_held = int(max(min_hold_bars, 1))
    cooldown_remaining = 0
    raw_policy_actions: list[int] = []
    requested_actions: list[int] = []
    executed_actions: list[int] = []
    constraint_adjusted_actions: list[int] = []
    selection_reasons: list[str] = []
    selected_rows: list[int] = []
    previous_row: int | None = None
    for row in split.valid_start_indices.detach().cpu().tolist():
        reset_reason = _path_reset_reason(split, previous_row, int(row))
        if reset_reason is not None:
            previous_action = int(initial_action)
            bars_held = int(max(min_hold_bars, 1))
            cooldown_remaining = 0
        decision_valid = split.action_valid_mask[row].clone()
        if not bool(decision_valid.any().item()):
            previous_action = int(initial_action)
            bars_held = int(max(min_hold_bars, 1))
            cooldown_remaining = 0
            previous_row = None
            continue
        raw_masked_q = q_values[row].masked_fill(~decision_valid, torch.finfo(q_values.dtype).min)
        raw_policy_action = int(raw_masked_q.argmax().item())
        valid = decision_valid.clone()
        if min_hold_bars > 1 and bars_held < min_hold_bars and previous_action < valid.numel():
            hold_only = torch.zeros_like(valid)
            hold_only[previous_action] = valid[previous_action]
            valid = hold_only if bool(hold_only.any().item()) else valid
        if cooldown_remaining > 0 and previous_action < valid.numel():
            hold_only = torch.zeros_like(valid)
            hold_only[previous_action] = valid[previous_action]
            valid = hold_only if bool(hold_only.any().item()) else valid
        masked_q = q_values[row].masked_fill(~valid, torch.finfo(q_values.dtype).min)
        requested_action = int(masked_q.argmax().item())
        executed_action = requested_action
        row_label_valid = (
            split.label_valid_mask[row].bool()
            if split.label_valid_mask is not None
            else torch.isfinite(split.action_returns[row])
        )
        if requested_action != 0 and (
            not bool(row_label_valid[requested_action].item())
            or not bool(torch.isfinite(split.action_returns[row, requested_action]).item())
        ):
            executed_action = 0
        if executed_action != requested_action:
            if requested_action != raw_policy_action:
                selection_reason = "constraint_adjusted_then_fallback_due_to_missing_label"
            else:
                selection_reason = "fallback_due_to_missing_label"
        elif requested_action != raw_policy_action:
            selection_reason = "constraint_adjusted"
        else:
            selection_reason = "selected_by_policy"
        if executed_action != previous_action:
            bars_held = 0
            cooldown_remaining = int(cooldown_bars)
        else:
            bars_held += 1
            cooldown_remaining = max(0, cooldown_remaining - 1)
        previous_action = executed_action
        raw_policy_actions.append(raw_policy_action)
        requested_actions.append(requested_action)
        executed_actions.append(executed_action)
        constraint_adjusted_actions.append(requested_action)
        selection_reasons.append(selection_reason)
        selected_rows.append(int(row))
        previous_row = int(row)
    selected_rows_tensor = torch.tensor(selected_rows, dtype=torch.long)
    metrics = _evaluate_action_path(
        split,
        torch.tensor(requested_actions, dtype=torch.long),
        row_indices=selected_rows_tensor,
        q_values=q_values[selected_rows_tensor] if selected_rows else q_values[:0],
        initial_action=initial_action,
        liquidate_at_end=liquidate_at_end,
        return_decision_logs=return_decision_logs,
    )
    metrics.update({"diagnostic_only": False, "cost_model": "sequential_switch_only_cost"})
    # Reportability LABEL (no P&L movement): this path prices fills from the close + an estimated cost bps,
    # so it is NOT a crossable quote fill model and cannot claim real executable trading. Stamp the honest
    # verdict, computed from the decision logs via the shared gate when available (the rows lack crossable /
    # real-fill fields, so the verdict is False with informative reasons) -- never overstating it as real.
    decision_logs = metrics.get("decision_logs")
    if decision_logs is not None:
        verdict = evaluate_decision_log_reportability(decision_logs, require_real_executable=True)
        base_reportable = verdict.base_reportable
        real_reportable = verdict.real_executable_trade_reportable
        reasons: tuple[str, ...] = verdict.missing_reportability_reasons  # already distinct "category:field" tokens
    else:
        base_reportable = False
        real_reportable = False
        reasons = ("missing:decision_logs",)
    # Expose BOTH tiers: mechanical (causal/complete/coherent) vs strict real-executable. A causal close-only
    # backtest is mechanically reportable but not real-executable; a log that fails even the base tier (e.g.
    # non-causal timestamps) is a non-reportable diagnostic.
    metrics["mechanically_reportable"] = bool(base_reportable)
    metrics["real_executable_trade_reportable"] = bool(real_reportable)
    metrics["sequential_evaluation_type"] = (
        "real_executable"
        if real_reportable
        else "close_based_research_backtest"
        if base_reportable
        else "non_reportable_research_diagnostic"
    )
    metrics["missing_reportability_reasons"] = reasons
    if return_selected_actions:
        metrics["selected_actions"] = executed_actions
        metrics["executed_actions"] = executed_actions
        metrics["requested_actions"] = requested_actions
        metrics["raw_policy_actions"] = raw_policy_actions
        metrics["constraint_adjusted_actions"] = constraint_adjusted_actions
        metrics["selection_reasons"] = selection_reasons
        metrics["selected_rows"] = selected_rows
    return metrics


def fixed_rollout_cost_stress(
    split: SecondContextDataSplit,
    actions: torch.Tensor,
    *,
    row_indices: torch.Tensor | None = None,
    cost_bps_values: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    initial_action: int = 0,
    liquidate_at_end: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        f"{cost:g}_bps": _evaluate_action_path(
            split,
            actions.long(),
            row_indices=row_indices,
            assume_prefix_rows=row_indices is None,
            cost_bps_override=cost,
            initial_action=initial_action,
            liquidate_at_end=liquidate_at_end,
        )
        for cost in cost_bps_values
    }


def evaluate_second_context_baselines(
    split: SecondContextDataSplit,
    *,
    reference_actions: torch.Tensor | None = None,
    seed: int = 17,
) -> dict[str, dict[str, Any]]:
    row_indices = split.valid_start_indices.detach().cpu().long()
    row_count = int(row_indices.numel())
    baselines: dict[str, dict[str, Any]] = {
        "CASH": _evaluate_action_path(split, torch.zeros(row_count, dtype=torch.long), row_indices=row_indices),
        "PreviousActionNoTrade": _evaluate_action_path(
            split,
            torch.zeros(row_count, dtype=torch.long),
            row_indices=row_indices,
        ),
    }
    for name in ("QQQ", "SPY"):
        if name in split.action_names:
            action_index = split.action_names.index(name)
            baselines[f"BuyAndHold_{name}"] = _evaluate_action_path(
                split,
                torch.full((row_count,), action_index, dtype=torch.long),
                row_indices=row_indices,
            )
    generator = torch.Generator().manual_seed(seed)
    if reference_actions is None:
        valid_counts = split.action_valid_mask.float().sum(dim=0)
        distribution = valid_counts / valid_counts.sum().clamp_min(1.0)
        switch_count = max(0, row_count // 10)
    else:
        reference_actions = reference_actions.long().cpu()
        distribution = torch.bincount(reference_actions.clamp_min(0), minlength=len(split.action_names)).float()
        distribution = distribution / distribution.sum().clamp_min(1.0)
        switch_count = int((reference_actions[1:] != reference_actions[:-1]).sum().item()) if reference_actions.numel() > 1 else 0

    def sample_valid_action(source_row: int, *, exclude_action: int | None = None) -> int:
        valid = split.action_valid_mask[source_row].detach().cpu().bool().clone()
        if exclude_action is not None and 0 <= exclude_action < valid.numel():
            valid[exclude_action] = False
        if not bool(valid.any().item()):
            return 0
        weights = distribution[: valid.numel()].clone()
        weights = weights.masked_fill(~valid, 0.0)
        if float(weights.sum().item()) <= 0.0:
            candidates = torch.nonzero(valid, as_tuple=False).flatten()
            return int(candidates[torch.randint(candidates.numel(), (1,), generator=generator)].item())
        weights = weights / weights.sum().clamp_min(1e-12)
        return int(torch.multinomial(weights, 1, replacement=True, generator=generator).item())

    sampled = torch.tensor([sample_valid_action(int(row)) for row in row_indices.tolist()], dtype=torch.long)
    baselines["RandomSameActionDistribution"] = _evaluate_action_path(split, sampled, row_indices=row_indices)
    turnover_actions = torch.zeros(row_count, dtype=torch.long)
    if row_count:
        current = sample_valid_action(int(row_indices[0].item()))
        turnover_actions[0] = current
        switch_positions = (
            torch.randperm(max(row_count - 1, 0), generator=generator)[:switch_count].add(1).sort().values.tolist()
        )
        switch_set = set(int(value) for value in switch_positions)
        for index in range(1, row_count):
            if index in switch_set:
                current = sample_valid_action(int(row_indices[index].item()), exclude_action=current)
            turnover_actions[index] = current
    baselines["RandomSameTurnover"] = _evaluate_action_path(split, turnover_actions, row_indices=row_indices)
    if reference_actions is not None and reference_actions.numel() == row_count:
        switch_positions = [
            index
            for index in range(1, row_count)
            if int(reference_actions[index].item()) != int(reference_actions[index - 1].item())
        ]
        same_timing = torch.zeros(row_count, dtype=torch.long)
        if row_count:
            current = sample_valid_action(int(row_indices[0].item()))
            same_timing[0] = current
            for index in range(1, row_count):
                if index in switch_positions:
                    current = sample_valid_action(int(row_indices[index].item()), exclude_action=current)
                same_timing[index] = current
        baselines["RandomSameTurnoverSameTiming"] = _evaluate_action_path(split, same_timing, row_indices=row_indices)

        same_segments = torch.zeros(row_count, dtype=torch.long)
        if row_count:
            segment_starts = [0, *switch_positions]
            segment_ends = [*switch_positions, row_count]
            for start, end in zip(segment_starts, segment_ends):
                valid = split.action_valid_mask[row_indices[start:end]].all(dim=0)
                candidates = torch.nonzero(valid, as_tuple=False).flatten()
                if candidates.numel():
                    chosen = int(candidates[torch.randint(candidates.numel(), (1,), generator=generator)].item())
                else:
                    chosen = 0
                same_segments[start:end] = chosen
        baselines["RandomSameSegments"] = _evaluate_action_path(split, same_segments, row_indices=row_indices)
    return baselines


def evaluate_second_context_policy(
    split: SecondContextDataSplit,
    model: SecondContextTransformerQNetwork,
    *,
    device: torch.device,
    reward_scale: float = 10_000.0,
) -> dict[str, float | int | None]:
    return evaluate_second_context_action_scorer(split, model, device=device, reward_scale=reward_scale)
