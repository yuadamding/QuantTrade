"""Frozen-policy and constraint-matched baseline evaluation through one ledger."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import math
from statistics import mean, stdev

import torch

from rl_quant.datasets.massive_raw_seconds_v1 import RawSecondCatalog
from rl_quant.datasets.raw_second_economics_v1 import SecondEconomicInputs
from rl_quant.envs.raw_second_portfolio_v1 import RawSecondPortfolioEnv, SecondExecutionConfig, _session
from rl_quant.evaluation.raw_second_policy_v1 import FrozenRawSecondPolicy
from rl_quant.rl.types import ActionBatch

D = Decimal


def economic_summary(env: RawSecondPortfolioEnv) -> dict:
    """Reconcile exact ledger quantities; use daily, not second, risk sampling."""
    initial = D(env.config.capital)
    prior, log_return = initial, 0.0
    daily = {}
    peak, drawdown = initial, D(0)
    gross = sum((abs(D(f.signed_shares)) * D(f.price) for f in env.fills), D(0))
    fees = sum((D(f.fee) for f in env.fills), D(0))
    terminal_fees = sum((D(r["terminal_liquidation_cost"]) for r in env.audit), D(0))
    terminal_notional = sum((D(r["terminal_liquidation_notional"]) for r in env.audit), D(0))
    for row in env.audit:
        before, after = D(row["equity_before"]), D(row["equity_after"])
        if before != prior or after <= 0 or not math.isclose(row["reward_net_log_equity"], math.log(float(after / before)), abs_tol=1e-12):
            raise ValueError("Equity/reward ledger does not reconcile")
        marked = sum((D(q) * D(row["marks"][a]) for a, q in row["holdings"]), D(0))
        receivables = sum((D(r["amount"]) for r in row["receivables"]), D(0))
        if not math.isclose(float(D(row["cash"]) + marked + receivables), float(after), rel_tol=1e-12):
            raise ValueError("Cash, shares, receivables and equity differ")
        prior = after
        log_return += row["reward_net_log_equity"]
        peak = max(peak, after)
        drawdown = max(drawdown, 1 - after / peak)
        daily[_session(row["end_ms"])] = after
    if not env.audit or prior != env.current_equity or not math.isclose(math.exp(log_return), float(prior / initial), rel_tol=1e-10):
        raise ValueError("Terminal wealth and cumulative log rewards differ")
    daily_returns, previous = [], initial
    for value in daily.values():
        daily_returns.append(float(value / previous - 1))
        previous = value
    volatility = stdev(daily_returns) * math.sqrt(252) if len(daily_returns) > 1 else None
    sharpe = mean(daily_returns) * 252 / volatility if volatility else None
    exposures = [float(D(r["risky_marked_notional"]) / D(r["pre_terminal_equity"])) for r in env.audit]
    cash_allocations = [float(D(r["pre_terminal_cash"]) / D(r["pre_terminal_equity"])) for r in env.audit]
    entry_requested = D(env.audit[0]["requested_notional"])
    entry_filled = D(env.audit[0]["filled_order_notional_at_decision_marks"])
    if entry_filled < 0 or entry_filled > entry_requested:
        raise ValueError("Initial order completion does not reconcile with fills")
    return dict(net_return=float(prior / initial - 1), terminal_equity=str(prior),
                cumulative_net_log_return=log_return, maximum_drawdown=float(drawdown),
                daily_equity={k: str(v) for k, v in daily.items()}, daily_returns=daily_returns,
                annualized_daily_volatility=volatility, daily_net_sharpe=sharpe,
                risk_sampling="last-scored-equity-per-Eastern-session;partial-boundary-days-included",
                gross_filled_notional=str(gross), terminal_mark_liquidation_notional=str(terminal_notional),
                turnover=float((gross + terminal_notional) / initial), turnover_definition="one-way-traded-plus-terminal-notional/initial-capital",
                fill_costs=str(fees), terminal_liquidation_costs=str(terminal_fees), total_costs=str(fees + terminal_fees),
                fill_count=len(env.fills), buy_fill_count=sum(D(f.signed_shares) > 0 for f in env.fills),
                sell_fill_count=sum(D(f.signed_shares) < 0 for f in env.fills),
                requested_notional=str(sum((D(r["requested_notional"]) for r in env.audit), D(0))),
                unfilled_order_intervals=sum(any(D(q) for q in r["unfilled_shares"].values()) for r in env.audit),
                average_risky_exposure=mean(exposures), maximum_risky_exposure=max(exposures),
                average_cash_allocation=mean(cash_allocations), maximum_cash_allocation=max(cash_allocations),
                exposure_sampling="unweighted-decision-interval-end;before-terminal-mark-liquidation;not-time-weighted",
                initial_entry_requested_notional=str(entry_requested),
                initial_entry_filled_notional_at_decision_marks=str(entry_filled),
                initial_entry_completion_fraction=float(entry_filled / entry_requested) if entry_requested else None,
                ending_cash=str(env.book.cash), ending_receivables=str(sum((r.amount for r in env.book.receivables), D(0))))


def evaluate_raw_second_policy(*, catalog: RawSecondCatalog, economic_inputs: SecondEconomicInputs,
                               execution_config: SecondExecutionConfig, device: str | torch.device,
                               policy: FrozenRawSecondPolicy | None = None, baseline: str | None = None) -> dict:
    if (policy is None) == (baseline is None) or baseline not in (None, "cash", "buy-and-hold"):
        raise ValueError("Choose exactly one frozen policy or predefined baseline")
    if policy is not None and policy.model.catalog.identity != catalog.identity:
        raise ValueError("Policy must be loaded for the evaluation catalog")
    splits, dividends, sessions = economic_inputs.load(catalog)
    env = RawSecondPortfolioEnv(catalog, config=execution_config, device=device,
                                splits=splits, dividends=dividends, sessions=sessions)
    while env.index < len(catalog.windows) - 1:
        observation = env.observation()
        if policy is not None:
            action = policy.act(observation)
        else:
            requested = torch.zeros_like(observation.action_mask, dtype=torch.float32)
            requested[:, 0] = 1
            if baseline == "buy-and-hold" and env.index == 0:
                # Fixed equal weight across the original universe; unavailable
                # names retain their cash, never reallocate using later returns.
                n = len(catalog.asset_ids)
                weight = min(D(execution_config.maximum_asset_weight), D(execution_config.maximum_gross_exposure) / n)
                requested[:, 1:] = float(weight) * observation.action_mask[:, 1:]
                requested[:, 0] = 1 - requested[:, 1:].sum(-1)
            action = ActionBatch(requested)
        env.step(action, submit_orders=baseline != "buy-and-hold" or env.index == 0 or env.risk_halted)
    if policy is not None:
        policy.validate_unchanged()
    # Reopen the coverage document/sources; a mid-run replacement is not allowed.
    economic_inputs.load(catalog)
    return dict(catalog_sha256=catalog.identity, economic_inputs=asdict(economic_inputs),
                execution_config=asdict(execution_config), baseline=baseline,
                entry_convention="one-shot-first-decision-interval;unfilled-remainder-expires;no-catch-up-buys" if baseline == "buy-and-hold" else None,
                baseline_comparability="same-constraints-not-necessarily-same-realized-exposure",
                policy_sha256=policy.artifact_sha256 if policy else None,
                parameter_sha256=policy.parameter_sha256 if policy else None,
                summary=economic_summary(env), ledger=env.audit,
                execution_proxy="later-second-open;whole-shares;volume-cap;terminal-mark-adjustment",
                executed_action_semantics="ending-book-state-not-fill-allocation")
