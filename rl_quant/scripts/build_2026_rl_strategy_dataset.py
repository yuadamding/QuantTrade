#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


def float_or_zero(value: str | None) -> float:
    if value is None:
        return 0.0
    text = value.strip()
    if not text:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return number if math.isfinite(number) else 0.0


def load_price_series(path: Path, column: str = "Adj Close") -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            text = row.get(column) or row.get("Close") or ""
            value = float_or_zero(text)
            if value > 0:
                out[row["Date"]] = value
    return out


def read_curves(path: Path) -> tuple[list[str], list[str], dict[str, list[float]]]:
    with path.open(newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        if not header or header[0] != "Date":
            raise ValueError(f"{path} must have Date as the first column")
        names = header[1:]
        dates: list[str] = []
        curves = {name: [] for name in names}
        for row in reader:
            if not row:
                continue
            dates.append(row[0])
            for i, name in enumerate(names, start=1):
                curves[name].append(float_or_zero(row[i] if i < len(row) else ""))
    return dates, names, curves


def read_metrics(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        return {row["Strategy"]: row for row in reader}


def read_strategy_order(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if "Strategy" not in (reader.fieldnames or []):
            return []
        return [row["Strategy"] for row in reader if row.get("Strategy")]


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: float) -> str:
    if not math.isfinite(value):
        value = 0.0
    return f"{value:.10f}"


def daily_returns(values: list[float]) -> list[float]:
    out = [0.0]
    for i in range(1, len(values)):
        previous = values[i - 1]
        current = values[i]
        if previous > 0 and current > 0:
            out.append(current / previous - 1.0)
        else:
            out.append(0.0)
    return out


def compounded_return(returns: list[float], end_i: int, window: int) -> float:
    start = max(1, end_i - window + 1)
    if end_i < start:
        return 0.0
    total = 1.0
    for i in range(start, end_i + 1):
        total *= 1.0 + returns[i]
    return total - 1.0


def window_stdev(values: list[float], end_i: int, window: int) -> float:
    start = max(1, end_i - window + 1)
    sample = values[start : end_i + 1]
    return pstdev(sample) if len(sample) >= 2 else 0.0


def rolling_drawdown(values: list[float], end_i: int, window: int) -> float:
    start = max(0, end_i - window + 1)
    sample = values[start : end_i + 1]
    if not sample:
        return 0.0
    peak = max(sample)
    current = sample[-1]
    return current / peak - 1.0 if peak > 0 else 0.0


def max_drawdown(equity: list[float]) -> float:
    peak = equity[0] if equity else 1.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def sharpe_ratio(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    sigma = pstdev(returns)
    if sigma <= 0:
        return 0.0
    return mean(returns) / sigma * math.sqrt(252.0)


def load_universe_stats(universe_dir: Path, dates: list[str]) -> list[dict[str, float]]:
    buckets: list[list[float]] = [[] for _ in dates]
    paths = sorted(universe_dir.glob("*_daily.csv"))
    for path in paths:
        series = load_price_series(path)
        for i in range(1, len(dates)):
            p0 = series.get(dates[i - 1])
            p1 = series.get(dates[i])
            if p0 is not None and p1 is not None and p0 > 0 and p1 > 0:
                buckets[i].append(p1 / p0 - 1.0)

    stats: list[dict[str, float]] = []
    for returns in buckets:
        if returns:
            stats.append(
                {
                    "available_symbols": float(len(returns)),
                    "universe_ew_ret_1d": mean(returns),
                    "universe_breadth_positive": sum(1 for ret in returns if ret > 0) / len(returns),
                    "universe_dispersion_1d": pstdev(returns) if len(returns) >= 2 else 0.0,
                }
            )
        else:
            stats.append(
                {
                    "available_symbols": 0.0,
                    "universe_ew_ret_1d": 0.0,
                    "universe_breadth_positive": 0.0,
                    "universe_dispersion_1d": 0.0,
                }
            )
    return stats


def select_actions(
    ordered_names: list[str],
    returns_by_name: dict[str, list[float]],
    max_actions: int,
) -> list[str]:
    if max_actions <= 0:
        return ordered_names

    selected: list[str] = []
    seen_return_paths: set[tuple[float, ...]] = set()

    def add(name: str) -> None:
        if name not in returns_by_name or name in selected or len(selected) >= max_actions:
            return
        fingerprint = tuple(round(value, 10) for value in returns_by_name[name][1:])
        if fingerprint in seen_return_paths:
            return
        selected.append(name)
        seen_return_paths.add(fingerprint)

    add("BH_QQQ")
    for name in ordered_names:
        add(name)
        if len(selected) >= max_actions:
            break
    return selected


def metric_value(metrics: dict[str, dict[str, str]], strategy: str, key: str) -> str:
    return metrics.get(strategy, {}).get(key, "")


def source_for_strategy(name: str) -> str:
    if name.startswith("BH_"):
        return "benchmark"
    if name.startswith("CS_"):
        return "cross_sectional_momentum"
    if name.startswith("DM_"):
        return "dual_momentum"
    return "strategy_library"


def build_action_manifest(
    selected: list[str],
    metrics: dict[str, dict[str, str]],
    variation_metrics: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, name in enumerate(selected):
        variation = variation_metrics.get(name, {})
        rows.append(
            {
                "ActionIndex": i,
                "Strategy": name,
                "Source": source_for_strategy(name),
                "IsBenchmark": int(name.startswith("BH_")),
                "Freq": metric_value(metrics, name, "Freq"),
                "TopN": metric_value(metrics, name, "TopN"),
                "Lookback": metric_value(metrics, name, "Lookback"),
                "Skip": metric_value(metrics, name, "Skip"),
                "Only Positive": metric_value(metrics, name, "Only Positive"),
                "Trend Filter": metric_value(metrics, name, "Trend Filter"),
                "Total Return [%]": metric_value(metrics, name, "Total Return [%]"),
                "Sharpe": metric_value(metrics, name, "Sharpe"),
                "Max DD [%]": metric_value(metrics, name, "Max DD [%]"),
                "Final Equity": metric_value(metrics, name, "Final Equity"),
                "Variation Adjusted Score": variation.get("Variation Adjusted Score", ""),
                "Portfolio Daily Variance": variation.get("Portfolio Daily Variance", ""),
                "Portfolio Annual Vol [%]": variation.get("Portfolio Annual Vol [%]", ""),
                "Constituent Annual Vol [%]": variation.get("Constituent Annual Vol [%]", ""),
                "Avg Intra Portfolio Dispersion [%]": variation.get("Avg Intra Portfolio Dispersion [%]", ""),
                "Avg Top Contribution Share [%]": variation.get("Avg Top Contribution Share [%]", ""),
                "Avg Effective Holdings": variation.get("Avg Effective Holdings", ""),
                "Max Abs Single Stock Symbol": variation.get("Max Abs Single Stock Symbol", ""),
                "Max Abs Single Stock Date": variation.get("Max Abs Single Stock Date", ""),
                "Max Abs Single Stock Return [%]": variation.get("Max Abs Single Stock Return [%]", ""),
            }
        )
    return rows


def build_action_return_rows(
    dates: list[str],
    selected: list[str],
    returns_by_name: dict[str, list[float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        row: dict[str, object] = {"Date": date}
        for name in selected:
            row[name] = format_float(returns_by_name[name][i])
        rows.append(row)
    return rows


def build_state_rows(
    dates: list[str],
    selected: list[str],
    curves: dict[str, list[float]],
    returns_by_name: dict[str, list[float]],
    universe_stats: list[dict[str, float]],
) -> tuple[list[str], list[dict[str, object]]]:
    qqq_curve = curves["BH_QQQ"]
    qqq_returns = returns_by_name["BH_QQQ"]
    feature_names = [
        "qqq_ret_1d",
        "qqq_ret_5d",
        "qqq_ret_21d",
        "qqq_vol_21d",
        "qqq_drawdown_63d",
        "universe_ew_ret_1d",
        "universe_breadth_positive",
        "universe_dispersion_1d",
        "available_symbols",
        "strategy_best_ret_1d",
        "strategy_median_ret_1d",
        "strategy_dispersion_1d",
        "strategy_best_trailing_5d",
        "strategy_median_trailing_5d",
        "strategy_best_trailing_21d",
        "strategy_median_trailing_21d",
    ]

    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        action_day_returns = [returns_by_name[name][i] for name in selected]
        trailing_5 = [compounded_return(returns_by_name[name], i, 5) for name in selected]
        trailing_21 = [compounded_return(returns_by_name[name], i, 21) for name in selected]
        row = {
            "Date": date,
            "qqq_ret_1d": qqq_returns[i],
            "qqq_ret_5d": compounded_return(qqq_returns, i, 5),
            "qqq_ret_21d": compounded_return(qqq_returns, i, 21),
            "qqq_vol_21d": window_stdev(qqq_returns, i, 21),
            "qqq_drawdown_63d": rolling_drawdown(qqq_curve, i, 63),
            "universe_ew_ret_1d": universe_stats[i]["universe_ew_ret_1d"],
            "universe_breadth_positive": universe_stats[i]["universe_breadth_positive"],
            "universe_dispersion_1d": universe_stats[i]["universe_dispersion_1d"],
            "available_symbols": universe_stats[i]["available_symbols"],
            "strategy_best_ret_1d": max(action_day_returns) if action_day_returns else 0.0,
            "strategy_median_ret_1d": median(action_day_returns) if action_day_returns else 0.0,
            "strategy_dispersion_1d": pstdev(action_day_returns) if len(action_day_returns) >= 2 else 0.0,
            "strategy_best_trailing_5d": max(trailing_5) if trailing_5 else 0.0,
            "strategy_median_trailing_5d": median(trailing_5) if trailing_5 else 0.0,
            "strategy_best_trailing_21d": max(trailing_21) if trailing_21 else 0.0,
            "strategy_median_trailing_21d": median(trailing_21) if trailing_21 else 0.0,
        }
        rows.append({"Date": date, **{name: format_float(float(row[name])) for name in feature_names}})
    return feature_names, rows


def evaluate_sequence(
    dates: list[str],
    selected: list[str],
    returns_by_name: dict[str, list[float]],
    action_sequence: list[int],
    *,
    switch_cost_bps: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    equity = 1.0
    equity_curve = [equity]
    daily: list[float] = []
    switches = 0
    records: list[dict[str, object]] = []
    previous_action = action_sequence[0] if action_sequence else 0

    for i in range(1, len(dates)):
        action = action_sequence[i]
        raw_return = returns_by_name[selected[action]][i]
        switch_cost = switch_cost_bps / 10_000.0 if action != previous_action else 0.0
        net_return = raw_return - switch_cost
        equity *= 1.0 + net_return
        equity_curve.append(equity)
        daily.append(net_return)
        if action != previous_action:
            switches += 1
        records.append(
            {
                "Date": dates[i],
                "ActionIndex": action,
                "Strategy": selected[action],
                "Daily Return": format_float(net_return),
                "Equity": format_float(equity * 10_000.0),
            }
        )
        previous_action = action

    return (
        {
            "Total Return [%]": (equity - 1.0) * 100.0,
            "Sharpe": sharpe_ratio(daily),
            "Max DD [%]": max_drawdown(equity_curve) * 100.0,
            "Final Equity": equity * 10_000.0,
            "Switches": switches,
        },
        records,
    )


def trailing_selector_sequence(
    selected: list[str],
    returns_by_name: dict[str, list[float]],
    *,
    dates: list[str],
    window: int,
    fallback_action: int,
) -> list[int]:
    sequence = [fallback_action] * len(dates)
    for i in range(1, len(dates)):
        signal_i = i - 1
        if signal_i < window:
            sequence[i] = fallback_action
            continue
        best_action = max(
            range(len(selected)),
            key=lambda action: compounded_return(returns_by_name[selected[action]], signal_i, window),
        )
        sequence[i] = best_action
    return sequence


def oracle_sequence(
    selected: list[str],
    returns_by_name: dict[str, list[float]],
    dates: list[str],
    fallback_action: int,
) -> list[int]:
    sequence = [fallback_action] * len(dates)
    for i in range(1, len(dates)):
        sequence[i] = max(range(len(selected)), key=lambda action: returns_by_name[selected[action]][i])
    return sequence


def write_readme(
    path: Path,
    *,
    curves_path: Path,
    universe_dir: Path,
    output_dir: Path,
    selected_count: int,
    total_strategy_count: int,
    start_date: str,
    end_date: str,
    switch_cost_bps: float,
) -> None:
    text = f"""# 2026 Daily Strategy RL Dataset

This dataset converts the 2026 strategy backtests into a reinforcement-learning
allocation problem.

## Policy Used

The policy is a daily strategy allocator. At the close of day `t`, the agent
observes market and strategy-state features through day `t`. It chooses one
action, where each action is a complete strategy curve such as `BH_QQQ`,
cross-sectional momentum, dual momentum, or a trend-filtered variant. The chosen
action receives the realized close-to-close return on day `t + 1`.

This is the right bridge for the existing framework because the current DQN code
already uses rolling state windows and Q-values. The difference is the action
space: intraday RL uses `short/flat/long`; this daily allocator uses strategy
IDs. Internal strategy trading costs are already included in the strategy curves.
The optional allocator switch cost used for diagnostics is {switch_cost_bps:.4f}
bps per strategy switch.

## Data Format

- `action_returns.csv`: one row per trading date. Column `Date` is followed by
  strategy action columns. Each value is the daily return realized from the
  previous trading close to this date. Row 0 is set to 0 because no prior 2026
  close exists inside the evaluation window.
- `state_features.csv`: one row per trading date. Features use only information
  available through that date, including QQQ returns/volatility/drawdown,
  top-1000 equal-weight breadth and dispersion, and cross-sectional behavior of
  the candidate strategy returns.
- `action_manifest.csv`: maps action index to strategy name and backtest
  metadata. When the variation-risk audit is available, the action set is
  ordered by variation-adjusted stability and the manifest includes both
  portfolio-level and single-stock-level variation fields.
- `baseline_policies.csv`: diagnostic policies over the same action space.
- `trailing21_policy.csv`: daily choices of the non-oracle 21-day trailing
  selector.

## Data Frequency

The dataset is daily, close-to-close. The evaluation period is {start_date} to
{end_date}. Longer lookback strategy signals may use pre-2026 prices only as
warmup; all reported rewards, returns, states, and policy diagnostics in this
folder are 2026 rows.

## Coverage

The source curve file is `{curves_path}`. It contains {total_strategy_count}
tested strategy curves. This RL-ready action set keeps {selected_count} actions
after removing duplicate daily return paths and applying the configured action
limit. The breadth features come from `{universe_dir}`.

## Loading From Python

```python
from pathlib import Path
from rl_quant.strategy_data import build_strategy_split

split = build_strategy_split(
    name="stock_top1000_2026",
    state_features_path=Path("{output_dir / "state_features.csv"}"),
    action_returns_path=Path("{output_dir / "action_returns.csv"}"),
    lookback=20,
)
```
"""
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 2026 daily RL dataset from strategy equity curves.",
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_all_curves_top1000.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_results_top1000.csv",
    )
    parser.add_argument(
        "--variation-metrics",
        type=Path,
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_variation_risk_top1000.csv",
    )
    parser.add_argument(
        "--selection-ranking",
        type=Path,
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_variation_stable_picks_top1000.csv",
        help="Optional Strategy-ranked CSV used before falling back to the curve file order.",
    )
    parser.add_argument(
        "--universe-dir",
        type=Path,
        default=PROJECT_ROOT / "derived" / "daily_ohlcv" / "top_us_market_cap_1000_2026-06-14",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "derived" / "rl_daily" / "stock_top1000_2026",
    )
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--switch-cost-bps", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.curves.exists():
        raise FileNotFoundError(
            f"Missing all-curves input {args.curves}. Run massive_2026_momentum_test.py with --all-curves-output first."
        )

    dates, ordered_names, curves = read_curves(args.curves)
    keep_indices = [
        i
        for i, date in enumerate(dates)
        if date >= args.start_date and (args.end_date is None or date <= args.end_date)
    ]
    dates = [dates[i] for i in keep_indices]
    curves = {name: [values[i] for i in keep_indices] for name, values in curves.items()}
    if "BH_QQQ" not in curves:
        raise ValueError("The curves file must contain BH_QQQ as the benchmark/fallback action.")

    returns_by_name = {name: daily_returns(values) for name, values in curves.items()}
    metrics = read_metrics(args.metrics)
    variation_metrics = read_metrics(args.variation_metrics)
    preferred_order = read_strategy_order(args.selection_ranking)
    variation_order = read_strategy_order(args.variation_metrics)
    selection_order: list[str] = []
    seen_selection_names: set[str] = set()
    for source_order in (preferred_order, variation_order, ordered_names):
        for name in source_order:
            if name in returns_by_name and name not in seen_selection_names:
                selection_order.append(name)
                seen_selection_names.add(name)
    selected = select_actions(selection_order, returns_by_name, args.max_actions)
    universe_stats = load_universe_stats(args.universe_dir, dates)
    feature_names, state_rows = build_state_rows(dates, selected, curves, returns_by_name, universe_stats)
    action_return_rows = build_action_return_rows(dates, selected, returns_by_name)
    manifest_rows = build_action_manifest(selected, metrics, variation_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "state_features.csv", ["Date", *feature_names], state_rows)
    write_rows(args.output_dir / "action_returns.csv", ["Date", *selected], action_return_rows)
    write_rows(
        args.output_dir / "action_manifest.csv",
        [
            "ActionIndex",
            "Strategy",
            "Source",
            "IsBenchmark",
            "Freq",
            "TopN",
            "Lookback",
            "Skip",
            "Only Positive",
            "Trend Filter",
            "Total Return [%]",
            "Sharpe",
            "Max DD [%]",
            "Final Equity",
            "Variation Adjusted Score",
            "Portfolio Daily Variance",
            "Portfolio Annual Vol [%]",
            "Constituent Annual Vol [%]",
            "Avg Intra Portfolio Dispersion [%]",
            "Avg Top Contribution Share [%]",
            "Avg Effective Holdings",
            "Max Abs Single Stock Symbol",
            "Max Abs Single Stock Date",
            "Max Abs Single Stock Return [%]",
        ],
        manifest_rows,
    )

    fallback_action = selected.index("BH_QQQ")
    policy_rows: list[dict[str, object]] = []
    policy_records: dict[str, list[dict[str, object]]] = {}
    policies = {
        "BH_QQQ": [fallback_action] * len(dates),
        "BestFixedHindsight": [
            max(
                range(len(selected)),
                key=lambda action: compounded_return(returns_by_name[selected[action]], len(dates) - 1, len(dates)),
            )
        ]
        * len(dates),
        "Trailing5": trailing_selector_sequence(
            selected,
            returns_by_name,
            dates=dates,
            window=5,
            fallback_action=fallback_action,
        ),
        "Trailing21": trailing_selector_sequence(
            selected,
            returns_by_name,
            dates=dates,
            window=21,
            fallback_action=fallback_action,
        ),
        "OracleNextDay": oracle_sequence(selected, returns_by_name, dates, fallback_action),
    }
    notes = {
        "BH_QQQ": "Benchmark buy-and-hold QQQ.",
        "BestFixedHindsight": "Diagnostic only; chooses the best fixed action with full-period hindsight.",
        "Trailing5": "Non-oracle; picks the best action by trailing 5 trading days.",
        "Trailing21": "Non-oracle; picks the best action by trailing 21 trading days.",
        "OracleNextDay": "Upper-bound diagnostic only; uses same-day future returns.",
    }
    for name, sequence in policies.items():
        metrics_row, records = evaluate_sequence(
            dates,
            selected,
            returns_by_name,
            sequence,
            switch_cost_bps=args.switch_cost_bps,
        )
        policy_records[name] = records
        policy_rows.append(
            {
                "Policy": name,
                "Total Return [%]": f"{metrics_row['Total Return [%]']:.6f}",
                "Sharpe": f"{metrics_row['Sharpe']:.6f}",
                "Max DD [%]": f"{metrics_row['Max DD [%]']:.6f}",
                "Final Equity": f"{metrics_row['Final Equity']:.2f}",
                "Switches": metrics_row["Switches"],
                "Note": notes[name],
            }
        )

    write_rows(
        args.output_dir / "baseline_policies.csv",
        ["Policy", "Total Return [%]", "Sharpe", "Max DD [%]", "Final Equity", "Switches", "Note"],
        policy_rows,
    )
    write_rows(
        args.output_dir / "trailing21_policy.csv",
        ["Date", "ActionIndex", "Strategy", "Daily Return", "Equity"],
        policy_records["Trailing21"],
    )
    write_rows(
        args.output_dir / "oracle_next_day_policy.csv",
        ["Date", "ActionIndex", "Strategy", "Daily Return", "Equity"],
        policy_records["OracleNextDay"],
    )

    summary = {
        "start_date": dates[0],
        "end_date": dates[-1],
        "frequency": "daily_close_to_close",
        "source_curves": str(args.curves),
        "source_metrics": str(args.metrics),
        "variation_metrics": str(args.variation_metrics),
        "selection_ranking": str(args.selection_ranking),
        "universe_dir": str(args.universe_dir),
        "total_curves": len(ordered_names),
        "selected_actions": len(selected),
        "state_features": feature_names,
        "switch_cost_bps": args.switch_cost_bps,
        "outputs": {
            "state_features": str(args.output_dir / "state_features.csv"),
            "action_returns": str(args.output_dir / "action_returns.csv"),
            "action_manifest": str(args.output_dir / "action_manifest.csv"),
            "baseline_policies": str(args.output_dir / "baseline_policies.csv"),
            "trailing21_policy": str(args.output_dir / "trailing21_policy.csv"),
            "oracle_next_day_policy": str(args.output_dir / "oracle_next_day_policy.csv"),
        },
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(
        args.output_dir / "README.md",
        curves_path=args.curves,
        universe_dir=args.universe_dir,
        output_dir=args.output_dir,
        selected_count=len(selected),
        total_strategy_count=len(ordered_names),
        start_date=dates[0],
        end_date=dates[-1],
        switch_cost_bps=args.switch_cost_bps,
    )

    print(f"Dates: {dates[0]} -> {dates[-1]} ({len(dates)} rows)")
    print(f"Curves: {len(ordered_names)} total, {len(selected)} selected RL actions")
    print(f"State features: {len(feature_names)}")
    print(f"Output -> {args.output_dir}")
    for row in policy_rows:
        print(
            f"{row['Policy']:<20} TR={float(row['Total Return [%]']):8.2f}% "
            f"Sharpe={float(row['Sharpe']):6.2f} MaxDD={float(row['Max DD [%]']):7.2f}% "
            f"Switches={row['Switches']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
