#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from massive_2026_momentum_test import (  # noqa: E402
    Spec,
    compute_rankings,
    load_price_series,
    load_universe,
    make_specs,
    month_end_indices,
    moving_average,
    target_for_spec,
    turnover,
    week_end_indices,
)


def float_or_nan(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_pstdev(values: list[float]) -> float:
    return pstdev(values) if len(values) >= 2 else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def daily_returns(values: list[float]) -> list[float]:
    returns: list[float] = []
    for i in range(1, len(values)):
        previous = values[i - 1]
        current = values[i]
        if previous > 0 and current > 0:
            returns.append(current / previous - 1.0)
    return returns


def max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as source:
        return {row["Strategy"]: row for row in csv.DictReader(source)}


def read_curves(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with path.open(newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        if not header or header[0] != "Date":
            raise ValueError(f"{path} must have Date as the first column")
        names = header[1:]
        dates: list[str] = []
        curves: dict[str, list[float]] = {name: [] for name in names}
        for row in reader:
            if not row:
                continue
            dates.append(row[0])
            for i, name in enumerate(names, start=1):
                value = float_or_nan(row[i] if i < len(row) else "")
                if math.isfinite(value):
                    curves[name].append(value)
                elif curves[name]:
                    curves[name].append(curves[name][-1])
                else:
                    curves[name].append(10_000.0)
    return dates, curves


def price_return_matrix(matrix: dict[str, list[float | None]]) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    for symbol, prices in matrix.items():
        returns: list[float | None] = []
        for i in range(len(prices) - 1):
            p0 = prices[i]
            p1 = prices[i + 1]
            if p0 is None or p1 is None or p0 <= 0:
                returns.append(None)
            else:
                returns.append(p1 / p0 - 1.0)
        out[symbol] = returns
    return out


def spec_lookup() -> dict[str, Spec]:
    return {spec.name: spec for spec in make_specs()}


def whole_portfolio_metrics(values: list[float]) -> dict[str, float]:
    returns = daily_returns(values)
    daily_mean = safe_mean(returns)
    daily_var = safe_pstdev(returns) ** 2
    daily_std = math.sqrt(daily_var)
    total_return = values[-1] / values[0] - 1.0 if values and values[0] > 0 else float("nan")
    sharpe = daily_mean / daily_std * math.sqrt(252.0) if daily_std > 0 else float("nan")
    return {
        "portfolio_total_return": total_return,
        "portfolio_daily_mean": daily_mean,
        "portfolio_daily_variance": daily_var,
        "portfolio_daily_std": daily_std,
        "portfolio_annual_vol": daily_std * math.sqrt(252.0),
        "portfolio_sharpe": sharpe,
        "portfolio_max_drawdown": max_drawdown(values),
        "portfolio_positive_rate": sum(1 for ret in returns if ret > 0) / len(returns) if returns else 0.0,
        "portfolio_worst_day": min(returns) if returns else 0.0,
        "portfolio_best_day": max(returns) if returns else 0.0,
    }


def variation_adjusted_score(row: dict[str, float]) -> float:
    total_return = row["portfolio_total_return"]
    sharpe = row["portfolio_sharpe"]
    positive_rate = row["portfolio_positive_rate"]
    max_dd = row["portfolio_max_drawdown"]
    if not all(math.isfinite(x) for x in (total_return, sharpe, positive_rate, max_dd)):
        return -float("inf")
    if total_return <= -0.999:
        return -float("inf")
    risk_penalty = (
        (1.0 + row["portfolio_annual_vol"])
        * (1.0 + 0.25 * row["constituent_annual_vol"])
        * (1.0 + 0.50 * row["intra_portfolio_dispersion_annualized"])
        * (1.0 + row["avg_top_contribution_share"])
        * (1.0 + row["max_weight"])
        * (1.0 + abs(max_dd))
    )
    return sharpe * math.log1p(total_return) * max(positive_rate, 0.0) / risk_penalty


def analyze_benchmark(
    *,
    name: str,
    curve: list[float],
    dates: list[str],
    qqq_values: list[float | None],
    test_start_i: int,
) -> dict[str, float | int | str]:
    metrics = whole_portfolio_metrics(curve)
    returns: list[float] = []
    for i in range(test_start_i, test_start_i + len(curve) - 1):
        p0 = qqq_values[i]
        p1 = qqq_values[i + 1]
        if p0 is not None and p1 is not None and p0 > 0:
            returns.append(p1 / p0 - 1.0)
    constituent_var = safe_pstdev(returns) ** 2
    if returns:
        max_index, max_return = max(enumerate(returns), key=lambda item: abs(item[1]))
        max_date = dates[test_start_i + max_index + 1]
    else:
        max_return = 0.0
        max_date = ""
    row: dict[str, float | int | str] = {
        "Strategy": name,
        "Variation Family": "benchmark",
        "Max Abs Single Stock Symbol": "QQQ",
        "Max Abs Single Stock Date": max_date,
        "Max Abs Single Stock Return": max_return,
        "Max Top Contribution Symbol": "QQQ",
        "Max Top Contribution Date": max_date,
        "Unique Symbols Held": 1,
        "Active Day Rate": 1.0,
        "Gross Portfolio Daily Variance": constituent_var,
        "Constituent Daily Variance": constituent_var,
        "Constituent Annual Vol": math.sqrt(constituent_var) * math.sqrt(252.0),
        "Avg Intra Portfolio Dispersion": 0.0,
        "Intra Portfolio Dispersion Annualized": 0.0,
        "Avg Abs Single Stock Day": safe_mean([abs(ret) for ret in returns]),
        "P95 Abs Single Stock Day": percentile([abs(ret) for ret in returns], 0.95),
        "Max Abs Single Stock Day": max([abs(ret) for ret in returns], default=0.0),
        "Avg Top Contribution Share": 1.0,
        "Max Top Contribution Share": 1.0,
        "Avg Effective Holdings": 1.0,
        "Min Effective Holdings": 1.0,
        "Avg Max Weight": 1.0,
        "Max Weight": 1.0,
        "Avg Turnover": 1.0,
    }
    row.update(
        {
            "Portfolio Total Return": metrics["portfolio_total_return"],
            "Portfolio Daily Mean": metrics["portfolio_daily_mean"],
            "Portfolio Daily Variance": metrics["portfolio_daily_variance"],
            "Portfolio Annual Vol": metrics["portfolio_annual_vol"],
            "Portfolio Sharpe": metrics["portfolio_sharpe"],
            "Portfolio Max Drawdown": metrics["portfolio_max_drawdown"],
            "Portfolio Positive Rate": metrics["portfolio_positive_rate"],
            "Portfolio Worst Day": metrics["portfolio_worst_day"],
            "Portfolio Best Day": metrics["portfolio_best_day"],
        }
    )
    numeric = {
        "portfolio_total_return": metrics["portfolio_total_return"],
        "portfolio_sharpe": metrics["portfolio_sharpe"],
        "portfolio_positive_rate": metrics["portfolio_positive_rate"],
        "portfolio_max_drawdown": metrics["portfolio_max_drawdown"],
        "portfolio_annual_vol": metrics["portfolio_annual_vol"],
        "constituent_annual_vol": row["Constituent Annual Vol"],
        "intra_portfolio_dispersion_annualized": 0.0,
        "avg_top_contribution_share": 1.0,
        "max_weight": 1.0,
    }
    row["Variation Adjusted Score"] = variation_adjusted_score(numeric)  # type: ignore[arg-type]
    return row


def analyze_strategy(
    spec: Spec,
    *,
    curve: list[float],
    dates: list[str],
    returns_matrix: dict[str, list[float | None]],
    rankings: dict[tuple[int, int, int], list[tuple[float, str]]],
    trend_ok: dict[int, list[bool]],
    weekly_rebal: set[int],
    monthly_rebal: set[int],
    test_start_i: int,
    cost_bps: float,
) -> dict[str, float | int | str]:
    rebal = weekly_rebal if spec.freq == "weekly" else monthly_rebal
    holdings: dict[str, float] = {}
    unique_symbols: set[str] = set()
    turnovers: list[float] = []
    gross_portfolio_returns: list[float] = []
    single_stock_returns: list[float] = []
    daily_dispersion: list[float] = []
    daily_abs_single: list[float] = []
    daily_max_abs_single: list[float] = []
    top_contribution_shares: list[float] = []
    effective_holdings: list[float] = []
    max_weights: list[float] = []
    active_days = 0
    max_abs_single_symbol = ""
    max_abs_single_date = ""
    max_abs_single_return = 0.0
    max_top_contribution_symbol = ""
    max_top_contribution_date = ""
    max_top_contribution_share = 0.0

    for i in range(len(dates) - 1):
        if i in rebal:
            target = target_for_spec(spec, rankings=rankings, trend_ok=trend_ok, rebal_i=i)
            if target is not None:
                if i >= test_start_i:
                    turnovers.append(turnover(holdings, target))
                holdings = target

        if i < test_start_i:
            continue

        valid_returns: list[float] = []
        valid_weights: list[float] = []
        observations: list[tuple[str, float, float, float]] = []
        gross_return = 0.0
        for symbol, weight in holdings.items():
            symbol_returns = returns_matrix.get(symbol)
            if symbol_returns is None or i >= len(symbol_returns):
                continue
            stock_return = symbol_returns[i]
            if stock_return is None:
                continue
            valid_returns.append(stock_return)
            valid_weights.append(weight)
            contribution = weight * stock_return
            observations.append((symbol, weight, stock_return, contribution))
            gross_return += contribution
            unique_symbols.add(symbol)
            if abs(stock_return) > abs(max_abs_single_return):
                max_abs_single_symbol = symbol
                max_abs_single_date = dates[i + 1]
                max_abs_single_return = stock_return

        gross_portfolio_returns.append(gross_return)
        if not valid_returns:
            continue

        active_days += 1
        single_stock_returns.extend(valid_returns)
        abs_returns = [abs(ret) for ret in valid_returns]
        abs_contributions = [abs(value) for _symbol, _weight, _ret, value in observations]
        abs_contribution_sum = sum(abs_contributions)
        daily_dispersion.append(safe_pstdev(valid_returns))
        daily_abs_single.extend(abs_returns)
        daily_max_abs_single.append(max(abs_returns))
        top_contribution_share = max(abs_contributions) / abs_contribution_sum if abs_contribution_sum > 0 else 0.0
        top_contribution_shares.append(top_contribution_share)
        if top_contribution_share > max_top_contribution_share and observations:
            max_top_contribution_share = top_contribution_share
            max_top_observation = max(observations, key=lambda item: abs(item[3]))
            max_top_contribution_symbol = max_top_observation[0]
            max_top_contribution_date = dates[i + 1]
        weight_sq_sum = sum(weight * weight for weight in valid_weights)
        effective_holdings.append(1.0 / weight_sq_sum if weight_sq_sum > 0 else 0.0)
        max_weights.append(max(valid_weights))

    whole = whole_portfolio_metrics(curve)
    gross_var = safe_pstdev(gross_portfolio_returns) ** 2
    constituent_var = safe_pstdev(single_stock_returns) ** 2
    row: dict[str, float | int | str] = {
        "Strategy": spec.name,
        "Variation Family": "dual_momentum" if spec.only_positive else "cross_sectional",
        "Max Abs Single Stock Symbol": max_abs_single_symbol,
        "Max Abs Single Stock Date": max_abs_single_date,
        "Max Abs Single Stock Return": max_abs_single_return,
        "Max Top Contribution Symbol": max_top_contribution_symbol,
        "Max Top Contribution Date": max_top_contribution_date,
        "Unique Symbols Held": len(unique_symbols),
        "Active Day Rate": active_days / max(len(gross_portfolio_returns), 1),
        "Gross Portfolio Daily Variance": gross_var,
        "Constituent Daily Variance": constituent_var,
        "Constituent Annual Vol": math.sqrt(constituent_var) * math.sqrt(252.0),
        "Avg Intra Portfolio Dispersion": safe_mean(daily_dispersion),
        "Intra Portfolio Dispersion Annualized": safe_mean(daily_dispersion) * math.sqrt(252.0),
        "Avg Abs Single Stock Day": safe_mean(daily_abs_single),
        "P95 Abs Single Stock Day": percentile(daily_abs_single, 0.95),
        "Max Abs Single Stock Day": max(daily_max_abs_single, default=0.0),
        "Avg Top Contribution Share": safe_mean(top_contribution_shares),
        "Max Top Contribution Share": max(top_contribution_shares, default=0.0),
        "Avg Effective Holdings": safe_mean(effective_holdings),
        "Min Effective Holdings": min(effective_holdings, default=0.0),
        "Avg Max Weight": safe_mean(max_weights),
        "Max Weight": max(max_weights, default=0.0),
        "Avg Turnover": safe_mean(turnovers),
    }
    row.update(
        {
            "Portfolio Total Return": whole["portfolio_total_return"],
            "Portfolio Daily Mean": whole["portfolio_daily_mean"],
            "Portfolio Daily Variance": whole["portfolio_daily_variance"],
            "Portfolio Annual Vol": whole["portfolio_annual_vol"],
            "Portfolio Sharpe": whole["portfolio_sharpe"],
            "Portfolio Max Drawdown": whole["portfolio_max_drawdown"],
            "Portfolio Positive Rate": whole["portfolio_positive_rate"],
            "Portfolio Worst Day": whole["portfolio_worst_day"],
            "Portfolio Best Day": whole["portfolio_best_day"],
        }
    )
    score_inputs = {
        "portfolio_total_return": whole["portfolio_total_return"],
        "portfolio_sharpe": whole["portfolio_sharpe"],
        "portfolio_positive_rate": whole["portfolio_positive_rate"],
        "portfolio_max_drawdown": whole["portfolio_max_drawdown"],
        "portfolio_annual_vol": whole["portfolio_annual_vol"],
        "constituent_annual_vol": row["Constituent Annual Vol"],
        "intra_portfolio_dispersion_annualized": row["Intra Portfolio Dispersion Annualized"],
        "avg_top_contribution_share": row["Avg Top Contribution Share"],
        "max_weight": row["Max Weight"],
    }
    row["Variation Adjusted Score"] = variation_adjusted_score(score_inputs)  # type: ignore[arg-type]
    return row


def merged_row(
    *,
    variation: dict[str, float | int | str],
    metrics: dict[str, str] | None,
) -> dict[str, object]:
    metrics = metrics or {}
    strategy = str(variation["Strategy"])
    topn = float_or_nan(metrics.get("TopN"))
    only_positive = metrics.get("Only Positive", "")
    trend = metrics.get("Trend Filter", "")
    portfolio_var = float(variation["Portfolio Daily Variance"])
    constituent_var = float(variation["Constituent Daily Variance"])
    row: dict[str, object] = {
        "Strategy": strategy,
        "Freq": metrics.get("Freq", "buy_hold" if strategy == "BH_QQQ" else ""),
        "TopN": int(topn) if math.isfinite(topn) else 1,
        "Lookback": metrics.get("Lookback", ""),
        "Skip": metrics.get("Skip", ""),
        "Only Positive": only_positive,
        "Trend Filter": trend,
        "Portfolio Total Return [%]": float(variation["Portfolio Total Return"]) * 100.0,
        "Portfolio Daily Variance": portfolio_var,
        "Portfolio Annual Vol [%]": float(variation["Portfolio Annual Vol"]) * 100.0,
        "Portfolio Sharpe": variation["Portfolio Sharpe"],
        "Portfolio Max DD [%]": float(variation["Portfolio Max Drawdown"]) * 100.0,
        "Portfolio Positive Day Rate [%]": float(variation["Portfolio Positive Rate"]) * 100.0,
        "Portfolio Worst Day [%]": float(variation["Portfolio Worst Day"]) * 100.0,
        "Portfolio Best Day [%]": float(variation["Portfolio Best Day"]) * 100.0,
        "Gross Portfolio Daily Variance": variation["Gross Portfolio Daily Variance"],
        "Constituent Daily Variance": constituent_var,
        "Constituent Annual Vol [%]": float(variation["Constituent Annual Vol"]) * 100.0,
        "Avg Intra Portfolio Dispersion [%]": float(variation["Avg Intra Portfolio Dispersion"]) * 100.0,
        "Annualized Intra Portfolio Dispersion [%]": float(
            variation["Intra Portfolio Dispersion Annualized"]
        )
        * 100.0,
        "Avg Abs Single Stock Day [%]": float(variation["Avg Abs Single Stock Day"]) * 100.0,
        "P95 Abs Single Stock Day [%]": float(variation["P95 Abs Single Stock Day"]) * 100.0,
        "Max Abs Single Stock Day [%]": float(variation["Max Abs Single Stock Day"]) * 100.0,
        "Max Abs Single Stock Symbol": variation["Max Abs Single Stock Symbol"],
        "Max Abs Single Stock Date": variation["Max Abs Single Stock Date"],
        "Max Abs Single Stock Return [%]": float(variation["Max Abs Single Stock Return"]) * 100.0,
        "Single To Portfolio Variance Ratio": (
            constituent_var / portfolio_var if portfolio_var > 0 else float("nan")
        ),
        "Avg Top Contribution Share [%]": float(variation["Avg Top Contribution Share"]) * 100.0,
        "Max Top Contribution Share [%]": float(variation["Max Top Contribution Share"]) * 100.0,
        "Max Top Contribution Symbol": variation["Max Top Contribution Symbol"],
        "Max Top Contribution Date": variation["Max Top Contribution Date"],
        "Avg Effective Holdings": variation["Avg Effective Holdings"],
        "Min Effective Holdings": variation["Min Effective Holdings"],
        "Avg Max Weight [%]": float(variation["Avg Max Weight"]) * 100.0,
        "Max Weight [%]": float(variation["Max Weight"]) * 100.0,
        "Active Day Rate [%]": float(variation["Active Day Rate"]) * 100.0,
        "Unique Symbols Held": variation["Unique Symbols Held"],
        "Avg Turnover": variation["Avg Turnover"],
        "Variation Adjusted Score": variation["Variation Adjusted Score"],
    }
    return row


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row))


def format_row(row: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, float):
            if not math.isfinite(value):
                out[key] = ""
            elif "Variance" in key:
                out[key] = f"{value:.12f}"
            elif key in {"Variation Adjusted Score", "Avg Turnover"}:
                out[key] = f"{value:.6f}"
            else:
                out[key] = f"{value:.6f}"
        else:
            out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze 2026 strategy risk using both portfolio and constituent-stock variation.",
    )
    parser.add_argument(
        "--universe-dir",
        type=Path,
        default=PROJECT_ROOT / "derived" / "daily_ohlcv" / "top_us_market_cap_1000_2026-06-14",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "derived" / "daily_ohlcv" / "max" / "QQQ_1980-01-01_2026-06-15_daily.csv",
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_variation_risk_top1000.csv",
    )
    parser.add_argument(
        "--stable-output",
        type=Path,
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_variation_stable_picks_top1000.csv",
    )
    parser.add_argument("--warmup-start", default="2025-01-01")
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--min-total-return", type=float, default=50.0)
    parser.add_argument("--min-sharpe", type=float, default=2.0)
    parser.add_argument("--max-dd", type=float, default=15.0, help="Maximum drawdown magnitude in percent.")
    parser.add_argument("--min-topn", type=int, default=10)
    parser.add_argument("--max-portfolio-annual-vol", type=float, default=120.0)
    parser.add_argument("--max-constituent-annual-vol", type=float, default=220.0)
    parser.add_argument("--max-top-contribution-share", type=float, default=55.0)
    parser.add_argument("--min-active-day-rate", type=float, default=80.0)
    parser.add_argument("--top", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    curve_dates, curves = read_curves(args.curves)
    metrics_by_name = read_metrics(args.metrics)
    specs_by_name = spec_lookup()

    universe = load_universe(args.universe_dir)
    qqq = load_price_series(args.benchmark)
    end_date = curve_dates[-1]
    dates = sorted(d for d in qqq if d >= args.warmup_start and d <= end_date)
    dates = [d for d in dates if any(d in series for series in universe.values())]
    if curve_dates[0] not in dates:
        raise ValueError(f"First curve date {curve_dates[0]} is absent from reconstructed date grid.")
    test_start_i = dates.index(curve_dates[0])
    if dates[test_start_i : test_start_i + len(curve_dates)] != curve_dates:
        raise ValueError("Curve dates do not match the reconstructed universe/benchmark trading dates.")

    symbols = sorted(universe)
    matrix = {symbol: [universe[symbol].get(date) for date in dates] for symbol in symbols}
    returns_matrix = price_return_matrix(matrix)
    qqq_values = [qqq.get(date) for date in dates]

    weekly = week_end_indices(dates)
    monthly = month_end_indices(dates)
    rebal_union = weekly | monthly
    lookbacks = (21, 42, 63, 84, 126, 168, 189, 252)
    skips = (0, 5, 10, 21)
    rankings = compute_rankings(
        dates=dates,
        symbols=symbols,
        matrix=matrix,
        rebal_indices=rebal_union,
        lookbacks=lookbacks,
        skips=skips,
    )

    trend_ok: dict[int, list[bool]] = {}
    for ma_window in (50, 100, 200):
        ma = moving_average(qqq_values, ma_window)
        flags = [False] * len(dates)
        for i in range(len(dates)):
            signal_i = i - 1
            if signal_i >= 0 and qqq_values[signal_i] is not None and ma[signal_i] is not None:
                flags[i] = bool(qqq_values[signal_i] > ma[signal_i])
        trend_ok[ma_window] = flags

    rows: list[dict[str, object]] = []
    for n, (name, curve) in enumerate(curves.items(), 1):
        if name == "BH_QQQ":
            variation = analyze_benchmark(
                name=name,
                curve=curve,
                dates=dates,
                qqq_values=qqq_values,
                test_start_i=test_start_i,
            )
        else:
            spec = specs_by_name.get(name)
            if spec is None:
                continue
            variation = analyze_strategy(
                spec,
                curve=curve,
                dates=dates,
                returns_matrix=returns_matrix,
                rankings=rankings,
                trend_ok=trend_ok,
                weekly_rebal=weekly,
                monthly_rebal=monthly,
                test_start_i=test_start_i,
                cost_bps=args.cost_bps,
            )
        rows.append(merged_row(variation=variation, metrics=metrics_by_name.get(name)))
        if n % 500 == 0:
            print(f"analyzed {n}/{len(curves)} strategy curves", flush=True)

    fields = [
        "Strategy",
        "Freq",
        "TopN",
        "Lookback",
        "Skip",
        "Only Positive",
        "Trend Filter",
        "Portfolio Total Return [%]",
        "Portfolio Daily Variance",
        "Portfolio Annual Vol [%]",
        "Portfolio Sharpe",
        "Portfolio Max DD [%]",
        "Portfolio Positive Day Rate [%]",
        "Portfolio Worst Day [%]",
        "Portfolio Best Day [%]",
        "Gross Portfolio Daily Variance",
        "Constituent Daily Variance",
        "Constituent Annual Vol [%]",
        "Avg Intra Portfolio Dispersion [%]",
        "Annualized Intra Portfolio Dispersion [%]",
        "Avg Abs Single Stock Day [%]",
        "P95 Abs Single Stock Day [%]",
        "Max Abs Single Stock Day [%]",
        "Max Abs Single Stock Symbol",
        "Max Abs Single Stock Date",
        "Max Abs Single Stock Return [%]",
        "Single To Portfolio Variance Ratio",
        "Avg Top Contribution Share [%]",
        "Max Top Contribution Share [%]",
        "Max Top Contribution Symbol",
        "Max Top Contribution Date",
        "Avg Effective Holdings",
        "Min Effective Holdings",
        "Avg Max Weight [%]",
        "Max Weight [%]",
        "Active Day Rate [%]",
        "Unique Symbols Held",
        "Avg Turnover",
        "Variation Adjusted Score",
    ]

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["Variation Adjusted Score"]),
            float(row["Portfolio Daily Variance"]),
            float(row["Avg Top Contribution Share [%]"]),
        ),
    )
    bh = next((row for row in rows if row["Strategy"] == "BH_QQQ"), None)
    bh_return = float(bh["Portfolio Total Return [%]"]) if bh is not None else args.min_total_return
    min_return = max(args.min_total_return, bh_return)
    stable = [
        row
        for row in ranked
        if row["Strategy"] != "BH_QQQ"
        and float(row["Portfolio Total Return [%]"]) >= min_return
        and float(row["Portfolio Sharpe"]) >= args.min_sharpe
        and float(row["Portfolio Max DD [%]"]) >= -args.max_dd
        and int(row["TopN"]) >= args.min_topn
        and float(row["Portfolio Annual Vol [%]"]) <= args.max_portfolio_annual_vol
        and float(row["Constituent Annual Vol [%]"]) <= args.max_constituent_annual_vol
        and float(row["Avg Top Contribution Share [%]"]) <= args.max_top_contribution_share
        and float(row["Active Day Rate [%]"]) >= args.min_active_day_rate
    ]

    write_rows(args.output, fields, ranked)
    write_rows(args.stable_output, fields, stable)

    print(f"Dates: {curve_dates[0]} -> {curve_dates[-1]} ({len(curve_dates)} rows)")
    print(f"Universe symbols: {len(symbols)}")
    print(f"Strategies analyzed: {len(rows)}")
    if bh is not None:
        print(
            "BH_QQQ: "
            f"TR={float(bh['Portfolio Total Return [%]']):.2f}% "
            f"PortVol={float(bh['Portfolio Annual Vol [%]']):.2f}% "
            f"ConstitVol={float(bh['Constituent Annual Vol [%]']):.2f}% "
            f"Sharpe={float(bh['Portfolio Sharpe']):.2f}"
        )
    print(f"Full variation audit -> {args.output}")
    print(f"Variation-stable picks -> {args.stable_output}")
    print()
    print("Top variation-adjusted stable picks:")
    for row in stable[: args.top]:
        print(
            f"{row['Strategy']:<34} "
            f"TR={float(row['Portfolio Total Return [%]']):8.2f}% "
            f"PortVol={float(row['Portfolio Annual Vol [%]']):6.2f}% "
            f"StockVol={float(row['Constituent Annual Vol [%]']):6.2f}% "
            f"Disp={float(row['Avg Intra Portfolio Dispersion [%]']):5.2f}% "
            f"TopContrib={float(row['Avg Top Contribution Share [%]']):5.1f}% "
            f"EffN={float(row['Avg Effective Holdings']):5.1f} "
            f"Score={float(row['Variation Adjusted Score']):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
