#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT


@dataclass(frozen=True)
class Spec:
    name: str
    freq: str
    top_n: int
    lookback: int
    skip: int
    only_positive: bool
    trend_ma: int | None


@dataclass
class Result:
    strategy: str
    freq: str
    top_n: int
    lookback: int
    skip: int
    only_positive: bool
    trend_filter: str
    start: str
    end: str
    days: int
    rebalances: int
    avg_holdings: float
    avg_turnover: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_dd_pct: float
    calmar: float
    final_equity: float


def symbol_from_path(path: Path) -> str:
    return path.name.split("_1980-01-01_")[0].split("_2025-01-01_")[0]


def load_price_series(path: Path, column: str = "Adj Close") -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            text = row.get(column) or row.get("Close") or ""
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                continue
            if value > 0:
                out[row["Date"]] = value
    return out


def load_universe(input_dir: Path) -> dict[str, dict[str, float]]:
    prices: dict[str, dict[str, float]] = {}
    for path in sorted(input_dir.glob("*_daily.csv")):
        symbol = symbol_from_path(path)
        series = load_price_series(path)
        if series:
            prices[symbol] = series
    return prices


def month_end_indices(dates: list[str]) -> set[int]:
    last: dict[str, int] = {}
    for i, date in enumerate(dates):
        last[date[:7]] = i
    return set(last.values())


def week_end_indices(dates: list[str]) -> set[int]:
    last: dict[str, int] = {}
    for i, date in enumerate(dates):
        # ISO week is stable for US trading calendars and handles year boundary.
        year, week, _weekday = __import__("datetime").date.fromisoformat(date).isocalendar()
        last[f"{year}-W{week:02d}"] = i
    return set(last.values())


def moving_average(values: list[float | None], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    running = 0.0
    valid = 0
    queue: list[float | None] = []
    for i, value in enumerate(values):
        queue.append(value)
        if value is not None:
            running += value
            valid += 1
        if len(queue) > window:
            old = queue.pop(0)
            if old is not None:
                running -= old
                valid -= 1
        if len(queue) == window and valid == window:
            out[i] = running / window
    return out


def make_specs() -> list[Spec]:
    specs: list[Spec] = []
    top_ns = (1, 2, 3, 5, 10, 15, 20, 30, 50, 75)
    lookbacks = (21, 42, 63, 84, 126, 168, 189, 252)
    skips = (0, 5, 10, 21)
    freqs = ("weekly", "monthly")
    only_positive_options = (False, True)
    trend_filters: tuple[int | None, ...] = (None, 50, 100, 200)

    for freq in freqs:
        for top_n in top_ns:
            for lookback in lookbacks:
                for skip in skips:
                    for only_positive in only_positive_options:
                        for trend_ma in trend_filters:
                            parts = [
                                "DM" if only_positive else "CS",
                                freq[0].upper(),
                                f"Top{top_n}",
                                f"LB{lookback}",
                            ]
                            if skip:
                                parts.append(f"Skip{skip}")
                            if trend_ma is not None:
                                parts.append(f"QMA{trend_ma}")
                            name = "_".join(parts)
                            specs.append(
                                Spec(
                                    name=name,
                                    freq=freq,
                                    top_n=top_n,
                                    lookback=lookback,
                                    skip=skip,
                                    only_positive=only_positive,
                                    trend_ma=trend_ma,
                                )
                            )
    return specs


def compute_rankings(
    *,
    dates: list[str],
    symbols: list[str],
    matrix: dict[str, list[float | None]],
    rebal_indices: set[int],
    lookbacks: tuple[int, ...],
    skips: tuple[int, ...],
) -> dict[tuple[int, int, int], list[tuple[float, str]]]:
    rankings: dict[tuple[int, int, int], list[tuple[float, str]]] = {}
    for lookback in lookbacks:
        for skip in skips:
            for i in rebal_indices:
                signal_i = i - 1 - skip
                base_i = signal_i - lookback
                if signal_i < 0 or base_i < 0:
                    rankings[(lookback, skip, i)] = []
                    continue
                scored: list[tuple[float, str]] = []
                for symbol in symbols:
                    series = matrix[symbol]
                    p0 = series[base_i]
                    p1 = series[signal_i]
                    p_entry = series[i]
                    if p0 is None or p1 is None or p_entry is None or p0 <= 0:
                        continue
                    scored.append((p1 / p0 - 1.0, symbol))
                scored.sort(reverse=True)
                rankings[(lookback, skip, i)] = scored
    return rankings


def target_for_spec(
    spec: Spec,
    *,
    rankings: dict[tuple[int, int, int], list[tuple[float, str]]],
    trend_ok: dict[int, list[bool]],
    rebal_i: int,
) -> dict[str, float] | None:
    if spec.trend_ma is not None and not trend_ok[spec.trend_ma][rebal_i]:
        return {}

    ranked = rankings.get((spec.lookback, spec.skip, rebal_i), [])
    if spec.only_positive:
        ranked = [item for item in ranked if item[0] > 0]
    if not ranked:
        return None
    picked = [symbol for _score, symbol in ranked[: spec.top_n]]
    weight = 1.0 / len(picked)
    return {symbol: weight for symbol in picked}


def turnover(current: dict[str, float], target: dict[str, float]) -> float:
    return sum(abs(target.get(s, 0.0) - current.get(s, 0.0)) for s in set(current) | set(target))


def portfolio_return(
    holdings: dict[str, float],
    matrix: dict[str, list[float | None]],
    i: int,
) -> float:
    ret = 0.0
    for symbol, weight in holdings.items():
        series = matrix[symbol]
        p0 = series[i]
        p1 = series[i + 1]
        if p0 is None or p1 is None or p0 <= 0:
            continue
        ret += weight * (p1 / p0 - 1.0)
    return ret


def metrics(
    *,
    spec: Spec,
    dates: list[str],
    equity: list[float],
    test_start_i: int,
    rebalances: int,
    turnovers: list[float],
    holding_counts: list[int],
) -> Result:
    base = equity[test_start_i]
    test_values = [value / base * 10_000.0 for value in equity[test_start_i:]]
    returns = [
        test_values[i] / test_values[i - 1] - 1.0
        for i in range(1, len(test_values))
        if test_values[i - 1] > 0
    ]
    total = test_values[-1] / test_values[0] - 1.0
    days = len(test_values) - 1
    cagr = (1.0 + total) ** (252.0 / days) - 1.0 if days > 0 and total > -1.0 else float("nan")
    sigma = pstdev(returns) if len(returns) >= 2 else 0.0
    sharpe = mean(returns) / sigma * math.sqrt(252.0) if sigma > 0 else float("nan")
    peak = test_values[0]
    max_dd = 0.0
    for value in test_values:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1.0)
    calmar = cagr / abs(max_dd) if max_dd < -1e-12 and math.isfinite(cagr) else float("nan")
    return Result(
        strategy=spec.name,
        freq=spec.freq,
        top_n=spec.top_n,
        lookback=spec.lookback,
        skip=spec.skip,
        only_positive=spec.only_positive,
        trend_filter="" if spec.trend_ma is None else f"QQQ>{spec.trend_ma}DMA",
        start=dates[test_start_i],
        end=dates[-1],
        days=days,
        rebalances=rebalances,
        avg_holdings=mean(holding_counts) if holding_counts else 0.0,
        avg_turnover=mean(turnovers) if turnovers else 0.0,
        total_return_pct=total * 100.0,
        cagr_pct=cagr * 100.0 if math.isfinite(cagr) else float("nan"),
        sharpe=sharpe,
        max_dd_pct=max_dd * 100.0,
        calmar=calmar,
        final_equity=test_values[-1],
    )


def run_spec(
    spec: Spec,
    *,
    dates: list[str],
    matrix: dict[str, list[float | None]],
    rankings: dict[tuple[int, int, int], list[tuple[float, str]]],
    trend_ok: dict[int, list[bool]],
    weekly_rebal: set[int],
    monthly_rebal: set[int],
    test_start_i: int,
    cost_bps: float,
) -> tuple[Result, list[float]]:
    rebal = weekly_rebal if spec.freq == "weekly" else monthly_rebal
    holdings: dict[str, float] = {}
    equity = [10_000.0]
    rebalances = 0
    turnovers: list[float] = []
    holding_counts: list[int] = []

    for i in range(len(dates) - 1):
        if i in rebal:
            target = target_for_spec(spec, rankings=rankings, trend_ok=trend_ok, rebal_i=i)
            if target is not None:
                to = turnover(holdings, target)
                current_equity = equity[-1] * (1.0 - to * cost_bps / 10_000.0)
                equity[-1] = current_equity
                holdings = target
                if i >= test_start_i:
                    rebalances += 1
                    turnovers.append(to)
                    holding_counts.append(len(holdings))
        equity.append(equity[-1] * (1.0 + portfolio_return(holdings, matrix, i)))

    return (
        metrics(
            spec=spec,
            dates=dates,
            equity=equity,
            test_start_i=test_start_i,
            rebalances=rebalances,
            turnovers=turnovers,
            holding_counts=holding_counts,
        ),
        equity,
    )


def benchmark_result(
    *,
    dates: list[str],
    qqq_values: list[float | None],
    test_start_i: int,
) -> Result:
    base = qqq_values[test_start_i]
    if base is None:
        raise ValueError("benchmark missing at test start")
    equity = [
        10_000.0 * (value / base)
        for value in qqq_values[test_start_i:]
        if value is not None
    ]
    spec = Spec("BH_QQQ", "buy_hold", 1, 0, 0, False, None)
    return metrics(
        spec=spec,
        dates=dates[test_start_i:],
        equity=equity,
        test_start_i=0,
        rebalances=1,
        turnovers=[1.0],
        holding_counts=[1],
    )


def write_results(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Strategy",
        "Freq",
        "TopN",
        "Lookback",
        "Skip",
        "Only Positive",
        "Trend Filter",
        "Start",
        "End",
        "Days",
        "Rebalances",
        "Avg Holdings",
        "Avg Turnover",
        "Total Return [%]",
        "CAGR [%]",
        "Sharpe",
        "Max DD [%]",
        "Calmar",
        "Final Equity",
    ]
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Strategy": r.strategy,
                    "Freq": r.freq,
                    "TopN": r.top_n,
                    "Lookback": r.lookback,
                    "Skip": r.skip,
                    "Only Positive": int(r.only_positive),
                    "Trend Filter": r.trend_filter,
                    "Start": r.start,
                    "End": r.end,
                    "Days": r.days,
                    "Rebalances": r.rebalances,
                    "Avg Holdings": f"{r.avg_holdings:.6f}",
                    "Avg Turnover": f"{r.avg_turnover:.6f}",
                    "Total Return [%]": f"{r.total_return_pct:.6f}",
                    "CAGR [%]": f"{r.cagr_pct:.6f}",
                    "Sharpe": f"{r.sharpe:.6f}",
                    "Max DD [%]": f"{r.max_dd_pct:.6f}",
                    "Calmar": f"{r.calmar:.6f}",
                    "Final Equity": f"{r.final_equity:.2f}",
                }
            )


def write_top_curves(
    path: Path,
    *,
    dates: list[str],
    test_start_i: int,
    curves: dict[str, list[float]],
    results: list[Result],
    top_n: int,
) -> None:
    selected = [r.strategy for r in results if r.strategy != "BH_QQQ"][:top_n]
    selected = ["BH_QQQ", *selected]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=["Date", *selected])
        writer.writeheader()
        for i, date in enumerate(dates[test_start_i:], start=test_start_i):
            row = {"Date": date}
            for name in selected:
                curve = curves[name]
                base = curve[test_start_i]
                row[name] = f"{curve[i] / base * 10_000.0:.6f}"
            writer.writerow(row)


def write_all_curves(
    path: Path,
    *,
    dates: list[str],
    test_start_i: int,
    curves: dict[str, list[float]],
    results: list[Result],
) -> None:
    selected: list[str] = []
    seen: set[str] = set()
    for result in results:
        if result.strategy in curves and result.strategy not in seen:
            selected.append(result.strategy)
            seen.add(result.strategy)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=["Date", *selected])
        writer.writeheader()
        for i, date in enumerate(dates[test_start_i:], start=test_start_i):
            row = {"Date": date}
            for name in selected:
                curve = curves[name]
                base = curve[test_start_i]
                row[name] = f"{curve[i] / base * 10_000.0:.6f}"
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Massively test daily momentum variants on 2026 YTD only.",
    )
    parser.add_argument(
        "--universe-dir",
        type=Path,
        default=PROJECT_ROOT / "derived" / "daily_ohlcv" / "us_largecap_current",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "derived" / "daily_ohlcv" / "max" / "QQQ_1980-01-01_2026-06-15_daily.csv",
    )
    parser.add_argument("--warmup-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end")
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_results.csv",
    )
    parser.add_argument(
        "--curves-output",
        type=Path,
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_top_curves.csv",
    )
    parser.add_argument(
        "--all-curves-output",
        type=Path,
        help="Optional CSV containing every tested 2026 equity curve, sorted like the results file.",
    )
    parser.add_argument("--top-curves", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    universe = load_universe(args.universe_dir)
    qqq = load_price_series(args.benchmark)
    dates = sorted(d for d in qqq if d >= args.warmup_start and (args.test_end is None or d <= args.test_end))
    dates = [d for d in dates if any(d in series for series in universe.values())]
    test_start_i = next(i for i, d in enumerate(dates) if d >= args.test_start)

    symbols = sorted(universe)
    matrix = {
        symbol: [universe[symbol].get(date) for date in dates]
        for symbol in symbols
    }
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

    specs = make_specs()
    results: list[Result] = []
    curves: dict[str, list[float]] = {}
    bh = benchmark_result(dates=dates, qqq_values=qqq_values, test_start_i=test_start_i)
    results.append(bh)
    curves["BH_QQQ"] = [
        10_000.0 * (value / qqq_values[test_start_i])
        if value is not None and qqq_values[test_start_i] is not None
        else 10_000.0
        for value in qqq_values
    ]

    for n, spec in enumerate(specs, 1):
        result, curve = run_spec(
            spec,
            dates=dates,
            matrix=matrix,
            rankings=rankings,
            trend_ok=trend_ok,
            weekly_rebal=weekly,
            monthly_rebal=monthly,
            test_start_i=test_start_i,
            cost_bps=args.cost_bps,
        )
        results.append(result)
        curves[spec.name] = curve
        if n % 500 == 0:
            print(f"tested {n}/{len(specs)} strategies", flush=True)

    results.sort(key=lambda r: r.total_return_pct, reverse=True)
    write_results(args.output, results)
    write_top_curves(
        args.curves_output,
        dates=dates,
        test_start_i=test_start_i,
        curves=curves,
        results=results,
        top_n=args.top_curves,
    )
    if args.all_curves_output:
        write_all_curves(
            args.all_curves_output,
            dates=dates,
            test_start_i=test_start_i,
            curves=curves,
            results=results,
        )

    print(f"Universe symbols: {len(symbols)}")
    print(f"Warmup dates: {dates[0]} -> {dates[test_start_i - 1]}")
    print(f"Test dates: {dates[test_start_i]} -> {dates[-1]} ({len(dates) - test_start_i} rows)")
    print(f"Tested strategies: {len(specs)} + BH_QQQ")
    print(f"Cost model: {args.cost_bps:.2f} bps per unit turnover")
    print(f"Results -> {args.output}")
    print(f"Top curves -> {args.curves_output}")
    if args.all_curves_output:
        print(f"All curves -> {args.all_curves_output}")
    print()
    print("Top 20 by 2026 YTD Total Return:")
    for result in results[:20]:
        print(
            f"{result.strategy:<32} TR={result.total_return_pct:8.2f}% "
            f"Sharpe={result.sharpe:5.2f} MaxDD={result.max_dd_pct:7.2f}% "
            f"Reb={result.rebalances:3d} Hold={result.avg_holdings:5.1f}"
        )
    beaters = [r for r in results if r.strategy != "BH_QQQ" and r.total_return_pct > bh.total_return_pct]
    print()
    print(
        f"Strategies beating BH_QQQ: {len(beaters)}/{len(specs)} "
        f"(BH_QQQ TR={bh.total_return_pct:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
