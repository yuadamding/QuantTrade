#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


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


def read_metrics(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="") as source:
        return {row["Strategy"]: row for row in csv.DictReader(source)}


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


def parse_strategy_family(name: str) -> str:
    if name.startswith("BH_"):
        return "benchmark"
    if name.startswith("CS_"):
        return "cross_sectional"
    if name.startswith("DM_"):
        return "dual_momentum"
    return "other"


def metric_number(metrics: dict[str, dict[str, str]], strategy: str, key: str) -> float:
    return float_or_nan(metrics.get(strategy, {}).get(key))


def metric_text(metrics: dict[str, dict[str, str]], strategy: str, key: str) -> str:
    return metrics.get(strategy, {}).get(key, "")


def topn_value(metrics: dict[str, dict[str, str]], strategy: str) -> int:
    value = metric_number(metrics, strategy, "TopN")
    return int(value) if math.isfinite(value) else 0


def stable_score(
    *,
    sharpe: float,
    total_return: float,
    max_dd: float,
    annual_vol: float,
    positive_rate: float,
) -> float:
    if annual_vol <= 0 or not all(math.isfinite(x) for x in (sharpe, total_return, max_dd, positive_rate)):
        return -float("inf")
    # Prefer smooth, positive compounding. The denominator penalizes both daily
    # variance and drawdown; the log return term prevents one huge winner from
    # overwhelming stability.
    return (
        sharpe
        * math.log1p(max(total_return, -0.999))
        * max(positive_rate, 0.0)
        / ((1.0 + annual_vol) * (1.0 + abs(max_dd)))
    )


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rank_strategies(
    *,
    curves_path: Path,
    metrics_path: Path | None,
) -> tuple[list[dict[str, object]], str]:
    metrics = read_metrics(metrics_path)
    with curves_path.open(newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        if not header or header[0] != "Date":
            raise ValueError(f"{curves_path} must have Date as the first column")
        names = header[1:]
        dates: list[str] = []
        curves = {name: [] for name in names}
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

    rows: list[dict[str, object]] = []
    for name in names:
        values = curves[name]
        returns = daily_returns(values)
        if len(values) < 2 or not returns:
            continue
        total_return = values[-1] / values[0] - 1.0 if values[0] > 0 else float("nan")
        daily_mean = mean(returns)
        daily_var = pstdev(returns) ** 2 if len(returns) >= 2 else 0.0
        daily_std = math.sqrt(daily_var)
        annual_vol = daily_std * math.sqrt(252.0)
        sharpe = daily_mean / daily_std * math.sqrt(252.0) if daily_std > 0 else float("nan")
        drawdown = max_drawdown(values)
        positive_rate = sum(1 for ret in returns if ret > 0) / len(returns)
        worst_day = min(returns)
        best_day = max(returns)
        score = stable_score(
            sharpe=sharpe,
            total_return=total_return,
            max_dd=drawdown,
            annual_vol=annual_vol,
            positive_rate=positive_rate,
        )
        rows.append(
            {
                "Strategy": name,
                "Family": parse_strategy_family(name),
                "Freq": metric_text(metrics, name, "Freq"),
                "TopN": topn_value(metrics, name),
                "Lookback": metric_text(metrics, name, "Lookback"),
                "Skip": metric_text(metrics, name, "Skip"),
                "Trend Filter": metric_text(metrics, name, "Trend Filter"),
                "Total Return [%]": total_return * 100.0,
                "Daily Mean [%]": daily_mean * 100.0,
                "Daily Variance": daily_var,
                "Daily Std [%]": daily_std * 100.0,
                "Annual Vol [%]": annual_vol * 100.0,
                "Sharpe": sharpe,
                "Max DD [%]": drawdown * 100.0,
                "Positive Day Rate [%]": positive_rate * 100.0,
                "Worst Day [%]": worst_day * 100.0,
                "Best Day [%]": best_day * 100.0,
                "Stable Score": score,
                "Final Equity": values[-1],
            }
        )
    return rows, f"{dates[0]} -> {dates[-1]} ({len(dates)} rows)"


def formatted_row(row: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, float):
            if key == "Daily Variance":
                out[key] = f"{value:.12f}"
            elif math.isfinite(value):
                out[key] = f"{value:.6f}"
            else:
                out[key] = ""
        else:
            out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank 2026 strategies by variance and stable performance.",
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
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_stability_rank_top1000.csv",
    )
    parser.add_argument(
        "--stable-output",
        type=Path,
        default=PROJECT_ROOT / "derived" / "backtests" / "massive_2026_momentum_stable_picks_top1000.csv",
    )
    parser.add_argument("--min-total-return", type=float, default=50.0)
    parser.add_argument("--min-sharpe", type=float, default=2.0)
    parser.add_argument("--max-dd", type=float, default=15.0, help="Maximum drawdown magnitude in percent.")
    parser.add_argument("--min-topn", type=int, default=10)
    parser.add_argument("--top", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, date_summary = rank_strategies(curves_path=args.curves, metrics_path=args.metrics)
    bh = next((row for row in rows if row["Strategy"] == "BH_QQQ"), None)
    bh_return = float(bh["Total Return [%]"]) if bh is not None else args.min_total_return
    min_return = max(args.min_total_return, bh_return)

    variance_ranked = sorted(
        rows,
        key=lambda row: (
            float(row["Daily Variance"]),
            -float(row["Total Return [%]"]),
        ),
    )
    stable_rows = [
        row
        for row in rows
        if row["Strategy"] != "BH_QQQ"
        and float(row["Total Return [%]"]) >= min_return
        and float(row["Sharpe"]) >= args.min_sharpe
        and float(row["Max DD [%]"]) >= -args.max_dd
        and int(row["TopN"]) >= args.min_topn
    ]
    stable_ranked = sorted(
        stable_rows,
        key=lambda row: (
            -float(row["Stable Score"]),
            float(row["Daily Variance"]),
            -float(row["Sharpe"]),
        ),
    )

    fields = [
        "Strategy",
        "Family",
        "Freq",
        "TopN",
        "Lookback",
        "Skip",
        "Trend Filter",
        "Total Return [%]",
        "Daily Mean [%]",
        "Daily Variance",
        "Daily Std [%]",
        "Annual Vol [%]",
        "Sharpe",
        "Max DD [%]",
        "Positive Day Rate [%]",
        "Worst Day [%]",
        "Best Day [%]",
        "Stable Score",
        "Final Equity",
    ]
    write_rows(args.output, fields, [formatted_row(row) for row in variance_ranked])
    write_rows(args.stable_output, fields, [formatted_row(row) for row in stable_ranked])

    print(f"Dates: {date_summary}")
    print(f"Strategies ranked: {len(rows)}")
    if bh is not None:
        print(
            "BH_QQQ: "
            f"TR={float(bh['Total Return [%]']):.2f}% "
            f"Var={float(bh['Daily Variance']):.8f} "
            f"Vol={float(bh['Annual Vol [%]']):.2f}% "
            f"Sharpe={float(bh['Sharpe']):.2f} "
            f"MaxDD={float(bh['Max DD [%]']):.2f}%"
        )
    print(
        "Stable filter: "
        f"TR>={min_return:.2f}%, Sharpe>={args.min_sharpe:.2f}, "
        f"MaxDD>=-{args.max_dd:.2f}%, TopN>={args.min_topn}"
    )
    print(f"Variance ranking -> {args.output}")
    print(f"Stable picks -> {args.stable_output}")
    print()
    print("Top stable picks:")
    for row in stable_ranked[: args.top]:
        print(
            f"{row['Strategy']:<34} "
            f"TR={float(row['Total Return [%]']):8.2f}% "
            f"Var={float(row['Daily Variance']):.8f} "
            f"Vol={float(row['Annual Vol [%]']):6.2f}% "
            f"Sharpe={float(row['Sharpe']):5.2f} "
            f"MaxDD={float(row['Max DD [%]']):7.2f}% "
            f"Pos={float(row['Positive Day Rate [%]']):5.1f}% "
            f"Score={float(row['Stable Score']):.4f}"
        )
    print()
    print("Lowest variance profitable strategies:")
    shown = 0
    for row in variance_ranked:
        if row["Strategy"] == "BH_QQQ":
            continue
        if float(row["Total Return [%]"]) < bh_return:
            continue
        print(
            f"{row['Strategy']:<34} "
            f"TR={float(row['Total Return [%]']):8.2f}% "
            f"Var={float(row['Daily Variance']):.8f} "
            f"Vol={float(row['Annual Vol [%]']):6.2f}% "
            f"Sharpe={float(row['Sharpe']):5.2f} "
            f"MaxDD={float(row['Max DD [%]']):7.2f}% "
            f"TopN={int(row['TopN']):2d}"
        )
        shown += 1
        if shown >= min(args.top, 10):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
