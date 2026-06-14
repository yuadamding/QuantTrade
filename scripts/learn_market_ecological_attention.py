#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT


FEATURE_NAMES = [
    "ret_1d_z",
    "mom_5d_z",
    "mom_21d_z",
    "vol_21d_z",
    "drawdown_63d_z",
    "log_dollar_volume_z",
    "log_market_cap_z",
]


@dataclass(frozen=True)
class TickerMeta:
    symbol: str
    rank: int
    name: str
    market_cap: float
    exchange: str


@dataclass
class StockDay:
    date: str
    symbol: str
    features: list[float]
    next_rel_return: float | None
    log_mass: float
    market_cap: float
    dollar_volume: float
    close: float


def symbol_from_path(path: Path) -> str:
    return path.name.split("_1980-01-01_")[0].split("_2025-01-01_")[0].removesuffix("_daily.csv")


def float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def load_metadata(path: Path) -> dict[str, TickerMeta]:
    out: dict[str, TickerMeta] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            symbol = row["symbol"]
            out[symbol] = TickerMeta(
                symbol=symbol,
                rank=int(float_or_none(row.get("rank")) or 0),
                name=row.get("short_name") or row.get("long_name") or symbol,
                market_cap=float_or_none(row.get("market_cap")) or 0.0,
                exchange=row.get("exchange") or "",
            )
    return out


def load_ohlcv(path: Path) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            adj = float_or_none(row.get("Adj Close"))
            close = float_or_none(row.get("Close"))
            price = adj if adj is not None and adj > 0 else close
            volume = float_or_none(row.get("Volume")) or 0.0
            if price is not None and price > 0:
                rows[row["Date"]] = (price, max(volume, 0.0))
    return rows


def choose_calendar(universe: dict[str, dict[str, tuple[float, float]]], start: str, end: str | None) -> list[str]:
    counts: dict[str, int] = {}
    for series in universe.values():
        for date in series:
            if date >= start and (end is None or date <= end):
                counts[date] = counts.get(date, 0) + 1
    # Keep dates with a real market cross-section. This avoids odd one-off OTC
    # dates while preserving the full daily US calendar from the top-1000 set.
    threshold = max(10, int(len(universe) * 0.50))
    return sorted(date for date, count in counts.items() if count >= threshold)


def pct_return(series: dict[str, tuple[float, float]], dates: list[str], i0: int, i1: int) -> float | None:
    if i0 < 0 or i1 < 0 or i0 >= len(dates) or i1 >= len(dates):
        return None
    p0 = series.get(dates[i0])
    p1 = series.get(dates[i1])
    if p0 is None or p1 is None or p0[0] <= 0:
        return None
    return p1[0] / p0[0] - 1.0


def trailing_returns(series: dict[str, tuple[float, float]], dates: list[str], end_i: int, window: int) -> list[float]:
    values: list[float] = []
    start = max(1, end_i - window + 1)
    for i in range(start, end_i + 1):
        ret = pct_return(series, dates, i - 1, i)
        if ret is not None:
            values.append(ret)
    return values


def trailing_drawdown(series: dict[str, tuple[float, float]], dates: list[str], end_i: int, window: int) -> float | None:
    prices: list[float] = []
    for i in range(max(0, end_i - window + 1), end_i + 1):
        row = series.get(dates[i])
        if row is not None and row[0] > 0:
            prices.append(row[0])
    if len(prices) < 2:
        return None
    peak = max(prices)
    return prices[-1] / peak - 1.0 if peak > 0 else None


def zscore_columns(rows: list[tuple[str, list[float]]]) -> dict[str, list[float]]:
    if not rows:
        return {}
    width = len(rows[0][1])
    cols = [[values[j] for _symbol, values in rows] for j in range(width)]
    means = [mean(col) for col in cols]
    stds = [pstdev(col) if len(col) >= 2 else 1.0 for col in cols]
    stds = [std if std > 1e-12 else 1.0 for std in stds]
    out: dict[str, list[float]] = {}
    for symbol, values in rows:
        out[symbol] = [max(min((values[j] - means[j]) / stds[j], 5.0), -5.0) for j in range(width)]
    return out


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    top = max(logits)
    exps = [math.exp(max(min(value - top, 60.0), -60.0)) for value in logits]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in exps]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def squared_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def learn_kmeans(vectors: list[list[float]], k: int, iterations: int, max_rows: int) -> list[list[float]]:
    if len(vectors) < k:
        raise ValueError(f"Need at least {k} vectors to learn mediators.")
    if max_rows > 0 and len(vectors) > max_rows:
        stride = len(vectors) / max_rows
        sample = [vectors[int(i * stride)] for i in range(max_rows)]
    else:
        sample = vectors
    ordered = sorted(sample, key=lambda row: (row[2], row[3], row[0]))
    centroids = [ordered[min(len(ordered) - 1, round(i * (len(ordered) - 1) / max(k - 1, 1)))][:] for i in range(k)]
    width = len(centroids[0])
    for _ in range(iterations):
        sums = [[0.0] * width for _ in range(k)]
        counts = [0] * k
        for row in sample:
            best = min(range(k), key=lambda idx: squared_distance(row, centroids[idx]))
            counts[best] += 1
            for j, value in enumerate(row):
                sums[best][j] += value
        for idx in range(k):
            if counts[idx]:
                centroids[idx] = [value / counts[idx] for value in sums[idx]]
    return centroids


def assignments(features: list[float], centroids: list[list[float]], temperature: float) -> list[float]:
    return softmax([-temperature * squared_distance(features, centroid) for centroid in centroids])


def build_stock_days(
    *,
    universe: dict[str, dict[str, tuple[float, float]]],
    metadata: dict[str, TickerMeta],
    dates: list[str],
    start_date: str,
    mass_mode: str,
) -> tuple[dict[str, list[StockDay]], list[list[float]]]:
    by_date: dict[str, list[StockDay]] = {}
    vectors: list[list[float]] = []
    market_next_by_date: dict[str, float] = {}

    raw_by_date: dict[str, list[tuple[str, list[float], float, float, float, float | None]]] = {}
    for i, date in enumerate(dates):
        raw_rows: list[tuple[str, list[float], float, float, float, float | None]] = []
        next_returns: list[float] = []
        for symbol, series in universe.items():
            today = series.get(date)
            if today is None:
                continue
            ret_1d = pct_return(series, dates, i - 1, i)
            mom_5d = pct_return(series, dates, i - 5, i)
            mom_21d = pct_return(series, dates, i - 21, i)
            returns_21d = trailing_returns(series, dates, i, 21)
            drawdown_63d = trailing_drawdown(series, dates, i, 63)
            next_ret = pct_return(series, dates, i, i + 1)
            if (
                ret_1d is None
                or mom_5d is None
                or mom_21d is None
                or drawdown_63d is None
                or len(returns_21d) < 10
            ):
                continue
            if next_ret is not None:
                next_returns.append(next_ret)
            meta = metadata.get(symbol, TickerMeta(symbol, 0, symbol, 0.0, ""))
            close, volume = today
            dollar_volume = close * volume
            market_cap = meta.market_cap
            log_market_cap = math.log(max(market_cap, 1.0))
            log_dollar_volume = math.log1p(max(dollar_volume, 0.0))
            raw = [
                ret_1d,
                mom_5d,
                mom_21d,
                pstdev(returns_21d) if len(returns_21d) >= 2 else 0.0,
                drawdown_63d,
                log_dollar_volume,
                math.log1p(max(market_cap, 0.0)),
            ]
            if mass_mode == "market_cap":
                log_mass = log_market_cap
            elif mass_mode == "dollar_volume":
                log_mass = log_dollar_volume
            elif mass_mode == "liquidity_adjusted":
                log_mass = 0.5 * log_market_cap + 0.5 * log_dollar_volume
            else:
                raise ValueError(f"Unknown mass_mode {mass_mode!r}")
            raw_rows.append((symbol, raw, log_mass, market_cap, dollar_volume, next_ret))
        raw_by_date[date] = raw_rows
        if next_returns:
            market_next_by_date[date] = mean(next_returns)

    for date, raw_rows in raw_by_date.items():
        if date < start_date:
            continue
        zscored = zscore_columns([(symbol, raw) for symbol, raw, *_rest in raw_rows])
        day_rows: list[StockDay] = []
        market_next = market_next_by_date.get(date)
        for symbol, _raw, log_mass, market_cap, dollar_volume, next_ret in raw_rows:
            features = zscored.get(symbol)
            if features is None:
                continue
            close = universe[symbol][date][0]
            next_rel = None if next_ret is None or market_next is None else next_ret - market_next
            day = StockDay(
                date=date,
                symbol=symbol,
                features=features,
                next_rel_return=next_rel,
                log_mass=log_mass,
                market_cap=market_cap,
                dollar_volume=dollar_volume,
                close=close,
            )
            day_rows.append(day)
            vectors.append(features)
        if day_rows:
            by_date[date] = day_rows
    return by_date, vectors


def learn_causal_betas(
    by_date: dict[str, list[StockDay]],
    centroids: list[list[float]],
    assignment_temperature: float,
) -> tuple[list[float], list[list[float]], list[float]]:
    k = len(centroids)
    width = len(centroids[0])
    sum_w = [0.0] * k
    sum_y = [0.0] * k
    sum_x2 = [[0.0] * width for _ in range(k)]
    sum_xy = [[0.0] * width for _ in range(k)]
    for rows in by_date.values():
        log_weights = softmax([row.log_mass for row in rows])
        for row, mass_weight in zip(rows, log_weights):
            if row.next_rel_return is None:
                continue
            probs = assignments(row.features, centroids, assignment_temperature)
            for m, prob in enumerate(probs):
                w = mass_weight * prob
                sum_w[m] += w
                sum_y[m] += w * row.next_rel_return
                for j, x in enumerate(row.features):
                    sum_x2[m][j] += w * x * x
                    sum_xy[m][j] += w * x * row.next_rel_return

    alphas = [sum_y[m] / sum_w[m] if sum_w[m] > 0 else 0.0 for m in range(k)]
    betas = [
        [sum_xy[m][j] / sum_x2[m][j] if sum_x2[m][j] > 1e-12 else 0.0 for j in range(width)]
        for m in range(k)
    ]
    support = sum_w
    return alphas, betas, support


def prediction(features: list[float], alpha: float, beta: list[float]) -> float:
    return alpha + sum(b * x for b, x in zip(beta, features))


def entropy(weights: list[float]) -> float:
    return -sum(w * math.log(max(w, 1e-30)) for w in weights)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: float) -> str:
    return f"{value:.10f}" if math.isfinite(value) else ""


def build_outputs(
    *,
    output_dir: Path,
    by_date: dict[str, list[StockDay]],
    metadata: dict[str, TickerMeta],
    centroids: list[list[float]],
    alphas: list[float],
    betas: list[list[float]],
    support: list[float],
    assignment_temperature: float,
    mass_temperature: float,
    causal_temperature: float,
    top_edges_per_day: int,
    source_summary: dict[str, object],
) -> None:
    k = len(centroids)
    context_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    mediator_daily_rows: list[dict[str, object]] = []

    for date in sorted(by_date):
        rows = by_date[date]
        mass_weights = softmax([row.log_mass for row in rows])
        probs_by_row = [assignments(row.features, centroids, assignment_temperature) for row in rows]
        q = [sum(mass_weight * probs[m] for mass_weight, probs in zip(mass_weights, probs_by_row)) for m in range(k)]

        context_row: dict[str, object] = {
            "Date": date,
            "ActiveSymbols": len(rows),
            "ContextEntropy": fmt(entropy(q)),
            "TotalLogMass": fmt(math.log(sum(math.exp(min(row.log_mass, 700.0)) for row in rows))),
        }
        day_edge_candidates: list[dict[str, object]] = []
        for m in range(k):
            pred_by_row = [prediction(row.features, alphas[m], betas[m]) for row in rows]
            logits = [
                mass_temperature * row.log_mass
                + assignment_temperature * probs[m]
                + causal_temperature * pred
                for row, probs, pred in zip(rows, probs_by_row, pred_by_row)
            ]
            attn = softmax(logits)
            pred_ret = sum(weight * pred for weight, pred in zip(attn, pred_by_row))
            actual_terms = [
                weight * row.next_rel_return
                for weight, row in zip(attn, rows)
                if row.next_rel_return is not None
            ]
            actual_next = sum(actual_terms) if actual_terms else float("nan")
            feature_context = [
                sum(weight * row.features[j] for weight, row in zip(attn, rows))
                for j in range(len(FEATURE_NAMES))
            ]
            context_row[f"q_m{m}"] = fmt(q[m])
            context_row[f"pred_rel_ret_m{m}"] = fmt(pred_ret)
            context_row[f"actual_next_rel_ret_m{m}"] = fmt(actual_next)
            for j, feature_name in enumerate(FEATURE_NAMES):
                context_row[f"ctx_m{m}_{feature_name}"] = fmt(feature_context[j])
            mediator_daily_rows.append(
                {
                    "Date": date,
                    "Mediator": m,
                    "ProgramMass": fmt(q[m]),
                    "PredNextRelReturn": fmt(pred_ret),
                    "ActualNextRelReturn": fmt(actual_next),
                    "AttentionEntropy": fmt(entropy(attn)),
                    "TopTicker": rows[max(range(len(rows)), key=lambda idx: attn[idx])].symbol,
                }
            )
            for row, probs, pred, weight in zip(rows, probs_by_row, pred_by_row, attn):
                edge_score = sigmoid(assignment_temperature * probs[m] + causal_temperature * pred)
                meta = metadata.get(row.symbol, TickerMeta(row.symbol, 0, row.symbol, 0.0, ""))
                day_edge_candidates.append(
                    {
                        "Date": date,
                        "Ticker": row.symbol,
                        "Company": meta.name,
                        "Exchange": meta.exchange,
                        "MarketCapRank": meta.rank,
                        "Mediator": m,
                        "Assignment": fmt(probs[m]),
                        "AttentionWeight": fmt(weight),
                        "EdgeScore": fmt(edge_score),
                        "PredNextRelReturn": fmt(pred),
                        "ActualNextRelReturn": fmt(row.next_rel_return if row.next_rel_return is not None else float("nan")),
                        "MarketCap": f"{row.market_cap:.0f}",
                        "DollarVolume": f"{row.dollar_volume:.2f}",
                        "Close": f"{row.close:.6f}",
                        "SortScore": weight * edge_score,
                    }
                )
        context_row["TopMediator"] = max(range(k), key=lambda idx: q[idx])
        context_rows.append(context_row)
        for candidate in sorted(day_edge_candidates, key=lambda row: row["SortScore"], reverse=True)[:top_edges_per_day]:
            candidate.pop("SortScore", None)
            edge_rows.append(candidate)

    program_rows: list[dict[str, object]] = []
    for m, centroid in enumerate(centroids):
        row: dict[str, object] = {
            "Mediator": m,
            "TrainingSupportWeight": fmt(support[m]),
            "AlphaNextRelReturn": fmt(alphas[m]),
        }
        for feature_name, value in zip(FEATURE_NAMES, centroid):
            row[f"centroid_{feature_name}"] = fmt(value)
        for feature_name, value in zip(FEATURE_NAMES, betas[m]):
            row[f"beta_{feature_name}"] = fmt(value)
        program_rows.append(row)

    context_fields = list(context_rows[0].keys()) if context_rows else ["Date"]
    program_fields = list(program_rows[0].keys()) if program_rows else ["Mediator"]
    mediator_daily_fields = [
        "Date",
        "Mediator",
        "ProgramMass",
        "PredNextRelReturn",
        "ActualNextRelReturn",
        "AttentionEntropy",
        "TopTicker",
    ]
    edge_fields = [
        "Date",
        "Ticker",
        "Company",
        "Exchange",
        "MarketCapRank",
        "Mediator",
        "Assignment",
        "AttentionWeight",
        "EdgeScore",
        "PredNextRelReturn",
        "ActualNextRelReturn",
        "MarketCap",
        "DollarVolume",
        "Close",
    ]
    write_rows(output_dir / "market_ecology_daily_context.csv", context_fields, context_rows)
    write_rows(output_dir / "market_ecology_mediators.csv", program_fields, program_rows)
    write_rows(output_dir / "market_ecology_mediator_daily.csv", mediator_daily_fields, mediator_daily_rows)
    write_rows(output_dir / "market_ecology_top_edges.csv", edge_fields, edge_rows)

    summary = {
        **source_summary,
        "feature_names": FEATURE_NAMES,
        "n_mediators": k,
        "assignment_temperature": assignment_temperature,
        "mass_attention_temperature": mass_temperature,
        "causal_temperature": causal_temperature,
        "outputs": {
            "daily_context": str(output_dir / "market_ecology_daily_context.csv"),
            "mediators": str(output_dir / "market_ecology_mediators.csv"),
            "mediator_daily": str(output_dir / "market_ecology_mediator_daily.csv"),
            "top_edges": str(output_dir / "market_ecology_top_edges.csv"),
        },
    }
    (output_dir / "market_ecology_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "README.md").write_text(readme_text(summary))


def readme_text(summary: dict[str, object]) -> str:
    return f"""# Whole-Market Ecological Attention Context

This folder adapts the CREDO causal ecological attention idea to the stock
market cross-section.

## CREDO Analogy

- CREDO groups are perturbations; here groups are stocks.
- CREDO finite-measure mass is cell/count mass; here mass is market cap, falling
  back to dollar volume when market cap is unavailable. The default run uses a
  liquidity-adjusted mass, approximately `sqrt(market_cap * dollar_volume)`, so
  inactive cross-listings do not dominate the ecological context.
- CREDO latent particles are cell states; here each stock-day is represented by
  trailing return, volatility, drawdown, liquidity, and size features.
- CREDO mediator edges are intervention-addressable ecological channels; here
  mediators are learned market regimes/programs and each stock receives
  stock-to-mediator attention/edge scores.
- CREDO requires full-context ecological caching for claim-grade attention; this
  script computes every daily context from the full active top-1000 cross-section,
  not from ticker chunks.

## Policy/Model

The model first learns mediator centroids from all 2026 stock-day feature
vectors. It then learns causal-lag coefficients from day `t` features to day
`t+1` stock return relative to the equal-weight market. Daily context is built
using only day `t` features, mass-biased attention, and the learned causal-lag
edge coefficients.

This is a research context learner, not a production causal claim. For live use,
the same code should be run in a rolling or expanding window so mediator betas
are learned only from dates prior to the prediction date.

## Data Format

- `market_ecology_daily_context.csv`: one row per trading date with global
  program mass `q_m*`, predicted/realized mediator next relative return, and
  mediator feature contexts.
- `market_ecology_mediators.csv`: mediator centroids and feature-to-next-return
  coefficients.
- `market_ecology_mediator_daily.csv`: compact date-mediator panel.
- `market_ecology_top_edges.csv`: strongest stock-to-mediator edges per date.

## Frequency

Daily close-to-close. Output dates are `{summary["start_date"]}` to
`{summary["end_date"]}`. Features may use pre-output warmup prices for trailing
windows, while context rows and learned stock-day states are 2026 rows.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn a whole-market causal ecological attention context from daily OHLCV data.",
    )
    parser.add_argument(
        "--universe-dir",
        type=Path,
        default=PROJECT_ROOT / "derived" / "daily_ohlcv" / "top_us_market_cap_1000_2026-06-14",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "derived" / "universes" / "top_us_market_cap_1000_2026-06-14.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "derived" / "market_ecology" / "top1000_2026",
    )
    parser.add_argument("--warmup-start", default="2025-09-01")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--n-mediators", type=int, default=8)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--max-training-rows", type=int, default=60000)
    parser.add_argument("--assignment-temperature", type=float, default=1.5)
    parser.add_argument("--mass-attention-temperature", type=float, default=0.08)
    parser.add_argument("--causal-temperature", type=float, default=50.0)
    parser.add_argument("--top-edges-per-day", type=int, default=25)
    parser.add_argument(
        "--mass-mode",
        choices=["liquidity_adjusted", "market_cap", "dollar_volume"],
        default="liquidity_adjusted",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    universe = {
        symbol_from_path(path): load_ohlcv(path)
        for path in sorted(args.universe_dir.glob("*_daily.csv"))
    }
    universe = {symbol: rows for symbol, rows in universe.items() if rows}
    dates = choose_calendar(universe, args.warmup_start, args.end_date)
    by_date, vectors = build_stock_days(
        universe=universe,
        metadata=metadata,
        dates=dates,
        start_date=args.start_date,
        mass_mode=args.mass_mode,
    )
    if not by_date:
        raise ValueError("No stock-day rows were built. Check date range and input directory.")
    centroids = learn_kmeans(
        vectors,
        k=args.n_mediators,
        iterations=args.kmeans_iterations,
        max_rows=args.max_training_rows,
    )
    alphas, betas, support = learn_causal_betas(
        by_date,
        centroids,
        assignment_temperature=args.assignment_temperature,
    )
    output_dates = sorted(by_date)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = {
        "source": "CREDO-inspired whole-market causal ecological attention",
        "universe_dir": str(args.universe_dir),
        "metadata": str(args.metadata),
        "start_date": output_dates[0],
        "end_date": output_dates[-1],
        "frequency": "daily_close_to_close",
        "active_symbols_min": min(len(rows) for rows in by_date.values()),
        "active_symbols_max": max(len(rows) for rows in by_date.values()),
        "stock_day_rows": sum(len(rows) for rows in by_date.values()),
        "calendar_rows": len(output_dates),
        "claim_boundary": "research context learner; causal-lag predictive coefficients, not structural causal proof",
        "mass_mode": args.mass_mode,
    }
    build_outputs(
        output_dir=args.output_dir,
        by_date=by_date,
        metadata=metadata,
        centroids=centroids,
        alphas=alphas,
        betas=betas,
        support=support,
        assignment_temperature=args.assignment_temperature,
        mass_temperature=args.mass_attention_temperature,
        causal_temperature=args.causal_temperature,
        top_edges_per_day=args.top_edges_per_day,
        source_summary=source_summary,
    )
    print(f"Dates: {output_dates[0]} -> {output_dates[-1]} ({len(output_dates)} rows)")
    print(f"Stock-day rows: {source_summary['stock_day_rows']}")
    print(
        "Active symbols/day: "
        f"{source_summary['active_symbols_min']} -> {source_summary['active_symbols_max']}"
    )
    print(f"Mediators learned: {args.n_mediators}")
    print(f"Output -> {args.output_dir}")
    print("Top learned mediator alphas:")
    for idx in sorted(range(len(alphas)), key=lambda m: alphas[m], reverse=True):
        print(
            f"  mediator {idx}: alpha_next_rel={alphas[idx] * 10000.0:8.3f} bps "
            f"support={support[idx]:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
