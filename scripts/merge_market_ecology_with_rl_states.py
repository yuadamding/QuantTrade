#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT


def read_by_date(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        return fieldnames, {row["Date"]: row for row in reader}


def ecology_feature_columns(fieldnames: list[str]) -> list[str]:
    out: list[str] = []
    for name in fieldnames:
        if name == "ContextEntropy" or name.startswith("q_m") or name.startswith("pred_rel_ret_m"):
            out.append(name)
        elif name.startswith("ctx_m") and not name.endswith("actual_next_rel_ret"):
            out.append(name)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append non-oracle market ecological context features to RL daily state features.",
    )
    parser.add_argument(
        "--state-features",
        type=Path,
        default=PROJECT_ROOT / "derived" / "rl_daily" / "stock_top1000_2026" / "state_features.csv",
    )
    parser.add_argument(
        "--ecology-context",
        type=Path,
        default=PROJECT_ROOT / "derived" / "market_ecology" / "top1000_2026" / "market_ecology_daily_context.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "derived" / "rl_daily" / "stock_top1000_2026" / "state_features_with_market_ecology.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_fields, state_rows = read_by_date(args.state_features)
    ecology_fields, ecology_rows = read_by_date(args.ecology_context)
    selected_ecology = ecology_feature_columns(ecology_fields)
    output_fields = list(state_fields) + [f"ecology_{name}" for name in selected_ecology]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dates = [date for date in state_rows if date in ecology_rows]
    with args.output.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=output_fields)
        writer.writeheader()
        for date in dates:
            row = dict(state_rows[date])
            ecology = ecology_rows[date]
            for name in selected_ecology:
                row[f"ecology_{name}"] = ecology.get(name, "")
            writer.writerow(row)

    print(f"Aligned dates: {len(dates)}")
    print(f"Base state features: {len(state_fields) - 1}")
    print(f"Ecology features added: {len(selected_ecology)}")
    print(f"Output -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
