#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline behavior on a second-context decision dataset.")
    parser.add_argument("--dataset", type=Path, default=DATA_ROOT / "rl_decision_datasets" / "stock_second_context_15m_v001" / "dataset.pt")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def summarize_returns(values) -> dict[str, float | None]:
    import torch

    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"total_return": 0.0, "mean_return": 0.0, "sharpe_like": None}
    total = float(torch.prod(1.0 + finite).item() - 1.0)
    mean = float(finite.mean().item())
    std = float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0
    return {"total_return": total, "mean_return": mean, "sharpe_like": None if std <= 0 else mean / std * math.sqrt(252.0)}


def main() -> int:
    import torch

    from rl_quant.features.stock_second_context import validate_second_context_payload

    args = parse_args()
    payload = torch.load(args.dataset, map_location="cpu", weights_only=True)
    validate_second_context_payload(payload)
    returns = payload["action_returns"].float()
    valid = payload["action_valid_mask"].bool()
    costs = payload["action_cost_bps"].float() / 10_000.0
    net = returns - costs
    best = net.masked_fill(~valid, -1e9).argmax(dim=1)
    best_returns = net[torch.arange(net.shape[0]), best]
    summary = {
        "rows": len(payload["decision_timestamps"]),
        "actions": payload["action_names"],
        "cash": summarize_returns(net[:, 0]),
        "oracle_best_valid_action": summarize_returns(best_returns),
        "valid_action_fraction": float(valid.float().mean().item()),
        "dataset_manifest": payload.get("dataset_manifest", {}),
    }
    output = args.output or args.dataset.with_name("evaluation_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"Evaluation -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
