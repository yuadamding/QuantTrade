"""Small committed provider-shaped SECOND responses; no patched economics."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch

from rl_quant.datasets.massive_raw_seconds_v1 import (
    CapturedSecondPage, RawSecondCatalog, RawSecondContract, RawSecondWindowRef,
    SecondCaptureRef, SecondQuery, publish_second_capture,
)
from rl_quant.envs.raw_second_portfolio_v1 import RawSecondPortfolioEnv, SecondExecutionConfig
from rl_quant.models.raw_second_policy_v1 import RawSecondActorCritic, RawSecondModelConfig
from rl_quant.rl.ppo import PPOConfig
from rl_quant.training.raw_second_ppo_v1 import RawSecondPPOTrainer

START = int(datetime(2017, 1, 3, 14, 30, tzinfo=timezone.utc).timestamp()) * 1000


def response(query, rows, *, next_url=None):
    body = dict(status="OK", ticker=query.ticker, adjusted=False, resultsCount=len(rows), results=rows,
                request_id="synthetic-engineering-only")
    if next_url is not None:
        body["next_url"] = next_url
    return json.dumps(body, separators=(",", ":")).encode()


def make_catalog(root: Path) -> RawSecondCatalog:
    captures = []
    for i, ticker in enumerate(("AAPL", "BRK.B")):
        query = SecondQuery(ticker, START, START + 63_000)
        rows = []
        for second in range(64):
            if second in (10, 20):
                continue  # complete source, explicitly known-empty intervals
            price = 100 + i * 25 + (1 if i == 0 else -1) * second * 0.125
            rows.append(dict(t=START + second * 1000, o=price, h=price + 0.25, l=price - 0.25,
                             c=price + 0.0625, v=3_000_000 + second, vw=price + 0.03125, n=300))
        next_url = query.url + "&cursor=page-two"
        pages = (CapturedSecondPage(query.url, START + 1_000_000, response(query, rows[:30], next_url=next_url)),
                 CapturedSecondPage(next_url, START + 1_000_001, response(query, rows[30:])))
        captures.append(publish_second_capture(root / ticker, query, pages))
    contract = RawSecondContract("historical-finalized-assumed-delay", 0)
    return RawSecondCatalog(tuple(RawSecondWindowRef(tuple(captures), ("fixture-issue-apple", "fixture-issue-berkshire-b"),
                                                    START, 64, START + step * 1000, contract)
                                  for step in (16, 24, 32, 40, 48)))


def save_catalog(catalog: RawSecondCatalog, path: Path):
    with path.open("x") as stream:
        json.dump([asdict(w) for w in catalog.windows], stream, sort_keys=True)


def reopen_catalog(path: Path) -> RawSecondCatalog:
    rows = json.loads(path.read_text())
    for row in rows:
        row["captures"] = tuple(SecondCaptureRef(**c) for c in row["captures"])
        row["asset_ids"] = tuple(row["asset_ids"])
        row["contract"]["market_fields"] = tuple(row["contract"]["market_fields"])
        row["contract"] = RawSecondContract(**row["contract"])
    return RawSecondCatalog(tuple(RawSecondWindowRef(**r) for r in rows))


def trainer(catalog: RawSecondCatalog, *, device="cuda:0") -> RawSecondPPOTrainer:
    model = RawSecondActorCritic(catalog, RawSecondModelConfig(d_model=16, attention_heads=2,
                                local_layers=1, temporal_layers=1, local_block_seconds=8,
                                max_context_seconds=64, asset_chunk_size=1, local_batch_blocks=4)).to(device)
    env = RawSecondPortfolioEnv(catalog, device=device,
          config=SecondExecutionConfig(capital="100000", maximum_asset_weight="0.8", decision_interval_seconds=8))
    return RawSecondPPOTrainer(model, env, PPOConfig(learning_rate=0.001, epochs=2, minibatch_sequences=1, seed=17))


def configure():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(17)
