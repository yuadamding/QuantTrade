from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from raw_second_fixture import START, configure, make_catalog, response, save_catalog, trainer
from rl_quant.datasets.massive_raw_seconds_v1 import (
    CapturedSecondPage, RawSecondCatalog, RawSecondContract, RawSecondWindowRef, SecondQuery,
    load_raw_second_window, publish_second_capture,
)
from rl_quant.envs.raw_second_portfolio_v1 import RawSecondPortfolioEnv, SecondExecutionConfig
from rl_quant.execution.qt200_aggregate_execution_v1 import Book, Dividend, Split, apply_actions, equity
from rl_quant.rl.types import ActionBatch
from rl_quant.workflows.massive_raw_second_rl_v1 import run_raw_second_engineering_episode


@pytest.fixture(autouse=True, scope="module")
def cuda_runtime():
    # This is acceptance, not a CPU fallback or a skip that can qualify a tree.
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Run on an assigned LSF GPU"
    configure()


@pytest.fixture
def catalog(tmp_path):
    return make_catalog(tmp_path / "sources")


def test_raw_projection_receives_only_unchanged_provider_ohlcv(catalog):
    agent = trainer(catalog)
    raw = catalog.load(0, device="cuda:0")
    projected = []
    handle = agent.algorithm.model.input_projection.register_forward_pre_hook(lambda m, args: projected.append(args[0].detach().clone()))
    agent.algorithm.act(agent.environment.observation(), deterministic=True)
    handle.remove()
    actual = torch.cat(projected, dim=1)
    assert torch.equal(actual[raw.observed_mask], raw.raw_ohlcv[raw.observed_mask])
    assert actual.shape[-1] == 5 and actual[0, 0, 0, 4] == 3_000_000
    query, pages = catalog.windows[0].captures[0].load()
    original = json.loads(pages[0].body)["results"][0]
    assert torch.equal(actual[0, 0, 0], torch.tensor([original[k] for k in ("o", "h", "l", "c", "v")], device="cuda:0", dtype=torch.float32))
    assert original["vw"] != original["c"] and query.ticker == "AAPL"


@pytest.mark.parametrize("change", [dict(multiplier=300), dict(timespan="minute"), dict(adjusted=True),
    dict(raw_input_normalization="zscore"), dict(resampling=300), dict(feature_engineering=True),
    dict(market_fields=("open", "high", "low", "close", "volume", "vwap")),
    dict(forward_fill_market_inputs=True), dict(covariates=True), dict(news=True)])
def test_forbidden_input_configuration_rejected(change):
    with pytest.raises(ValueError, match="Forbidden"):
        RawSecondContract("historical-finalized-assumed-delay", 0, **change)


def test_explicit_availability_required():
    with pytest.raises(TypeError):
        RawSecondContract()
    with pytest.raises(ValueError, match="15 minutes"):
        RawSecondContract("developer-delayed-assumption", 1000)


def test_incomplete_pagination_is_unknown_not_empty(tmp_path):
    q = SecondQuery("AAPL", START, START + 1000)
    page = CapturedSecondPage(q.url, START + 5000, response(q, [], next_url=q.url + "&cursor=next"))
    with pytest.raises(ValueError, match="pagination incomplete"):
        publish_second_capture(tmp_path / "incomplete", q, (page,))
    assert not (tmp_path / "incomplete").exists()


def test_empty_failed_duplicate_adjusted_and_credentialed_pages(tmp_path):
    q = SecondQuery("AAPL", START, START + 1000)
    empty = CapturedSecondPage(q.url, START + 5000, response(q, []))
    assert publish_second_capture(tmp_path / "empty", q, (empty,)).load()[0] == q
    for index, patch in enumerate((dict(status="ERROR"), dict(adjusted=True), dict(resultsCount=1),
                                   dict(next_url=q.url + "&apiKey=forbidden-placeholder"))):
        payload = {**json.loads(empty.body), **patch}
        with pytest.raises(ValueError):
            publish_second_capture(tmp_path / f"bad-{index}", q, (replace(empty, body=json.dumps(payload).encode()),))
    duplicate_json = empty.body[:-1] + b',"ticker":"AAPL"}'
    with pytest.raises(ValueError, match="Duplicate JSON"):
        publish_second_capture(tmp_path / "duplicate", q, (replace(empty, body=duplicate_json),))


def test_missingness_and_masked_payload_invariance(catalog):
    agent = trainer(catalog)
    raw = catalog.load(0, device="cuda:0")
    assert raw.known_mask[0, 0, 10] and not raw.observed_mask[0, 0, 10] and not raw.padding_mask[0, 0, 10]
    assert raw.padding_mask[0, 0, 16] and not raw.observed_mask[0, 0, 16]
    changed = raw.raw_ohlcv.clone()
    changed[~raw.observed_mask] = float("nan")
    with torch.no_grad():
        first = agent.algorithm.model.encode_market(raw)
        second = agent.algorithm.model.encode_market(replace(raw, raw_ohlcv=changed))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    unknown = raw.known_mask.clone()
    unknown[:, :, 10] = False
    with pytest.raises(ValueError, match="Unknown coverage"):
        replace(raw, known_mask=unknown).validate()


def test_temporal_causality_and_partial_block(catalog):
    agent = trainer(catalog)
    raw = catalog.load(0, device="cuda:0")
    future = raw.raw_ohlcv.clone()
    future[:, :, 16:] = 1e30
    obs = agent.environment.observation()
    with torch.no_grad():
        a = agent.algorithm.model.forward_raw(raw, obs.tensors["account_state"], obs.action_mask)
        b = agent.algorithm.model.forward_raw(replace(raw, raw_ohlcv=future), obs.tensors["account_state"], obs.action_mask)
    torch.testing.assert_close(a.distribution.concentration, b.distribution.concentration, rtol=0, atol=0)
    partial = load_raw_second_window(replace(catalog.windows[0], decision_ms=START + 13_000), device="cuda:0")
    assert partial.observed_mask[:, :, 13:].sum() == 0
    bad = partial.available_at_ms.clone()
    bad[:, :, 2] = START + 999_000
    with pytest.raises(ValueError, match="Unavailable"):
        replace(partial, available_at_ms=bad).validate()


def test_raw_catalog_tampering_and_no_embedding_fallback(catalog):
    agent = trainer(catalog)
    observation = agent.environment.observation()
    for name in ("forecast", "feature_mean", "embeddings", "holding_age"):
        with pytest.raises(ValueError, match="engineered inputs"):
            agent.algorithm.model({**observation.tensors, name: torch.ones(1, device="cuda:0")}, action_mask=observation.action_mask)
    with pytest.raises(ValueError, match="cache"):
        agent.algorithm.model(observation.tensors, action_mask=observation.action_mask, recurrent_state={"kv": torch.ones(1, device="cuda:0")})
    bad = dict(observation.tensors)
    bad["raw_catalog_sha256"] = torch.zeros_like(bad["raw_catalog_sha256"])
    with pytest.raises(ValueError, match="catalog changed"):
        agent.algorithm.model(bad, action_mask=observation.action_mask)
    path = Path(catalog.windows[0].captures[0].path) / "page-000000.json"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="content changed"):
        agent.algorithm.act(observation)


def test_end_to_end_ppo_updates_projection_and_each_attention_tier(catalog):
    agent = trainer(catalog)
    model = agent.algorithm.model
    names = ("input_projection.weight", "local.0.attention.in_proj_weight", "temporal.0.attention.in_proj_weight",
             "cross_stock.attention.in_proj_weight", "actor.2.weight", "critic.2.weight")
    before = {name: dict(model.named_parameters())[name].detach().clone() for name in names}
    gradients = {name: [] for name in names}
    hooks = [dict(model.named_parameters())[name].register_hook(lambda g, key=name: gradients[key].append(float(g.norm()))) for name in names]
    buffer = agent.collect(steps=2)
    batch = buffer.as_batch()
    assert set(batch.observations) == {"raw_window_index", "raw_catalog_sha256", "account_state"}
    assert batch.observations["raw_window_index"].numel() == 2  # no copied raw tensors or learned embeddings
    metrics = agent.update(buffer)
    for h in hooks:
        h.remove()
    assert metrics["minibatches"] == 4 and all(math.isfinite(float(v)) for v in metrics.values())
    for name in names:
        assert gradients[name] and all(math.isfinite(g) for g in gradients[name]) and max(gradients[name]) > 0
        assert not torch.equal(before[name], dict(model.named_parameters())[name])
    assert any(float(state["step"]) == 4 for state in agent.algorithm.optimizer.state.values())
    assert agent.environment.fills and sum(Decimal(f.fee) for f in agent.environment.fills) > 0


def test_rollout_and_optimizer_recomputation_agree(catalog):
    agent = trainer(catalog)
    batch = agent.collect(steps=2).recurrent_sequences(sequence_length=1)
    agent.algorithm.model.train()
    result = agent.algorithm.model(batch.observations, action_mask=batch.action_masks, valid_mask=batch.valid_mask)
    torch.testing.assert_close(result.distribution.log_prob(batch.actions), batch.old_log_probs, rtol=0, atol=0)
    torch.testing.assert_close(result.value, batch.old_values, rtol=0, atol=0)


def test_same_ledger_replays_costs_and_terminal_compounding(catalog):
    first, replay = trainer(catalog), trainer(catalog)
    actions = (torch.tensor([[0.2, 0.4, 0.4]], device="cuda:0"), torch.tensor([[1., 0., 0.]], device="cuda:0"),
               torch.tensor([[0.2, 0.4, 0.4]], device="cuda:0"), torch.tensor([[0.2, 0.4, 0.4]], device="cuda:0"))
    log_wealth = 0.0
    for request in actions:
        a = first.environment.step(ActionBatch(request))
        b = replay.environment.step(ActionBatch(request))
        assert first.environment.book == replay.environment.book and first.environment.fills == replay.environment.fills
        assert torch.equal(a.executed_action, b.executed_action) and torch.equal(a.reward, b.reward)
        log_wealth += float(a.reward)
        assert float(a.reward) == pytest.approx(math.log(float(a.info["equity_after"] / a.info["equity_before"])), abs=2e-7)
    env = first.environment
    assert env.book.holdings == () and float(a.info["terminal_liquidation_cost"]) > 0
    assert any(Decimal(f.signed_shares) > 0 for f in env.fills) and any(Decimal(f.signed_shares) < 0 for f in env.fills)
    assert math.exp(log_wealth) == pytest.approx(float(env.current_equity / Decimal(env.config.capital)), rel=2e-7)
    # Independent dollar oracle: buys/sells at actual recorded fills, fees, then
    # marked terminal adjustment; no market return is patched or supplied.
    cash = Decimal(env.config.capital)
    shares = {}
    for fill in env.fills:
        q, p, fee = map(Decimal, (fill.signed_shares, fill.price, fill.fee))
        assert fee == abs(q) * p * Decimal("0.002")
        cash -= q * p + fee
        shares[fill.asset_id] = shares.get(fill.asset_id, Decimal(0)) + q
        assert fill.second_start_ms > START + 16_000
    marked = sum(q * env.last_marks[asset] for asset, q in shares.items())
    assert env.book.cash == cash + marked * Decimal("0.998")


def test_split_and_dividend_share_ledger_does_not_modify_market_values(catalog):
    d = Decimal
    book = Book(d(0), (("fixture-issue-apple", d(10)),), action_date="2017-01-03")
    before = equity(book, {"fixture-issue-apple": d(100)})
    split = Split("split-1", "fixture-issue-apple", "2017-01-04", d(1), d(2))
    after = apply_actions(book, "2017-01-04", splits=(split,))
    assert equity(after, {"fixture-issue-apple": d(50)}) == before
    paid = apply_actions(after, "2017-01-05", dividends=(Dividend("div-1", "fixture-issue-apple", "2017-01-05", "2017-01-06", d(1)),))
    assert equity(paid, {"fixture-issue-apple": d(49)}) == before
    raw = catalog.load(0, device="cuda:0")
    assert raw.raw_ohlcv[0, 0, 0, 0] == 100


def test_carried_episode_split_reconciles_without_adjusting_raw_seconds(tmp_path):
    tomorrow = START + 86_400_000
    q = SecondQuery("AAPL", START, tomorrow + 63_000)
    rows = [dict(t=day + second * 1000, o=price, h=price, l=price, c=price, v=3_000_000)
            for day, price in ((START, 100), (tomorrow, 50)) for second in range(64)]
    capture = publish_second_capture(tmp_path / "two-sessions", q,
              (CapturedSecondPage(q.url, tomorrow + 100_000, response(q, rows)),))
    contract = RawSecondContract("historical-finalized-assumed-delay", 0)
    refs = tuple(RawSecondWindowRef((capture,), ("issue-apple",), day, 64, day + second * 1000, contract)
                 for day, second in ((START, 16), (START, 24), (tomorrow, 16), (tomorrow, 24)))
    catalog = RawSecondCatalog(refs)
    env = RawSecondPortfolioEnv(catalog, device="cuda:0", config=SecondExecutionConfig(
          capital="100000", maximum_asset_weight="1", decision_interval_seconds=8),
          splits=(Split("split", "issue-apple", "2017-01-04", Decimal(1), Decimal(2)),))
    env.step(ActionBatch(torch.tensor([[0.2, 0.8]], device="cuda:0")))
    before = env.current_equity
    cash_weight = float(env.book.cash / before)
    prior_count = len(env.fills)
    transition = env.step(ActionBatch(torch.tensor([[cash_weight, 1 - cash_weight]], device="cuda:0")))
    fees = sum(Decimal(f.fee) for f in env.fills[prior_count:])
    assert env.current_equity == before - fees
    assert env.current_equity > before * Decimal("0.99")  # no fictitious 50% split loss
    assert env.book.applied_events == ("split",)
    assert dict(env.book.holdings)["issue-apple"] > 1500
    assert catalog.load(2, device="cuda:0").raw_ohlcv[0, 0, 0, 0] == 50
    assert not transition.terminated.item()


def test_checkpoint_resume_and_fresh_process_replay(catalog, tmp_path):
    agent = trainer(catalog)
    agent.update(agent.collect(steps=2))
    checkpoint = tmp_path / "policy.pt"
    checkpoint_sha = agent.save(checkpoint)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog, catalog_path)
    obs = agent.environment.observation()
    deterministic = agent.algorithm.act(obs, deterministic=True).action.cpu().tolist()
    second = agent.collect(steps=2)
    metrics = agent.update(second)
    final = {k: v.detach().cpu() for k, v in agent.algorithm.model.named_parameters()}
    reference = tmp_path / "reference.pt"
    torch.save(dict(model=final, rewards=second.as_batch().rewards.cpu(), actions=second.as_batch().actions.cpu(),
                    metrics=metrics, deterministic=deterministic), reference)
    script = Path(__file__).with_name("raw_second_resume_probe.py")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for mode in ("resume", "cpu-inference"):
        child_env = {**env, **({"CUDA_VISIBLE_DEVICES": ""} if mode == "cpu-inference" else {})}
        result = subprocess.run([sys.executable, "-B", str(script), str(catalog_path), str(checkpoint), checkpoint_sha,
                                 str(reference), mode], env=child_env, text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha
    with pytest.raises(FileExistsError):
        agent.save(checkpoint)


def test_persisted_raw_second_workflow(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw-second-engineering")
    catalog = make_catalog(root / "sources")
    agent = trainer(catalog)
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in (root / "sources").rglob("*") if p.is_file()}
    result = run_raw_second_engineering_episode(catalog=catalog, output=root / "run", device="cuda:0",
             model_config=agent.algorithm.model.config, execution_config=agent.environment.config,
             ppo_config=agent.algorithm.config, rollout_steps=2)
    assert result["engineering_execution_complete"] and result["optimizer_updates"] == 2
    assert result["fills"] and len(result["trajectory"]) == 4
    assert result["hardware"]["peak_cuda_allocated_bytes"] > 0
    assert not any(result[k] for k in ("real_data_training_ready", "native_v5_authorized", "positive_profitability_authorization_eligible"))
    assert before == {str(p): sha256(p.read_bytes()).hexdigest() for p in (root / "sources").rglob("*") if p.is_file()}
    assert sha256((root / "run/checkpoint.pt").read_bytes()).hexdigest() == result["checkpoint_sha256"]
    persisted = json.loads((root / "run/COMPLETE.json").read_text())
    assert persisted["terminal_equity"] == result["terminal_equity"]
