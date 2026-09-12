"""Frozen, deliberately easy synthetic market; not real-price-scale evidence.

The generator writes provider-shaped raw observations, not model features.
Completed high/low excursions predict subsequent opposing equity movements.
Diagnostic regime labels are never attached to a catalog, tensor or account.
The production runner owns all actions, fills, rewards, selection and reports.
"""

import cProfile
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import pstats
from random import Random
from statistics import mean, median
import time

import torch

from raw_second_fixture import START, event_coverage, response, save_catalog
from rl_quant.datasets.massive_raw_seconds_v1 import (
    CapturedSecondPage, RawSecondCatalog, RawSecondContract, RawSecondWindowRef,
    SecondQuery, publish_second_capture,
)
from rl_quant.datasets.raw_second_economics_v1 import SecondEconomicInputs
from rl_quant.envs.raw_second_portfolio_v1 import SecondExecutionConfig
from rl_quant.models.raw_second_policy_v1 import RawSecondModelConfig
from rl_quant.rl.ppo import PPOConfig
from rl_quant.workflows.raw_second_experiment_v1 import run_raw_second_experiment

SCHEMA = "rl-quant.raw-second-controlled-economic-learning-v1"
SEEDS = (17, 29, 43)
ROLES = (("train", 256, 11017), ("validation", 32, 22029), ("test", 32, 33043))
ASSETS = ("synthetic-economic-issue-a", "synthetic-economic-issue-b")
INTERVAL_SECONDS = 8
ROLLOUT_STEPS = 16


def model_config() -> RawSecondModelConfig:
    return RawSecondModelConfig(d_model=16, attention_heads=2, local_layers=1,
        temporal_layers=1, local_block_seconds=4, max_context_seconds=8,
        asset_chunk_size=2, local_batch_blocks=4)


def execution_config() -> SecondExecutionConfig:
    return SecondExecutionConfig(capital="10000000", maximum_asset_weight="0.8",
        maximum_fill_participation="0.02", maximum_drawdown="0.25",
        decision_interval_seconds=INTERVAL_SECONDS, observation_session="regular-only",
        execution_session="regular-only", order_expiry="session-close")


def ppo_config(seed: int) -> PPOConfig:
    return PPOConfig(seed=seed, learning_rate=0.003, epochs=4,
                     minibatch_sequences=None, entropy_coefficient=0.001)


def _publish(path: Path, document: dict) -> str:
    body = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with path.open("xb") as stream:
        stream.write(body)
    return sha256(body).hexdigest()


def specification() -> dict:
    """Written before source generation or model initialization; no tuning API."""
    return dict(schema=SCHEMA, policy_seeds=SEEDS, roles=ROLES, asset_ids=ASSETS,
        generator="python.random.Random/MT19937;separate-from-policy-RNG",
        initial_price_range=[9500, 10500], share_volume_per_second=20000,
        future_price_multiplier=[0.97, 1.03], favorable_hl=[1.5, 0.99],
        unfavorable_hl=[1.01, 0.5], cue_seconds_before_decision=[2, 1],
        payoff_second_after_decision=3, balanced_winner_block=16,
        market_streams_common_across_policy_seeds=True,
        model_config=asdict(model_config()), execution_config=asdict(execution_config()),
        ppo_configs=[asdict(ppo_config(seed)) for seed in SEEDS], rollout_steps=ROLLOUT_STEPS,
        favorable_weight_margin_definition="requested weights before caps/repair;not executed exposure",
        acceptance=dict(minimum_improved_seed_count=2, minimum_median_net_improvement=0.01,
                        minimum_median_trained_net_return=0.0,
                        minimum_median_favorable_weight_margin=0.05,
                        minimum_median_margin_improvement=0.02),
        constraints="same ledger, costs, cash, shares, participation and drawdown; no reward override",
        limitations=["extreme synthetic OHLC excursions and balanced arbitrary raw units",
                     "not realistic source, real-price-scale, H100, or alpha qualification",
                     "one chronological pass; model seeds share market histories",
                     "positive compounding can bind liquidity; retain clipping and drawdown failures",
                     "no per-seed adjustment or reranking by held-out outcomes"])


@dataclass(frozen=True)
class EconomicFixture:
    root: Path
    catalogs: tuple[RawSecondCatalog, ...]
    economics: SecondEconomicInputs
    opportunities: tuple[tuple[int, ...], ...]
    specification_sha256: str


def _catalog(root: Path, *, start: int, transitions: int, data_seed: int):
    rng = Random(data_seed)  # independent of torch/PPO and other scored roles
    initial_rng = sha256(json.dumps(rng.getstate()).encode()).hexdigest()
    winners = []
    for _ in range(transitions // ROLLOUT_STEPS):
        block = [0] * 8 + [1] * 8
        rng.shuffle(block)
        winners.extend(block)
    captures = []
    duration = (transitions + 1) * INTERVAL_SECONDS
    for asset, ticker in enumerate(("SYN.A", "SYN.B")):
        price = rng.uniform(9500, 10500)
        rows = []
        for second in range(duration):
            opened = price
            step, phase = divmod(second - INTERVAL_SECONDS, INTERVAL_SECONDS)
            if 0 <= step < transitions and phase == 3:
                price *= 1.03 if winners[step] == asset else 0.97
            high, low = max(opened, price), min(opened, price)
            # These are completed raw-second paths BEFORE the next decision.
            # O=C at the cue: no favorable retrospective entry price is used.
            next_step = second // INTERVAL_SECONDS
            if second % INTERVAL_SECONDS in (6, 7) and next_step < transitions:
                high *= 1.5 if winners[next_step] == asset else 1.01
                low *= 0.99 if winners[next_step] == asset else 0.5
            rows.append(dict(t=start + second * 1000, o=opened, h=high,
                             l=low, c=price, v=20000))
        query = SecondQuery(ticker, start, start + (duration - 1) * 1000)
        # Every role remains below the acquisition client's one-hour boundary.
        captures.append(publish_second_capture(root / ticker, query,
            (CapturedSecondPage(query.url, query.end_ms + 5000, response(query, rows)),)))
    contract = RawSecondContract("historical-finalized-assumed-delay", 0)
    windows = tuple(RawSecondWindowRef(tuple(captures), ASSETS,
        start + step * INTERVAL_SECONDS * 1000, INTERVAL_SECONDS,
        start + (step + 1) * INTERVAL_SECONDS * 1000, contract)
        for step in range(transitions + 1))
    catalog = RawSecondCatalog(windows)
    generator = dict(generator_seed=data_seed, initial_rng_state_sha256=initial_rng,
        final_rng_state_sha256=sha256(json.dumps(rng.getstate()).encode()).hexdigest(),
        realization_catalog_sha256=catalog.identity)
    return catalog, tuple(winners), generator


def prepare_economic_fixture(root: Path) -> EconomicFixture:
    root.mkdir(parents=True, exist_ok=False)
    spec_sha = _publish(root / "specification.json", specification())
    catalogs, opportunities, generators = [], [], []
    for index, (role, transitions, data_seed) in enumerate(ROLES):
        catalog, winners, generator = _catalog(root / role, start=START + index * 86_400_000,
                                               transitions=transitions, data_seed=data_seed)
        save_catalog(catalog, root / f"{role}-catalog.json")
        catalogs.append(catalog)
        opportunities.append(winners)
        generators.append(dict(role=role, **generator))
    economics = event_coverage(root / "events", tuple(catalogs))
    # These are diagnostic labels ONLY. Neither the catalog nor production
    # runner receives them; nothing here supplies actions, rewards or returns.
    _publish(root / "generator-diagnostics.json", dict(schema=SCHEMA,
        specification_sha256=spec_sha, opportunities=opportunities, generator_identities=generators,
        catalog_sha256=[c.identity for c in catalogs]))
    return EconomicFixture(root, tuple(catalogs), economics, tuple(opportunities), spec_sha)


def favorable_weight_margin(ledger: list[dict], winners: tuple[int, ...]) -> float:
    """Requested pre-cap/pre-repair preference, not filled or realized exposure."""
    if len(ledger) != len(winners):
        raise ValueError("Diagnostic labels differ from actual scored decisions")
    return mean(row["requested_weights"][winner + 1] - row["requested_weights"][2 - winner]
                for row, winner in zip(ledger, winners, strict=True))


def host_profile(profile: cProfile.Profile) -> dict:
    """Compact observed host costs, not disjoint stages or CUDA kernel timing."""
    names = {"_validate_sources", "_validation", "_test", "save", "freeze", "parameter_hash",
             "load_raw_second_window", "resolve_second_interval", "_marks", "collect", "update"}
    records = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profile).stats.items():
        if name in names and "rl_quant" in filename:
            records.append(dict(function=name, source=filename.split("rl_quant/", 1)[-1], line=line,
                primitive_calls=primitive, total_calls=calls, self_host_seconds=own,
                cumulative_host_seconds=cumulative))
    return dict(interpretation="profiled-host-wall-time;nested-not-additive;not-CUDA-kernel-time",
                functions=sorted(records, key=lambda row: (row["source"], row["line"])))


def _source_inventory(fixture: EconomicFixture) -> dict[str, str]:
    files = [path for name in ("train", "validation", "test", "events")
             for path in (fixture.root / name).rglob("*") if path.is_file()]
    files.extend(fixture.root / name for name in
                 ("train-catalog.json", "validation-catalog.json", "test-catalog.json",
                  "specification.json", "generator-diagnostics.json"))
    return {str(path.relative_to(fixture.root)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def run_economic_benchmark(fixture: EconomicFixture) -> dict:
    """Persist all three actual outcomes before any learning-quality assertion."""
    before = _source_inventory(fixture)
    before_sha = _publish(fixture.root / "source-inventory-before.json", dict(files=before))
    rows = []
    for seed in SEEDS:
        output = fixture.root / f"seed-{seed}"
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        profile = cProfile.Profile()
        try:
            result = profile.runcall(run_raw_second_experiment,
                training=fixture.catalogs[0], validation=fixture.catalogs[1],
                test=fixture.catalogs[2], economic_inputs=fixture.economics, output=output, device="cuda:0",
                model_config=model_config(), execution_config=execution_config(), ppo_config=ppo_config(seed),
                rollout_steps=ROLLOUT_STEPS)
        finally:
            _publish(fixture.root / f"seed-{seed}-run-profile.json", host_profile(profile))
        torch.cuda.synchronize()
        fitted = json.loads((output / "training.json").read_text())
        comparisons = result["test"]["20"]["comparisons"]
        margins = {label: favorable_weight_margin(comparisons[label]["ledger"], fixture.opportunities[2])
                   for label in ("trained", "untrained")}
        row = dict(seed=seed, report_file=f"seed-{seed}/report.json", report_sha256=result["report_sha256"],
            selected_update=result["selection"]["selected"]["update"],
            summaries={label: value["summary"] for label, value in comparisons.items()},
            favorable_weight_margin=margins, margin_improvement=margins["trained"] - margins["untrained"],
            net_improvement=result["test"]["20"]["net_return_differences"]["untrained"],
            collected_transitions=sum(u["transitions"] for u in fitted["updates"]),
            rollout_updates=len(fitted["updates"]),
            optimizer_minibatches=sum(u["metrics"]["minibatches"] for u in fitted["updates"]),
            rollout_seconds=sum(u["rollout_seconds"] for u in fitted["updates"]),
            update_seconds=sum(u["update_seconds"] for u in fitted["updates"]),
            host_profile_file=f"seed-{seed}-run-profile.json",
            complete_experiment_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), runtime=fitted["runtime"])
        _publish(fixture.root / f"seed-{seed}-outcome.json", row)
        rows.append(row)
    after = _source_inventory(fixture)
    after_sha = _publish(fixture.root / "source-inventory-after.json", dict(files=after))
    measures = dict(improved_seed_count=sum(r["net_improvement"] > 0 for r in rows),
        median_net_improvement=median(r["net_improvement"] for r in rows),
        median_trained_net_return=median(r["summaries"]["trained"]["net_return"] for r in rows),
        median_favorable_weight_margin=median(r["favorable_weight_margin"]["trained"] for r in rows),
        median_margin_improvement=median(r["margin_improvement"] for r in rows))
    thresholds = specification()["acceptance"]
    failed = [name for name, value in measures.items()
              if (value < thresholds[f"minimum_{name}"] if name == "improved_seed_count"
                  else value <= thresholds[f"minimum_{name}"])]
    summary = dict(schema=SCHEMA, specification_sha256=fixture.specification_sha256,
        rows=rows, measures=measures, acceptance=thresholds, failed_criterion_names=failed,
        source_inventory_unchanged=before == after,
        source_inventory_before_sha256=before_sha, source_inventory_after_sha256=after_sha,
        favorable_weight_margin_definition="requested weights before caps/repair;not executed exposure",
        benchmark_criterion_passed=not failed, all_three_experiments_complete=True,
        real_data_training_ready=False, native_v5_authorized=False,
        positive_profitability_authorization_eligible=False,
        interpretation="fixed synthetic mechanism diagnostic; a failed criterion is retained evidence")
    _publish(fixture.root / "benchmark-summary.json", summary)
    return summary
