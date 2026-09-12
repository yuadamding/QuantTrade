"""Controlled economic learning on one LSF GPU; never local CPU fallback.

The criterion is fixed before execution. A failure must retain every seed's
completed report and must not trigger automatic favorable retuning or retries.
"""

import cProfile
from decimal import Decimal, localcontext
from hashlib import sha256
import json
import math
import time

import pytest
import torch

from raw_second_economic_fixture import (
    INTERVAL_SECONDS, ROLES, ROLLOUT_STEPS, SEEDS, prepare_economic_fixture,
    _publish, host_profile, run_economic_benchmark, specification,
)
from raw_second_fixture import configure
from rl_quant.workflows.raw_second_experiment_v1 import verify_raw_second_experiment

pytestmark = pytest.mark.lsf_gpu


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    configure()


@pytest.fixture(scope="module")
def economic_benchmark(tmp_path_factory):
    parent = tmp_path_factory.mktemp("raw-second-economic-learning")
    fixture = prepare_economic_fixture(parent / "experiment")
    summary = run_economic_benchmark(fixture)
    return fixture, summary


def test_source_process_has_causal_cues_and_balanced_disjoint_realizations(economic_benchmark):
    fixture, _ = economic_benchmark
    stored = json.loads((fixture.root / "specification.json").read_text())
    assert stored == json.loads(json.dumps(specification()))
    assert len({c.identity for c in fixture.catalogs}) == 3
    generators = json.loads((fixture.root / "generator-diagnostics.json").read_text())["generator_identities"]
    assert [row["generator_seed"] for row in generators] == [11017, 22029, 33043]
    assert [row["realization_catalog_sha256"] for row in generators] == [c.identity for c in fixture.catalogs]
    assert len({row["initial_rng_state_sha256"] for row in generators}) == 3
    for (_, count, _), catalog, winners in zip(ROLES, fixture.catalogs, fixture.opportunities, strict=True):
        assert len(winners) == count and len(catalog.windows) == count + 1
        for start in range(0, count, ROLLOUT_STEPS):
            assert winners[start:start + ROLLOUT_STEPS].count(0) == 8
        first = catalog.windows[0]
        observation = catalog.load(0, device="cuda:0")
        assert bool(observation.observed_mask.all())
        for asset, capture in enumerate(first.captures):
            query, pages = capture.load()
            assert query.end_ms - query.start_ms + 1000 <= 3_600_000
            records = [row for page in pages for row in json.loads(page.body)["results"]]
            assert all(set(row) == {"t", "o", "h", "l", "c", "v"} for row in records)
            for step, winner in enumerate(winners):
                decision = catalog.windows[step].decision_ms
                decision_row = (step + 1) * INTERVAL_SECONDS
                entry = records[decision_row + 1]
                payoff = records[decision_row + 3]
                favorable = winner == asset
                for offset in (2, 1):
                    cue = records[decision_row - offset]
                    assert cue["t"] == decision - offset * 1000
                    assert cue["t"] + 1000 <= decision < entry["t"] < payoff["t"]
                    assert cue["o"] == cue["c"] == entry["o"]
                    assert cue["h"] == pytest.approx(cue["c"] * (1.5 if favorable else 1.01))
                    assert cue["l"] == pytest.approx(cue["c"] * (0.99 if favorable else 0.5))
                assert entry["t"] == decision + 1000
                assert payoff["t"] == decision + 3000
                assert payoff["o"] == entry["o"]
                factor = 1.03 if favorable else 0.97
                assert payoff["c"] == pytest.approx(entry["o"] * factor)
            raw = torch.tensor([[row[k] for k in ("o", "h", "l", "c", "v")]
                                for row in records[:INTERVAL_SECONDS]], dtype=torch.float32, device="cuda:0")
            assert torch.equal(observation.raw_ohlcv[0, asset], raw)


def test_all_seeds_complete_persisted_frozen_experiments(economic_benchmark):
    fixture, summary = economic_benchmark
    assert [row["seed"] for row in summary["rows"]] == list(SEEDS)
    assert summary["all_three_experiments_complete"]
    assert summary["source_inventory_unchanged"]
    assert summary["source_inventory_before_sha256"] == summary["source_inventory_after_sha256"]
    for row in summary["rows"]:
        root = fixture.root / f"seed-{row['seed']}"
        body = (root / "report.json").read_bytes()
        assert sha256(body).hexdigest() == row["report_sha256"]
        report = json.loads(body)
        plan = json.loads((root / "plan.json").read_text())
        fitted = json.loads((root / "training.json").read_text())
        assert report["execution_complete"] and not report["selection"]["test_outcomes_used"]
        assert not report["positive_profitability_authorization_eligible"]
        assert fitted["initial_parameter_sha256"] != fitted["final_parameter_sha256"]
        assert plan["rollout_schedule"] == [16] * 16
        assert row["collected_transitions"] == 256 and row["rollout_updates"] == 16
        assert row["optimizer_minibatches"] == 64
        best = min(report["validation"], key=lambda r: (-r["evaluation"]["summary"]["net_return"], r["candidate"]["update"]))
        assert report["selection"]["selected"] == best["candidate"]
        assert all(math.isfinite(float(v)) for u in fitted["updates"] for v in u["metrics"].values())


def test_reports_reconcile_economics_and_exercise_actual_trading(economic_benchmark):
    fixture, summary = economic_benchmark
    buy_count = sell_count = 0
    for row in summary["rows"]:
        report = json.loads((fixture.root / row["report_file"]).read_text())
        for rung in report["test"].values():
            assert rung["comparisons"]["cash"]["summary"]["net_return"] == 0
            for label in ("trained", "untrained"):
                evaluated = rung["comparisons"][label]
                stats, ledger = evaluated["summary"], evaluated["ledger"]
                net = float(Decimal(stats["terminal_equity"]) / Decimal("10000000") - 1)
                assert net == pytest.approx(stats["net_return"], abs=1e-12)
                assert math.expm1(sum(r["reward_net_log_equity"] for r in ledger)) == pytest.approx(net, abs=1e-10)
                with localcontext() as context:
                    context.prec = 34
                    rate = Decimal(evaluated["execution_config"]["cost_basis_points"]) / Decimal(10000)
                    for interval in ledger:
                        for fill in interval["fills"]:
                            independent_fee = abs(Decimal(fill["signed_shares"])) * Decimal(fill["price"]) * rate
                            assert Decimal(fill["fee"]) == independent_fee
                fees = sum((Decimal(f["fee"]) for r in ledger for f in r["fills"]), Decimal(0))
                fees += sum((Decimal(r["terminal_liquidation_cost"]) for r in ledger), Decimal(0))
                assert fees == Decimal(stats["total_costs"]) and fees > 0
                assert len(ledger) == 32
                buy_count += stats["buy_fill_count"]
                sell_count += stats["sell_fill_count"]
        selected = report["selection"]["selected"]
        policy = fixture.root / f"seed-{row['seed']}" / selected["policy_file"]
        assert sha256(policy.read_bytes()).hexdigest() == selected["policy_sha256"]
    assert buy_count > 0 and sell_count > 0


def test_one_completed_benchmark_report_replays_without_evidence_writes(economic_benchmark):
    fixture, summary = economic_benchmark
    row = summary["rows"][0]  # predeclared seed17, not the best test result
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in fixture.root.rglob("*") if p.is_file()}
    started = time.monotonic()
    profile = cProfile.Profile()
    replay = profile.runcall(verify_raw_second_experiment, output=fixture.root / f"seed-{row['seed']}",
        expected_report_sha256=row["report_sha256"], training=fixture.catalogs[0],
        validation=fixture.catalogs[1], test=fixture.catalogs[2], economic_inputs=fixture.economics,
        device="cuda:0")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    assert replay["frozen_report_replayed"] and replay["nonmaterializing"]
    assert before == {str(p): sha256(p.read_bytes()).hexdigest() for p in fixture.root.rglob("*") if p.is_file()}
    # Diagnostic publication follows the verified no-write comparison; it is
    # not part of replay and does not modify any experimental source/report.
    _publish(fixture.root / "seed-17-verification.json", dict(**replay, elapsed_seconds=elapsed,
        host_profile=host_profile(profile), process_scope="same-process;not-fresh-process-qualification"))


def test_preregistered_economic_learning_criterion(economic_benchmark):
    fixture, summary = economic_benchmark
    # Never replace this with a skip/xfail, favorable seed selection or a
    # structure-only assertion. Non-improvement is valid retained diagnostic
    # evidence, but cannot be labeled successful economic-learning acceptance.
    persisted = json.loads((fixture.root / "benchmark-summary.json").read_text())
    assert persisted == summary
    assert summary["benchmark_criterion_passed"], (
        f"Fixed economic-learning criterion failed: {summary['failed_criterion_names']}; "
        f"all three reports and measures remain at {fixture.root / 'benchmark-summary.json'}"
    )
