"""Behavioral bounded-partition, split-basis and exposure acceptance on LSF."""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json

import pytest
import torch

from raw_second_fixture import START, configure, event_coverage, response, trainer
from rl_quant.data_sources.massive import raw_second_capture_v1 as acquisition
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.datasets.massive_raw_seconds_v1 import (
    CapturedSecondPage, RawSecondCatalog, RawSecondContract, RawSecondWindowRef,
    SecondCaptureRef, SecondPartitionSet, SecondQuery, publish_second_capture,
    resolve_second_interval,
)
from rl_quant.envs.raw_second_portfolio_v1 import RawSecondPortfolioEnv, SecondExecutionConfig
from rl_quant.evaluation.raw_second_profitability_v1 import economic_summary, evaluate_raw_second_policy
from rl_quant.execution.qt200_aggregate_execution_v1 import Split
from rl_quant.rl.types import ActionBatch
from rl_quant.workflows.raw_second_experiment_v1 import run_raw_second_experiment, verify_raw_second_experiment

pytestmark = pytest.mark.lsf_gpu
D = Decimal


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    configure()


def _row(stamp, price=100, volume=3_000_000):
    return dict(t=stamp, o=price, h=price, l=price, c=price, v=volume)


def _capture(root, start, end, rows, *, received=None, ticker="AAPL"):
    query = SecondQuery(ticker, start, end - 1000)
    return publish_second_capture(root, query, (CapturedSecondPage(query.url, received or end + 5000, response(query, rows)),))


def _bounded_handoff(root, queries, rows):
    """Only the HTTP transport is synthetic; publication/decoding stay real."""
    root.mkdir(parents=True)
    query_map = {q.url: q for q in queries}

    class Reply:
        code = 200
        headers = {"Content-Type": "application/json"}

        def __init__(self, url):
            self.url = url

        def geturl(self):
            return self.url

        def read(self, size):
            query = query_map[self.url]
            return response(query, rows[self.url])[:size]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Authorization") == "Bearer synthetic-partition-transport"
            assert timeout == 45
            return Reply(request.full_url)

    plan = acquisition.publish_second_capture_plan(root=root / "http", queries=queries)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(transport.request, "build_opener", lambda *args: Opener())
        patch.setattr(transport.time, "sleep", lambda *_: None)
        acquisition.capture_seconds(root=root / "http", plan_sha256=plan, api_key="synthetic-partition-transport")
    complete = sha256((root / "http/COMPLETE.json").read_bytes()).hexdigest()
    result = acquisition.materialize_second_sources(root=root / "http", plan_sha256=plan,
        completion_sha256=complete, output=root / "raw")
    return {ref.load()[0].url: ref for ref in (SecondCaptureRef(**r) for r in result["captures"])}


def _continuous_catalog(root, start, *, next_price=100, next_rows=True, delay=0, volume=3_000_000):
    close, opened = start + 23_400_000, start + 86_400_000
    # Every query uses the actual bounded acquisition interface, never an
    # invented multi-day response. The first context crosses two partitions.
    queries = (SecondQuery("AAPL", close - 3_600_000, close - 17_000),
               SecondQuery("AAPL", close - 16_000, close - 1000),
               SecondQuery("AAPL", opened, opened + 15_000))
    market = {queries[0].url: [_row(t, volume=volume) for t in range(close - 32_000, close - 16_000, 1000)],
              queries[1].url: [_row(t, volume=volume) for t in range(close - 16_000, close, 1000)],
              queries[2].url: [_row(t, next_price, volume) for t in range(opened + delay, opened + 16_000, 1000)] if next_rows else []}
    captures = _bounded_handoff(root, queries, market)
    first = SecondPartitionSet(tuple(captures[q.url] for q in queries[:2]))
    second = SecondPartitionSet((captures[queries[2].url],))
    contract = RawSecondContract("historical-finalized-assumed-delay", delay)
    return RawSecondCatalog((
        RawSecondWindowRef((first,), ("issue-apple",), close - 32_000, 32, close - 8_000, contract),
        RawSecondWindowRef((first,), ("issue-apple",), close - 32_000, 32, close, contract),
        RawSecondWindowRef((second,), ("issue-apple",), opened, 16, opened + 8_000, contract),
        RawSecondWindowRef((second,), ("issue-apple",), opened, 16, opened + 16_000, contract)))


@pytest.fixture(scope="module")
def persisted_partitions(tmp_path_factory):
    root = tmp_path_factory.mktemp("continuous-partition-run")
    catalogs = tuple(_continuous_catalog(root / role, START + offset * 86_400_000)
                     for role, offset in (("train", 0), ("validation", 2), ("test", 6)))
    events = event_coverage(root / "events", catalogs)
    agent = trainer(catalogs[0])
    config = replace(agent.environment.config, execution_session="regular-only", order_expiry="session-close")
    result = run_raw_second_experiment(training=catalogs[0], validation=catalogs[1], test=catalogs[2],
        economic_inputs=events, output=root / "run", device="cuda:0", model_config=agent.algorithm.model.config,
        execution_config=config, ppo_config=agent.algorithm.config, rollout_steps=2)
    return root, catalogs, events, config, result


def test_bounded_partitions_train_select_and_test_across_session_close(persisted_partitions):
    root, catalogs, _, _, report = persisted_partitions
    assert report["execution_complete"] and not report["positive_profitability_authorization_eligible"]
    fitted = json.loads((root / "run/training.json").read_text())
    plan = json.loads((root / "run/plan.json").read_text())
    assert plan["rollout_schedule"] == [3]  # absorb the tail before any update
    assert [row["metrics"]["learning_samples"] for row in fitted["updates"]] == [3]
    assert fitted["initial_parameter_sha256"] != fitted["final_parameter_sha256"]
    assert fitted["ledger"][0]["holdings"] == fitted["ledger"][1]["holdings"]
    assert fitted["ledger"][1]["holdings"]
    for catalog in catalogs:
        for part in catalog.sources[0].partitions:
            query, _ = part.load()
            assert query.end_ms - query.start_ms + 1000 <= 3_600_000
        raw = catalog.load(0, device="cuda:0")
        assert torch.all(raw.raw_ohlcv[raw.observed_mask] == torch.tensor([100., 100., 100., 100., 3_000_000.], device="cuda:0"))
    for scenario in report["test"].values():
        assert scenario["comparisons"]["trained"]["summary"]["fill_count"] > 0


def test_partition_report_replays_without_source_or_evidence_writes(persisted_partitions):
    root, catalogs, events, _, report = persisted_partitions
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    result = verify_raw_second_experiment(output=root / "run", expected_report_sha256=report["report_sha256"],
        training=catalogs[0], validation=catalogs[1], test=catalogs[2], economic_inputs=events, device="cuda:0")
    assert result["frozen_report_replayed"]
    assert before == {str(p): sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("conflict", ["price", "observed-versus-empty", "ticker"])
def test_conflicting_partition_vintages_fail_closed(tmp_path, conflict):
    original = _capture(tmp_path / "first", START, START + 8000, [_row(START)])
    row = [] if conflict == "observed-versus-empty" else [_row(START, 101 if conflict == "price" else 100)]
    other = _capture(tmp_path / "second", START, START + 8000, row, ticker="BRK.B" if conflict == "ticker" else "AAPL")
    with pytest.raises(ValueError, match="Conflicting|Mixed ticker"):
        resolve_second_interval(SecondPartitionSet((original, other)), START, START + 8000)


def test_equal_overlap_retains_original_rows_and_availability(tmp_path):
    first = _capture(tmp_path / "first", START, START + 8000, [_row(START)], received=START + 10_000)
    second = _capture(tmp_path / "second", START, START + 8000, [_row(START)], received=START + 20_000)
    result = resolve_second_interval(SecondPartitionSet((first, second)), START, START + 8000)
    assert result.rows == ((START, (100., 100., 100., 100., 3_000_000.), START + 10_000),)
    assert result.source_hashes == (first.manifest_sha256, second.manifest_sha256)
    assert result.coverage_receipt(START + 1000) == START + 10_000


def test_unknown_gap_is_not_an_empty_second_or_overnight_exemption(tmp_path):
    first = _capture(tmp_path / "first", START, START + 8000, [_row(START)])
    second = _capture(tmp_path / "second", START + 9000, START + 16_000, [_row(START + 9000)])
    parts = SecondPartitionSet((first, second))
    with pytest.raises(ValueError, match="Unknown source coverage"):
        resolve_second_interval(parts, START, START + 16_000)
    window = RawSecondWindowRef((parts,), ("issue-apple",), START, 16, START + 16_000,
                                RawSecondContract("historical-finalized-assumed-delay", 0))
    with pytest.raises(ValueError, match="Unknown coverage"):
        RawSecondCatalog((window,)).load(0, device="cuda:0")


def test_execution_store_can_include_sources_not_in_model_context(tmp_path):
    parts = tuple(_capture(tmp_path / str(i), START + i * 8000, START + (i + 1) * 8000,
                           [_row(t) for t in range(START + i * 8000, START + (i + 1) * 8000, 1000)]) for i in range(3))
    contract = RawSecondContract("historical-finalized-assumed-delay", 0)
    windows = (RawSecondWindowRef((parts[0],), ("issue-apple",), START, 8, START + 8000, contract),
               RawSecondWindowRef((parts[2],), ("issue-apple",), START + 16_000, 8, START + 24_000, contract))
    config = SecondExecutionConfig(capital="100000", maximum_asset_weight="1", decision_interval_seconds=16)
    incomplete = RawSecondPortfolioEnv(RawSecondCatalog(windows), config=config, device="cuda:0")
    action = ActionBatch(torch.tensor([[0., 1.]], device="cuda:0"))
    with pytest.raises(ValueError, match="Unknown source coverage"):
        incomplete.step(action)
    assert not incomplete.fills and incomplete.index == 0
    complete = RawSecondPortfolioEnv(RawSecondCatalog(windows, execution_sources=(SecondPartitionSet((parts[1],)),)),
                                     config=config, device="cuda:0")
    complete.step(action)
    assert complete.fills[0].second_start_ms == START + 9000
    assert complete.catalog.identity != incomplete.catalog.identity


@pytest.mark.parametrize("delayed", [False, True])
def test_split_marks_and_order_sizing_use_current_share_basis(tmp_path, delayed):
    catalog = _continuous_catalog(tmp_path / "sources", START, next_price=50, next_rows=delayed, delay=4000 if delayed else 0)
    events = event_coverage(tmp_path / "events", (catalog,),
        splits=(Split("split", "issue-apple", "2017-01-04", D(1), D(2)),))
    terms = events.load(catalog)
    config = SecondExecutionConfig(capital="100000", maximum_asset_weight="1", decision_interval_seconds=8,
                                   execution_session="regular-only", order_expiry="session-close")
    env = RawSecondPortfolioEnv(catalog, config=config, device="cuda:0", splits=terms[0], sessions=terms[2])
    env.step(ActionBatch(torch.tensor([[0.5, 0.5]], device="cuda:0")))
    initial_shares = dict(env.book.holdings)["issue-apple"]
    env.step(ActionBatch(torch.tensor([[1., 0.]], device="cuda:0")))
    assert dict(env.book.holdings)["issue-apple"] == 2 * initial_shares
    assert env.last_marks["issue-apple"] == env.known_marks["issue-apple"] == 50
    assert env.audit[1]["equity_after"] == env.audit[0]["equity_after"]
    assert env.known_mark_state["issue-apple"].second_start_ms < START + 86_400_000
    if delayed:
        assert env.accounting_mark_state["issue-apple"].second_start_ms >= START + 86_400_000
    env.step(ActionBatch(torch.tensor([[0.5, 0.5]], device="cuda:0")))
    expected = (D(env.audit[1]["equity_after"]) * D("0.5") / 50).to_integral_value(rounding="ROUND_FLOOR") - 2 * initial_shares
    assert D(env.audit[-1]["requested_orders"]["issue-apple"]) == expected
    economic_summary(env)
    raw = catalog.load(0, device="cuda:0")
    assert torch.all(raw.raw_ohlcv[raw.observed_mask][:, :4] == 100)


def test_split_basis_is_restored_by_exact_training_resume(tmp_path):
    catalog = _continuous_catalog(tmp_path / "sources", START, next_rows=False)
    last = catalog.windows[-1]
    # A second genuine post-split transition, not a singleton learning update.
    extra = _capture(tmp_path / "later-empty", last.decision_ms, last.decision_ms + 8000, [])
    parts = SecondPartitionSet((*last.captures[0].partitions, extra))
    catalog = RawSecondCatalog((*catalog.windows,
        replace(last, captures=(parts,), seconds=24, decision_ms=last.decision_ms + 8000)))
    events = event_coverage(tmp_path / "events", (catalog,),
        splits=(Split("split", "issue-apple", "2017-01-04", D(1), D(2)),))
    terms = events.load(catalog)
    agent = trainer(catalog)
    config = replace(agent.environment.config, execution_session="regular-only", order_expiry="session-close")
    agent.environment = RawSecondPortfolioEnv(catalog, config=config, device="cuda:0", splits=terms[0], sessions=terms[2])
    buffer = agent.collect(steps=2)
    agent.update(buffer)
    path = tmp_path / "resume.pt"
    receipt = agent.save(path)
    first_marks = agent.environment.known_mark_state.copy()
    first = agent.collect(steps=2)
    metrics = agent.update(first)
    restored = trainer(catalog)
    restored.environment = RawSecondPortfolioEnv(catalog, config=config, device="cuda:0", splits=terms[0], sessions=terms[2])
    restored.load(path, receipt)
    assert restored.environment.known_mark_state == first_marks
    again = restored.collect(steps=2)
    assert restored.update(again) == metrics
    assert restored.environment.audit == agent.environment.audit


def test_one_shot_baseline_reports_actual_exposure_and_entry_shortfall(tmp_path):
    catalog = _continuous_catalog(tmp_path / "sources", START, volume=50)
    events = event_coverage(tmp_path / "events", (catalog,))
    config = SecondExecutionConfig(capital="100000", maximum_asset_weight="1", decision_interval_seconds=8,
                                   execution_session="regular-only", order_expiry="session-close")
    result = evaluate_raw_second_policy(catalog=catalog, economic_inputs=events, execution_config=config,
                                        device="cuda:0", baseline="buy-and-hold")
    assert result["entry_convention"].startswith("one-shot-first-decision-interval")
    summary = result["summary"]
    assert D(summary["initial_entry_requested_notional"]) == 100000
    assert D(summary["initial_entry_filled_notional_at_decision_marks"]) == 700
    assert summary["initial_entry_completion_fraction"] == 0.007
    assert 0 < summary["average_risky_exposure"] < 0.01
    assert summary["average_cash_allocation"] > 0.99
    assert all(not row["fills"] for row in result["ledger"][1:])
    values = [float(D(row["risky_marked_notional"]) / D(row["pre_terminal_equity"])) for row in result["ledger"]]
    assert summary["average_risky_exposure"] == sum(values) / len(values)
