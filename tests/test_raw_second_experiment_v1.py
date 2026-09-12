"""Persisted chronological acceptance on an assigned GPU, no economic mocks."""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from raw_second_fixture import START, configure, event_coverage, make_catalog, response, trainer
from rl_quant.datasets import massive_raw_seconds_v1 as source
from rl_quant.datasets.massive_raw_seconds_v1 import CapturedSecondPage, RawSecondCatalog, RawSecondContract, RawSecondWindowRef, SecondQuery, publish_second_capture
from rl_quant.envs.raw_second_portfolio_v1 import RawSecondPortfolioEnv, SecondExecutionConfig
from rl_quant.evaluation.raw_second_policy_v1 import load_frozen_raw_second_policy, parameter_hash
from rl_quant.evaluation.raw_second_profitability_v1 import evaluate_raw_second_policy, economic_summary
from rl_quant.execution.qt200_aggregate_execution_v1 import Dividend, Split
from rl_quant.rl.types import ActionBatch
from rl_quant.training.raw_second_ppo_v1 import RawSecondPPOTrainer
from rl_quant.workflows.massive_raw_second_rl_v1 import run_raw_second_engineering_episode
from rl_quant.workflows.raw_second_experiment_v1 import run_raw_second_experiment, verify_raw_second_experiment

pytestmark = pytest.mark.lsf_gpu


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    configure()


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw-second-experiment")
    catalogs = tuple(make_catalog(root / role, start=START + index * 86_400_000, falling=role == "test")
                     for index, role in enumerate(("train", "validation", "test")))
    events = event_coverage(root / "events", catalogs)
    agent = trainer(catalogs[0])
    config = replace(agent.environment.config, observation_session="regular-only", execution_session="regular-only", order_expiry="session-close")
    # Observe actual file reads, not fabricated outcomes/authorization. The
    # test capture must stay unopened until selection is persisted.
    original = source._read
    opened = []
    def read(path, expected):
        if str(path).startswith(str(root / "test")):
            assert (root / "run/selection.json").is_file()
            opened.append(str(path))
        return original(path, expected)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source, "_read", read)
        result = run_raw_second_experiment(training=catalogs[0], validation=catalogs[1], test=catalogs[2],
            economic_inputs=events, output=root / "run", device="cuda:0", model_config=agent.algorithm.model.config,
            execution_config=config, ppo_config=agent.algorithm.config, rollout_steps=2)
    assert opened
    return root, catalogs, events, config, result


def test_frozen_checkpoint_evaluates_new_catalog_and_costs(experiment):
    root, catalogs, events, config, result = experiment
    selected = result["selection"]["selected"]
    path = root / "run" / selected["policy_file"]
    policy = load_frozen_raw_second_policy(path, selected["policy_sha256"], catalog=catalogs[2], device="cuda:0")
    assert policy.training_provenance["catalog_sha256"] == catalogs[0].identity != catalogs[2].identity
    original = parameter_hash(policy.model)
    for cost in (20, 40):
        observed = evaluate_raw_second_policy(catalog=catalogs[2], economic_inputs=events,
            execution_config=replace(config, cost_basis_points=cost), device="cuda:0", policy=policy)
        assert observed["summary"]["fill_count"] > 0
        assert parameter_hash(policy.model) == original == observed["parameter_sha256"]
    assert sha256(path.read_bytes()).hexdigest() == selected["policy_sha256"]


def test_frozen_policy_has_no_training_route_and_detects_mutation(experiment):
    root, catalogs, _, _, result = experiment
    selected = result["selection"]["selected"]
    policy = load_frozen_raw_second_policy(root / "run" / selected["policy_file"], selected["policy_sha256"], catalog=catalogs[2], device="cuda:0")
    assert not hasattr(policy, "update") and not hasattr(policy, "optimizer")
    with pytest.raises(ValueError, match="Frozen"):
        policy.model.train()
    with pytest.raises(ValueError, match="Frozen"):
        policy.model.requires_grad_(True)
    with pytest.raises(ValueError, match="Frozen"):
        RawSecondPPOTrainer(policy.model, RawSecondPortfolioEnv(catalogs[2], config=trainer(catalogs[2]).environment.config, device="cuda:0"))
    with torch.no_grad():
        next(policy.model.parameters()).add_(1)
    with pytest.raises(ValueError, match="changed"):
        policy.validate_unchanged()


def test_resume_remains_strict_for_new_dates_and_costs(experiment):
    root, catalogs, _, config, result = experiment
    chosen = result["selection"]["selected"]
    agent = trainer(catalogs[2])
    with pytest.raises(ValueError, match="configuration differs"):
        agent.load(root / "run" / chosen["resume_file"], chosen["resume_sha256"])
    same = trainer(catalogs[0])
    same.environment.config = replace(config, cost_basis_points=40)
    with pytest.raises(ValueError, match="configuration differs"):
        same.load(root / "run" / chosen["resume_file"], chosen["resume_sha256"])
    with pytest.raises(ValueError, match="load_frozen"):
        same.load(root / "run" / chosen["resume_file"], chosen["resume_sha256"], inference_only=True)


def test_validation_only_selection_and_negative_test_report(experiment):
    root, _, _, _, result = experiment
    rows = result["validation"]
    best = min(rows, key=lambda r: (-r["evaluation"]["summary"]["net_return"], r["candidate"]["update"]))
    assert result["selection"]["selected"] == best["candidate"]
    assert result["selection"]["test_outcomes_used"] is False
    assert result["execution_complete"] and not result["positive_test_net_return"]
    assert (root / "run/report.json").is_file()
    fitted = json.loads((root / "run/training.json").read_text())
    assert fitted["initial_parameter_sha256"] != fitted["final_parameter_sha256"]
    for rung in result["test"].values():
        comparisons = rung["comparisons"]
        assert set(comparisons) == {"trained", "untrained", "cash", "buy-and-hold"}
        assert comparisons["cash"]["summary"]["net_return"] == 0
        assert comparisons["trained"]["summary"]["net_return"] < 0
        assert len({json.dumps(r["execution_config"], sort_keys=True) for r in comparisons.values()}) == 1
        assert len({r["catalog_sha256"] for r in comparisons.values()}) == 1


def test_report_replay_is_nonmaterializing_and_missing_sources_fail(experiment):
    root, catalogs, events, _, result = experiment
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    replay = verify_raw_second_experiment(output=root / "run", expected_report_sha256=result["report_sha256"],
        training=catalogs[0], validation=catalogs[1], test=catalogs[2], economic_inputs=events, device="cuda:0")
    assert replay["nonmaterializing"] and replay["frozen_report_replayed"]
    assert before == {str(p): sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    broken = replace(events, path=str(root / "absent.json"))
    with pytest.raises(ValueError, match="inputs differ"):
        verify_raw_second_experiment(output=root / "run", expected_report_sha256=result["report_sha256"],
            training=catalogs[0], validation=catalogs[1], test=catalogs[2], economic_inputs=broken, device="cuda:0")
    assert not Path(broken.path).exists()


def test_wrong_input_semantics_and_issue_order_rejected(experiment):
    root, catalogs, _, _, result = experiment
    selected = result["selection"]["selected"]
    for change in (dict(asset_ids=tuple(reversed(catalogs[2].asset_ids))),
                   dict(contract=RawSecondContract("historical-finalized-assumed-delay", 1000))):
        wrong = RawSecondCatalog(tuple(replace(w, **change) for w in catalogs[2].windows))
        with pytest.raises(ValueError, match="raw semantics or instrument order"):
            load_frozen_raw_second_policy(root / "run" / selected["policy_file"], selected["policy_sha256"], catalog=wrong, device="cuda:0")


def test_overlap_rejected_before_run_publication(experiment, tmp_path):
    _, catalogs, events, config, _ = experiment
    agent = trainer(catalogs[0])
    with pytest.raises(ValueError, match="overlap"):
        run_raw_second_experiment(training=catalogs[0], validation=catalogs[0], test=catalogs[2],
            economic_inputs=events, output=tmp_path / "run", device="cuda:0", model_config=agent.algorithm.model.config,
            execution_config=config, ppo_config=agent.algorithm.config, rollout_steps=2)
    assert not (tmp_path / "run").exists()


def _event_catalog(root, prices, *, after_hours=False):
    query = SecondQuery("AAPL", START, START + (len(prices) - 1) * 86_400_000 + 63_000)
    rows = [dict(t=START + d * 86_400_000 + s * 1000, o=p, h=p, l=p, c=p, v=0 if after_hours else 3_000_000)
            for d, p in enumerate(prices) for s in range(64)]
    if after_hours:
        rows.append(dict(t=START + 27_000_000, o=100, h=100, l=100, c=100, v=3_000_000))
    rows.sort(key=lambda r: r["t"])
    cap = publish_second_capture(root, query, (CapturedSecondPage(query.url, query.end_ms + 5000, response(query, rows)),))
    return RawSecondCatalog(tuple(RawSecondWindowRef((cap,), ("issue-apple",), START + d * 86_400_000, 64,
                            START + d * 86_400_000 + s * 1000, RawSecondContract("historical-finalized-assumed-delay", 0))
                           for d in range(len(prices)) for s in (16, 24)))


def test_top_level_runner_delivers_split_dividend_and_later_cash_payment(tmp_path):
    catalog = _event_catalog(tmp_path / "sources", (100, 50, 49, 49))
    events = event_coverage(tmp_path / "events", (catalog,),
        splits=(Split("split", "issue-apple", "2017-01-04", Decimal(1), Decimal(2)),),
        dividends=(Dividend("dividend", "issue-apple", "2017-01-05", "2017-01-06", Decimal(1)),))
    agent = trainer(catalog)
    result = run_raw_second_engineering_episode(catalog=catalog, output=tmp_path / "run", device="cuda:0",
        model_config=agent.algorithm.model.config, execution_config=replace(agent.environment.config,
            execution_session="regular-only", order_expiry="session-close"), ppo_config=agent.algorithm.config,
        rollout_steps=2, economic_inputs=events)
    assert Decimal(result["terminal_equity"]) > Decimal("95000")
    ledger = result["ledger"]
    ex_rows = [r for r in ledger if r["receivables"]]
    assert ex_rows and ex_rows[0]["receivables"][0]["payable_date"] == "2017-01-06"
    assert not ledger[-1]["receivables"] and "dividend" in ledger[-1]["applied_events"]
    # Payment adds the existing entitlement to cash, not a second economic gain.
    pay_index = next(i for i in range(1, len(ledger)) if ledger[i - 1]["receivables"] and not ledger[i]["receivables"])
    prior, paid = ledger[pay_index - 1], ledger[pay_index]
    entitlement = sum(Decimal(r["amount"]) for r in prior["receivables"])
    fill_delta = sum(Decimal(f["signed_shares"]) * Decimal(f["price"]) + Decimal(f["fee"]) for f in paid["fills"])
    assert Decimal(paid["cash"]) == Decimal(prior["cash"]) + entitlement - fill_delta
    assert catalog.load(2, device="cuda:0").raw_ohlcv[0, 0, 0, 0] == 50


def test_after_hours_expiry_does_not_force_position_liquidation(tmp_path):
    catalog = _event_catalog(tmp_path / "sources", (100, 100), after_hours=True)
    events = event_coverage(tmp_path / "events", (catalog,))
    terms = events.load(catalog)
    config = SecondExecutionConfig(capital="100000", maximum_asset_weight="1", decision_interval_seconds=8,
                                   execution_session="regular-only", order_expiry="session-close")
    env = RawSecondPortfolioEnv(catalog, config=config, device="cuda:0", splits=terms[0], dividends=terms[1], sessions=terms[2])
    for _ in range(2):
        env.step(ActionBatch(torch.tensor([[0., 1.]], device="cuda:0")))
    assert not env.fills and env.audit[1]["expiry_ms"] == START + 23_400_000
    all_hours = RawSecondPortfolioEnv(catalog, config=replace(config, execution_session="all-captured", order_expiry="next-decision"), device="cuda:0")
    for _ in range(2):
        all_hours.step(ActionBatch(torch.tensor([[0., 1.]], device="cuda:0")))
    assert all_hours.fills and all_hours.fills[0].second_start_ms == START + 27_000_000
    assert all_hours.book.holdings  # no mandatory session-end liquidation
    all_hours.step(ActionBatch(torch.tensor([[1., 0.]], device="cuda:0")))
    economic_summary(all_hours)


def test_missing_or_tampered_event_coverage_is_not_event_free(experiment, tmp_path):
    _, catalogs, events, _, _ = experiment
    with pytest.raises(FileNotFoundError):
        replace(events, path=str(tmp_path / "missing.json")).load(catalogs[0])
    body = json.loads(Path(events.path).read_text())
    body["unsupported_events"] = ["unresolved-merger"]
    encoded = json.dumps(body).encode()
    path = tmp_path / "unsupported.json"
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="unsupported"):
        replace(events, path=str(path), sha256=sha256(encoded).hexdigest()).load(catalogs[0])
