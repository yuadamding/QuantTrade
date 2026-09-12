"""One frozen chronological train/validation/test development experiment.

No native-V5 promotion, real-data readiness override, or test-based selection.
Returns from changing-policy training are never substituted for held-out P&L.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time

import torch

from rl_quant.datasets.massive_raw_seconds_v1 import RawSecondCatalog, _json, _read, _write, digest
from rl_quant.datasets.raw_second_economics_v1 import SecondEconomicInputs
from rl_quant.envs.raw_second_portfolio_v1 import RawSecondPortfolioEnv, SecondExecutionConfig
from rl_quant.evaluation.raw_second_policy_v1 import load_frozen_raw_second_policy, parameter_hash
from rl_quant.evaluation.raw_second_profitability_v1 import evaluate_raw_second_policy
from rl_quant.models.raw_second_policy_v1 import RawSecondActorCritic, RawSecondModelConfig
from rl_quant.rl.ppo import PPOConfig
from rl_quant.training.raw_second_ppo_v1 import RawSecondPPOTrainer

EXPERIMENT_SCHEMA = "rl-quant.raw-second-chronological-experiment-v2"


def _publish(path: Path, body: dict) -> str:
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _write(path, raw)
    return sha256(raw).hexdigest()


def _split_metadata(catalogs: dict[str, RawSecondCatalog]) -> dict:
    if tuple(catalogs) != ("train", "validation", "test"):
        raise ValueError("Exactly train, validation, test in that order are required")
    rows = {}
    reference = catalogs["train"]
    previous_end = -1
    for role, catalog in catalogs.items():
        times = [w.decision_ms for w in catalog.windows]
        if (len(times) < 2 or times != sorted(set(times)) or times[0] < previous_end
                or catalog.asset_ids != reference.asset_ids or catalog.windows[0].contract != reference.windows[0].contract):
            raise ValueError("Scored splits overlap or raw contracts/universe differ")
        rows[role] = dict(catalog_sha256=catalog.identity, scored_start_ms=times[0], scored_end_ms=times[-1],
                          decision_count=len(times) - 1, context_rule="earlier-context-allowed;only-role-transitions-scored")
        previous_end = times[-1]
    if len({r["catalog_sha256"] for r in rows.values()}) != 3:
        raise ValueError("Separate chronological catalogs required")
    return rows


def _validate_sources(catalog: RawSecondCatalog, *, next_split_start: int | None, device) -> None:
    checked = set()
    for source in catalog.sources:
        for capture in source.partitions:
            if capture.manifest_sha256 not in checked:
                query, _ = capture.load()
                if next_split_start is not None and query.end_ms >= next_split_start:
                    raise ValueError("Earlier-role capture contains later scored outcomes")
                checked.add(capture.manifest_sha256)
    for index in range(len(catalog.windows)):
        catalog.load(index, device=device)


def _validation(output, candidates, catalog, economic_inputs, config, device) -> tuple[dict, list]:
    evaluations = []
    for candidate in candidates:
        policy = load_frozen_raw_second_policy(output / candidate["policy_file"], candidate["policy_sha256"], catalog=catalog, device=device)
        if policy.training_provenance["last_reward_timestamp_ms"] > catalog.windows[0].decision_ms:
            raise ValueError("Candidate trained on validation rewards")
        result = evaluate_raw_second_policy(catalog=catalog, economic_inputs=economic_inputs,
                                            execution_config=config, device=device, policy=policy)
        evaluations.append(dict(candidate=candidate, evaluation=result))
    winner = min(evaluations, key=lambda row: (-row["evaluation"]["summary"]["net_return"], row["candidate"]["update"]))
    selection = dict(criterion="maximum-validation-net-return;earliest-update-breaks-ties",
                     selected=winner["candidate"], validation_catalog_sha256=catalog.identity,
                     validation_results_sha256=digest(evaluations), test_outcomes_used=False)
    return selection, evaluations


def _test(output, selection, untrained, catalog, economic_inputs, config, device, cost_rungs) -> dict:
    results = {}
    for cost in cost_rungs:
        scenario = replace(config, cost_basis_points=cost)
        comparisons = {}
        for label, artifact in (("trained", selection["selected"]), ("untrained", untrained)):
            policy = load_frozen_raw_second_policy(output / artifact["policy_file"], artifact["policy_sha256"], catalog=catalog, device=device)
            comparisons[label] = evaluate_raw_second_policy(catalog=catalog, economic_inputs=economic_inputs,
                execution_config=scenario, device=device, policy=policy)
        for baseline in ("cash", "buy-and-hold"):
            comparisons[baseline] = evaluate_raw_second_policy(catalog=catalog, economic_inputs=economic_inputs,
                execution_config=scenario, device=device, baseline=baseline)
        net = comparisons["trained"]["summary"]["net_return"]
        results[str(cost)] = dict(comparisons=comparisons, net_return_differences={
            name: net - row["summary"]["net_return"] for name, row in comparisons.items() if name != "trained"})
    return results


def run_raw_second_experiment(*, training: RawSecondCatalog, validation: RawSecondCatalog, test: RawSecondCatalog,
                             economic_inputs: SecondEconomicInputs, output: Path, device: str,
                             model_config: RawSecondModelConfig, execution_config: SecondExecutionConfig,
                             ppo_config: PPOConfig, rollout_steps: int, cost_rungs: tuple[int, ...] = (10, 20, 40)) -> dict:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Training requires one assigned CUDA GPU; no CPU fallback")
    if (not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32
            or type(rollout_steps) is not int or rollout_steps < 1):
        raise ValueError("Deterministic FP32 and a positive rollout bound required")
    if cost_rungs != (10, 20, 40) or execution_config.cost_basis_points != 20:
        raise ValueError("Freeze primary 20-bp selection and 10/20/40-bp evaluation")
    catalogs = dict(train=training, validation=validation, test=test)
    splits = _split_metadata(catalogs)  # metadata only: no test prices read
    events = economic_inputs.load(training)
    plan = dict(schema=EXPERIMENT_SCHEMA, splits=splits, asset_ids=training.asset_ids,
        economic_inputs=asdict(economic_inputs), input_contract=asdict(training.windows[0].contract),
        model_config=asdict(model_config), execution_config=asdict(execution_config), ppo_config=asdict(ppo_config),
        rollout_steps=rollout_steps, cost_rungs=cost_rungs,
        selection_rule="maximum-validation-net-return;earliest-update-breaks-ties",
        baseline_rule="cash;one-shot-entry-cap-matched-equal-weight-buy-and-hold;same-seed-untrained-policy")
    output.mkdir(parents=True, exist_ok=False)
    plan_sha = _publish(output / "plan.json", plan)
    _validate_sources(training, next_split_start=validation.windows[0].decision_ms, device=target)
    torch.manual_seed(ppo_config.seed)
    model = RawSecondActorCritic(training, model_config).to(target)
    env = RawSecondPortfolioEnv(training, config=execution_config, device=target, splits=events[0], dividends=events[1], sessions=events[2])
    agent = RawSecondPPOTrainer(model, env, ppo_config)
    initial_parameters = parameter_hash(model)
    untrained = dict(policy_file="untrained.pt", policy_sha256=agent.freeze(output / "untrained.pt"), update=0)
    candidates, updates = [], []
    while env.index < len(training.windows) - 1:
        count = min(rollout_steps, len(training.windows) - 1 - env.index)
        started = time.monotonic()
        buffer = agent.collect(steps=count)
        torch.cuda.synchronize(target)
        collected = time.monotonic()
        metrics = agent.update(buffer)
        torch.cuda.synchronize(target)
        finished = time.monotonic()
        update = agent.algorithm.update_count
        resume_name, policy_name = f"resume-{update:06d}.pt", f"policy-{update:06d}.pt"
        candidates.append(dict(update=update, resume_file=resume_name, resume_sha256=agent.save(output / resume_name),
                               policy_file=policy_name, policy_sha256=agent.freeze(output / policy_name)))
        updates.append(dict(metrics=metrics, transitions=count, rollout_seconds=collected - started,
                            update_seconds=finished - collected, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(target)))
    training_record = dict(candidates=candidates, untrained=untrained, updates=updates,
        initial_parameter_sha256=initial_parameters, final_parameter_sha256=parameter_hash(model),
        training_episode_only=True, ledger=env.audit, runtime=agent.runtime_profile())
    training_sha = _publish(output / "training.json", training_record)
    # No optimizer is used beyond this boundary.
    del agent, model, env, buffer
    _validate_sources(validation, next_split_start=test.windows[0].decision_ms, device=target)
    selection, validated = _validation(output, candidates, validation, economic_inputs, execution_config, target)
    selection_sha = _publish(output / "selection.json", selection)
    # Selection is durably committed BEFORE the first test capture is opened.
    _validate_sources(test, next_split_start=None, device=target)
    tested = _test(output, selection, untrained, test, economic_inputs, execution_config, target, cost_rungs)
    report = dict(schema=EXPERIMENT_SCHEMA, plan_sha256=plan_sha, training_sha256=training_sha,
        selection_sha256=selection_sha, selection=selection, validation=validated, test=tested,
        execution_complete=True, positive_test_net_return=tested["20"]["comparisons"]["trained"]["summary"]["net_return"] > 0,
        real_data_training_ready=False, native_v5_authorized=False, positive_profitability_authorization_eligible=False,
        interpretation="fixed-panel-finalized-second-bar-development;declared-availability-and-execution-assumptions")
    report_sha = _publish(output / "report.json", report)
    return {**report, "report_sha256": report_sha}


def verify_raw_second_experiment(*, output: Path, expected_report_sha256: str,
                                training: RawSecondCatalog, validation: RawSecondCatalog, test: RawSecondCatalog,
                                economic_inputs: SecondEconomicInputs, device: str) -> dict:
    """Read-only frozen validation/test replay; never train or regenerate files."""
    report = _json(_read(output / "report.json", expected_report_sha256))
    plan = _json(_read(output / "plan.json", report["plan_sha256"]))
    fitted = _json(_read(output / "training.json", report["training_sha256"]))
    selected = _json(_read(output / "selection.json", report["selection_sha256"]))
    if (plan["schema"] != EXPERIMENT_SCHEMA or plan["splits"] != _split_metadata(dict(train=training, validation=validation, test=test))
            or plan["economic_inputs"] != asdict(economic_inputs)):
        raise ValueError("Verification inputs differ from the frozen plan")
    for catalog in (training, validation, test):
        economic_inputs.load(catalog)
    for candidate in fitted["candidates"]:
        _read(output / candidate["resume_file"], candidate["resume_sha256"])
    config = SecondExecutionConfig(**plan["execution_config"])
    selection, validated = _validation(output, fitted["candidates"], validation, economic_inputs, config, device)
    if digest(selection) != digest(selected) or digest(selected) != digest(report["selection"]):
        raise ValueError("Validation selection failed exact reconstruction")
    tested = _test(output, selection, fitted["untrained"], test, economic_inputs, config, device, tuple(plan["cost_rungs"]))
    if digest(validated) != digest(report["validation"]) or digest(tested) != digest(report["test"]):
        raise ValueError("Held-out economic report failed exact reconstruction")
    return dict(report_sha256=expected_report_sha256, frozen_report_replayed=True, nonmaterializing=True,
                positive_test_net_return=report["positive_test_net_return"], native_v5_authorized=False)
