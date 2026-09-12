"""Reward-objective regressions on one assigned LSF GPU, never CPU fallback."""

from dataclasses import replace
from hashlib import sha256
import io
import json
import math

import pytest
import torch

from raw_second_fixture import configure, event_coverage, make_catalog, trainer
from rl_quant.datasets.massive_raw_seconds_v1 import RawSecondCatalog
from rl_quant.evaluation.raw_second_policy_v1 import parameter_hash
from rl_quant.training.raw_second_ppo_v1 import LEARNING_SCHEMA, RawSecondPPOTrainer, plan_raw_second_rollouts
from rl_quant.workflows.massive_raw_second_rl_v1 import run_raw_second_engineering_episode
from rl_quant.workflows.raw_second_experiment_v1 import run_raw_second_experiment

pytestmark = pytest.mark.lsf_gpu


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    configure()


@pytest.mark.parametrize("transitions,nominal,expected", [
    (2, 2, (2,)), (3, 2, (3,)), (4, 2, (2, 2)), (5, 2, (2, 3)),
    (63, 63, (63,)), (64, 63, (64,)), (127, 63, (63, 64)),
])
def test_rollout_plan_covers_every_transition_without_singletons(transitions, nominal, expected):
    assert plan_raw_second_rollouts(transitions, nominal) == expected
    # Exhaustive small boundaries, without importing/executing tests locally.
    for count in range(2, 129):
        schedule = plan_raw_second_rollouts(count, nominal)
        assert sum(schedule) == count and min(schedule) >= 2
        assert max(schedule) <= nominal + 1


@pytest.mark.parametrize("transitions,nominal", [(1, 2), (0, 2), (3, 1), (3, True), (3, 2.0)])
def test_invalid_learning_plan_is_rejected(transitions, nominal):
    with pytest.raises(ValueError, match="requires"):
        plan_raw_second_rollouts(transitions, nominal)


def test_singleton_rejected_before_optimizer_or_parameter_mutation(tmp_path):
    agent = trainer(make_catalog(tmp_path / "sources"))
    buffer = agent.collect(steps=1)  # collection alone remains legal for diagnostics
    batch = buffer.recurrent_sequences(sequence_length=1)
    advantages = batch.advantages[batch.loss_mask]
    assert advantages.numel() == 1
    assert torch.equal(advantages - advantages.mean(), torch.zeros_like(advantages))
    before = parameter_hash(agent.algorithm.model)
    random_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state().clone()
    ledger = list(agent.environment.audit)
    with pytest.raises(ValueError, match="Singleton PPO"):
        agent.update(buffer)
    assert parameter_hash(agent.algorithm.model) == before
    assert not agent.algorithm.optimizer.state and agent.algorithm.update_count == 0
    assert torch.equal(torch.get_rng_state(), random_state)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_state)
    assert agent.environment.audit == ledger


@pytest.mark.parametrize("runner", ["engineering", "experiment"])
def test_runner_rejects_singleton_configuration_before_sources_or_publication(tmp_path, runner):
    catalog = make_catalog(tmp_path / "sources")
    events = event_coverage(tmp_path / "events", (catalog,))
    agent = trainer(catalog)
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    # Bad paths establish that rejection precedes economic-source access.
    missing_events = replace(events, path=str(tmp_path / "absent-events.json"))
    kwargs = dict(output=tmp_path / "run", device="cuda:0", model_config=agent.algorithm.model.config,
                  execution_config=agent.environment.config, ppo_config=agent.algorithm.config,
                  economic_inputs=missing_events, rollout_steps=1)
    with pytest.raises(ValueError, match="rollout_steps >= 2"):
        if runner == "engineering":
            run_raw_second_engineering_episode(catalog=catalog, **kwargs)
        else:
            run_raw_second_experiment(training=catalog, validation=catalog, test=catalog, **kwargs)
    assert not (tmp_path / "run").exists()
    assert before == {str(p): sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_policy_objective_alone_updates_actor_and_market_encoder(tmp_path, seed):
    torch.manual_seed(seed)
    catalog = make_catalog(tmp_path / "sources")
    prototype = trainer(catalog)
    config = replace(prototype.algorithm.config, seed=seed, entropy_coefficient=0.0,
                     value_coefficient=0.0, epochs=1, minibatch_sequences=None)
    agent = RawSecondPPOTrainer(prototype.algorithm.model, prototype.environment, config)
    model = agent.algorithm.model
    names = ("input_projection.weight", "local.0.attention.in_proj_weight",
             "temporal.0.attention.in_proj_weight", "cross_stock.attention.in_proj_weight",
             "actor.2.weight", "cash_actor.weight")
    params = dict(model.named_parameters())
    before = {name: p.detach().clone() for name, p in params.items()}
    gradients = {name: [] for name in (*names, "critic.2.weight")}
    hooks = [params[name].register_hook(lambda g, key=name: gradients[key].append(float(g.norm())))
             for name in gradients]
    try:
        buffer = agent.collect(steps=4)
        batch = buffer.recurrent_sequences(sequence_length=1)
        advantages = batch.advantages[batch.loss_mask]
        assert advantages.numel() == 4 and advantages.std(unbiased=False) > config.advantage_epsilon
        # Actual ledger rewards/GAE, sampled actions and behavior probabilities;
        # no patched actor, reward, target, transition or gradient.
        metrics = agent.update(buffer)
    finally:
        for hook in hooks:
            hook.remove()
    assert metrics["learning_samples"] == 4 and metrics["minibatches"] == 1
    assert all(math.isfinite(float(v)) for v in metrics.values())
    assert metrics["loss"] == pytest.approx(metrics["policy_loss"], abs=1e-8)
    for name in names:
        assert gradients[name] and all(math.isfinite(v) for v in gradients[name])
        assert max(gradients[name]) > 0, name
        assert not torch.equal(params[name], before[name]), name
    assert all(v == 0 for v in gradients["critic.2.weight"])
    assert torch.equal(params["critic.2.weight"], before["critic.2.weight"])
    # A centered objective can have scalar value zero at ratio=1 while its
    # derivative is nonzero. Do not confuse the reported scalar with gradient.


def test_engineering_workflow_plans_tail_before_collecting(tmp_path):
    source = make_catalog(tmp_path / "sources")
    catalog = RawSecondCatalog(source.windows[:4])  # three scored transitions
    agent = trainer(catalog)
    result = run_raw_second_engineering_episode(catalog=catalog, output=tmp_path / "run", device="cuda:0",
        model_config=agent.algorithm.model.config, execution_config=agent.environment.config,
        ppo_config=agent.algorithm.config, rollout_steps=2,
        economic_inputs=event_coverage(tmp_path / "events", (catalog,)))
    plan = json.loads((tmp_path / "run/learning-plan.json").read_text())
    assert plan["rollout_schedule"] == [3] and plan["learning_schema"] == LEARNING_SCHEMA
    assert result["optimizer_updates"] == 1 and len(result["trajectory"]) == 3
    assert result["updates"][0]["learning_samples"] == 3


def test_old_learning_resume_state_cannot_silently_change_contract(tmp_path):
    catalog = make_catalog(tmp_path / "sources")
    agent = trainer(catalog)
    path = tmp_path / "resume.pt"
    agent.save(path)
    state = torch.load(path, weights_only=True, map_location="cuda:0")
    assert state.pop("learning_schema") == LEARNING_SCHEMA
    stream = io.BytesIO()
    torch.save(state, stream)
    body = stream.getvalue()
    old = tmp_path / "legacy.pt"
    old.write_bytes(body)
    with pytest.raises(ValueError, match="configuration differs"):
        agent.load(old, sha256(body).hexdigest())
