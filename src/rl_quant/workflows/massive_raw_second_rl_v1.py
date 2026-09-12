"""Bounded end-to-end raw-second engineering execution, not V5 promotion.

Data are immutable REST-second references. This operation collects actual
ledger transitions and updates the raw Transformer through PPO, then persists
the checkpoint and economic diagnostics. It is deliberately not a replacement
for a preregistered four-fold study's access and report authorities.
"""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

import torch

from rl_quant.datasets.massive_raw_seconds_v1 import RawSecondCatalog, _write
from rl_quant.datasets.raw_second_economics_v1 import SecondEconomicInputs
from rl_quant.envs.raw_second_portfolio_v1 import RawSecondPortfolioEnv, SecondExecutionConfig
from rl_quant.models.raw_second_policy_v1 import RawSecondActorCritic, RawSecondModelConfig
from rl_quant.rl.ppo import PPOConfig
from rl_quant.training.raw_second_ppo_v1 import LEARNING_SCHEMA, RawSecondPPOTrainer, plan_raw_second_rollouts


def run_raw_second_engineering_episode(
    *, catalog: RawSecondCatalog, output: Path, device: str,
    model_config: RawSecondModelConfig, execution_config: SecondExecutionConfig,
    ppo_config: PPOConfig, rollout_steps: int,
    economic_inputs: SecondEconomicInputs,
) -> dict:
    """One numerical integration episode, no economic eligibility assertion.

    Must be run on assigned GPU compute. No CPU-training fallback, frozen
    context cache, scaler fitting, forecast fitting, or outer-selection loop.
    """
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Raw-second engineering training requires one assigned CUDA GPU")
    schedule = plan_raw_second_rollouts(len(catalog.windows) - 1, rollout_steps)
    if not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Establish deterministic FP32 runtime before the workflow")
    if output.exists():
        raise FileExistsError("Existing evidence must not be overwritten")
    # Validate all immutable raw dependencies before model creation/publication.
    for index in range(len(catalog.windows)):
        catalog.load(index, device=target)
    torch.manual_seed(ppo_config.seed)
    model = RawSecondActorCritic(catalog, model_config).to(target)
    splits, dividends, sessions = economic_inputs.load(catalog)
    env = RawSecondPortfolioEnv(catalog, config=execution_config, device=target,
                                splits=splits, dividends=dividends, sessions=sessions)
    agent = RawSecondPPOTrainer(model, env, ppo_config)
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(learning_schema=LEARNING_SCHEMA, rollout_steps=rollout_steps, rollout_schedule=schedule)
    _write(output / "learning-plan.json", json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())
    updates, trajectory = [], []
    for steps in schedule:
        buffer = agent.collect(steps=steps)
        batch = buffer.as_batch()
        for row in range(steps):
            trajectory.append(dict(raw_window_index=int(batch.observations["raw_window_index"][row, 0, 0]),
                                   requested_action=batch.actions[row, 0].cpu().tolist(),
                                   executed_action=batch.executed_actions[row, 0].cpu().tolist(),
                                   reward_net_log_equity=float(batch.rewards[row, 0])))
        updates.append(agent.update(buffer))
    checkpoint_sha = agent.save(output / "checkpoint.pt")
    summary = dict(schema="rl-quant.massive-raw-second-engineering-run-v2", catalog_sha256=catalog.identity,
                   **plan,
                   asset_ids=catalog.asset_ids, input_contract=asdict(catalog.windows[0].contract),
                   model_config=asdict(model_config), execution_config=asdict(execution_config),
                   economic_inputs=asdict(economic_inputs), ledger=env.audit,
                   ppo_config=asdict(ppo_config), checkpoint_sha256=checkpoint_sha,
                   optimizer_updates=len(updates), updates=updates, trajectory=trajectory,
                   fills=[asdict(fill) for fill in env.fills], terminal_equity=str(env.current_equity),
                   net_return=str(env.current_equity / Decimal(env.config.capital) - 1),
                   engineering_execution_complete=True, real_data_training_ready=False,
                   native_v5_authorized=False, positive_profitability_authorization_eligible=False,
                   hardware=dict(device=str(target), name=torch.cuda.get_device_name(target),
                                 torch=str(torch.__version__), cuda=torch.version.cuda,
                                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(target)))
    _write(output / "COMPLETE.json", json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return summary
