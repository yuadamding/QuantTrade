"""Two-stage training for the learning framework, decoupled to enforce the context/policy split.

  * context_pretrain -- Stage 1: self-supervised context learning, then freeze + encode windows to detached
    context embeddings while carrying raw bars forward.
  * decision_policy  -- Stage 2: differentiable-portfolio policy learning on detached context plus a trainable
    raw-second policy encoder.

Both expose resumable training routines (start_step + optimizer + checkpoint callback); persistence/resume
orchestration is the caller's (the driver's) job.
"""
from __future__ import annotations

from rl_quant.training.context_pretrain import (
    encode_days,
    freeze_encoder,
    ssl_targets,
    ssl_targets_daily,
    ssl_targets_perstock,
    train_context_encoder,
)
from rl_quant.training.daily_policy import (
    daily_cost_paid_baselines,
    daily_policy_telemetry,
    evaluate_daily_detailed,
    train_daily_policy,
)
from rl_quant.training.decision_policy import (
    cost_paid_baselines,
    evaluate_policy,
    evaluate_policy_detailed,
    policy_telemetry,
    train_decision_policy,
)
from rl_quant.training.designs import DEFAULT_DESIGN, DESIGNS, SWEEP, Phase1Design

__all__ = [
    "DEFAULT_DESIGN",
    "DESIGNS",
    "Phase1Design",
    "SWEEP",
    "cost_paid_baselines",
    "daily_cost_paid_baselines",
    "daily_policy_telemetry",
    "encode_days",
    "evaluate_daily_detailed",
    "evaluate_policy",
    "evaluate_policy_detailed",
    "freeze_encoder",
    "policy_telemetry",
    "ssl_targets",
    "ssl_targets_daily",
    "ssl_targets_perstock",
    "train_context_encoder",
    "train_daily_policy",
    "train_decision_policy",
]
