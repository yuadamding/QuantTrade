"""Backward-compatibility shim. The strategy DQN workflow split into the envs layer (rl_quant.envs.strategy: the env) and the training layer (rl_quant.training.strategy: configs + eval + train loop) in the protocol-first reorganization; this re-export keeps the old import path working (see architecture_migration_plan.md)."""

from rl_quant.training.strategy import *  # noqa: F401,F403
