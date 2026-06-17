"""Backward-compatibility shim. The intraday DQN workflow split into the envs layer (rl_quant.envs.intraday: the env + ACTION_TO_POSITION) and the training layer (rl_quant.training.intraday: configs + pretraining + eval + train loop) in the protocol-first reorganization; this re-export keeps the old import path working (see architecture_migration_plan.md)."""

from rl_quant.training.intraday import *  # noqa: F401,F403
