"""Backward-compatibility shim. The strategy dataset builder moved to the datasets layer
(``rl_quant.datasets.strategy``) in the protocol-first reorganization; this re-export keeps the old import
path working (see architecture_migration_plan.md)."""

from rl_quant.datasets.strategy import *  # noqa: F401,F403
