"""Backward-compatibility shim. The intraday dataset builder moved to the datasets layer
(``rl_quant.datasets.intraday``) in the protocol-first reorganization; this re-export keeps the old import
path working (see architecture_migration_plan.md)."""

from rl_quant.datasets.intraday import *  # noqa: F401,F403
