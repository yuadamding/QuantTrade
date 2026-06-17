"""Backward-compatibility shim. The action-universe / risk metadata moved to the features layer
(``rl_quant.features.action_risk``) in the protocol-first reorganization; this re-export keeps the old import
path working (see architecture_migration_plan.md)."""

from rl_quant.features.action_risk import *  # noqa: F401,F403
