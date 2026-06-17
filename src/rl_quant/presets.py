"""Backward-compatibility shim. Run presets moved to the workflows layer (``rl_quant.workflows.presets``) in
the protocol-first reorganization; this re-export keeps the old import path working
(see architecture_migration_plan.md)."""

from rl_quant.workflows.presets import *  # noqa: F401,F403
