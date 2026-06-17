"""Backward-compatibility shim. The CLI entry point moved to the workflows layer (``rl_quant.workflows.cli``)
in the protocol-first reorganization; this re-export keeps the old import path working
(see architecture_migration_plan.md)."""

from rl_quant.workflows.cli import *  # noqa: F401,F403
