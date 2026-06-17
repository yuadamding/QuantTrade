"""Backward-compatibility shim. The runtime config (RuntimeConfig + runtime-arg helpers) moved to the
workflows layer (``rl_quant.workflows.config``) in the protocol-first reorganization; this re-export keeps
the old import path working (see architecture_migration_plan.md)."""

from rl_quant.workflows.config import *  # noqa: F401,F403
