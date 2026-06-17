"""Backward-compatibility shim. The research/holdout protocol moved to the evaluation layer
(``rl_quant.evaluation.research_protocol``) in the protocol-first reorganization; this re-export keeps the
old import path working (see architecture_migration_plan.md)."""

from rl_quant.evaluation.research_protocol import *  # noqa: F401,F403
