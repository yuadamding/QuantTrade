"""Backward-compatibility shim. The decision framework (point-in-time causal asserts + market/feature
manifests) moved to the evaluation layer (``rl_quant.evaluation.decision_framework``), co-located with the
research protocol it builds on, in the protocol-first reorganization; this re-export keeps the old import
path working (see architecture_migration_plan.md)."""

from rl_quant.evaluation.decision_framework import *  # noqa: F401,F403
