"""Backward-compatibility shim. The confidence/calibration utilities moved to the evaluation layer
(``rl_quant.evaluation.confidence``) in the protocol-first reorganization; this re-export keeps the old
import path working (see architecture_migration_plan.md)."""

from rl_quant.evaluation.confidence import *  # noqa: F401,F403
