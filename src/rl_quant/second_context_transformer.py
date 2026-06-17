"""Backward-compatibility shim. The second-context transformer workflow split into datasets/ (rl_quant.datasets.second_context) and evaluation/ (rl_quant.evaluation.second_context) in the protocol-first reorganization; these re-exports keep the old import path working (see architecture_migration_plan.md)."""

from rl_quant.datasets.second_context import *  # noqa: F401,F403
from rl_quant.evaluation.second_context import *  # noqa: F401,F403
