"""Backward-compatibility shim. The partition protocol moved to the protocol layer
(``rl_quant.protocol.partition``) in the protocol-first reorganization; this re-export keeps the old import
path working (see architecture_migration_plan.md)."""

from rl_quant.protocol.partition import *  # noqa: F401,F403
