"""Backward-compatibility shim. The trading-constraints contract (action masks, transition/constraint
feature schema, decision-tensor mask semantics) moved to the protocol layer (``rl_quant.protocol.constraints``)
in the protocol-first reorganization -- it is the contract, so it now lives in the protocol layer. This
re-export keeps the old import path (used by 16 modules) working (see architecture_migration_plan.md)."""

from rl_quant.protocol.constraints import *  # noqa: F401,F403
