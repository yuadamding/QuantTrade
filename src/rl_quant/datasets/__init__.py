"""Datasets layer: gold dataset builders / loaders that turn vendor data + features into compact, causal
training tensors and splits. Part of the protocol-first layered architecture
(see architecture_migration_plan.md). Submodules: ``intraday`` (intraday split builder), ``strategy``
(strategy split builder). Old top-level import paths (``rl_quant.intraday_data`` / ``rl_quant.strategy_data``)
remain working via shims."""
