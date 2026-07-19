# ADR-0004: Only the env/execution layer mutates portfolio state and computes reward

**Status:** Partial (implemented for the new RL environment; legacy Phase-1 migration remains)

## Context

If more than one place computes P&L, cost, cash handling, terminal liquidation, or action-validity semantics,
workflow-specific reward becomes untrustworthy — a trainer and an evaluator can silently disagree on what a
"return" is. The architecture migration plan names this as the core governance rule.

## Decision

Portfolio-state mutation and trading-reward computation are owned by the env/execution layer. Trainers,
evaluators, decision-log replay, and stress tests must consume that one authority rather than hand-rolling
reward math.

## Status / Consequences

- `rl_quant.envs.VectorPortfolioEnv` now satisfies this decision for the new RL path: it projects requested
  actions, owns state/equity, delegates cost primitives to `execution`, realizes one chronological return,
  drifts holdings, decomposes reward, and liquidates at a true terminal.
- `rl_quant.training.decision_policy` and `daily_policy` still run differentiable portfolio loops. They share
  pure accounting primitives (`drift_weights`, `one_way_turnover`, forced-unavailability handling), but they do
  not step the environment. Their existence keeps this ADR Partial.
- Evaluation for the general RL path is not yet artifact-driven. Before this ADR becomes fully Accepted, route
  sequential scoring through the environment, add deterministic action-trace parity tests against the direct
  baseline, and persist the environment's requested/executed actions and reward ledger.
- Reward-changing migration requires a new run/config identity and fresh paired evaluation. A compatibility flag
  cannot turn an old result into evidence for new semantics.
- Until migration completes, reward computed outside the environment is a named legacy baseline or a liability
  to remove, not a pattern for new algorithms.
