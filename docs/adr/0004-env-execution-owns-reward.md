# ADR-0004: Only the env/execution layer mutates portfolio state and computes reward

**Status:** Partial (agreed direction; staged, not fully realized)

## Context

If more than one place computes P&L, cost, cash handling, terminal liquidation, or action-validity semantics,
workflow-specific reward becomes untrustworthy — a trainer and an evaluator can silently disagree on what a
"return" is. The architecture migration plan names this as the core governance rule.

## Decision

Portfolio-state mutation and trading-reward computation are owned by the env/execution layer. Trainers,
evaluators, decision-log replay, and stress tests must consume that one authority rather than hand-rolling
reward math.

## Status / Consequences

- **Direction accepted; not fully realized.** Today some evaluation still recomputes ledger logic outside the
  env, and full execution-reward training is staged behind result-moving flags (`use_execution_env_reward`,
  with `execution_env_reward_shadow` for shadow logging). These flags are result-moving and default off.
- **Recommended path to "Accepted":** (1) extract pure `execution.reward` / `execution.ledger` primitives;
  (2) add parity tests against the current evaluator/trainer reward outputs; (3) run shadow logging; (4) wire
  evaluation through the shared primitive; (5) only then flip training reward behind `use_execution_env_reward`.
  Do it staged, not as a big-bang — the flag is result-moving (affects reward, P&L, TD targets), so promotion
  must be gated by parity evidence.
- Until then, treat reward computed outside the env as a migration liability to be removed, not a pattern to
  extend.
