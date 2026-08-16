# Architecture Decision Records

Short, durable records of the *policy* decisions behind QuantTrade — distinct from the migration-state notes in
`docs/architecture_migration_plan.md`. An ADR captures a decision that future contributors should treat as
settled (or know is staged), so the reasoning is not buried in code comments or commit messages.

Each ADR states its **Status** (`Accepted` = current policy; `Partial` = agreed direction, not fully realized),
the **Context**, the **Decision**, and the **Consequences**. Most are *enforced in CI* — the enforcing test is
named so a reader can see the policy is not just aspirational.

| ADR | Decision | Status | Enforced by |
|-----|----------|--------|-------------|
| [0001](0001-protocol-first-package-ownership.md) | Protocol-first layered package; the package owns logic, layers form a DAG | Accepted | `tests/test_import_boundaries.py` |
| [0002](0002-strict-reportability-default.md) | `reportable: true` requires the strict tier (`reportable_tier`) | Accepted | `tests/test_decision_framework.py::...classify_reportability...` |
| [0003](0003-return-basis-content-hash.md) | A canonical `ReturnBasis` + content hash is the return-economics contract | Accepted; manifest agreement tested, known-digest golden test still needed | `tests/test_research_protocol.py` |
| [0004](0004-env-execution-owns-reward.md) | Only the env/execution layer may mutate portfolio state and compute reward | Partial | `tests/test_portfolio_env.py`; legacy gaps in `tests/test_daily_runtime_accounting.py` |
| [0005](0005-scripts-to-package-migration.md) | `scripts/` are thin wrappers; the package owns implementation | Accepted | `tests/test_scripts_are_wrappers.py` |
| [0006](0006-daily-decision-soft-30-session-holding.md) | One daily decision, carried positions, soft 30-session holding target | Accepted; implemented mechanics, governed PIT launch blocked | `tests/test_hold30_accounting.py`, `tests/test_hold30_alpha_m03r_v7_protocol.py`, `tests/test_hold30_alpha_m03r_v7_objective.py`, and `tests/test_top2000_m03r_v7_dev_semantics.py` |
| [0007](0007-benchmark-relative-hold30-alpha-objective.md) | Historical C1-relative alpha-v3 decision; superseded before launch by the active-alpha M03R generations | Superseded; retained as immutable design history | `tests/test_hold30_alpha_v3_protocol.py`, `tests/test_hold30_alpha_core.py`, `tests/test_hold30_alpha_evaluation.py`, and `tests/test_hold30_alpha_workflow.py` |
| [0008](0008-parameterized-soft-holding-target.md) | New generic holding APIs default to calibrated soft target 3; legacy Hold-30 protocols bind 30 explicitly | Accepted; core specification and v16 artifact binding implemented, broader generic replay/CLI migration staged | `tests/test_hold_target_protocol.py`, `tests/test_top2000_m03r_v16_initial_state.py`, `tests/test_top2000_m03r_v16_checkpoint.py` |
