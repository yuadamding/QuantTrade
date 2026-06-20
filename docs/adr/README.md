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
| [0003](0003-return-basis-content-hash.md) | A canonical `ReturnBasis` + content hash is the return-economics contract | Accepted | `tests/test_golden_artifacts.py`, `test_research_protocol.py` |
| [0004](0004-env-execution-owns-reward.md) | Only the env/execution layer may mutate portfolio state and compute reward | Partial | (staged behind `use_execution_env_reward`) |
| [0005](0005-scripts-to-package-migration.md) | `scripts/` are thin wrappers; the package owns implementation | Accepted | `tests/test_scripts_are_wrappers.py` |
