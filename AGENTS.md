# QuantTrade

QuantTrade is a non-PHI research library for reinforcement-learning and direct
portfolio-policy experiments. It is not a live-trading service, investment
product, or business-production system.

## Start here

- Read [the documentation index](docs/README.md) before changing a Hold-30 or
  M03R contract.
- Treat the [M03R v7 RFC](docs/prelockbox_hold30_active_alpha_m03r_v7.md) and
  [experiment specification](docs/prelockbox_hold30_active_alpha_m03r_v7_experiment.md)
  as the current canonical scientific direction.
- Read [the M03R v7 revision and training guide](docs/m03r_v7_revision_and_training_guide.md)
  before changing the objective, cohort accounting, TOP2000 runtime, package,
  or recovery behavior.
- The [seed-17 TOP2000 diagnostic](docs/top2000_m03r_v7_seed17_diagnostic.md)
  is development-only and nonreportable. It cannot satisfy the canonical
  five-seed ensemble or promotion contracts.

Repository documentation describes contracts and implementation state. It
does not prove that a remote Job used this checkout, completed successfully,
or produced valid performance. Those claims require the exact package/source,
data, image, application, terminal, and cleanup receipts for that run.

## Structure

- `src/rl_quant/rl/` — domain-neutral RL contracts and algorithms.
- `src/rl_quant/envs/` and `src/rl_quant/execution/` — authoritative portfolio state,
  cohort accounting, constraints, costs, and requested-to-executed actions.
- `src/rl_quant/protocol/` — immutable scientific generations and identities.
- `src/rl_quant/training/` — objectives, routes, schedules, workers, packages,
  and lifecycle primitives.
- `src/rl_quant/evaluation/` — statistical and reportability components.
- `src/rl_quant/workflows/` — package-owned CLI surfaces; keep wrappers thin.
- `tests/` — blocking contract and regression tests.
- `docs/adr/` — durable architecture decisions; historical ADRs may be
  superseded without rewriting their original decision body.

## Local verification

Use the Python 3.11 `quanttrade` environment. Focus tests on the changed
boundary before running the full suite.

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest \
  tests/test_hold30_accounting.py \
  tests/test_hold30_alpha_m03r_v7_protocol.py \
  tests/test_hold30_alpha_m03r_v7_objective.py \
  tests/test_hold30_alpha_m03r_v7_routes.py \
  tests/test_hold30_alpha_m03r_v7_schedule.py \
  tests/test_top2000_m03r_v7_dev_training.py \
  tests/test_top2000_m03r_v7_seed17_generation.py -q

conda run -n quanttrade ruff check src scripts tests
```

The complete commands supported by project configuration are documented in
[the README](README.md). Do not treat the quarantined scripts under `legacy/`
as runnable package entrypoints.

## Scientific conventions

- Protocol, design, setting, schema, source, and receipt identities are
  immutable. A result-moving semantic change requires a new generation; never
  relabel an older artifact.
- Environment/execution code owns portfolio mutation, cost, cohort age, and
  reward accounting. Keep requested and executed actions distinct.
- Thirty sessions is a soft one-sided persistence preference. It is not a
  minimum hold, sell mask, expiry, turnover proxy, or promotion gate.
- Promotion evidence is active return relative to C1, including active
  multifactor alpha. Portfolio factor alpha is not a substitute.
- Preserve point-in-time data and policy/evaluator access boundaries. The
  future-selected TOP2000 cache is mechanism-diagnostic only.
- Holding telemetry, folds, and seeds have different statistical roles. Seeds
  are algorithmic replications on shared history, not independent market
  paths.

## Training and recovery gotchas

- Keep canonical PIT Active-300 v7 separate from the executable TOP2000
  compatibility route. Shared causal questions do not make their artifact
  identities or evidentiary status interchangeable.
- A four-update seed-17 qualification sentinel proves startup, wiring,
  validation, parity, and capacity surfaces—not checkpoint restart, fit, or
  underfitting. The one-seed panel is a mechanism screen, not five-seed
  ensemble evidence.
- Positive FP32 cohort notionals below machine epsilon remain economically
  real. Preserve exact forward sale accounting while using bounded backward
  derivatives; reject non-finite gradients before any optimizer step.
- A checkpoint written after a non-finite update is poisoned. Do not resume it.
  If a numerical source defect may affect completed cells, preserve the failed
  evidence and default to a fresh source-homogeneous panel.
- Local package inspection paths and bound in-container runtime paths are
  separate trust boundaries. Validate both; do not rewrite one as the other.
- Immutable runtime terminal receipts must not share output paths across a
  sentinel and a later qualification phase. Consume an existing receipt by
  exact hash or use a disjoint phase output identity.

Remote GPU work is never authorized by this file. Use the environment's
approved Seadragon/Kubernetes research runbook, exact Job/run identity, and
receipt-gated lifecycle. Do not record live Job status, cluster credentials,
or machine-specific secrets in repository documentation.
