# QuantTrade

QuantTrade is a non-PHI research library for reinforcement-learning and direct
portfolio-policy experiments. It is not a live-trading service, investment
product, or business-production system.

## Start here

- Read [the documentation index](docs/README.md) before changing a Hold-30 or
  M03R contract.
- Read [the QuantTrade training knowledge base](docs/quanttrade_training_knowledge_base.md)
  before packaging, launching, recovering, evaluating, or interpreting an
  M03R training run.
- Treat [M03R-v12 rank/scale-decoupled predictive research](docs/top2000_m03r_v12_rank_scale_decoupled.md)
  as completed negative future-selected TOP2000 development evidence. Its
  exact 3-session run did not pass the predictive gate and cannot authorize
  economic training or 2026 access. V11 and its a15 audit are immutable
  predecessor evidence; V10 was superseded before launch. Repository text
  never establishes live remote state or authorizes GPU work.
- Treat the [v12 post-hoc inference audit](docs/top2000_m03r_v12_posthoc_inference_audit.md)
  as explanatory reuse of exact frozen checkpoints only. It compares rank and
  economic heads under corrected causal masks and chronology, but cannot
  select a v12 model, mint an economic generation, or access 2026.
- Treat [M03R-v14 executable-score-aligned h3 research](docs/top2000_m03r_v14_context_matched_h3.md)
  as completed negative development evidence. Neither setting passed the
  predictive or tradeability gate, so it cannot authorize economic training.
- Treat [M03R-v15 corrected executable-score h3 research](docs/top2000_m03r_v15_executable_score_corrected_h3.md)
  as the current local predictive implementation boundary. It corrects v14
  provenance, rank-gradient, ablation, capacity, and checkpoint-selection
  defects. Local source and tests do not authorize GPU or 2026 access.
- Treat the [M03R v7 RFC](docs/prelockbox_hold30_active_alpha_m03r_v7.md) and
  [experiment specification](docs/prelockbox_hold30_active_alpha_m03r_v7_experiment.md)
  as the canonical PIT Active-300 objective and promotion contract. That
  scientific role is distinct from v11's executable TOP2000 development role.
- Read [the M03R v7 revision and training guide](docs/m03r_v7_revision_and_training_guide.md)
  before changing the objective, cohort accounting, TOP2000 runtime, package,
  or recovery behavior.
- The [seed-17 TOP2000 diagnostic](docs/top2000_m03r_v7_seed17_diagnostic.md)
  is development-only and nonreportable. It cannot satisfy the canonical
  five-seed ensemble or promotion contracts.
- Read the [2026-YTD retrospective guide](docs/top2000_m03r_v7_seed17_2026_ytd_retrospective.md)
  before opening 2026 TOP2000 outcomes or changing checkpoint, carry, factor,
  censoring, inference, or one-GPU execution semantics. Its results remain
  development-only, nonreportable, and nonpromotable.

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
# V11 predecessor and a15 evidence boundary
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v11_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v11_*.py \
  tests/test_cost_aware_active_policy_v3.py

# Completed v12 predictive boundary
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v12_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v12_*.py \
  tests/test_cost_aware_active_policy_v4.py

# V12 exact-checkpoint post-hoc audit boundary
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_top2000_m03r_v12_posthoc_inference_audit.py

# V13 context, direct-score, target/action, and schedule boundary
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v13_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v13_*.py

# V14 executable-score, causal-support, and numerical boundary
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v14_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v14_*.py

# V15 package-owned preflight, corrected objective, and checkpoint-selection boundary
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v15_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v15_*.py

# Canonical v7 contract boundary
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

The core repository commands are documented in [the README](README.md). Do not
treat the quarantined scripts under `legacy/` as runnable package entrypoints.

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
- The v7/v8 four-update seed-17 qualification sentinel proves startup, wiring,
  validation, parity, and capacity surfaces—not checkpoint restart, fit, or
  underfitting. V11 instead uses a disjoint two-rank capacity Job followed by
  update-64 predictive qualification. Never transfer a sentinel shape or its
  scientific meaning between generations. A one-seed panel remains a mechanism
  screen, not five-seed ensemble evidence.
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
