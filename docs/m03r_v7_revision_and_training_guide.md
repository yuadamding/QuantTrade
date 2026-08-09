# M03R v7 revision and training guide

**Status:** non-normative implementation and operations orientation

**Scope:** non-PHI research only

**Normative contracts:** [v7 RFC](prelockbox_hold30_active_alpha_m03r_v7.md)
and [experiment specification](prelockbox_hold30_active_alpha_m03r_v7_experiment.md)

This guide records durable lessons from revising M03R and qualifying its
TOP2000 training route. It does not authorize GPU work, outer-data access,
checkpoint promotion, or investment claims. When this guide and a typed
protocol disagree, the protocol and its package-owned validator win.

## Keep the three evidence layers separate

| Layer | Purpose | What it cannot prove |
|---|---|---|
| Canonical PIT Active-300 v7 | Scientific objective, causal panel, five-seed ensemble, inference, and promotion contract | That the production driver, PIT data, or H100 study is qualified |
| TOP2000 v7 compatibility route | Executable mechanism and numerical qualification on a future-selected universe | Point-in-time validity, canonical factor/sector evidence, or promotion |
| Seed-17 TOP2000 diagnostic | Efficient twelve-setting, six-fold, one-seed screen | Seed robustness, a five-seed ensemble, reportability, or investment performance |

Never relabel evidence between these layers. Shared model code or causal
questions do not make protocol, design, setting, data, or receipt identities
interchangeable.

## What v7 changed scientifically

The v7 revision closes a set of recurring specification-to-execution gaps:

- The economic target is cost-adjusted **active** return relative to C1.
  Portfolio and benchmark factor regressions remain useful diagnostics, but
  promotion requires active multifactor alpha and its lower confidence bound.
- There is no tracking-error floor. The policy may abstain when evidence is
  weak. Risk control uses a 6% annual tracking-error ceiling and active-market-
  beta equivalence around zero rather than total-beta targeting.
- The 42-session recent raw branch, 252-session learned temporal context,
  63-session controlled rollout, and exactly 30 post-fill economic returns are
  separate concepts. Auxiliary 5/21/30/63 labels do not extend economic credit.
- Thirty sessions is a weak age-aware prior. Early sale is legal at every age,
  partial sale is legal, and favorable positions may remain beyond day 30.
  Exact HOLD is supported but optional; it is not activated by age alone.
- Confidence controls initiation or enlargement of active risk. Low confidence
  does not itself liquidate a feasible book to C1; learned exits and hard risk
  repair have separate causes.
- Checkpoint eligibility is driven by economics, risk, integrity, and evidence
  coverage. Survival and RMST are diagnostics and late tie-breakers, not
  compulsory 30-session promotion gates.

The exact twelve setting IDs and one-causal-change semantics live in
`rl_quant.protocol.hold30_alpha_m03r_v7`. Do not maintain an independent
setting registry in an operator script or infer scientific setting from a
Kubernetes completion index.

## Persistence objective invariant

For discretionary sold notional as a fraction of NAV, the canonical term is

```text
coefficient * warmup * mean_valid_sessions(
    sum_age(sold_notional[session, age] * max(0, 1 - age / 30)^2)
)
```

There is no denominator based on total sold notional. This is essential:

- selling 1% young notional must cost 1% as much as selling 100%;
- mature sales at age 30 or older have zero persistence cost;
- mature sales must not dilute the cost of a young sale;
- unavailable, hard-risk, corporate-action, and terminal exits remain exempt;
- the coefficient is a content-bound setting field, not a free runtime knob.

The canonical coefficient is 5 bp per unit NAV sold at age zero. P00 and P10
own separate 0-bp and 10-bp setting identities. Warm-up is the first 10% of the
frozen optimizer-step plan.

## Canonical and diagnostic training geometry

Canonical v7 pairs twelve settings over six folds and five seeds. Seeds are
algorithmic replications on the same market history. For each fold, selected
seed outputs are combined before one authoritative projection and one
chronological portfolio execution; the 30 fold-seed cells are not independent
return histories.

The seed-17 diagnostic deliberately reduces this to:

```text
12 settings x 6 folds x seed 17 = 72 cells
64 alpha-core optimizer updates per full cell
2 H100 ranks in one worker Pod per setting
at most 8 concurrent workers = at most 16 requested H100s
```

Its completion-to-scientific-setting order is content-bound and is not simple
numeric order: `(0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11)`. Use the package map.
The one-member fold execution reuses the exact seed-validation trace; it must
not execute the model again to imitate a one-member ensemble.

Each two-rank TOP2000 worker keeps the full cross-section on both ranks and
splits loss-bearing origins across ranks. Do not shard the stock axis without
separate qualification of cross-sectional attention, benchmark, projection,
and gradient semantics. Two 80-GB GPUs are two address spaces, not a pooled
160-GB tensor heap.

## TOP2000 compatibility limits

The executable cache contains 1001 daily rows, 1999 actions including CASH,
and five OHLCV channels for 2022-01-03 through 2025-12-29. It is aggregated
daily data from 300-second bars, not the canonical PIT Active-300 raw-data
contract.

The compatibility route also differs from canonical evidence:

- the universe is future-selected and therefore nonreportable;
- policy and benchmark returns use a single 20-bp cost rather than canonical
  10/20/40-bp validation evidence;
- its four causal projection controls are not a canonical factor/sector
  manifest;
- non-A12 confidence sizing is raw `sigmoid(logit) * 0.04`, not governed
  post-freeze calibration;
- a full cell uses a content-bound development profile, not a free training
  configuration.

These differences are useful for mechanism qualification but cannot be hidden
behind canonical v7 names.

## Numerical accounting lessons

### Preserve exact forward economics below machine epsilon

A TOP2000 book can contain hundreds of legitimate positive FP32 positions
smaller than `torch.finfo(float32).eps`. Using epsilon as an absolute divisor
floor under-sells those positions, moves the requested cash amount anyway, and
breaks cohort-to-weight reconciliation.

For forward accounting, divide by the exact positive denominator and replace
only a true zero denominator with one. Apply the rule to total sale value,
proposed release, residual release, and retention-unit removal. Raising the
reconciliation tolerance is not a fix.

### Bound the backward derivative without changing the forward value

Repeated partial releases can drive a positive cohort into FP32 subnormals.
The exact quotient remains economically correct, but differentiating through
`1 / subnormal` can produce an infinite or NaN gradient. The implementation
therefore returns the detached exact quotient in the forward pass while taking
the derivative from an epsilon-bounded quotient.

Regression coverage must include:

- many TOP2000-shaped sub-epsilon positions;
- full and partial sales;
- proposed-release and ordinary pro-rata paths;
- exact economic-value and retention-unit removal;
- repeated partial sales into positive FP32 subnormals; and
- finite backward gradients.

### Reject a bad update before optimizer mutation

Gradient clipping must use `error_if_nonfinite=True`. A rejected update must
leave parameters and optimizer state unchanged. A finite encoder output at
step `n+1` is too late to protect a checkpoint written after a non-finite step
`n`.

## Packaging and qualification lessons

- A local artifact path used by an operator and the bound in-container plan
  path used by a worker are distinct trust boundaries. Preserve both in the
  package; do not rewrite the local path into a container path during local
  validation.
- Run the six-fold C1 benchmark feasibility and 1% cap audit on CPU before GPU
  allocation. Reuse its exact matching receipt; a GPU Pod should not
  rediscover an infeasible benchmark.
- The seed-17 package requires both its exact validation sentinel and ordered
  all-setting qualification artifacts. Do not add a legacy same-Job sentinel
  and scale ramp after the matching capacity receipt already exists.
- Sentinel and all-setting qualification runtime terminal receipts are not
  idempotent plan files. Consume the sentinel by exact hash or give the later
  phase a disjoint output root; never rerun into the same immutable terminal
  path.
- Create, bind, activate, retry, and cleanup are serialized lifecycle edges.
  Parallelize only independent local tests, rendering, hashing, and safe
  archive verification.
- A four-update seed-17 validation sentinel proves the route starts, validates,
  and executes under the bound two-rank shape. The current seed-17 worker does
  not intentionally exercise checkpoint restart. The sentinel is not a
  64-update fit experiment and cannot diagnose underfitting.

## Fit diagnosis

Operational health and scientific fit answer different questions. Ready GPUs,
VRAM consumption, elapsed time, moving checkpoints, and a successful sentinel
prove execution—not learning.

Use this evidence ladder:

| Evidence | Allowed interpretation |
|---|---|
| Four-update seed-17 qualification sentinel | Startup, wiring, validation, parity, and capacity only; no restart or fit conclusion |
| One completed seed-17 cell plus validation | Directional diagnostic only |
| Six completed seed-17 cells and fold executions for one setting | One-seed across-fold mechanism diagnostic; no seed-robustness conclusion |
| All 72 paired seed-17 cells | Complete development panel screen; still nonreportable and nonpromotable |
| Future canonical-identity five-seed validations plus one chronological fold ensemble | Minimum preliminary canonical fit screen; the seed-17 route cannot emit this evidence |
| Future canonical-identity fold ensembles and paired contrasts | Robust development-panel conclusion, subject to a generation-qualified canonical evaluator |

Near-zero active risk is not automatically underfitting because v7 has no
tracking-error floor; it may be correct abstention. Weak training and weak
validation can also indicate a broken gradient route, optimizer failure, or no
usable signal. Use same-fold/same-seed causal contrasts and matching validation
receipts before changing capacity or update count.

The current seed-17 worker retains `last_metrics` in rotating checkpoints but
does not publish a durable per-update learning curve. Adjacent checkpoint slots
use different deterministic episodes, so two adjacent metrics do not prove a
plateau.

## Recovery and source lineage

Classify a failure before choosing reuse:

- An orchestration-only failure before model execution can preserve the exact
  scientific package when its source and evidence remain valid.
- A numerical source defect that can affect economic accounting or gradients
  invalidates the execution-source key. Preserve terminal evidence and exact-
  clean the failed Job, then build a fresh source-bound qualification.
- Never resume a checkpoint whose parameters or optimizer state may already
  contain non-finite values.
- Prefer a fresh all-setting source-homogeneous rerun when the defect could
  affect already completed cells. Partial patched recovery is development-only
  and requires an explicit mixed-source lineage contract; never silently merge
  old and patched evidence.

A local Git commit or clean worktree does not prove which source a remote run
used. Require the exact package plan, source archive, source manifest, runtime
manifest, image digest, and per-cell receipts.

## Implementation map and focused verification

The package-owned source/test inventory in
`rl_quant.workflows.hold30_alpha_prelockbox.V3_LATER_GENERATION_SOURCE_TESTS`
is authoritative for registered later-generation surfaces. Important entry
points include:

- `rl_quant.protocol.hold30_alpha_m03r_v7` — canonical identities;
- `rl_quant.training.hold30_alpha_m03r_v7` — proportional persistence;
- `rl_quant.training.top2000_m03r_v7_dev` — compatibility trainer;
- `rl_quant.envs.hold30` — cohort accounting;
- `rl_quant.workflows.top2000_m03r_v7_package_builder` — package build;
- `rl_quant.workflows.top2000_m03r_v7_seed17_dev` — one-seed worker; and
- `rl_quant.training.top2000_m03r_v7_seadragon_lifecycle` — typed lifecycle.

Run the single maintained focused command in the repository
[AGENTS.md](../AGENTS.md#local-verification). Add the smallest blocking test
for the changed boundary rather than copying that command into another guide.

Remote status and scientific completion must be read from the exact run's
receipts. Do not write transient Job progress into this guide.
