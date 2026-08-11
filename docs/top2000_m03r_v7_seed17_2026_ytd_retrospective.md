# TOP2000 M03R-v7 seed-17 2026-YTD retrospective

**Status:** development-only retrospective evaluation contract and operator
orientation

**Scope:** non-PHI research only

**Source panel:** [TOP2000 M03R-v7 seed-17 diagnostic](top2000_m03r_v7_seed17_diagnostic.md)

This document explains how the completed twelve-setting, six-fold, seed-17
panel may be evaluated on the available 2026 TOP2000 history. It does not turn
that history into a lockbox, authorize promotion, or make the future-selected
universe point-in-time valid. Typed protocol and receipt validators remain
authoritative when this guide and executable code disagree.

## Evidence limits

The available evaluation interval is 2026-01-02 through 2026-06-23. It is a
YTD interval, not a full-calendar-year result. The static TOP2000 universe was
selected on 2026-06-12 using information from 2026, so every output is:

- retrospective and development-only;
- affected by future selection and survivorship;
- based on one algorithmic seed;
- unsuitable for scientific reporting or investment claims; and
- ineligible for checkpoint or setting promotion.

The evaluation may diagnose mechanism behavior and execution economics. It
cannot establish the canonical PIT Active-300 result or five-seed robustness.

## Freeze before opening outcomes

The evaluation has an explicit pre-access boundary. Before any 2026 bar or
factor-return value is read, freeze and content-bind:

1. the completed 12-by-6 training coverage and rank runtime proofs;
2. all 72 setting/fold checkpoint, cell, validation, and fold-execution
   receipts;
3. the pre-2026 cache, action axis, and training package identities;
4. the exact evaluation source inventory and protocol contract;
5. the TOP2000 metadata namespace and declared YTD interval;
6. the headline and sensitivity checkpoint rule; and
7. bootstrap, contrast, cost, and reportability semantics.

The frozen plan must record that outcome partitions and factor archives were
not opened while it was constructed. Later materialization accepts the exact
plan file hash and semantic receipt hash; a caller-supplied digest is not a
substitute for hashing the bound files.

## Checkpoint roles and leakage-safe carry

For each setting, fold 5 / seed 17 / rank 0 is the sole headline checkpoint.
Folds 0 through 4 are cutoff sensitivities. They are reported separately and
must never be averaged, pooled, or treated as an ensemble.

The encoder may consume the full causal 252-session context available at each
decision. Economic portfolio state is different: it begins strictly after the
checkpoint's training cutoff. For the current fold-5 geometry, this means the
economic trace begins at local context offset 93. Earlier folds begin at the
start of the retained 252-session context because their cutoffs predate it.

Every economic trace starts from C1 at age zero with no inherited policy
retention units. This prevents a checkpoint from seeding the January 2026 book
with positions chosen on dates that were in its own training sample. The trace
then runs once, chronologically, with no fold reset before the scored YTD
suffix.

## Cost and execution semantics

The authoritative policy path is a closed-loop 20-bp one-way-cost execution.
The 10-bp and 40-bp rows reprice that exact executed turnover path. The policy
is not rerun at the sensitivity costs, because doing so would change actions
and confound cost sensitivity with a second policy trajectory.

Requested actions, executed actions, constraint repair, forced exits, and
discretionary exits remain cause-typed. The evaluator consumes executed
turnover and the chronological cohort ledger; it does not infer holding
behavior from aggregate turnover.

## Holding and censoring evidence

Survival and RMST are telemetry, not eligibility gates. Censoring-aware RMST60
uncertainty requires complete cohort trajectories rather than a date-by-age
snapshot bootstrap. The retained artifact therefore binds, by origin date:

- entry notional or units;
- partial discretionary event units and age;
- forced-censor units, age, and cause; and
- terminal right-censored units.

Primary RMST uncertainty resamples origin-date clusters while carrying each
sampled cohort's complete entry-to-event-or-censor trajectory. A calendar
block bootstrap of aggregate risk sets may be reported only as a sensitivity;
it is not labeled censoring-aware cohort uncertainty.

This RMST estimand covers entries whose origins fall inside the scored YTD
window. It deliberately excludes positions carried into scoring from the
pre-score economic prefix. S(10), S(20), and S(30) are separate descriptive
date-by-age notional-snapshot product-limit values; they are not
censoring-aware cohort RMST evidence.

## Factor evidence

Available factor attribution uses official daily Kenneth French FF5 plus
Momentum data with exact date joins, percent-to-decimal conversion, and no
imputation. The package-owned retrieval step binds the frozen HTTPS URLs,
response URL, response status, archive bytes, selected members, and the frozen
evaluation-plan identities. Redirects are rejected.

The parser extracts only the exact scored dates. Rows after 2026-06-23 may be
present in a retrieved source container, but their return cells are never
parsed or used; their presence and count are recorded explicitly. Arbitrary
caller-staged archives cannot be labeled official evidence.

## Statistical families

Each checkpoint fold owns one separate inference family. The primary 20-bp
return path uses 10,000 joint circular moving-block draws of 21 trading
sessions, with 10 and 30 as sensitivities. The same date indices apply across
settings, metrics, and paired contrasts within that fold. The 10-bp and 40-bp
cost rows are frozen-path point sensitivities; they are not separately
bootstrapped in v1.

The eleven declared causal contrasts use a joint maximum-absolute-centered
familywise procedure. Raw one-sided bootstrap p-values are computed from
null-centered draws. Sharpe, information ratio, and tracking error use sample
standard deviation. Factor alpha, active beta equivalence, and Sharpe
difference remain unavailable unless the exact factor evidence is complete.

## Efficient GPU shape

Inference needs one visible GPU, not the two-rank training shape. The efficient
panel layout is one worker per scientific setting:

```text
12 logical setting workers x 1 H100 request each
one resident retrospective cache per worker
fold order: headline fold 5, then sensitivity folds 0, 1, 2, 3, 4
six immutable checkpoint artifacts per worker
```

Twelve is the logical panel request ceiling. A full setting-0 worker serves as
the single-GPU startup/capacity sentinel and its six outputs are reused; after
that Pod is terminal and exact-cleaned, the remaining phase requests at most
eleven H100s. This avoids 72 independent Pod startups, repeated cache loads,
and a scientifically redundant all-setting qualification panel. The runtime
proves one visible CUDA device; the operational startup gate must additionally
prove the exact H100 80-GB model, memory range, and compute capability before
the sentinel is accepted.

Each checkpoint artifact records peak allocated and reserved VRAM, allocator
retry/OOM deltas, device identity, a bound model-state hash and explicit
no-change proof, the economic execution offset, evaluator arrays,
cause/age/action telemetry, and cohort-survival evidence. A per-setting retry
may reuse only artifacts that independently validate against the exact frozen
plan and source inventory.

## Receipt graph and failure rules

The minimum successful graph is:

```text
completed training coverage
  -> frozen pre-access plan and source inventory
  -> immutable 2026-YTD cache and official-factor retrieval/data receipts
  -> 72 checkpoint execution artifacts
  -> 12 setting-completion receipts
  -> 6 separate fold evaluation receipts
  -> one research-only panel summary
```

At operational boundaries, retain the exact source archive, image digest,
rendered/admitted manifest, Job UID, startup proof, terminal evidence, and
UID/resourceVersion cleanup receipts. Unknown create or activation state is
not absence. A failed or ambiguous evaluation Job must not modify or recreate
the completed training Job.

Stop before outcome access when source, checkpoint, or plan validation fails.
After outcome access, fix only implementation defects that can be resolved
without tuning to observed performance; otherwise mint a new explicitly
consumed retrospective generation. Preserve failure receipts rather than
rewriting them.

## Required outputs

For every setting and checkpoint fold, retain at least:

- 10/20/40-bp net active return and the 20-bp bootstrap LCB;
- information ratio, policy and benchmark Sharpe, Sharpe difference and LCB;
- active market beta and equivalence bound;
- active multifactor alpha and LCB when official factors are available;
- annualized tracking error and maximum drawdown;
- cause-typed discretionary and forced turnover and transaction cost;
- RMST60, survival at 10/20/30, censoring-aware uncertainty, and age exits;
- HOLD/CONTINUOUS/EXIT frequencies and hazard saturation diagnostics;
- requested-to-executed projection distance; and
- exact source, data, checkpoint, execution, and inference receipt hashes.

The final summary must state the YTD dates, future-selection contamination,
single-seed limitation, fold roles, and that no row is promotion eligible.
Live Job status belongs in operational receipts, not in this document.
