# M03R v7 twelve-setting experiment specification

**Protocol:** `prelockbox-hold30-active-alpha-m03r-v7`  
**Design:** `daily_raw_pit300_hold30_m03r_v7`  
**Canonical:** `M03R-soft-persistence-active-alpha-hold30-v7`  
**Status:** implementation-qualified locally; real two-H100 qualification required before launch
**Normative RFC:** [M03R v7 active-alpha Hold-30 RFC](prelockbox_hold30_active_alpha_m03r_v7.md)

This document specifies the paired experiment, GPU topology, admission order,
reporting, and promotion rules. It does not launch Kubernetes work and cannot
serve as a launch receipt.

## Experimental unit and pairing

All twelve settings use the same six folds and five seeds:

```text
validation/outer folds     6
seeds per setting         5
fold-seed cells/setting  30
primary settings         12
total training cells    360
```

The seed tuple, fold boundaries, data order, initialization convention,
confidence-calibration procedure, optimizer schedule, update budget, numeric
precision, and execution chronology are identical across settings. Each
setting differs from canonical only through the field declared in the RFC.

Seeds are algorithmic replications on a shared market history, not independent
investment histories. For every setting and fold:

1. train the five paired seeds;
2. select each seed's checkpoint using inner validation only;
3. fit and freeze each seed's inner-validation confidence calibrator where the
   route uses calibrated confidence;
4. aggregate the five frozen seed outputs in output space;
5. apply risk projection and constraints once after aggregation;
6. execute one continuing chronological ensemble portfolio;
7. use that single deployed return path for fold-level inference.

Inference must not treat the 30 fold-seed cells as 30 independent return paths.
The primary evidence is the chronological deployed ensemble series pooled
across the six predeclared fold segments using the sealed fold-aware bootstrap
and regression contract.

## Two-H100 worker topology

Every primary setting uses one Kubernetes worker Pod with two NVIDIA H100 80GB
GPUs:

```text
torchrun --nproc_per_node=2
DDP or qualified synchronized two-rank trajectory training
full asset cross-section present on both ranks
origin/trajectory effective batch partitioned across ranks
one gradient all-reduce per effective batch
activation checkpointing enabled
raw-stock chunking enabled
```

The two devices provide 160 GB of aggregate device capacity, not one 160-GB
address space. The model and complete cross-section are replicated on each
rank; trajectory/origin batches, not the stock axis, are partitioned. Stock-axis
sharding is prohibited until cross-sectional attention, benchmark construction,
risk projection, gradient equivalence, and replay semantics receive a separate
distributed qualification.

One setting worker owns that setting's complete frozen 30-cell fold/seed
inventory. It emits an independent checkpoint and receipt for every cell and
does not advance past an incomplete cell without an exact-restart receipt.
Cells may be ordered for efficient immutable-data reuse, but they cannot be
mixed with another setting or silently skipped. The five completed seed
receipts are the prerequisite for that fold's ensemble execution.

Each Pod requests exactly two GPUs and both must be admitted before training
starts. A one-GPU fallback or independently scheduled one-GPU Pods cannot claim
the two-H100 setting receipt.

## Admission order under the 16-H100 ceiling

Twelve simultaneous workers would request 24 H100s. The approved experiment
ceiling admits at most eight two-GPU workers (16 H100s) concurrently. Four
settings remain capacity-pending and backfill one-for-one as slots become free.

Scientific setting index is not admission rank. In particular, scientific
index 8 belongs in Wave 1 while index 7 belongs in Wave 2. The immutable launch
manifest must therefore bind an explicit worker-index map:

| Admission rank | Scientific index | Setting ID | Wave |
|---:|---:|---|---:|
| 0 | 0 | `M03R-soft-persistence-active-alpha-hold30-v7` | 1 |
| 1 | 1 | `P00-no-soft-persistence-v7` | 1 |
| 2 | 2 | `P10-soft-persistence-10bp-v7` | 1 |
| 3 | 3 | `A08-fixed-exit-hazard-v7` | 1 |
| 4 | 4 | `A11-no-exact-hold-atom-v7` | 1 |
| 5 | 5 | `A09-no-long-context-v7` | 1 |
| 6 | 6 | `M02-active-risk-no-alpha-heads-v7` | 1 |
| 7 | 8 | `A12-fixed-2pct-active-risk-budget-v7` | 1 |
| 8 | 7 | `A04-no-downside-score-adjustment-v7` | 2 |
| 9 | 9 | `A10-no-factor-neutral-projection-v7` | 2 |
| 10 | 10 | `A06-sharpe-overlay-v7` | 2 |
| 11 | 11 | `A07-direct-sharpe-v7` | 2 |

Wave 1 establishes the core persistence, learned-exit, long-context,
residual-alpha, and adaptive-risk mechanisms. Wave 2 provides downside,
factor-attribution, and Sharpe diagnostics.

A launcher must explicitly guarantee this admission order. It must not rely on
an undocumented Kubernetes assumption that scientific numeric indices are
created or scheduled in the desired priority. Automatic backfill is allowed
only after the admitted-spec receipt proves that no more than eight workers
are running and every worker maps to the expected setting.

Pending because of the 16-H100 ceiling is normal capacity state, not failure.
A Pending worker must not be replaced, mutated, or assigned another setting
without a new content-bound plan.

## Required causal contrasts

### Persistence and exit behavior

```text
canonical - P00   effect of the 5-bp soft persistence cost
P10 - canonical  effect of stronger 10-bp encouragement
canonical - A08  value of learned state-dependent exits over fixed prior
canonical - A11  value of the exact-HOLD atom
```

These comparisons use actual age-cohort outcomes. Aggregate turnover or a
`1/30` turnover limit is never treated as evidence of a 30-session holding
period.

### Alpha extraction

```text
canonical - M02  incremental value of 5/21/30/63 residual-alpha heads
canonical - A04  incremental value of downside-aware stock scoring
```

### Risk and attribution

```text
canonical - A12  calibrated 0%-4% sizing versus fixed 2% active risk
canonical - A10  factor/sector-neutral stock selection versus unconstrained tilts
```

For A10, higher raw active return is not evidence of superior stock-selection
alpha unless active multifactor alpha also improves.

### Sharpe

```text
canonical - A06  separate total-risk overlay
canonical - A07  direct two-pass Sharpe objective
```

A06 must report both the unchanged alpha core and the final overlaid portfolio.
A07 is diagnostic and nonpromotable even if its point estimate is highest.

## Checkpoint and ensemble contract

Checkpoint eligibility and ranking use the order frozen in the RFC. Selection
occurs independently per seed using inner validation. No outer result may alter
the checkpoint, persistence coefficient, confidence calibrator, ensemble
weight, active-risk budget, cost rung, or causal inventory.

The five-seed ensemble is frozen before chronological evaluation. Absolute
portfolio weights are not averaged independently after execution. Aggregate
the declared output-space intents, then apply one authoritative constraint and
execution layer to produce one portfolio path.

Incomplete seed coverage fails closed. A four-seed substitute, a retrained seed,
or a seed-specific setting change requires a new plan and receipt.

## Required report

Every setting/fold and pooled setting report contains:

```text
10-, 20-, and 40-bp net portfolio and active return
20-bp and 40-bp net active return used by gates
active-return moving-block bootstrap lower confidence bound
information ratio
portfolio and C1 Sharpe
portfolio-minus-C1 Sharpe difference and lower confidence bound
active market beta and equivalence upper bound
portfolio, C1, and active multifactor regressions
active multifactor alpha and lower confidence bound
annualized tracking error
maximum portfolio and active drawdown
discretionary turnover by cause
forced turnover by cause
spread/impact or frozen cost charge
RMST60 and censoring-aware uncertainty
notional survival S(10), S(20), and S(30)
discretionary exit notional by age
forced exit notional by age and cause
HOLD/CONTINUOUS/EXIT action frequencies
continuous-hazard quantiles and saturation
requested-to-executed projection distance
performance during predeclared reversal episodes
startup, availability, risk-repair, and terminal accounting separately
```

Holding metrics are descriptive and late-ranking evidence. They cannot make an
otherwise ineligible policy eligible or displace a materially stronger active-
return checkpoint merely because its duration is closer to 30 sessions.

## Promotion and multiplicity

Only `M03R-soft-persistence-active-alpha-hold30-v7` can be promoted. All other
primary rows, the M01 short control, and any activated A05 reserve diagnostic
are registered trials in the scientific family even though they are
nonpromotable.

Promotion remains false unless the canonical setting satisfies the sealed
20/40-bp economics, tracking-error, active-beta equivalence, active
multifactor-alpha, Sharpe noninferiority, data-integrity, execution, and
multiplicity gates. Seeds are not counted as independent market paths.

An ablation win cannot be used to relabel, tune, or promote that row after
outer access. It creates a hypothesis for a new immutable generation.

## Prelaunch receipt checklist

The governed launcher must fail closed unless it receives all of the following:

```text
v7 protocol/design/setting registry hashes
source archive, clean Git tree, dependency lock, and image digest
PIT data/universe/C1/risk-free/factor/sector manifests
six-fold boundary and five-seed inventory
optimizer/update and precision plan
per-seed staged confidence-calibration plan
12-setting causal-route inventory
explicit admission-rank-to-setting map
two-H100 DDP and restart qualification
peak-memory and throughput capacity evidence
execution, cost, and constraint manifests
checkpoint/ensemble/evaluator receipt schemas
bootstrap, factor-alpha, and multiplicity contracts
outer-data access authorization and audit path
ownership-scoped cleanup and exact-UID recovery procedure
```

The currently implemented v6 primitives do not satisfy this v7 checklist by
themselves. No Kubernetes Job should be rendered or submitted from this
document alone.

## Development-only TOP2000 Kubernetes lifecycle

The future-selected TOP2000 compatibility experiment has a disjoint,
nonreportable identity. Its immutable package and lifecycle schemas live in:

```text
src/rl_quant/training/hold30_alpha_m03r_v7_package.py
src/rl_quant/training/hold30_alpha_m03r_v7_kubernetes.py
```

The package binds separate source-archive, source-manifest, dependency-lock,
cache-artifact, cache-manifest, data-manifest, execution-model, image-digest,
and package-plan hashes. No aggregate hash substitutes for any of them. Each
Indexed-Job completion maps by the frozen admission order to one development
setting and owns its exact six-fold by five-seed inventory.

The current cache binding is the existing pre-2026 TOP2000 artifact:

```text
host: /rsrch8/home/bcb/yding4/quant/training/caches/pre2026-bars-f08931bae1d0.pt
sha256: 0ba73414c3adea7712f7a68b1e76d934a17694a27671f35b8aa191bcc6aa1ee0
shape: [1001, 1999, 5] daily OHLCV, including CASH on the action axis
dates: 2022-01-03 through 2025-12-29
role: future-selected TOP2000 development-only, nonreportable
```

Every training replay has 378 state rows: 251 observation-only warmup
decisions, 63 loss-bearing origins, and enough detached support for the full
63-return auxiliary label. Each origin still owns exactly 30 post-fill
economic returns. Validation scores exactly 63 chronological transitions and
uses one five-seed output-space ensemble path per fold.

The Kubernetes module has no cluster client or hidden mutation path. It first
emits a suspended, all-twelve `batch/v1` qualification Job from the unqualified
immutable package. That Job runs the exact final worker command for four real
optimizer updates per setting, deliberately interrupts after the first
checkpoint under `torchrun --max-restarts=1`, and must resume the same
checkpoint before producing evidence. The verifier replays each terminal,
cell, and execution-plan receipt from bytes; caller-authored peak-memory or
parity claims cannot qualify a package.

Every one of the twelve settings must independently prove:

```text
world size and rank set                 2 and {0, 1}
physical accelerator                   NVIDIA H100 80GB HBM3
measured optimizer updates              4
intentional torchrun restart count       1
peak allocated memory per rank       60--75 GiB
peak reserved memory per rank        allocated--75 GiB
minimum physical headroom per rank       5 GiB
allocator OOM/retry counters             0 / 0
final model and optimizer rank parity    exact
```

Only after all twelve verified artifacts exist may the package emit the final
suspended Job. That final renderer requires both receipts below to match the
exact package execution surface:

1. executable worker qualification derived from all twelve artifacts,
   including two-rank parity and exact restart evidence;
2. an all-setting capacity receipt showing 120--150 GiB aggregate peak
   allocated HBM for every setting.

The manifest uses `completionMode: Indexed`, `completions: 12`, two H100s in
one container per completion, `backoffLimitPerIndex: 0`, and
`maxFailedIndexes: 0`. Parallelism is recomputed from a fresh live RBAC/cap
receipt and is always:

```text
min(8, floor((16 - protected_H100) / 2))
```

Both qualification and final manifests request two GPUs per completion. The
qualification batch uses the same twelve-entry admission map and at most eight
concurrent completions, so it tests every causal route while respecting the
same 16-H100 ceiling. Live free capacity is observational rather than an
entitlement gate. Capacity
that is not immediately available remains scheduler-Pending for automatic
backfill; zero instantaneous free GPUs is not misreported as a failed Job.

The API server must pass a fresh server-side dry run for the indexed-backoff
fields before create. Product scheduling is restricted to the approved
`nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3` selector. If node-list RBAC is
denied, qualification does not claim that the label was observed; the actual
runtime device name, physical memory, and compute capability provide the
artifact-backed product proof required by the final launch. Node names,
hostname selectors, host paths, private-key copying, and service-account token
automounting are prohibited.

The proven namespace topology uses the `yding4-gpu-home` PVC, `default`
service account, `kai-scheduler`, queue
`yding4-yn-gpu-workload-queue`, and PVC subpaths below
`quant/training/{packages,runs}`. The package subtree is mounted read-only and
the exact run subtree is the only writable research output mount.

Attachment is receipt-gated: read each newly admitted suspended Job twice,
verify stable UID and execution-bearing spec, verify zero UID-owned Pods, and
bind the admitted `template.spec` and selector hashes. Activation may only use
a full JSON Patch whose `test` operations bind the fresh UID, resourceVersion,
run-ID annotation, suspension state, parallelism, selector, template metadata,
and complete Pod spec before its sole mutation changes `suspend` to `false`.
Every successful final completion must later publish one receipt containing all
thirty fold/seed receipts; complete batch evidence requires exact indices
0--11.

Cleanup is also non-mutating in the package. Immediately before deletion it
must re-read the exact Job and render foreground `DeleteOptions` with that
fresh UID and `resourceVersion` as preconditions. A cleanup
receipt is complete only after two observations show that the exact Job and
all Pods owned by that UID are absent. Broad name-prefix or label-only deletion
is not an acceptable recovery method.
