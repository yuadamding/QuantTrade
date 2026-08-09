# TOP2000 M03R-v7 seed-17 diagnostic

**Protocol:** `top2000-dev-hold30-active-alpha-m03r-v7-seed17-v1`

**Design:** `daily_ohlcv_aggregated_top2000_dev_hold30_m03r_v7_seed17_v1`

**Status:** executable development-only diagnostic; nonreportable,
nonpromotable, and not five-seed-ensemble eligible

**Canonical scientific contract:**
[M03R v7 RFC](prelockbox_hold30_active_alpha_m03r_v7.md)

**Implementation lessons:**
[M03R v7 revision and training guide](m03r_v7_revision_and_training_guide.md)

The package-owned protocol and validators are authoritative if this narrative
drifts. This document is not a Kubernetes launch receipt and does not describe
live Job status.

## Purpose

This is a fresh, development-only diagnostic generation for the twelve M03R-v7
TOP2000 settings. Each setting trains the same six frozen development folds and
only seed 17. It answers whether the mechanism panel can train, validate, and
execute chronologically without paying the five-seed replication cost before
the numerical route has earned that expansion.

It does not reuse a prior model, optimizer state, checkpoint, validation trace,
or fold return path. It also cannot satisfy the existing five-seed package or
ensemble contracts.

## Frozen geometry

| Field | Value |
| --- | ---: |
| Settings | 12 |
| Folds per setting | 6 |
| Seeds per fold | 1 (`17`) |
| Fold/seed cells | 72 |
| Alpha-core optimizer updates per cell | 64 |
| Panel alpha-core updates | 4,608 |
| GPUs per worker | 2 H100 |
| Maximum concurrent workers | 8 |
| Maximum concurrent H100 request ceiling | 16 |
| Fold execution | one chronological seed-17 path |
| Five-seed ensemble eligibility | false |
| Promotion eligibility | false |

The 16-H100 value is a maximum request ceiling, not proof of namespace quota,
admission, Running+Ready allocation, hardware startup, or measured VRAM.

The legacy five-seed panel owns 360 cells and 23,040 alpha-core updates. The
seed-17 diagnostic therefore removes four fifths of the replication work while
leaving the per-cell runtime, data, setting route, two-rank topology, and
validation chronology unchanged. A06 still owns its separately bound overlay
optimizer.

## Immutable identities

The diagnostic has separate protocol, design, setting, package-plan, training,
validation, fold-execution, completion, Kubernetes, and receipt schemas. A
seed-17 setting ID maps to the corresponding legacy numerical setting ID only
inside the governed worker. The result remains labelled with the seed-17 ID.

The package critical-source inventory includes the seed-17 protocol, package,
worker, Kubernetes renderer/operator, and Seadragon attach lifecycle. The
package hash therefore changes if any launch-bearing implementation changes.

## Pre-GPU benchmark gate

Package build and independent package validation both reconstruct, on CPU, the
authoritative validation and factor-calibration slices for all six folds. Every
slice uses the v2 C1 benchmark adapter and an explicit 1% maximum stock weight.
The adapter checks every fill-time row for causal availability, per-name caps,
and gross feasibility.

The package binds:

- each validation and calibration slice receipt;
- each benchmark weights and full trace hash;
- the v2 benchmark ID and risk-repair rule;
- the 1% cap;
- a combined benchmark trace/cap identity;
- the complete CPU preflight file hash.

An infeasible benchmark prevents package publication. No Kubernetes Pod should
be used to discover that failure again.

## Qualification and launch order

1. Build and independently validate a new clean-source seed-17 package.
2. Render the two-H100 validation sentinel; server-dry-run it, create it
   suspended, bind two exact zero-Pod read-backs, activate it once, capture its
   terminal evidence, and exact-clean it.
3. Build the sentinel artifact from its actual validation and one-member
   fold-execution receipts.
4. Render the all-setting qualification Job under a phase-disjoint output
   identity; server-dry-run it, create it suspended, bind it, activate it once,
   validate all twelve routes under the same two-rank shape, capture terminal
   evidence, and exact-clean it.
5. Build the twelve ordered qualification artifacts, capacity receipt, and
   execution-qualification receipt from the sentinel plus all-setting evidence.
6. Render the final twelve-index Job only from that exact qualified package and
   capacity/execution evidence.
7. Server-dry-run and create the final Job suspended, then bind two exact
   read-backs, stable UID/admitted spec, and zero UID-owned Pods before
   producing the UID/resourceVersion-tested activation request.
8. Attach the detached lifecycle to that exact final Job and activate it once.

At most eight workers may run concurrently. The remaining four completions
remain controller-queued when no Pod has been instantiated; an admitted but
unschedulable Pod may instead be Pending. Both states may backfill as capacity
is released and neither is a reason to create another Job.

## Completion evidence

Each setting must publish exactly six ordered coordinates:

```text
(fold 0, seed 17) ... (fold 5, seed 17)
```

Every coordinate binds the cell completion, seed validation, and one-member
fold execution. The final batch receipt requires twelve successful setting
receipts and 72 fold executions. No statistic may treat the six folds as a
five-seed output-space ensemble, and no result from this future-selected
TOP2000 universe is reportable or promotion eligible.

## Interpretation and reuse

- The four-update validation sentinel proves startup, route wiring, validation,
  execution, and the exact two-rank surface. The current seed-17 worker does
  not intentionally exercise checkpoint restart. It is not a full 64-update
  cell and cannot diagnose underfitting.
- A complete six-fold setting is a one-seed across-fold mechanism diagnostic.
  The full 72-cell panel supports paired development comparisons only; it does
  not establish seed robustness.
- One-member fold execution consumes the exact seed-validation trace. It must
  not run the model a second time merely to imitate an ensemble.
- Reuse the exact CPU benchmark/cap preflight and typed GPU qualification only
  when every bound key matches. A changed source, image, runtime, data, or
  execution-shape key invalidates only its affected gate.
- If a numerical defect may change accounting or gradients, preserve the old
  evidence and default to a fresh source-homogeneous panel. Never resume a
  checkpoint that may already contain non-finite parameters or optimizer
  state.

The seed-17 protocol, package, worker, Kubernetes renderer, lifecycle, and
their blocking tests are registered in the package-owned later-generation
source/test inventory. Do not maintain a second operator-authored file map.
