# TOP2000 M03R-v7 seed-17 diagnostic

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
| Maximum concurrent H100 allocation | 16 |
| Fold execution | one chronological seed-17 path |
| Five-seed ensemble eligibility | false |
| Promotion eligibility | false |

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
2. Render a suspended two-H100 validation sentinel.
3. Bind its actual seed-validation and one-member fold-execution receipts.
4. Render the suspended all-setting qualification batch and validate all twelve
   numerical routes under the same two-rank shape.
5. Bind two independent API read-backs of the exact suspended Job and require
   zero UID-owned Pods.
6. Produce an exact UID/resourceVersion-tested activation request.
7. Render the final suspended twelve-index Job only from the qualified package.
8. Attach the detached lifecycle to that exact bound Job, then activate it.

At most eight workers may run concurrently. The remaining four setting indices
should remain scheduler-pending and backfill as capacity is released; a Pending
index is not a reason to create another Job.

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

