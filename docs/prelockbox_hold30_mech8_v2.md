# Experiment delta: Pre-lockbox Hold-30 mechanism-8 v2

**Status:** Superseded before launch; retained as an audit and implementation-history record

**Date:** 2026-08-04

**Protocol generation:** `prelockbox-hold30-mech8-v2`

**Base design:** `daily_raw_pit300_hold30`

**Mandate and design:** [Daily-decision Hold-30 RFC](daily_hold30_policy_rfc.md)

**Normative base experiment:** [Pre-lockbox Hold-30 H0–H3 v1](prelockbox_hold30_h0_h3_experiment.md)

**Supersedes for every future launch:** `prelockbox-hold30-h0-h3-v1`

**Superseded by:**
[`prelockbox-hold30-alpha-mech8-v3`](prelockbox_hold30_alpha_mech8_v3.md)

> V2 never received authority for a scientific launch. Its accounting,
> age-state, distributed, receipt, and qualification artifacts may be reused
> only after explicit porting to v3 identities and tests. A v2 generation ID,
> setting ID, checkpoint, or result row is invalid in every artifact-producing
> v3 path.

## Scope and precedence

This document is a narrow v2 delta. It incorporates the linked RFC and the
complete v1 experiment specification by reference; their chronology, economic
ledger, model, optimizer, point-in-time data, fold geometry, controls,
statistics, qualification thresholds, evidence boundary, and artifact
requirements remain normative unless this document explicitly overrides them.

For `prelockbox-hold30-mech8-v2`, precedence is:

1. this v2 delta for generation identity, setting inventory, trial count,
   compute allocation, and qualification lifecycle;
2. the Hold-30 RFC for implementation semantics; and
3. the v1 experiment specification for every unchanged scientific and
   evaluation field.

The v1 document remains an auditable design record but MUST NOT authorize a new
Job, manifest, checkpoint family, or result after this delta exists. A rendered
or executed artifact that uses the v1 generation ID is invalid for v2. The
consumed 2026 evidence remains unavailable for debugging, selection, or
evaluation exactly as specified in v1.

The v2 base design is exactly `daily_raw_pit300_hold30`. A TOP2000 or other
future-selected-universe design is not an alias and MUST fail manifest
validation.

## Frozen eight-setting inventory

Stable IDs are artifact identity. Aliases, renumbering, silent substitutions,
or post-launch setting additions are prohibited.

| Index | Stable setting ID | Frozen role/change | Promotion eligible |
|---:|---|---|---|
| 0 | `hold30-m00-legacy-gate` | H0 ported scalar-gate mechanism control under the common corrected ledger | No |
| 1 | `hold30-m01-slow-gate` | H1 slow-gate control with actual discretionary-turnover regularization | No |
| 2 | `hold30-m02-age-hazard` | Canonical H2 age-aware hazard, entry-score, and risky-exposure policy | **Yes** |
| 3 | `hold30-m03-sleeve30` | H3 structural staggered 30-sleeve duration control | No |
| 4 | `hold30-a04-no-age-input` | H2 with only actor position-age inputs disabled | No |
| 5 | `hold30-a05-no-early-penalty` | H2 with only the discretionary early-exit penalty disabled | No |
| 6 | `hold30-a06-no-turn-penalty` | H2 with only excess-discretionary-turnover regularization disabled | No |
| 7 | `hold30-a07-no-exp-timing` | H2 with only the risky-exposure timing residual fixed to zero | No |

Rows 4–7 are exact one-factor ablations of
`hold30-m02-age-hazard`. Every other H2 field MUST remain byte-for-byte or
canonically equal to the canonical row in the frozen manifest. Rows 0, 1, and
3 are mechanism controls, not one-factor H2 ablations. H2 remains the sole
promotion-eligible setting; an ablation or control cannot be selected merely
because its observed score is higher.

## Trials and compute

Freeze:

| Field | Value |
|---|---:|
| Settings | 8 |
| Outer folds | 6 (`0..5`) |
| Paired seeds | 5 (`17, 29, 43, 71, 101`) |
| Training trials | `8 * 6 * 5 = 240` |
| Accelerator product | `NVIDIA-H100-80GB-HBM3` |
| H100s per active setting worker | 2 |
| Concurrent setting workers | At most 8 |
| Namespace-wide protocol cap | At most 16 H100s |

The two-device allocation is a systems field, not two seeds or two independent
market samples. The manifest MUST bind the exact mechanically equivalent
sharding or distributed strategy and prove that it does not change scientific
fields. A fold/seed trial is counted once regardless of ranks, gradient
accumulation, restarts, or GPU count.

The expected training inventory is exactly 240 rows keyed by
`(protocol_generation, setting_id, fold_index, seed)`. Controls, null branches,
and evaluator simulations required by v1 are separately inventoried and MUST
NOT be mislabeled as additional training trials. Admission may be staggered by
capacity, but no setting may receive a different scientific configuration or
selection opportunity because it starts earlier.

## Frozen compact model correction

V2 MUST NOT inherit the `daily_raw_top2000` d512/8-layer Stage-1 encoder. The
common representation is the raw-OHLCV-only compact graph incorporated from
v1: a width-128, two-layer, four-head context encoder with width-256
feed-forward sublayers, plus the width-64/two-layer raw actor encoder and
width-128/two-layer policy context. News and stock covariates are disabled;
the covariate axis has width zero rather than a fitted or synthetic feature.

The package-owned constructor and manifest record non-overlapping context,
actor-path, and total-unique parameter counts for every stable setting. In the
current graph, actor-path counts range from 660,994 to 679,429 and total unique
trainable counts range from 927,490 to 945,925. Any actor-path count above five
million, total count above seven million, or change in these exact per-setting
counts invalidates the software receipt and blocks launch.

## Inference delta for the added ablations

The four v1 mechanism contrasts remain planned. Add these paired diagnostic
contrasts, always written as canonical H2 minus its ablation:

```text
hold30-m02-age-hazard - hold30-a04-no-age-input
hold30-m02-age-hazard - hold30-a05-no-early-penalty
hold30-m02-age-hazard - hold30-a06-no-turn-penalty
hold30-m02-age-hazard - hold30-a07-no-exp-timing
```

Apply the v1 joint within-fold bootstrap and Holm procedure to the resulting
eight planned contrasts. These four additional contrasts diagnose components;
their signs are reported but are not independent promotion gates.

For v2, the White Reality Check and Hansen SPA family is exactly the eight
stable setting traces plus v1 controls C2–C5. The simultaneous learned-policy
max-T family is exactly H1, canonical H2, and rows 4–7. H0 remains the ported
mechanism control and H3 remains structural, so neither enters that max-T
family. All v1 block lengths, replicate counts, null centering,
studentization, pairing, and thresholds remain unchanged. This expansion is
mandatory even though only canonical H2 can qualify.

### Frozen joint-inference mechanics

The package-owned inference plan completes byte-level choices that v1 required
to be frozen but did not enumerate. It is part of the executable-manifest
identity and MUST be retained before outer access:

- use exactly 10,000 noncircular moving-block replicates independently inside
  each of the six 63-decision outer folds at block lengths 5, 10, and 30;
- draw a block start from `0..fold_length-block_length` inclusive, concatenate
  blocks, and truncate only the final block; blocks never wrap or cross folds;
- reuse the same sampled fold/time indices jointly for every setting, control,
  and planned contrast, and pool decisions with equal per-decision weight;
- derive every block start by SHA-256 counter modulo from an explicit
  manifest-bound 32-byte seed plus unsigned big-endian block length,
  replicate, fold, and block counters under the versioned RNG domain recorded
  in `inference-plan.json`; there is no implicit or wall-clock seed;
- use an uncentered nearest-rank 5th percentile as the one-sided 95% H2 lower
  bound and uncentered nearest-rank 2.5th/97.5th percentiles for planned
  contrast intervals;
- use add-one p-values and count ties as exceedances;
- White uses the empirical-mean-centered, unstudentized maximum; SPA uses
  bootstrap-standard-error studentization and Hansen consistent recentering;
  max-T and planned-contrast tests use empirical-mean-centered,
  bootstrap-standard-error-studentized null draws; and
- Holm tests the eight frozen left-minus-right contrasts against the one-sided
  alternative `> 0` at 0.05, ordering equal raw p-values by lexical contrast
  ID.

The exact machine-readable plan is emitted by
`rl_quant.evaluation.hold30_inference.Hold30InferencePlan`. Any different
replicate count, block convention, family, centering rule, studentization,
quantile convention, seed encoding, or tie behavior is a new protocol rather
than a compatible rerun.

### Frozen Stage-1 ownership and two-rank sharding

V2 trains exactly one compact Stage-1 context encoder per outer fold using only
that fold's permitted expanding-training rows. All eight settings and all five
policy seeds in the fold consume the same frozen encoder receipt; Stage-1 is
not retrained per setting or policy seed. The exact normalization-date list and
all 1,000 update date schedules are explicit manifest artifacts hashed before
training and cannot contain inner-validation, outer, support, or embargo rows.

The Stage-1 seed is the unsigned integer represented by the first eight bytes
of:

```text
SHA256(UTF8(protocol_generation) || 0x00 || fold_index:u16be ||
       0x00 || UTF8("stage1"))
```

Retain the initialized checkpoint, every 50th optimizer update, and update
1,000. There is no Stage-1 validation selection or early stop.

The declared two-H100 worker preserves the frozen global SSL microbatch of
three dates and 12-microbatch accumulation without duplicating a date. For
zero-based microbatch `m`, local date counts are rank0/rank1=`2/1` when `m` is
even and `1/2` when `m` is odd. Thus each rank processes 18 of the 36 global
dates per optimizer update. Each local loss contributes sums and a valid-target
denominator; gradients are SUM-reduced and divided once by the global
denominator before the single AdamW step. Sample-level masking/augmentation
RNG is counter-keyed by fold, date, and update so changing rank partition does
not change a sample. CPU/Gloo one-rank-versus-two-rank update and exact-resume
parity are blocking software tests; H100 numeric parity remains a separate
capacity-stage receipt.

## Qualification lifecycle

Stages are ordered and fail closed. A later stage cannot waive or retroactively
repair an earlier stage.

| Stage | Required evidence before advancing | Current status |
|---|---|---|
| 1. Component | Accounting conservation; age, forced-exit, no-liquidation, action-builder, sleeve, loss, fold, manifest, and resume tests | In progress |
| 2. Software | Package-owned end-to-end runner; deterministic small-fixture trace; streaming/in-memory and restart parity; complete artifact schemas; pinned source/runtime | In progress |
| 3. Data | Retained point-in-time pre-2026 active-300 snapshot; `N >= 1811`; legal decision/fill masks; corporate actions; benchmark; six materialized folds; all required hashes | Not qualified |
| 4. Capacity | Approved controller-to-seadragon preflight; namespace/RBAC and quota evidence; bounded 2-H100 pilot; eight-worker/16-H100 admission proof; detached recovery and cleanup receipts | Not qualified |
| 5. Scientific | One frozen 240-trial execution, receipt-complete unseal, v1 controls/statistics, and predeclared holding/economic gates | Blocked |

Component tests alone do not constitute software qualification. Software or
capacity qualification MUST use synthetic or systems-only inputs and MUST NOT
inspect outer scientific outcomes. Capacity availability cannot relax data,
source, model, fold, or statistical requirements.

Scientific qualification begins only from a single immutable executable
manifest after stages 1–4 pass. It ends only after the v1 promotion gates and
family-level inference are complete. Job completion, GPU utilization, or a
nominal best return is not scientific qualification.

## Receipt-gated launch and unseal

No training Job may be created until one prelaunch root receipt binds all of:

- this rendered v2 specification and the incorporated RFC/v1 hashes;
- repository commit/tree, clean-worktree state or dirty-patch digest, retained
  source archive, dependency lock, and container image digest;
- the exact eight-setting table and 240-row trial-inventory digest;
- component, software, data, and capacity qualification receipts;
- point-in-time data, universe, benchmark, decision-axis, split-array, and
  corporate-action digests;
- the exact 2-H100 worker template, eight-worker concurrency ceiling,
  16-H100 cap, namespace, service account, and admitted Job template; and
- expected artifact paths, schemas, ownership metadata, recovery policy, and
  receipt graph.

A dry-run renderer MAY produce non-executable manifests while implementation
is in progress. Rendering, unit-test success, an available GPU count, or a
partially populated receipt MUST NOT grant launch authority. The executable
approval transition must be explicit, content-addressed, and independently
auditable.

After launch, no scientific result may be unsealed or used for checkpoint,
setting, or threshold selection until every expected trial has either a valid
terminal success receipt or a predeclared fail-closed disposition. A missing,
duplicate, replaced, or lineage-incomplete row blocks the family. Recovery
must resume or rerun the same immutable trial identity; it cannot create a new
setting or selective retry opportunity.

## Current disposition

Implementation artifacts and component tests are being assembled. This
document does **not** assert that any qualification stage has passed, that the
point-in-time dataset exists, that 16 H100s are currently admissible, or that
an executable receipt has been issued.

Therefore `prelockbox-hold30-mech8-v2` is **superseded before launch**. No new
or resumed v2 Job, v2 checkpoint family, v2 scientific evaluation, or 2026
reuse is authorized. Historical implementation artifacts remain audit
evidence only; the current experiment is the disjoint v3 generation.
