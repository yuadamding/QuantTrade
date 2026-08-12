# QuantTrade training knowledge base

Status: durable, non-normative orientation for future research work.

QuantTrade is a non-PHI research system. It is not a live-trading service,
investment product, or business-production system. Older artifacts may use
the word `production` for an operationally hardened research lifecycle; that
word does not change the scientific or business status of the work.

This guide consolidates the most reusable lessons from the Hold-30 M03R-v7
revision, the TOP2000 seed-17 diagnostic, the Phase-0 forensic audit, the
M03R-v8 alpha-discovery work, receipt-gated GPU execution, and the 2026-YTD
retrospective design. It explains how the pieces fit together. It does not
replace a protocol, a package-owned validator, or a run receipt.

## 1. Evidence and authority hierarchy

Use the narrowest authoritative source for each question:

1. A protocol or experiment specification defines the scientific question,
   causal settings, seeds, folds, metrics, gates, and reportability.
2. A package plan and source manifest identify the exact implementation and
   immutable data/cache inputs.
3. A rendered manifest identifies the requested compute and container
   surface.
4. A live-admission binding identifies the admitted Job UID and actual
   suspended Pod template before execution.
5. Activation, startup, terminal, and cleanup receipts identify what the
   cluster actually did.
6. Evaluator receipts identify which checkpoints, dates, traces, costs, and
   statistical rules produced a performance result.
7. Repository documentation explains the design but proves none of the
   execution facts above.

Never infer a remote run's source from the current branch name, local working
tree, or latest commit. Compare the run's source archive and manifest to the
claimed commit or checkout file by file. A later documentation-only commit
does not alter an already sealed training package, but it also must not be
described as that package's runtime source without an exact inventory match.

## 2. End-to-end lifecycle

| Boundary | Required result | What it does not prove |
| --- | --- | --- |
| Scientific freeze | Immutable protocol, design, settings, folds, seeds, and gates | That code implements them |
| Local qualification | Focused semantic, numerical, route, package, and lifecycle tests | That the cluster admits the same bytes |
| Immutable package | Source, data/cache, runtime image, plan, and file inventory hashes | That a Job ran |
| Static gate | Same-image imports, plan loading, mounts, indices, and zero-GPU assertions | GPU capacity or model fit |
| Qualification | Short exact-shape execution, rank parity, validation, and startup proof | Convergence, restart, or scientific success |
| Capacity gate | Exact rank/GPU shape can start under the approved ceiling | That every queued worker is already running |
| Suspended binding | Exact admitted UID, resource versions, Pod template, and zero-Pod state | Successful activation |
| Activation/startup | One activation attempt and per-Pod device/model proof | Training completion or alpha |
| Terminal/cleanup | Bounded evidence followed by UID/resourceVersion-conditioned deletion and absence proof | Scientific value |
| Evaluation | Frozen checkpoint and trace results under an exact evaluator contract | Reportability when the dataset is contaminated |

The safe order is therefore:

```text
freeze -> qualify -> package -> dry-run -> create suspended once
       -> bind admitted identity -> activate once -> detach and supervise
       -> capture terminal evidence -> exact cleanup -> evaluate
```

Do not skip a boundary because a later boundary happened to succeed. Do not
repeat a completed boundary when its full immutable key still matches.

## 3. Scientific lessons from M03R-v7

### Holding duration was not the limiting problem

The completed twelve-setting seed-17 benchmark showed discretionary sale ages
near 40–50 sessions for most useful variants. Changing the extra age-zero
persistence coefficient among 0, 5, and 10 basis points barely changed the
result. The structural hazard prior, exact-HOLD option, transaction costs, and
carried portfolio state already produced persistent holdings.

Thirty sessions should remain:

- a soft, one-sided inductive bias;
- a telemetry reference;
- never a minimum-hold mask, forced expiry, turnover proxy, or promotion gate.

Future work should not spend a large setting budget encouraging still longer
holding unless a new diagnostic shows that persistence has again become the
bottleneck.

### Predictive alpha and signal-to-turnover efficiency were the bottlenecks

All twelve M03R-v7 settings had negative mean active return across six folds.
The canonical setting had roughly -0.56% annualized active return, mean
information ratio near -1.01, 0.75% tracking error, 1.63% daily one-way
turnover, and a sale age near 50 sessions. Exact trace replay later measured
about +0.04% gross active return at zero cost but about -0.61% at 20 basis
points, with a break-even one-way cost near 1.2 basis points.

The durable conclusion is:

```text
learn predictive cross-sectional alpha
-> preserve it through portfolio construction
-> trade only when it clears cost and uncertainty
-> establish net and factor-adjusted alpha
```

Optimizing portfolio Sharpe before establishing positive active alpha can
hide the core failure. The first scientific gate should be predictive and
gross-active alpha; cost survival comes next.

### Exact-HOLD and residual heads were useful, but partly as churn controls

Removing exact-HOLD or removing the residual alpha heads increased turnover,
shortened sale age, increased tracking error, and materially worsened active
return. Retain both mechanisms. However, gross-versus-net decomposition is
necessary before calling their benefit predictive: a component that primarily
suppresses trading can improve net performance without improving raw signal.

Useful follow-ups compare:

- exact-HOLD with different initialization or temperature;
- a continuous no-trade gate;
- turnover-matched policies with and without exact-HOLD;
- residual heads trained but detached from execution versus trained and used.

### A named ablation is invalid if the economic policy is unchanged

The Phase-0 trace audit found multiple nominally causal settings with identical
requested and executed economic books. Rounded fold metrics were not enough to
detect this. Every future causal comparison should bind, at minimum:

```text
selected checkpoint and model-state hash
requested-weight trace hash
post-hazard trace hash
post-projection trace hash
executed-weight trace hash
gross-return trace hash
net-return and active-return trace hashes
```

Pairwise maximum weight differences, exact-equality fractions, and return
correlations are blocking integrity checks. If two settings produce identical
executed-weight hashes without a mathematical reason, do not interpret their
performance as an ablation.

### The unconstrained-risk setting was not a clean factor ablation

The setting that removed factor/sector neutral projection combined nearly zero
turnover with roughly 15% tracking error and deeply negative active return.
That behavior is consistent with a largely static portfolio far from the C1
benchmark, a startup-accounting issue, or removal of more than the intended
constraint. It does not justify removing the whole risk stack.

Future tests should retain benchmark anchoring, tracking-error and active-beta
controls, long-only and asset caps, and availability repair while relaxing
factor/sector bounds gradually, for example 1.0x versus 1.5x.

### Projection attribution is essential

The canonical policy used little of the nominal tracking-error ceiling. In
several folds, only a small fraction of the requested active book survived the
projection pipeline. Record requested active weights and each subsequent
stage:

```text
requested -> exact-HOLD/hazard -> confidence sizing
          -> factor/sector projection -> beta repair
          -> turnover limiting -> executed
```

Measure ex-ante tracking error, projection distance, binding constraints, and
requested-to-executed signal retention at every stage. Do not add a mandatory
tracking-error floor. First determine whether the requested signal predicts
returns and where it is lost.

The complete benchmark and audit are in the
[twelve-setting performance benchmark](top2000_m03r_v7_seed17_12_setting_performance_benchmark.md)
and [Phase-0 forensic audit](top2000_m03r_v7_seed17_phase0_forensic_audit.md).

## 4. M03R-v8 alpha-discovery direction

M03R-v8 responds to the v7 failure by changing the order of learning:

1. pretrain the raw encoder and multi-horizon residual-alpha heads;
2. require predictive evidence before economic policy training;
3. express the policy as an incremental active proposal around C1;
4. use a cost- and uncertainty-aware no-trade region;
5. retain exact-HOLD, long context, downside scoring, confidence sizing, and
   bounded factor/risk controls;
6. use gross, net, and factor-adjusted active alpha as the progression gates.

The frozen pretraining horizons are 5, 21, 30, and 63 sessions with intended
weights 0.10, 0.35, 0.40, and 0.15. The 21- and 30-session horizons carry 75%
of the objective because they are closest to the cross-day policy horizon.

The primary predictive gate remains:

- mean validation rank IC at least 0.02 for the 21- or 30-session horizon;
- positive rank IC in at least four of six chronological folds.

Do not lower this gate to make a failed run pass. Preserve failed evidence,
diagnose the causal defect, change source under a new immutable identity, and
rerun only the invalidated surface.

### Objective-integrity lesson

An early v8 attempt imported the correct protocol weights but the executable
loss still averaged horizons equally. The shorter horizons therefore received
more influence than intended. The corrected rule is:

1. compute each horizon loss with date-balanced aggregation;
2. apply the frozen horizon weights in executable code;
3. renormalize only across horizons genuinely unavailable for that example;
4. test with intentionally asymmetric per-horizon losses so equal weighting
   cannot accidentally pass.

This generalizes beyond M03R: importing a protocol field is not proof that the
runtime objective uses it. A contract must be checked at the last executable
boundary before reduction and backpropagation.

### Qualification is not fit evidence

A four-update seed-17 qualification proves startup, exact rank shape, imports,
validation, deterministic wiring, and bounded resource use. It does not prove:

- restart or checkpoint continuation unless restart is explicitly exercised;
- convergence or adequate optimizer updates;
- predictive-gate success;
- active-alpha success.

The authoritative v8 design remains the
[M03R-v8 alpha-discovery document](top2000_m03r_v8_alpha_discovery.md).
Remote run state must be established from receipts, not from this guide.

## 5. Numerical and accounting safety

### Preserve exact forward accounting and stabilize only the backward path

Hold-30 cohort weights can remain economically positive while falling below
normal FP32 ranges after repeated partial sales. Replacing an exact denominator
with `max(denominator, eps)` in the forward calculation changes accounting and
can break conservation.

The accepted pattern is an exact current quotient in the forward pass with a
mathematically zero straight-through correction whose derivative follows the
epsilon-bounded quotient. This preserves exact forward reconciliation while
keeping gradients finite. Apply the rule consistently to every cohort-ratio
site, including proposed release, pro-rata sale, residual sale, and retention
removal.

Regression tests should drive a positive weight into the FP32 subnormal range,
exercise all sale paths, reconcile after every transition, and backpropagate a
finite gradient.

### Reject non-finite gradients before optimizer mutation

Distributed gradients must be reduced first, then checked and clipped with a
non-finite error before either optimizer step. A regression should prove that
the exception prevents both parameter and AdamW state mutation.

A checkpoint written after a non-finite update is poisoned. It cannot be made
safe by fixing the source and resuming. Start from the last checkpoint proven
to predate the bad update, or use a fresh source-homogeneous run.

## 6. Immutable source, package, and version-control discipline

For any training claim, bind:

```text
commit and tree identity
source archive hash and tracked-file manifest
package-plan file and semantic hashes
data/cache manifest and chronology hashes
container image digest and runtime manifest
execution and setting identities
```

Before publishing a branch as the source of an active run, compare the runtime
source inventory in the package against the committed checkout. Do not rely on
`git status` alone: a clean tree can still be the wrong commit, and a dirty tree
can include unrelated documentation that is absent from the sealed package.

If a source change can move scientific results, it requires a fresh package
and run identity. Orchestration-only recovery may reuse the scientific package
only when the exact package bytes remain valid and all new lifecycle identities
and evidence paths are disjoint.

Never modify packaged runtime source in place while a Job is running. Fixes
for the next generation belong in a new immutable source archive. Preserve the
old terminal and cleanup evidence even when the failure is operational.

## 7. Efficient receipt-gated Kubernetes execution

The operational objective is not maximal activity; it is the minimum work
that establishes each new boundary exactly once.

### Session and preflight

- Verify the approved remote chain once per session.
- Use one reviewed Kubernetes context, namespace, kubeconfig, and client.
- Run mutation/RBAC preflight once per session and bind its receipt.
- Reuse an immutable receipt only when every key it covers still matches.
- Do not repeat broad cluster, directory, or node scans to rediscover known
  identities.

### Rendering and creation

- Render and server-dry-run each distinct manifest hash once.
- Create a Job exactly once and reconcile timeout or transport ambiguity for
  the full request window. Unknown is not absence; never issue a second create
  while acceptance is uncertain.
- Create suspended, then take two exact reads proving the same UID and admitted
  spec with zero owned Pods.
- Bind admission-injected fields explicitly. Rendered, dry-run, created, and
  actual Pod surfaces may have different approved allowlists; they need not be
  byte-identical.
- Activate once. A transport error followed by an observed unsuspended Job is
  attach-required ambiguity, not proof that this controller successfully
  activated it.

### Output and receipt identities

Runtime terminal receipts are not generally idempotent because elapsed time,
memory, and other observed fields can differ. A sentinel and a later
qualification must not write a terminal receipt to the same physical path.
Use phase-disjoint output roots, or consume the existing sentinel by exact
hash without rerunning it.

Keep plan and binding receipts idempotent only when their complete content is
identical. Never delete, move, or archive a published terminal receipt merely
to make its canonical path writable again.

### Detached work and status checks

Once a receipt-bound supervisor has taken ownership, release the agent turn
and let it run. Do not keep a manual sleep loop, scheduled poll, or duplicate
controller alive. Resume only on an explicit status request, a predeclared
terminal notification, or an anomaly returned by the current call.

A healthy status request should read the exact known top-level phase receipt
or take one compact snapshot of the exact Job. If a qualification Job is
expectedly absent after cleanup, validate the bound phase-success transition
and retarget at most one snapshot to the successor Job. Do not recreate the
qualifier or race the multi-phase supervisor.

### Capacity interpretation

Keep these numbers distinct:

- the user-authorized GPU ceiling;
- one Job's total requested ceiling;
- namespace quota;
- Running and Ready allocations;
- startup-proved working GPUs.

An Indexed Job can be correctly deployed while excess completions remain
quota-Pending. Let them backfill as existing workers finish. Creating another
Job duplicates work and can exceed the intended allocation.

Call GPUs working only after the exact Pod startup guard proves the device
count and model. Ready establishes admitted allocation, not necessarily a
completed CUDA/model startup check. Multi-GPU memory is per rank and is not a
single automatically pooled memory space.

### Failure and cleanup

Once an accepted UID exists, every fallible boundary must either:

- produce a cleanup-capable binding and exact-clean it; or
- publish an attach-required receipt with the observed identity and stop.

Exact cleanup uses the run ID, Job UID, and a fresh resourceVersion
precondition, captures terminal evidence first, issues one foreground delete,
and proves Job plus UID-owned Pod absence twice. A fresh resourceVersion may
equal a previously bound value for a never-activated suspended Job; two fresh
exact reads and zero-Pod proof make that equality legitimate.

Cleanup failure, absent Job with remaining owned Pods, identity drift, or a
live detached child is attach-required. Never claim absence or success from an
inconclusive read.

### Bound checkpoint retention under storage quotas

Treat filesystem capacity, per-user/project quota, and Kubernetes GPU quota as
different resources. `df` can show ample free space while a task-owned PVC
write still raises `EDQUOT`. Before launch, estimate the worst-case checkpoint
footprint across parallel workers, folds, save cadence, model state, optimizer
state, and RNG state. A frequent checkpoint interval multiplied by many folds
and workers can exceed a quota even when each checkpoint is small.

An operational continuation may reclaim a completed fold's intermediate
restart checkpoints only when the frozen worker contract proves all of the
following:

- the final fold model and canonical fold receipt exist and revalidate;
- the resume path skips that completed fold without reading its intermediate
  checkpoints;
- every checkpoint and manifest is an exact regular-file pair whose hashes,
  setting, fold, update, plan, and source identities match;
- a durable prune intent records every path, size, and hash before unlink;
- a success receipt proves the exact paths absent while the final model and
  fold receipt remain valid.

Never prune the current incomplete fold, an unpaired or drifting checkpoint,
or unrelated project data. Run retention under the same detached lifecycle
owner so each newly completed fold is reclaimed before later folds can refill
the quota. Make interrupted pruning idempotent by replaying the immutable
intent and deleting only still-present exact paths.

If the quota is so full that the normal recovery bundle or receipt cannot be
published, use a minimal bootstrap only: store the current-session preflight
and one exact completed-fold prune intent in the unique durable work-mac stage,
copy those small receipts to the approved project root, remove one fully
verified redundant pair, publish its success receipt, and then resume the
normal receipt-gated recovery. Do not use the bootstrap to delete arbitrary
old runs or bypass terminal capture and UID-bound Job cleanup.

Checkpoint retention is an operational fix only when it cannot change the
scientific model, optimizer trajectory, selected update, or metrics. A corrupt
or non-finite checkpoint is a source/scientific boundary and follows the fresh
source-homogeneous recovery rules instead.

## 8. Efficiency rules that preserve evidence

Efficient research execution means reusing proven immutable work and avoiding
scientifically empty steps:

- load immutable data and encoder state once per setting or worker when the
  protocol permits it;
- use focused semantic and mocked lifecycle tests before expensive GPU work;
- reuse one exact static gate per static-surface key and one capacity gate per
  GPU execution shape;
- use one startup proof per Pod UID;
- leave quota-Pending indexed completions queued for backfill;
- avoid redundant host dependency probes when the dependency is qualified in
  the exact runtime image;
- stage one safe archive through one unique transfer path, verify an
  intermediate hash once, then perform one complete safe-member audit at the
  destination;
- distinguish a source defect, which requires a new package, from an
  orchestration defect, which may require only new lifecycle identities;
- keep detached supervision running without occupying an interactive agent
  turn.

Optimization must not weaken a gate. Remove duplicate proof, not necessary
proof.

## 9. Evaluation and promotion boundaries

### Minimum frozen-checkpoint audit

Before further training, reuse existing frozen checkpoints to produce:

- full-precision action and return trace hashes;
- gross and net policy/C1 returns;
- 0, 10, 20, and 40 basis-point repricing;
- policy cost, C1 cost, incremental active cost, and break-even cost;
- 5/21/30/63-session rank IC and spread diagnostics;
- requested-to-executed projection attribution;
- setting-specific integrity audits such as the unconstrained-risk route.

This distinguishes representation failure, execution-cost failure, projection
signal destruction, regime concentration, and route/configuration failure
without retraining.

### Seeds, folds, and ensembles

Chronological folds measure performance across time regimes. Seeds measure
algorithmic instability on the same market history. They are not independent
market samples. Select each seed checkpoint using inner-validation evidence,
combine seed outputs into one deployed chronological path, and perform
inference on that path.

### 2026-YTD retrospective

The 2026-YTD TOP2000 surface is a retrospective mechanism diagnostic only. It
uses an inclusive 2026-01-02 through 2026-06-23 score window and must freeze
model, data, action, factor, cost, and statistical rules before opening outcome
data. Fold 5/seed 17/rank 0 is the headline checkpoint; folds 0–4 are separate
sensitivity analyses, never pooled as independent histories.

The static TOP2000 universe was selected using 2026 information. That creates
lookahead and survivorship contamination even though no 2026 row entered the
pre-2026 optimizer training. Results therefore remain development-only,
nonreportable, and nonpromotable.

The retrospective contract requires:

- one continuous no-reset chronology;
- leakage-safe separation of encoder context from the economic ledger;
- one authoritative 20-basis-point closed-loop path;
- frozen-trace 10- and 40-basis-point repricing;
- official factor retrieval evidence with exact-date coverage and no
  imputation;
- typed cohort trajectories for censoring-aware RMST uncertainty;
- exact 10,000-draw joint moving-block inference under the frozen family;
- one-GPU execution with exact checkpoint and source binding.

See the [2026-YTD retrospective guide](top2000_m03r_v7_seed17_2026_ytd_retrospective.md).

## 10. Interpreting outcomes correctly

| Observation | Correct interpretation | Next action |
| --- | --- | --- |
| Job completed and cleaned | Operational lifecycle succeeded | Validate scientific receipts |
| Qualification passed | Runtime shape and wiring work | Run the frozen scientific workload |
| Predictive gate failed | Model did not establish required IC | Audit objective and representation; do not lower gate |
| Gross alpha positive, net alpha negative | Signal is consumed by costs | Improve no-trade and turnover efficiency |
| Requested alpha positive, executed alpha negative | Construction removes signal | Audit confidence and projections |
| Same executed trace across causal settings | Ablation is inactive or mathematically redundant | Fix route/configuration before interpretation |
| One fold carries the result | Regime-specific evidence | Diagnose regime and require broader fold support |
| One seed differs | Optimization instability | Use additional seeds after the one-seed gate passes |
| Nonreportable cache performs well | Mechanism evidence only | Rebuild and confirm on point-in-time data |

Do not call a completed training Job a successful alpha result. Operational,
numerical, predictive, economic, statistical, and reportability success are
separate gates.

## 11. Verification levels and known debt

### V9 predictive-interface lessons

When the predictive distribution is a scientific input to execution, do not
train one uncertainty head and trade another. Bind the selected mean, selected
scale, selected horizon, both head-state hashes, and the distribution contract
in every checkpoint. Qualification, checkpoint selection, and execution must
name the same horizon.

Factor-residual labels and the executable risk projector must share one exact
point-in-time exposure inventory and asset order. Keep a previously
materialized artifact immutable when its only missing boundary is a later
projector contract: issue a separate content-bound binding that closes only
the known blockers, and preserve the original receipt. Do not suppress its
historical blocker codes or silently relabel the old artifact as ready.

For a 2,000-asset policy, qualify and transfer risk once per fold. A frozen
factor-plus-diagonal estimator avoids a dense covariance in the hot path;
inside the chronology, check only immutable identities and tensor version
counters. Construct the alpha proposal in the same exposure-null space used
for target residualization, then keep the final projector as a safety layer.
Record both signal-null retention and requested-to-executed book retention.

A tradeability gate should use a deterministic nonlearned sleeve before RL
fine-tuning. Start from C1, carry one chronological book, disable the learned
hazard, use the same selected mean/scale and risk projector, and publish gross
returns, policy/benchmark turnover, requested/projected books, and retention.
Rank IC by itself is not economic evidence.

Learned-exit proceeds and portfolio de-risking are different quantities. Only
`max(0, learned_exit_proceeds - explicit_derisk)` should fund same-step
replacements, and a released stock must be excluded from the same fill's buy
mask. Use the evaluator's exact one-way turnover convention in the optimizer.

Finally, a passing predictive receipt is not an economic-training protocol.
Keep the economic entrypoint fail-closed until the selected predictor,
horizon, calibration, action, and risk identities are frozen in a new
economic generation.

For a new multi-GPU execution shape, make the capacity receipt terminal and
cleanup-bound rather than startup-only. A valid two-H100 capacity
qualification binds the exact startup file, application terminal receipt,
captured terminal Job/Pod/log evidence, and final cleanup receipt. Publish it
only after one UID/resourceVersion-conditioned delete and two absence reads.
This prevents a short-lived CUDA startup from being reused after its Job or
Pods were left ambiguous.

Keep Kubernetes mutation ownership narrow and explicit. The v9 predictive
path uses a pure-file preparer, one create-capable operator, and attach-only
supervisors. The operator performs one server dry-run and exactly one
suspended create request. A successful create response without an exact UID,
a transport timeout, identity drift, or failure after acceptance is reconciled
without another create. Once a safe binding exists, a failed activation
handoff or receipt publication exact-cleans it; otherwise it publishes
attach-required evidence. Detached supervisors begin their hard wall at child
start, require a hash-bound process handshake before activation, accept only a
direct activation response as success, and issue at most one cleanup delete.

Keep verification claims precise:

1. A focused test suite validates only the changed contract boundary.
2. Package validation proves the sealed run inventory and package-owned
   invariants.
3. Runtime qualification proves the container and execution shape.
4. The complete repository suite detects integration and hygiene drift that a
   package-focused suite can miss.

Do not summarize focused green tests as “all tests pass.” At the time this
knowledge base was written, the active package had passed its focused
qualification, while the broader repository suite still exposed separate
integration debt, including later-generation source-inventory registration,
package-entrypoint thinness, a legacy import-boundary expectation, a float32
pipeline reconciliation edge, and a repository-hygiene literal. These are
repository qualification issues, not evidence that an already running sealed
package should be edited in place.

Resolve runtime-affecting debt at the next source-freeze boundary, rebuild a
fresh package, and rerun the invalidated qualification. Documentation-only
repairs may be committed separately because they do not alter sealed runtime
bytes.

## 12. Reusable checklists

### Before packaging

- [ ] Scientific protocol, setting inventory, folds, seeds, and gates frozen.
- [ ] Runtime objective demonstrably consumes every result-moving protocol
      field.
- [ ] Focused semantic, numerical, route, and lifecycle tests pass.
- [ ] Non-finite gradients fail before optimizer mutation.
- [ ] Source manifest, data/cache, image, and plan hashes recorded.
- [ ] Package receipts use relocation-safe identities; operator-local absolute
      paths are not treated as worker mount identities.
- [ ] Transfer archive has one top-level directory, deterministic metadata,
      no links/devices/traversal, and exact receipt-bound regular-file inventory.
- [ ] Package source inventory matches the intended commit exactly.
- [ ] Existing user changes are preserved and unrelated files excluded.

### Before GPU activation

- [ ] Approved session and mutation preflight receipt valid.
- [ ] Package publication and safe inventory valid.
- [ ] Output and lifecycle paths absent or explicitly reusable by exact hash.
- [ ] Server dry-run completed once for the exact manifest hash.
- [ ] Suspended Job accepted once and reconciled.
- [ ] Two exact UID/spec reads and zero-owned-Pod proof captured.
- [ ] Admission-injected fields conform to the layer-specific allowlist.
- [ ] Capacity and startup shape remain within the authorized ceiling.

### During healthy detached training

- [ ] One supervisor owns phase transitions.
- [ ] No duplicate Job or manual polling loop exists.
- [ ] Quota-Pending workers are allowed to backfill.
- [ ] Status requests use one exact receipt or compact Job snapshot.
- [ ] Working GPU counts come from startup-proved Pods, not requested totals.

### After terminal state

- [ ] Terminal Job, Pods, conditions, bounded logs, and application receipts
      captured promptly.
- [ ] Exact UID/resourceVersion cleanup completed.
- [ ] Job and UID-owned Pod absence proven twice.
- [ ] Scientific artifacts validated independently of Kubernetes success.
- [ ] Failures classified as source, data, model, admission, capacity,
      orchestration, or evaluator defects.
- [ ] Only the invalidated surface is rerun under fresh identities.

### Before interpreting performance

- [ ] Every setting produced a distinct policy trace or documented equivalence.
- [ ] Gross and 0/10/20/40-basis-point net results are available.
- [ ] Policy and C1 transaction costs are separated.
- [ ] Predictive IC and spread diagnostics are available by fold.
- [ ] Requested-to-executed signal loss is attributed.
- [ ] Seed and fold roles are not conflated.
- [ ] Universe and data provenance support the claimed reportability.

## 13. Canonical follow-on reading

- [Documentation index](README.md)
- [M03R-v7 RFC](prelockbox_hold30_active_alpha_m03r_v7.md)
- [M03R-v7 twelve-setting experiment](prelockbox_hold30_active_alpha_m03r_v7_experiment.md)
- [M03R-v7 revision and training guide](m03r_v7_revision_and_training_guide.md)
- [TOP2000 seed-17 diagnostic](top2000_m03r_v7_seed17_diagnostic.md)
- [Twelve-setting performance benchmark](top2000_m03r_v7_seed17_12_setting_performance_benchmark.md)
- [Phase-0 forensic audit](top2000_m03r_v7_seed17_phase0_forensic_audit.md)
- [M03R-v8 alpha discovery](top2000_m03r_v8_alpha_discovery.md)
- [M03R-v9 predictive stage](top2000_m03r_v9_predictive_stage.md)
- [2026-YTD retrospective](top2000_m03r_v7_seed17_2026_ytd_retrospective.md)
