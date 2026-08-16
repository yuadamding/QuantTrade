# QuantTrade training knowledge base

Status: durable, non-normative orientation for future research work.

QuantTrade is a non-PHI research system. It is not a live-trading service,
investment product, or business-production system. Older artifacts may use
the word `production` for an operationally hardened research lifecycle; that
word does not change the scientific or business status of the work.

This guide consolidates the most reusable lessons from the Hold-30 M03R-v7
revision, the TOP2000 seed-17 diagnostic and Phase-0 audit, the v8-v16
predictive-first lineage, receipt-gated GPU execution, and the separately
frozen 2026-YTD retrospective design. It explains how the pieces fit together.
It does not replace a protocol, package-owned validator, data manifest, or run
receipt.

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

### Route future questions to the right evidence

| Question | First authority | Do not substitute |
| --- | --- | --- |
| What is the scientific objective or gate? | Immutable protocol and generation document | Latest branch, old benchmark, or operator notes |
| Which source and data did an attempt use? | Package plan, source manifest, and data/cache receipts | A clean local checkout or matching filename |
| Is a remote attempt running or complete? | Exact external lifecycle receipts and one owned-Job snapshot | Repository prose or a stale Job name |
| Is a date, symbol, or field admissible? | Frozen split/data contract plus row-usage evidence | Merely having downloaded the row |
| Did a model qualify? | Exact published checkpoint, evaluator inputs, and gate receipt | GPU startup, training loss, or rounded summary |
| Can another stage start? | Explicit successor authorization receipt | Inference from a prior stage's success |

Repository documents intentionally contain no live status. A future session
should identify the generation first, then follow only its exact receipts.

### Scientific-generation ledger

This table records lineage, not current cluster state:

| Generation | Durable outcome or role | What follows |
| --- | --- | --- |
| Canonical PIT Active-300 v7 | Normative objective, five-seed design, inference, and promotion contract; still distinct from executable TOP2000 evidence | Qualify its own PIT data and research-grade surfaces before any canonical study |
| TOP2000 v7 seed-17 | Completed one-seed mechanism diagnostic; holding duration was adequate, while predictive alpha, costs, and projection retention failed | Preserve as negative development evidence; do not call it promotion evidence |
| v8 alpha discovery | Introduced predictive pretraining and cost-aware incremental action; the first predictive attempt failed and an objective-weight defect invalidated reuse | Preserve the failed artifacts; use a fresh identity for result-moving corrections |
| v9 predictive | Bound one mean/scale distribution and horizon; all setting-horizon candidates failed the frozen gate | No economic training; use its negative evidence to narrow the next question |
| v10 rank geometry | Superseded before launch after review exposed paired-schedule, target-validity, sleeve-sizing, and inference defects | Never package or launch it |
| v11 corrected rank geometry | Completed TOP2000 predictive predecessor: paired samples, qualified residuals, magnitude-preserving actions, block inference, and exact checkpoint reload | Preserve its result and the separate a15 audit; neither authorizes economic training |
| v11 a15 inference audit | Exact-checkpoint controls confirmed a directional P0 effect, but every cap saturated and no 10-bp net or spread lower bound passed; P1 scale collapsed | No economic training; preserve the audit as exploratory evidence |
| v12 rank/scale decoupling | Completed 3-session predictive study: dedicated rank score, separate economic mean/scale, bounded rank influence on the encoder, and nonsaturating turnover utilization; all three settings failed the frozen predictive gate | Preserve as negative evidence; do not start economic training or open 2026 outcomes |
| v13 context-matched h3 | Corrected full-context and h3-only geometry, but still optimized and diagnosed a different tensor from the score traded after residualization | Preserve as superseded design evidence; do not launch it |
| v14 executable-score h3 | Completed two-setting study with one action-projected score; both settings failed IC, net-cost, and break-even gates | Preserve as negative evidence; do not start economic training |
| v15 corrected executable-score h3 | Completed all twelve A04 folds under the corrected package and runtime; both settings failed with IC below 0.011 and break-even cost below 2 bp | End daily h3 loss tuning; no economic or 2026 access |
| v16 holding-aligned selection | Current local v4 successor comparing h21, h30, and reference-Hold-30 value on common 30-session support; selection-only with causal action masks, self-financing signal/risk-repair cohorts, balanced blocks, validated slab caching, nontrivial capacity, and checkpoint-owned qualification | Complete and validate package, evaluator, worker, and lifecycle before any remote authorization |
| v7 2026-YTD retrospective | Separate, fixed 2026-01-02 through 2026-06-23 mechanism-evaluation contract | It is not a universal 2026 cutoff and cannot select v11 |

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

### Treat compute topology as generation-specific

An Indexed completion is a scientific worker, not automatically a rank. In
v11, each of three completions owns one two-rank process group and requests two
H100s, so the Job ceiling is six H100 requests—not one world-size-six model and
not the historical v7 sixteen-H100 planning ceiling. Each worker runs its six
folds under its own exact setting identity.

Keep the user-authorized aggregate cap, one Job's request ceiling, namespace
quota, Ready allocation, and startup-proved working devices separate. Never
copy a GPU number from another generation or infer working devices from
parallelism alone.

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

## 4. Predictive-first lineage from v8 through v16

M03R-v8 introduced the durable ordering that remains relevant:

1. pretrain the raw encoder and multi-horizon residual-alpha heads;
2. require predictive evidence before economic policy training;
3. express the policy as an incremental active proposal around C1;
4. use a cost- and uncertainty-aware no-trade region;
5. retain exact-HOLD, long context, downside scoring, confidence sizing, and
   bounded factor/risk controls;
6. use gross, net, and factor-adjusted active alpha as the progression gates.

The predictive designs use 5-, 21-, 30-, and 63-session heads. The eligible
execution candidates are separately bound 21- and 30-session horizons; a
model cannot qualify one horizon and trade another.

The broad-rank gate begins with:

- mean validation rank IC at least 0.02 for the 21- or 30-session horizon;
- positive rank IC in at least four of six chronological folds.

V11 additionally requires date-level IC breadth, positive spread, block-LCB
economics, aggregate break-even cost, noncollapsed dispersion, and projection
retention. Do not lower any frozen gate to make a failed run pass. Preserve
failed evidence, diagnose the causal defect, change source under a new
immutable identity, and rerun only the invalidated surface.

V9 established that factor-residual targets were preferable to subtracting a
common benchmark return, but neither broad IC nor a profitable deterministic
sleeve was established. V10 was never launched. Its review showed that the
settings must share one panel-level episode schedule, factor-ineligible assets
must not enter losses as artificial zero residuals, weak signals must not spend
the turnover cap, and fold-level confidence summaries cannot be averaged into
an aggregate confidence claim. V11 corrects those boundaries under a fresh
identity.

V12 showed that binding a selected horizon does not make it primary when most
shared-representation loss weight remains on other targets. Its post-hoc audit
also showed that action/return chronology, future-conditioned action support,
and raw-versus-executed score mismatches can materially change the measured
result. V13 corrected context and horizon geometry but still optimized and
diagnosed a different tensor from the residualized score traded by its sleeve.
V14 therefore makes the action-projected score the single loss, diagnostic,
and execution object and intersects label support with origin action
eligibility.

The completed v14 screen still failed: P0/P1 projected mean IC was only
`0.01184`/`0.01334`, 10-bp net active return was negative, and break-even cost
was `1.07`/`1.39 bp`. Review then found that v14's detached rank denominator
had zero loss but a nonzero radial gradient, its nominal no-rank ablation also
rescaled economic losses, its preflight was not bound to the package-owned
risk/projector/source, and its load-only capacity job did not exercise the
real workload. V15 corrects those boundaries and adds training-only
checkpoint selection; it does not reinterpret v14 as passing.

The corrected v15 A04 screen then completed both settings and all twelve
folds. P0/P1 projected IC was `0.00789`/`0.01013`, annualized 10-bp net active
return was `-0.0711%`/`-0.0573%`, and break-even cost was only `0.30`/`1.84`
basis points. Risk projection retention was one for both settings, so the
failure cannot be assigned to final portfolio projection. The paired Huber
control exceeded the rank setting, leaving no evidence that another h3 rank
or loss-weight experiment is the highest-value direction.

V16 therefore changes the information horizon rather than relabeling the h3
failure. It compares cumulative h21, cumulative h30, and an unnormalized
1–30-session value derived from the actual age-dependent Hold-30 release prior.
All three settings use one common 30-session label mask and residual operator,
and dimensionless heads prevent target units from changing gradient geometry.
H3 timing and uncertainty calibration are excluded from the target screen.
Five outer scored supports are disjoint, epoch count is fixed at eight, and the
Hold-30-prior setting is the sole predeclared primary hypothesis. This preserves
a clean information-alpha gate before any portfolio optimizer or RL controller.

### Train, diagnose, and trade one causal score

A predictive gate is invalid when its IC or spread is computed from a raw head
but execution trades a projected or otherwise transformed score. Define the
executable score once:

\[
s_t=P_t^A f_\theta(x_{\le t}),
\]

then use that exact tensor for rank/regression loss, validation,
qualification, and action construction. Record the raw score only for
attribution. Bind both hashes so raw-to-executable retention is observable
rather than hidden inside the final portfolio projector.

Label support must be causal at the origin. An asset belongs in the label
regression and loss only when it was action-eligible at the decision origin
and has complete declared outcome support. Future availability may remove a
label; it must never add an asset to the decision universe or influence the
origin action operator.

When a fixed exposure-null transform is applied repeatedly, precompute its
small QR-derived coefficient map once, retain differentiability with respect
to the score, and use immutable identity plus tensor-version checks in the hot
path. Repeating QR, full tensor hashing, or CPU validation per score adds cost
without changing the scientific object.

Materialize the operator and target slab from the exact cache, risk, projector,
and source bytes already owned by the package. A preflight receipt without the
actual reusable tensors only proves that QR succeeded once; it does not prevent
the training loop from rebuilding a different object. Load large slabs through
one no-follow descriptor, hash before deserialization, verify descriptor
identity again afterward, and revalidate every reconstructed operator.

A horizon-matched cohort diagnostic must close the last decision. If the new
cohort earns its first return on the decision/fill step, an h30 cohort needs 29
additional no-new-decision transitions—not 30—and must earn all 30 returns.
Charge terminal liquidation for any remaining active risk and publish absolute
policy cost separately from incremental active cost.

The label-valid mask and the action-valid mask are different scientific
objects. Complete future support may remove an observation from IC or loss, but
it cannot remove an origin-actionable asset from the policy's candidate set.
Actions use origin eligibility and fill-time availability only. Receipt both
masks and test that changing an asset's future availability cannot change the
current requested book.

A cohort diagnostic must carry executed economic positions, not yesterday's
requested target vectors. After every risk projection, reconcile the aggregate
executed active book back to cohort rows; after every return, drift those rows
and require their sum to equal the carried policy-minus-benchmark book. Cohort
age, entry, and release telemetry otherwise describe intent rather than actual
holdings and daily target reconstruction silently creates extra turnover.

Every optimizer update must represent comparable date mass. Splitting 129
origins as `63,63,3` and averaging each block gives the final three dates one
full Adam step and about 21 times the per-date influence of a full block.
Balance blocks so sizes differ by at most one, and test both exact origin
coverage and per-epoch weighting.

Validate immutable operator slabs deeply once at package/worker/fold
boundaries, then issue a private validated authority with O(1) origin lookup,
version checks, and cached device operators. Revalidating and rehashing every
operator on every row defeats precomputation and can make an H100 workload
CPU-bound without improving scientific safety.

### Parameterize holding targets without rewriting experiment horizons

New generic holding APIs default to a soft neutral expected duration of three
earned sessions through `HoldTargetSpec`. Existing Hold-30 names and immutable
protocols explicitly bind `legacy-hold30-v1` and target 30. Holding target,
prediction horizon, label support, age cap, purge, and cohort evaluation horizon
are separate values and must never inherit from one another implicitly.

The generic hazard location is deterministically calibrated against the actual
finite-state process, including the geometric tail created when the `60+` bin
repeats the age-60 hazard. A finite survival-prefix sum is not the runtime
expectation. Merely substituting 3 for 30 in the legacy logit does not create a
three-session expectation. Protocols, initial state, head identity, checkpoint,
replay, and evaluation must bind the resolved hold-target receipt; target
mismatch fails even when tensor shapes agree.

### Test objective gradients, not only objective values

Value invariance is not enough for a scale-invariant loss. If a prediction RMS
is detached, a perfectly ranked prediction can have zero reported loss while
retaining a nonzero radial gradient that keeps inflating the economic mean.
For normalized rank objectives, require all of the following above the frozen
dispersion floor:

```text
positive rescaling leaves the loss unchanged
perfect positive alignment has near-zero loss and gradient
the gradient has no radial component
negative alignment points toward sign correction
near-collapse inputs retain a finite anti-collapse gradient
```

For a loss-only ablation, preserve the absolute coefficients of every shared
component. Removing a rank term and renormalizing robust or scale weights is
not a pure rank ablation because it changes gradient magnitudes, clipping, and
optimizer moments.

Keep raw and executable prediction types distinct. A raw model output should
not expose aliases named `rank_score` or `execution_score`; only the
operator-bound executable type may expose those meanings. Validate the core
mask subset and recompute the projected score at the built-batch boundary so a
stale alternate builder cannot silently revive a raw-versus-traded mismatch.

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

The v7/v8 four-update seed-17 qualification sentinel proves startup, exact
rank shape, imports, validation, deterministic wiring, and bounded resource
use. V11 uses a different contract: a disjoint two-rank capacity Job proves the
exact device/runtime shape, while the untouched-tail predictive decision is
made only from update-64 checkpoints. Never copy a sentinel geometry or its
meaning from one generation into another. Qualification does not prove:

- restart or checkpoint continuation unless restart is explicitly exercised;
- convergence or adequate optimizer updates;
- predictive-gate success;
- active-alpha success.

Generation-specific details are in the [v8 design](top2000_m03r_v8_alpha_discovery.md),
[completed v9 record](top2000_m03r_v9_predictive_stage.md),
[superseded v10 proposal](top2000_m03r_v10_rank_geometry.md), and
[corrected v11 protocol](top2000_m03r_v11_rank_geometry_corrected.md),
[completed v12 rank/scale-decoupled study](top2000_m03r_v12_rank_scale_decoupled.md),
[v13 context-matched design](top2000_m03r_v13_context_matched_h3.md), and
[completed v14 executable-score study](top2000_m03r_v14_context_matched_h3.md),
[completed v15 corrected executable-score study](top2000_m03r_v15_executable_score_corrected_h3.md),
and [v16 holding-aligned selection design](top2000_m03r_v16_holding_aligned_selection.md).
Remote run state must be established from receipts, not from any of these
documents.

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

### Keep residual CASH out of risky-signal retention

CASH is the simplex residual account, not a requested risky alpha signal.
Recomputing it after projection can move its weight by several floating-point
ULPs even when the risky active book is unchanged. For a weak requested signal,
including that bookkeeping delta in the full-book norm can report
requested-to-executed retention above one and invalidate an otherwise sound
qualification trace.

Define book retention on risky active weights only. Reject any material
amplification, and canonicalize only a predeclared sub-tolerance numerical
overshoot to one. Keep an exact weak-signal regression whose old full-book
ratio exceeds one; this proves the guard covers CASH closure rather than
silently relaxing the scientific gate.

The same principle applies inside the 61-bin cohort ledger. FP32 reduction
residue must be reconciled to the already approved risky target while
preserving cohort proportions and age, before implicit CASH, turnover, or
drift is measured. Do not widen cohort or cause-turnover tolerances to absorb a
manufactured CASH balance.

### Use one canonical portfolio-growth scalar

Portfolio drift and age-cohort drift must normalize by the same scalar:

```text
portfolio_growth = 1 + sum(weight * asset_return)
```

Computing `sum(weight * (1 + asset_return))` is algebraically equivalent only
under exact arithmetic. Across thousands of FP32 assets it can differ by enough
ULPs to disagree with cohort accounting, manufacture residual CASH, and make a
healthy chronology fail reconciliation.

Simplex repair should perform its reduction and radial correction in a higher
precision work dtype when the economic output uses FP32. Casting back is a new
numerical boundary: recheck nonnegativity, risky-gross limits, and simplex
closure after the cast instead of assuming a float64 repair remains feasible in
float32. Use the repair before turnover measurement so numerical closure is not
misclassified as trading.

Every conservation comparison must reduce both sides in the same work dtype.
For a roughly 2,000-asset FP32 book, comparing a float64 requested-weight sum
with an FP32-reduced anchor can manufacture an error of several `1e-8` even
when the float64 buys and sells balance exactly. Do not widen the scientific
tolerance; cast the unchanged anchor to the work dtype before reducing it.

Keep diagnostic operands on the same device as the result-bearing tensor.
PyTorch operations such as `torch.quantile` require a tensor-valued probability
argument to share the input device; constructing that small constant without
an explicit device works on CPU tests but fails only after CUDA training reaches
qualification. Construct diagnostic constants with the input tensor's dtype
and device, and cover the boundary with a fake-CUDA or same-image regression
so a controller without a physical GPU can still detect CPU/GPU drift.

Immutable scientific metadata often remains CPU-resident while the result
tensor is on CUDA. A validator must not call a binary tensor operation on those
mixed devices. Reconcile a complete mask once through a device-independent
tensor identity, or move one stacked immutable slab explicitly; do not transfer
one mask per asset/date inside a Python loop. The exact-shape capacity update
must exercise this validation path because CPU-only tests cannot prove it.

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

### Data freeze, conversion, and split discipline

Downloaded data and admissible data are different objects. Keep provider raw
responses append-only and intact; materialize converted caches under new,
no-clobber identities. A training or evaluation package must bind:

```text
provider/source receipt and raw-member hashes
conversion source and schema identities
exact inclusive/exclusive date coverage
row counts, gaps, duplicates, and adjustment rules
symbol and asset-axis hashes
decision-time availability and point-in-time classifications
converted array/file hashes and row-usage hashes
```

Never put an API key in source, documentation, a manifest, a command argument,
or a receipt. A transferred archive is not trusted until its exact final bytes,
safe member inventory, and converted semantic manifest validate at the
destination.

For the current TOP2000 compatibility lineage, the frozen training cache has
1,001 daily states from 2022-01-03 through 2025-12-29 and rejects every
2026-or-later cache date. Its daily OHLCV states are aggregated from five-minute
bars; they are not a learned intraday-token sequence. Newer provider rows may
exist in a separately bound delta while contributing exactly zero rows to the
training cache. “Use all available data” never overrides a frozen chronology,
purge, label-support, untouched qualification tail, or test boundary.

Point-in-time sector classification means the provider classification whose
effective/availability record was knowable at the decision origin. Use an
explicit unknown category when no eligible record exists. Never future-fill a
later SIC or sector label backward. The sector/exposure asset axis, names, and
origin-time availability must match the residual operator and risk manifest.

Training context, optimizer origins, label support, validation, qualification,
and outer evaluation are different date roles. Hash their actual row usage;
calendar overlap or a downloaded source range alone cannot prove isolation.

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

Precreate every writable PVC `subPath` on the approved host before activating
its suspended Job. If the path is absent, kubelet may create it as a
root-owned private directory even though the container itself is configured
with a non-root UID/GID. Derive the host path from the exact rendered writable
mount; walk it without following symlinks; require task scope, worker UID/GID,
private searchable permissions, and an empty final phase root. Do not fix this
after activation with broad `chmod`/`chown`, and do not reuse the failed
static-gate evidence. Preserve and exact-clean the failed attempt, then use a
fresh source/package/Job/output identity.

Publish the exact directory the container mounts, not the transfer wrapper
that contains it. Before Job creation, resolve the rendered read-only PVC
`subPath` and require the package plan, `source/src` import root, and runtime
entrypoint at that exact host directory. A safe archive and correct inner
inventory do not prove that the mount points at the inner package level.

Keep host-side lifecycle modules dependency-light. Rendering, attach, cleanup,
and detached supervision run under the approved host Python, which need not
contain PyTorch, PyArrow, or the workload image's scientific stack. Put shared
receipt dataclasses in a small contract module and test that importing the
lifecycle surface does not import heavy scientific dependencies. The exact
runtime image, not the login host, qualifies those dependencies in the static
and capacity gates.

Keep plan and binding receipts idempotent only when their complete content is
identical. Never delete, move, or archive a published terminal receipt merely
to make its canonical path writable again.

Client output is not the state transition. A controller yield, truncated
stdout, or an exception in post-action diagnostic printing does not prove that
the preceding remote file-producing command failed. Before retrying, reconcile
the one exact expected no-clobber artifact and its hash. If it exists and fully
validates, consume it and continue; if it is absent, follow the command's
bounded recovery contract. Never repeat a build, create, or activation merely
because its final convenience output was lost.

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

Progress receipts are terminal publication boundaries, not hidden update
counters. During a long first fold, zero published fold receipts is compatible
with healthy Running/Ready workers. Later, a partial count means only that
those exact folds reached immutable publication. Do not interpolate optimizer
updates from wall time or read logs solely to manufacture a percentage.

Call a GPU hardware-verified only once the startup guard proves the exact
visible device count and model for the current Pod UID. A completion-scoped
startup receipt that does not bind Pod UID cannot prove a replacement Pod;
use one bounded UID-owned startup-log read for that new Pod and cache the proof.
Ready status alone establishes request occupancy, not the H100 contract.

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

Failure capture must precede success-only topology validation. An Indexed Job
can fail after the cluster has already removed one or more terminal Pods; in
that case, preserve the exact retained Job, owned Pods, and bounded logs, then
exact-clean. Require the full predeclared Pod cardinality only when validating
a successful scientific terminal.

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

This audit is possible only while the exact checkpoint, qualification, and
trace artifacts still exist. Terminal and cleanup receipts prove that a run
completed and was removed from Kubernetes; they do not preserve the run output
tree. Before deleting or reclaiming a completed generation, retain a
hash-bound audit inventory containing every selected/final checkpoint and the
prediction, score, probability, requested-book, projected-book, turnover, and
return traces needed by the declared post-hoc analyses. If the receipt-bound
output root is absent, report the audit as unavailable and never scan unrelated
storage or reconstruct evidence from rounded summaries.

### Diagnose rank/economics disagreement before another training generation

The completed v11 a15 panel exposed a reusable diagnostic pattern. One setting
can have broad positive gross economics while failing mean rank IC, tail-spread
uncertainty, and net-cost lower bounds; another can improve rank IC while its
prediction magnitude collapses and produces no economic action. Neither result
authorizes economic training. They answer different questions:

- a coherent full cross-sectional ordering is not the same as useful extreme-
  tail separation;
- positive mean gross return is not a cost-surviving result when turnover is
  frequently at its action cap;
- a rank loss can dominate clipped gradients when prediction dispersion is
  tiny, starving the magnitude and uncertainty heads even when weight decay is
  negligible; and
- nearly identical traces across scientific settings show that the ablation is
  no longer adding information.

Before changing model weights, run a separately identified, inference-only
audit of the exact published checkpoints. Predeclare zero, sign-flipped, and
deterministically shuffled signal controls; a downward action-cap ladder; full
decile and vigintile curves; scale and probability calibration; carry,
anchor-repair, and alpha-signal attribution; and common fold-bounded block
draws. Outcomes may score these paths but must never construct the actions.
The audit cannot retroactively change the original gate, select a checkpoint,
or mint an economic generation.

Bind such a follow-up to the complete parent closure: exact package and
authorization files, worker and fold terminals, checkpoint file and semantic
state hashes, terminal evidence, and cleanup receipt. Reuse large parent data,
risk, and checkpoint surfaces through exact read-only mounts when possible;
package only the new source, plan, and small lifecycle evidence. This reduces
transfer and storage without weakening lineage.

Finally, “unseen by the latest generation” is not the same as scientifically
untouched. If earlier development generations already consumed the remaining
pre-2026 chronology, later use is exploratory robustness evidence. Do not call
it a fresh holdout, and do not open 2026 to choose the follow-up architecture,
threshold, horizon, or action cap.

### Seeds, folds, and ensembles

Chronological folds measure performance across time regimes. Seeds measure
algorithmic instability on the same market history. They are not independent
market samples. Select each seed checkpoint using inner-validation evidence,
combine seed outputs into one deployed chronological path, and perform
inference on that path.

### Frozen v7 2026-YTD retrospective

The implemented v7 2026-YTD TOP2000 surface is a retrospective mechanism
diagnostic only. It
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
July, August, or any later 2026 coverage is outside that immutable contract.
Using it requires a fresh evaluation protocol, data/coverage identity, and
pre-access freeze. It must not enter v11 training, validation, qualification,
horizon selection, calibration, or debugging. Once 2026 outcomes are opened,
they remain consumed adaptive evidence and cannot become a fresh lockbox.

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

Distinguish exact semantic inputs from derived floating-point byte identity.
Cache bytes, asset order, exposure bytes, projector and estimator contracts,
origin indexes, source code, image, and checkpoint bytes remain exact gates.
A risk tensor recomputed on another Kubernetes node may nevertheless have a
different byte hash because CPU math libraries or instruction paths need not
be bitwise portable. Preserve both the parent and recomputed tensor hashes,
bind a separate semantic construction receipt over all exact inputs, and make
the cross-node byte-match status explicit. Never drop the numeric hash or
silently call it equal. Exact same-process/rank agreement is still required;
cross-node acceptance is valid only when the protocol predeclares semantic
recomputation and every upstream identity matches.

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

### Paired-ablation and residual-validity lesson from v9/v10 review

A shared random seed is not a paired experiment when a setting-specific worker
receipt enters the sampling hash. All causal settings must bind one panel-level
episode schedule derived only from common data, fold geometry, seed, and update
cursor. Rank shards must be complementary views of the same global origin set,
and a loss-only change must preserve every input tensor hash.

Pair the initial model bytes as well as the data. A nominal seed and identical
architecture do not guarantee byte-identical initializers across different
PyTorch/runtime environments. Package one immutable CPU `state_dict`, bind its
serialized-file hash and semantic model-state hash, strict-load it for every
setting and fold, and rehash the loaded state before the first update. The
same-image capacity qualification should prove that exact load. Without this
boundary, a supposed loss-only ablation may also compare different starting
parameters.

Future-return availability also does not imply that a factor residual is
estimable. Intersect the target mask with the decision-origin positive
regression-weight mask before any loss, IC, rank transform, or tail diagnostic.
Never encode an unestimable residual as a valid zero. Record qualified fraction,
effective design rank, and residual degrees of freedom.

Target residualization and executable signal projection must use one exact
operator: the same mask, weights, exposure order, collinearity convention,
solver, and tolerance. Retain an intercept and drop one reference sector (or
omit the intercept and retain all sectors); do not combine an intercept with a
complete mutually exclusive sector basis.

Data-dependent linear algebra needs a real-data structural gate. Synthetic
unit fixtures can cover formulas yet miss an unsupported sector, insufficient
qualified history, or an origin-specific rank defect. Before packaging, replay
the earliest real scheduled fold against the frozen risk tensors over every
scheduled origin and target horizon. Bind the qualified masks, supported
exposures, effective ranks, and residual-orthogonality checks in a distinct
structural receipt. This is CPU validation and should fail before scarce GPU
allocation.

Run that structural gate from the package-owned copies, after exact source,
cache, risk, exposure, and projector files have been staged. An external
preflight that matches only the cache and fold geometry can be paired with a
different risk model or source implementation. Bind both file hashes and
semantic receipts, including the operator source and projector binding, then
seal the package only after the package-local sweep passes.

### State-row versus return-transition boundary lesson

Do not size a maximum-horizon origin from state rows without accounting for the
missing terminal transition. An episode with `N` state rows has `N - 1` return
rows. If targets begin at `origin + 1` and consume `h` returns, the legal local
origin must satisfy:

```text
local_origin + 1 + h <= N - 1
local_origin <= N - h - 2
```

For the 378-state M03R episode and 63-session target, the maximum is 313, not
314. Test this boundary over the complete deterministic schedule, including
every setting, fold, update, rank shard, origin, and target horizon. A synthetic
single-origin unit test is insufficient because a sparse hash schedule may hit
the terminal origin only in later folds or updates. After a source-changing
fix, preserve the failed receipts, exact-clean any accepted Kubernetes state,
and mint a fresh source/package/run identity; never resume the old checkpoints.

### Context, horizon, and head-alignment lesson from v12

A selected-horizon constructor argument does not by itself make that horizon
the model's primary learning problem. Freeze and test the entire consequence
chain: direct loss weight, longest auxiliary target, legal origin range,
qualification geometry, execution horizon, and evaluator head. In v12 the
selected three-session head received only 10% of horizon loss while 80% went
to 21/30/63-session targets; the 63-session support still removed recent
origins. A future corrected three-session study must either train that horizon
alone or give it a declared primary weight with horizon-specific support.

Training and qualification must also use the same local temporal context and
positional-index distribution. A global origin being legal is insufficient.
If qualification scores local positions 251 through 313 with a complete
252-session context, loss-bearing training origins must have the same minimum
local position. Record the distribution of context length and local position
for every fold; reject a schedule when nearly all training rows are short
context but all qualification rows are full context. Sample eligible global
origins uniformly once per declared epoch rather than treating repeated
overlapping windows as new market histories.

Every head used by a ranking loss must be scored directly, and any head claimed
to improve execution must reach the action boundary. An auxiliary rank head
that influences only a capped share of shared-encoder gradient while IC,
spreads, and actions consume a different mean head is not an economic rank
ablation. Publish head-specific IC, dispersion, tail curves, action hashes,
and requested books.

Do not call daily OHLCV aggregated from five-minute bars an intraday sequence.
A one-token intraday branch cannot learn within-session path geometry, and
level normalization can make close and centered-volume coordinates exactly
zero. Preserve the daily control, but compare it with a true ordered
five-minute token sequence before further loss-only tuning when daily models
cannot recover even simple origin-known baselines.

### Action-mask and action/return chronology lesson

Future label validity is diagnostic evidence, not an action input. Construct
the action signal universe only from decision-origin availability and
decision-origin regression qualification. Apply fill-time availability as an
execution repair, and keep the future label-path mask on the target side only.
Hash these masks separately so an evaluator cannot reuse a future-conditioned
residual operator for execution.

Map decisions to returns explicitly. When a decision at state `t` fills at
`t + 1`, that action earns the `t + 1` return. A `D`-decision audit therefore
requires `D` aligned post-fill returns; it must not earn one pre-action return
and charge a final action that never receives exposure. Test the final action
with a distinctive final return so endpoint errors cannot hide in aggregate
metrics.

Requested-to-executed retention after the final projector does not measure
how much raw learned signal survives factor residualization. Record both
raw-to-residual signal norm retention and requested-to-executed book
retention. A final value of one can coexist with near-total upstream signal
removal.

### Sleeve sizing and inference lesson

A turnover cap is not a target. The desired proximal buy and sell notionals
must bound common trade mass, so scaling all alpha means to zero also sends
turnover to zero. Avoid applying uncertainty twice through both a probability
hurdle and a full standard-deviation shrinkage. Prefer differentiable calibrated
probability gates plus the exact transaction-cost penalty, and scale the risk
model to the selected return horizon.

Do not average fold-level lower confidence bounds or break-even ratios. Build
one ordered out-of-sample chronology, use common fold-bounded moving-block
draws, and compute aggregate break-even from summed gross active return and
summed incremental turnover with explicit sign categories.

Qualification must evaluate the exact published checkpoint bytes through a
fresh strict loader. An in-memory clone does not test serialization completeness
or buffer identity. Publish the checkpoint with no-clobber semantics and an
fsync before its receipt. The loader must open a regular non-symlink file
without following links, hash the bytes it actually reads, and verify that
inode, size, and modification time did not change across the read. Load tensor
state with `weights_only=True`, validate every scientific identity, strict-load
all keys and buffers, and rehash the semantic model state. Both the file hash
and semantic state hash are part of checkpoint identity.

For a new multi-GPU execution shape, make the capacity receipt terminal and
cleanup-bound rather than startup-only. A valid two-H100 capacity
qualification binds the exact startup file, application terminal receipt,
captured terminal Job/Pod/log evidence, and final cleanup receipt. Publish it
only after one UID/resourceVersion-conditioned delete and two absence reads.
This prevents a short-lived CUDA startup from being reused after its Job or
Pods were left ambiguous.

A load-only capacity check does not qualify training. The disposable capacity
path must exercise one exact-shape forward, objective, backward, distributed
gradient reduction, clipping, and optimizer mutation plus one qualification
risk projection. It must prove rank equality, record peak device memory,
publish no scientific checkpoint, and be terminally cleaned. This catches
unsupported deterministic kernels, NCCL behavior, operator transfer, and
peak-memory failures before the scientific Job exists.

Completion of a fixed epoch count is not convergence evidence. When the outer
qualification tail must remain untouched, reserve a chronological
training-only validation slice, evaluate the executable score at every
predeclared epoch, and select the immutable checkpoint with a frozen rule. A
clean rule can maximize projected IC, then projected spread, then minimize
robust loss, with exact ties favoring the earlier epoch. Bind every candidate
receipt and prove the qualification tail was never read for selection.

Once training itself consumes the action-projected score, raw-to-projected
norm retention is attribution rather than a promotion gate: the raw null-space
component receives no predictive gradient and is not scientifically
identified. If reported, use the projection's weighted full-action-space norm.
Keep requested-to-executed portfolio retention as the actual execution gate.

Derive lifecycle receipt inventories from the immutable protocol rather than
copying an earlier generation's horizon or setting list into a validator. A
completed workload can otherwise be misclassified after the scientific work
has ended. When historical evidence is intact and exact-clean, repair only the
validator under a fresh CPU-only continuation that binds the original terminal
evidence, exact failure, cleanup receipt, corrected source, and prior failed
continuations; never recreate the training Job. Compare reconstructed typed
receipts through canonical JSON bytes, not Python container equality: JSON
arrays legitimately reconstruct as typed tuples and are semantically equal
only after canonical serialization.

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

Do not summarize focused green tests as “all tests pass.” Report the exact
command, source identity, scope, test count, and result. Put concrete
outstanding integration debt in the affected generation record or tracked
issue rather than this durable guide. A focused package suite, full repository
suite, same-image qualification, and remote execution each prove different
boundaries.

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
- [ ] All causal settings bind the same initial parameter file and semantic
      state hashes, and the same-image runtime proves the exact load.
- [ ] Real frozen fold/risk data pass the complete scheduled-origin and
      target-horizon structural sweep.
- [ ] Non-finite gradients fail before optimizer mutation.
- [ ] Source manifest, data/cache, image, and plan hashes recorded.
- [ ] Raw-to-converted data receipt binds schema, dates, row counts, gaps,
      duplicates, asset axis, availability, and actual row usage.
- [ ] Every train/validation/qualification/test date role is disjoint exactly
      as required by the frozen protocol.
- [ ] Package receipts use relocation-safe identities; operator-local absolute
      paths are not treated as worker mount identities.
- [ ] Transfer archive has one top-level directory, deterministic metadata,
      no links/devices/traversal, and exact receipt-bound regular-file inventory.
- [ ] Package source inventory matches the intended commit exactly.
- [ ] Existing user changes are preserved and unrelated files excluded.

### Before GPU activation

- [ ] Approved session and mutation preflight receipt valid.
- [ ] Package publication and safe inventory valid.
- [ ] The rendered package `subPath` resolves to the inner package root and
      contains the bound plan, `source/src`, and runtime entrypoint.
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
- [M03R-v10 superseded rank proposal](top2000_m03r_v10_rank_geometry.md)
- [M03R-v11 corrected rank geometry](top2000_m03r_v11_rank_geometry_corrected.md)
- [2026-YTD retrospective](top2000_m03r_v7_seed17_2026_ytd_retrospective.md)
