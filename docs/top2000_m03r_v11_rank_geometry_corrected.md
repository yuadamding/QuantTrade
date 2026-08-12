# TOP2000 M03R-v11 corrected rank-geometry stage

Status: local implementation, optimizer-bound structural qualification, a
governed two-rank worker, and a content-bound package builder. The fresh a14
receipt-gated predictive development run is active after its same-image static
gate and two-H100 capacity gate passed. Its detached supervisor owns the
remaining startup, terminal, and cleanup lifecycle. Economic optimization and
2026 outcome access remain forbidden.

## Why v11 exists

The completed v9 predictive evidence remains an immutable negative result. The
unlaunched v10 proposal correctly narrowed the next scientific question to
rank geometry, but review found result-moving defects in the common training
sample, factor-residual target validity, simple-sleeve sizing, and aggregate
inference. Those semantics cannot be changed under the v10 identity.

V11 therefore has a new immutable generation:

```text
top2000-dev-hold30-active-alpha-m03r-v11-rank-geometry-corrected-v1
```

It retains the three scientific comparisons:

| Setting | Objective |
| --- | --- |
| P0 | factor-residual standardized-return listwise control |
| P1 | factor-residual rank-Gaussian correlation |
| P2 | factor-residual rank-Gaussian correlation with only 21/30-session losses |

All settings use seed 17, six chronological folds, 64 predictive updates,
two ranks, and zero economic updates. The predictive mean/scale architecture
is initialized fresh; v9/v10 model and optimizer state cannot enter v11.

## Paired training sample

One panel-level episode schedule binds the common data identity, cache,
six fold geometries, seed, update count, and rule
`paired-across-settings-v1`. Setting ID, objective, output path, and worker
receipt do not enter episode selection.

For every fold and update:

```text
episode_start = hash(panel_schedule, fold, update) % admissible_start_count
first_training_origin = max(episode_start, minimum_risk_history_sessions)
```

The episode may begin before factor-risk history is estimable because those
early states are still useful causal encoder context. They are not training
origins. The ordered origin set begins no earlier than the frozen 20-session
risk-history threshold. That ordered global origin set is identical for
P0/P1/P2. Rank 0 and rank 1
receive complementary interleaved shards of that one origin set. Worker plans
also bind one common initial-parameter-state hash. A change to the setting must
not change any input tensor or origin hash.

A 378-state episode contains only 377 forward return transitions. Therefore,
the largest legal local origin for the 63-session target is:

```text
378 - 63 - 2 = 313
```

The earlier a13 implementation used 314. A small number of rank-1 schedules
therefore reached an origin whose 63-session target required transition 378,
which does not exist. The failure was preserved as immutable development
evidence, all accepted Job and Pod state was exact-cleaned, and no a13 model or
optimizer state entered a14. The corrected scheduler rejects any shard with a
local origin above 313. Its regression enumerates all three settings, six
folds, 64 updates, both rank shards, every origin, and all four horizons.

The executable fold runtime builds the episode exactly once on CPU, binds its
full raw decision-input tensors before rank sharding, transfers the reviewed
sequence to the requested device, and then replays only the complementary
rank-local origin states. It also requires the externally hash-bound risk
manifest, cache/action axis, fold identity, two-rank geometry, and common
initial parameter state before the first update.

The optimizer mutation boundary consumes the exact training-shard receipt and
rejects a batch unless its setting, fold, update cursor, and ordered rank-local
origins match that receipt. It binds the batch, residual-operator root, panel
schedule, model state, and optimizer state before and after the single update.
Non-finite gradients fail before `optimizer.step()` with both states unchanged.
Before sharding, a setting-neutral paired-input receipt hashes the full episode
tensors, source array, asset axis, global origins, and both rank shards. Changing
only the scientific setting leaves this receipt identical; changing an input
tensor invalidates it.

## Factor-qualified shared residual operator

Future-return availability is necessary but not sufficient for a
factor-residual label. The valid mask is now:

```text
future path available
AND decision-origin regression weight > 0
AND not CASH
```

An unestimable residual is never represented as a valid zero target. Each
origin records available risky count, factor-qualified count and fraction,
effective design rank, and weighted residual degrees of freedom.

One content-bound operator is used for both target and execution signal:

```text
P_t y_t       for target residuals
P_t alpha_t   for executable signal residuals
```

The operator retains an intercept, omits sector columns with no qualified
support at that origin, and drops one supported reference sector. This removes
both zero-column rank deficiency and the exact intercept/complete-sector-one-hot
collinearity. The receipt records the omitted unsupported sectors and the
supported reference sector. It uses the same qualified mask, positive weights,
exposure names/order, weighted QR solver, and no-ridge convention for both
objects. Weighted exposure error must remain below `1e-9`.

## Launch-hardening lessons

Recreating a model from the same seed is not a cross-environment byte identity.
PyTorch initializer bytes can differ across the local package builder and the
pinned CUDA runtime even when the nominal seed and architecture are identical.
The package therefore stores one immutable CPU `state_dict`; every training
fold loads that exact artifact with strict key/shape checks and rehashes the
loaded semantic state before the first optimizer update. The same-image
two-H100 capacity sentinel must prove that exact load before the predictive Job
can be rendered.

Static unit fixtures are also insufficient for data-dependent linear algebra.
Before launch, the earliest real scheduled fold must be replayed against the
frozen risk tensors across every scheduled origin and horizon. This catches
warm-up support and exposure-rank defects without consuming GPUs. A failed
remote attempt remains immutable evidence; a result-moving source correction
requires a fresh package, run ID, static gate, capacity gate, and Job identity.

## Magnitude-preserving simple action

The corrected allocator no longer expands any nonzero two-sided signal to the
full turnover allowance. Its common reallocation mass is bounded by:

```text
desired buy mass
desired sell mass
confidence-scaled turnover cap
buy capacity
sell capacity
```

Therefore, holding signal ranks fixed while scaling means toward zero drives
incremental turnover toward zero.

Uncertainty is used once through differentiable calibrated-probability gates.
The proximal objective keeps the exact one-way transaction-cost penalty but
does not subtract a second full predicted-standard-deviation penalty. Entry,
expansion, retention, and exit thresholds remain declared, with a frozen
probability temperature. Same-step repurchase after learned release remains
forbidden. Daily factor-plus-diagonal risk is scaled by the selected 21- or
30-session horizon before it is compared with horizon return predictions.

## Correct aggregate inference

The six untouched fold chronologies are concatenated in chronological order
while block sampling remains fold-bounded. All metrics use the same
precomputed deterministic draws.

```text
primary block      21 sessions
sensitivities      10 and 30 sessions
replicates         10,000
```

The primary gate uses 95% moving-block lower confidence bounds for gross
active return, 10-bp net active return, and top-minus-bottom spread. It does
not average fold-level lower bounds.

Aggregate break-even cost is:

```text
10,000 * sum(gross active return) / sum(policy turnover - C1 turnover)
```

The receipt distinguishes finite positive break-even, favorable cost
dominance when gross alpha is positive and incremental turnover is nonpositive,
and no positive break-even when gross alpha is nonpositive. It never averages
fold-specific ratios.

## Broad-rank and execution gates

One bound 21- or 30-session horizon must pass every condition:

```text
mean Spearman IC >= 0.020
positive mean-IC folds >= 4/6
positive median date-IC folds >= 4/6
positive-date IC fraction > 0.50 in >= 4/6 folds
positive-spread folds >= 4/6
block-bootstrap spread LCB > 0
gross active return LCB > 0
10-bp net active return LCB > 0
aggregate break-even cost >= 10 bp or favorable cost dominance
median requested-to-executed retention >= 0.50
minimum fold median retention >= 0.20
noncollapsed prediction dispersion and predeclared target ratio
```

Failure of any condition blocks economic training. Thresholds cannot be
lowered after outcome inspection.

## Exact checkpoint evaluation

Qualification evaluates the published checkpoint artifact rather than an
in-memory clone:

```text
write immutable update-64 checkpoint
release the candidate policy reference
open and hash the exact regular non-symlink artifact
load into a fresh policy with strict state mapping
recompute model identity
evaluate that loaded policy
```

The checkpoint's source-array and residual-operator roots describe the final
training update available before qualification. The untouched-tail evaluator
publishes separate qualification source-array and residual-operator roots only
after both 21- and 30-session checkpoint files exist. Conflating these two
lineages would require opening qualification data before checkpoint
publication and is therefore forbidden.

Omitted buffers, wrong horizon or schedule, state-dict mismatch, or file drift
fails before evaluation.

Fold qualification evidence is then joined back to the loaded checkpoint, not
merely to a caller-supplied digest. The join requires exact setting, fold,
horizon, checkpoint file, panel schedule, and residual-operator root, while
also binding the loaded model state, source array, asset axis, and evaluation
trace. The six-fold aggregate gate accepts only six distinct, ordered
round-trip lineages.

The local package contract binds one source archive/manifest, dependency
lock, pre-2026 cache and manifest, point-in-time risk artifact and manifest,
projector manifest and binding, worker source, fresh common initial state,
setting-neutral episode schedule, exact mounted paths, and digest-pinned image.
The immutable scientific package retains `package_authorized=false` and
`kubernetes_launch_authorized=false`; those booleans cannot be inferred from
local unit tests or connectivity. After local qualification, the package
builder minted a separate execution-authorization receipt for only the exact
validated predictive entrypoint, three Indexed completions, two H100 requests
per completion, and a six-H100 request ceiling. Static, capacity, admission,
activation, and startup receipts then authorized and proved the active
predictive run. None of those receipts authorize economic training or 2026
access, and they do not rewrite the scientific package.

## Current authorization boundary

The current worker plan deliberately sets:

```text
package_authorized             false
kubernetes_launch_authorized   false
economic_panel_authorized      false
outer_2026_access_authorized   false
```

Local structural gates were completed before the execution authorization was
minted. The governed entrypoint writes both update-64 horizon checkpoints
before opening the qualification tail, reloads the exact bytes, publishes six
fold terminals, and treats a failed predictive gate as a valid scientific
terminal. The active Seadragon lifecycle has exact source, data, image,
capacity, admission, activation, and startup receipts; terminal and cleanup
receipts remain pending while training runs. Hardware allocation is evidenced
by those external receipts, not by this scientific document alone.

## Governed Seadragon execution surface

The execution bundle includes three deliberately disjoint boundaries:

```text
pure-file renderer/config builder
single-attempt suspended-Job creator
attach-only activation/terminal/cleanup supervisor
```

The creator is the sole Kubernetes create surface and never activates. It
server-dry-runs one exact manifest, proves the exact Job name and name-scoped
Pods absent twice, creates once, reconciles ambiguity for the complete request
window, then publishes a stable admitted UID binding and activation request.
The supervisor cannot create or replace a Job; it revalidates the exact
package-plan file and the separate execution-authorization file before
activation, then binds every worker terminal back to the three package worker
receipts.

A disjoint one-completion capacity Job first proves exactly two ranks on two
`NVIDIA H100 80GB HBM3` devices. Its terminal and exact cleanup receipt become
inputs to the predictive render. The predictive Job is one Indexed Job with
three completions, at most three active Pods, two H100 requests per Pod, and a
maximum request ceiling of six. A completed scientific gate failure is a valid
terminal and still authorizes neither an economic panel nor 2026 access.

The detached supervisor owns terminal capture and UID/resourceVersion-bound
foreground cleanup. Once activation and the launch-success receipt are
published, routine operation is intentionally left to Kubernetes and the
supervisor; no controller-side polling loop is part of the research protocol.
