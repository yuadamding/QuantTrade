# TOP2000 M03R-v16 paired holding-aligned selection research

## Decision

V16 is the predictive-only successor to the completed V15 h3 screen. V15's
best mean projected IC was `0.01013`, its best break-even cost was `1.84 bp`,
and neither setting survived 10-bp costs. Further daily h3 loss or
learning-rate tuning is prohibited.

V16 asks one narrower question: does the existing daily representation contain
a slower selection signal aligned with multiweek holdings? It cannot reuse V15
model or optimizer state, train timing or uncertainty, run an economic/RL
optimizer, or access 2026 outcomes. The future-selected TOP2000 surface remains
development-only, nonreportable, and nonpromotable.

No V16 package, remote run, GPU result, or performance result is claimed. The
current local protocol is the result-moving v5 revision; older v16 source and
artifact identities may not be mixed with it.

## Scientific settings

| Index | Setting | Numerical target | Role |
| ---: | --- | --- | --- |
| R0 | `V16-R0-h21-selection-control` | cumulative h21 factor-residual log return | explanatory control |
| R1 | `V16-R1-h30-selection-control` | cumulative h30 factor-residual log return | explanatory control |
| R2 | `V16-R2-hold30-prior-selection-primary` | truncated cumulative day-1..30 value under the reference Hold-30 survival clock | predeclared primary hypothesis |

R2 is the only promotion-eligible setting. R0/R1 explain horizon behavior; the
outer tails may not be used to choose a different target after results are
opened.

All three settings share exactly the same:

```text
initial parameter bytes
30-session future-valid asset/date mask
label residual operator
action residual operator
date schedule and rank shards
optimizer groups and learning rates
fixed epoch count
qualification rules and bootstrap draws
```

Only the numerical target tensor and its frozen economic-unit scale differ.

## Common causal support

For every origin `t`:

\[
A_t=\text{origin available}\cap\text{origin regression eligible},
\]

\[
L_t=A_t\cap F_{t,30}.
\]

One label operator is built on `L_t` for all settings. H21 is still calculated
from 21 returns, but it is scored on the common 30-session support. Native-h21
support may later be reported only as a secondary diagnostic.

The model emits one dimensionless raw score `z`. The action-time operator
creates the executable score:

\[
z_t^A=P_t^A z_t.
\]

Loss, validation IC, qualification IC, spread, and sleeves must use that same
projected object. Economic units are restored only as:

\[
\mu_t^A=\tau_s z_t^A.
\]

Returned FP32 score and target residuals receive a second orthogonality check
after the float64 solver result is cast back.

## Hold-30-prior target

The primary target uses the repository's actual neutral release clock:

\[
\beta(a)=-2+\frac{a-30}{4},
\]

with the normalized `hold30_release_hazard(age, 0)` transform. Survival is:

\[
S(1)=1,\qquad S(k+1)=S(k)(1-h_0(k)).
\]

The development target is the unnormalized truncated cumulative value:

\[
y_t^{S,30}=\sum_{k=1}^{30}S(k)r^{\mathrm{res}}_{t+k}.
\]

It is not a normalized alpha rate, mandatory 30-day hold, forced expiry, or
complete infinite-horizon holding value. About 55% of reference-prior survival
remains after day 30 and is explicitly receipt-bound as truncation evidence.
V16 explicitly binds the immutable legacy Hold-30 specification: holding target
30, age cap 60, and `legacy-hold30-v1`. The generic framework default of three
sessions does not alter this experiment, its label support, or its cohort tail.

## Dimensionless optimization

Each setting's frozen scale is:

\[
\tau_s=0.02\sqrt{\sum_k w_{s,k}^2}.
\]

The single head predicts `z`, and the loss is:

\[
\mathcal L_s=
\operatorname{Huber}\left(P_t^A z_t,\frac{y_t^s}{\tau_s}\right).
\]

This prevents the target's units from changing the effective head gradient or
clipping probability. There is no h3, rank, or distributional loss in this
screen. Timing and uncertainty require later identities after a selection
target passes.

LayerNorm parameters, module biases, and `cash_bias` receive zero weight decay;
classification is based on module identity rather than name fragments.

## Split and checkpoint geometry

V16 uses five outer folds with qualification starts:

```text
535, 628, 721, 814, 907
```

Each fold has 63 decision origins and 30 future-support sessions. The 93-session
advance makes adjacent scored return-transition sets disjoint. The final fold
ends at the exact 1,001-state pre-2026 cache boundary.

Each training episode contains:

```text
252 causal context states
63 possible origin positions
30 future-support transitions
```

All eligible training origins appear once per each of eight fixed epochs.
Blocks are balanced within each fold, so their sizes differ by at most one; a
three-date remainder can no longer receive one full optimizer step and twenty-
one times the per-date weight of a full block. A
63-origin inner-validation slice publishes fit diagnostics only; it cannot
select an epoch. The immutable terminal epoch-8 checkpoint is evaluated after
strict write/reload. This avoids searching many checkpoints on highly
overlapping h30 labels.

Primary moving-block inference uses 42 sessions, with 30- and 63-session
sensitivities, while preserving fold boundaries.

## Qualification path

The local source now binds the package-owned structural slab, slab-backed fold
updates, exact-workload disposable capacity primitive, terminal-checkpoint
qualification, horizon-matched cohort accounting, fold-preserving statistical
aggregate, and primary-R2 decision rule end to end. It also provides a sealed
local package builder, same-image static validator, deterministic safe transfer
archive, two-rank image worker, and receipt-gated suspended Job renderers. No
real package has been built and no Kubernetes object has been created; real-data
package validation, server-side dry runs, live admission, capacity evidence,
and scientific execution remain required.

Qualification retains two distinct diagnostics:

1. a small fixed-rank ordering sleeve for projected IC and tail attribution;
2. a horizon-matched cohort sleeve, with 21-/30-session carry or the exact
   reference release clock truncated at return 30. The final 30-session cohort
   earns once on its decision/fill step and then through 29 no-new-decision
   transitions. Any remaining active risk is terminally liquidated and charged.

The cohort action mask is the causal origin-action mask intersected with
fill-time availability. The common 30-session label mask is diagnostic-only
and cannot influence action construction. Every signal cohort remains
self-financing after projection, release, and drift. Projector-created
exposures enter a separate age-independent risk-repair cohort, so they cannot
inherit an arbitrary signal age. Prior repair remains in the carried portfolio
for P&L and actual-turnover accounting but is excluded from the next desired
signal request; the current projector must recreate every still-required
repair. Same-direction executed excess beyond compatible signal request also
remains repair. Cohort rows sum to the carried executed active book after every
transition. Prior-repair unwind mass and request-to-execution projection
distance are explicitly non-additive telemetry, separate from actual policy
turnover.

The cost ladder is `0, 1, 2, 3, 5, 10, 20, 40 bp`. Report absolute policy cost,
benchmark cost, incremental active cost, gross/net policy return, gross/net
active return, break-even cost, capacity, concentration, and cohort ages.

The primary R2 setting may advance only if it passes the existing predictive
and tradeability gates, including projected IC `>= 0.020`, breadth in at least
four of five folds, positive spread/gross/net lower bounds, and break-even cost
`>= 10 bp`. Controls cannot be substituted after outer outcomes are read.

If the corrected daily screen fails, stop daily target/loss tuning and move to
an ordered five-minute stock-day encoder under a new protocol. Quotes and
trades follow only after the five-minute control establishes incremental
projected signal.

## Current implementation boundary

The repository implements the revised protocol, five disjoint folds, paired
every-origin schedules, common initial bytes, dimensionless selection policy,
common-support causal batch, action-projected loss, module-aware optimizer,
fail-closed mutation, FP32 orthogonality telemetry, immutable epoch
checkpoints, and diagnostic-only inner validation.

It additionally materializes all scheduled action/label operators and all
three targets once from exact cache/risk bytes. The no-clobber slab loader
hashes and loads through one no-follow descriptor, revalidates every operator,
and binds cache, asset axis, risk, exposure, projector, and source identities.
The canonical fold-update path consumes a validated-slab authority and performs
no QR or full-slab semantic validation in the optimizer hot path. Origin lookup
is O(1), device operators are cached, and scalar exposure telemetry is reduced
only at the batch boundary.

The exact-workload capacity primitive performs one disposable 345-state,
balanced first-fold two-rank BF16 forward/backward, NCCL gradient sum, clipped
Adam mutation, and an internally constructed nontrivial qualification
projection. The request must have positive active mass, projection must change
it while retaining nonzero mass, and the typed terminal requires at least 8 GiB
or 10% unreserved device memory plus equal post-update model and optimizer
identities across ranks. It cannot publish a scientific checkpoint or authorize
training.

The cohort primitive publishes separate absolute policy, benchmark, and
incremental costs and net returns; 21-/30-session or truncated age-clock
signal-cohort mass reduction after execution; executed signal ages;
risk-repair mass; prior-repair unwind mass; request-to-execution projection
distance; risk-projection retention;
final-horizon chronology; and terminal liquidation.
The authoritative fold qualifier accepts the reloaded policy, cache, risk,
fold, and validated slab—not a caller-supplied score batch. It rebuilds origin
states, raw scores, and action-projected scores internally, recomputes the
projection once at the qualification boundary, and binds all arrays to the
checkpoint before cohort economics can run. Qualification additionally requires
an issuer-bound terminal authority that reconciles fixed epoch 8, the complete
checkpoint-selection receipt, paired panel schedule, fold geometry, cache axis,
and structural-slab receipt. Every trace binds the terminal-checkpoint and
qualified-score authority receipts.

The local package builder copies the exact source, cache, risk, and projector
inputs; builds the structural slab from those package-owned copies in an
isolated interpreter; seals the common initial state, panel schedule, package
plan, execution authorization, and full inventory; and can emit a deterministic
transfer archive that rejects traversal, duplicates, links, devices, and hash
drift. The same-image static validator explicitly performs no training and
claims zero GPU visibility only. The worker requires exactly two visible H100
80GB devices, runs one disposable exact-shape capacity update plus a nontrivial
risk projection, or serially trains and strictly reloads all five folds before
checkpoint-owned qualification.

The aggregate uses fold-preserving 42-session bootstrap blocks with frozen
30-/63-session sensitivities. Only R2 can authorize a three-seed predictive
confirmation; a failure routes to ordered five-minute research. It can never
authorize an economic generation, reinforcement learning, or 2026 access. A
no-follow file aggregator verifies the exact three worker-terminal hashes,
package/authorization/worker identities, common bootstrap receipt, and each
setting qualification before publishing the no-clobber panel decision.

The local Kubernetes layer renders three distinct `batch/v1` Jobs: explicit
zero-GPU static validation, one two-H100 capacity sentinel, and a suspended
three-completion predictive panel requesting at most six H100s. Capacity and
predictive rendering require the preceding immutable gate receipts. All Jobs
remain suspended, fail-fast, token-unmounted, nonprivileged, digest-pinned,
bounded by deadline/TTL, and mount only the scoped package/output PVC paths.
Activation, live admission binding, detached supervision, UID-bound cleanup,
and any remote mutation are intentionally not implemented by this source-only
revision.

No real-data package or slab receipt, same-image result, capacity result,
remote Job, GPU result, or performance result is claimed. Those evidence gates
still block H100 scientific execution.

## Local verification

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold_target_protocol.py \
  tests/test_hold30_alpha_m03r_v16_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v16_*.py

conda run -n quanttrade ruff check \
  src/rl_quant/protocol/hold30_alpha_m03r_v16_top2000_dev.py \
  src/rl_quant/training/top2000_m03r_v16_*.py \
  tests/test_hold30_alpha_m03r_v16_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v16_*.py
```
