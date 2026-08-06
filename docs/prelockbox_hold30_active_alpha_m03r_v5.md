# M03R v5 active-alpha Hold-30 redesign and experiment contract

**Protocol generation:** `prelockbox-hold30-active-alpha-m03r-v5`
**Schema version:** 5
**Design identity:** `daily_raw_pit300_hold30_m03r_v5`
**Status:** implementation qualification; governed PIT real-data/H100 launch blocked
**Supersedes:** frozen M03R v4 at commit `eadcfafacbcb49125c7a76d8f0952787071cf9ea`

This is the single normative RFC for M03R v5. The committed
[v4 RFC](prelockbox_hold30_active_alpha_m03r_v4.md) and v4 protocol module
remain unchanged for historical replay. No v4 artifact may be relabeled as
v5.

Shared code resolves this generation through the explicit
`resolve_m03r_v5_setting` and `resolve_hold30_m03r_v5_model_switches` APIs.
The historical unsuffixed model resolver continues to resolve overlapping
M03R IDs as frozen v4 solely for backward compatibility; new v5 code must not
use that ambiguous surface.

## 1. Scientific question

M03R v5 tests whether a compact, age-aware, raw-market policy can generate
positive cost-adjusted active return versus C1 while:

- carrying positions for roughly 30 trading sessions;
- avoiding compulsory active variance when evidence is weak;
- keeping active market beta and declared PIT factor exposures controlled;
- maintaining total-portfolio Sharpe noninferiority;
- separating learned exits, explicit de-risking, and risk-forced repair.

The canonical policy optimizes active return. Sharpe overlay and direct-Sharpe
training remain non-promotable causal ablations.

## 2. Temporal and information contract

The implemented model has:

| Quantity | Frozen value | Meaning |
|---|---:|---|
| Trainable high-resolution raw branch | 42 sessions | Full trainable intraday raw-day encoding |
| Learned temporal context | 252 sessions | Cross-day learned context; older 210 sessions do not load trainable intraday raw bars |
| Controlled rollout | 63 sessions | Recurrent optimization geometry |
| Economic origin | 30 post-fill returns | Hold-30 credit support |
| Auxiliary horizons | 5, 21, 30, 63 | Residual-alpha representation targets |
| Evaluation warm-up / score | 63 / 63 sessions | Continuous-book outer geometry |

The present implementation is therefore a **42-session trainable raw branch
plus 252-session learned temporal context**. It is not described as a
252-session raw-data branch. A lightweight old-day raw tokenizer would require
a later named generation and separate capacity qualification.

## 3. V5 causal inventory

| Index | Setting | Exact role |
|---:|---|---|
| 0 | `M00-absolute-return` | Absolute-net-log-return baseline |
| 1 | `M01-benchmark-subtraction` | Gradient-null governance control for M00 |
| 2 | `M02-active-risk-no-alpha-heads` | Active-risk controls without residual-alpha heads |
| 3 | `M03R-active-alpha-hold30` | Sole promotion candidate |
| 4 | `A04-no-downside-score-adjustment` | Disables downside adjustment only; confidence budget stays enabled |
| 5 | `A05-fixed-te-floor` | Restores the rejected 2% compulsory TE floor |
| 6 | `A06-sharpe-overlay` | Separate total-risk/Sharpe overlay |
| 7 | `A07-direct-sharpe` | Two-pass direct-Sharpe gradient ablation |
| 8 | `A08-fixed-exit-hazard` | Fixed 30-session exit prior |
| 9 | `A09-no-long-context` | Truncates learned context from 252 to 63 sessions |
| 10 | `A10-no-factor-neutral-projection` | Disables factor/sector projection |

### M01 is intentionally gradient-null

For detached C1 returns,

\[
\nabla_\theta[-E\log(1+r^P)]
=
\nabla_\theta[-E\{\log(1+r^P)-\log(1+r^{C1})\}].
\]

M01 is not evidence that benchmark subtraction changes learning. Under the
same batch, initialization, and optimizer state, M00 and M01 must produce
bit-identical policy gradients and parameter updates. Any observed difference
is an implementation or control-flow defect.

### A04 has one causal switch

V5 separates:

```text
use_downside_adjusted_stock_score
use_confidence_scaled_active_risk_budget
```

Canonical M03R enables both. A04 disables only the first. Its confidence
calibrator and confidence-dependent new-risk budget remain active.

## 4. Typed confidence calibration

The actor emits `uncalibrated_signal_confidence_logit`. The only authorized
conversion to confidence is a content-verified inner-validation calibrator:

\[
c_t = \sigma(z_t/T+b).
\]

V5 defines confidence as:

\[
P\left(\text{30-session net active log return versus C1}>0\mid\mathcal F_t\right).
\]

The manifest binds:

```text
schema and protocol generation
setting identity
seed, checkpoint SHA-256, and model-state SHA-256
source score-array and binary target-array SHA-256
target and raw-score definitions
inner-validation fold IDs and fit dates
outer_data_used = false
temperature and intercept
fit observation count
Brier score, expected calibration error, and observed target rate
manifest SHA-256
```

Governed construction follows the frozen two-stage protocol in
[`m03r_confidence_calibration_protocol.md`](m03r_confidence_calibration_protocol.md):
the policy checkpoint is trained and frozen first; the package-owned fitter
then derives the calibrator from actual detached inner-validation logits and
targets; the calibrator is frozen before validation or deployment; and no
policy update may follow calibration. The replay receipt content-binds the
row-level folds/dates, deterministic optimizer, calibrated probabilities, and
fixed ten-bin ECE evidence. Caller-authored temperatures, intercepts, Brier
scores, or ECE values are not governed fit evidence.

The model recomputes the manifest digest and applies its transform in the
forward path. A digest-shaped placeholder, a mismatched setting, altered
parameter, checkpoint/seed mismatch, source-array mismatch, missing typed
payload, or outer-data use fails closed. Calibration is per seeded checkpoint:
the five-member ensemble manifest binds five separate calibration digests,
not one common calibrator. Each ensemble member also carries the typed
calibration manifest. Aggregation reapplies that member's transform to its raw
decision-time logit and requires exact agreement with the emitted calibrated
confidence before accepting the member. Only the data manifest is common
across members.

## 5. One confidence channel and neutral-action semantics

Confidence is applied exactly once:

\[
\text{new-active-risk budget}_t = 0.04\,c_t.
\]

It is **not** multiplied into entry scores or learned exit hazards. Entry
scores retain their relative magnitudes, and the learned hazard first releases
notional into cash. Let the hard-feasible hazard anchor be

\[
w^{anchor}_t=\Pi_{hard}\!\left(
w^{carry}_t+\Delta w^{hazard}_t
+d_t\big[w^{C1}_t-(w^{carry}_t+\Delta w^{hazard}_t)\big]
\right),
\]

where \(d_t\) is the independent benchmark-de-risk request. Let
\(w^{proposal}_t\) be a second, independently hard-feasible projection of the
full score-driven replacement proposal. V5 computes

\[
q_t=\sqrt{252\,(w^{proposal}_t-w^{anchor}_t)^T
\Sigma_t(w^{proposal}_t-w^{anchor}_t)},
\qquad
u_t=\frac12\left\|w^{proposal}_t-w^{anchor}_t\right\|_1,
\]

\[
s^{cov}_t=\min\left(1,\frac{0.04c_t}{\max(q_t,\epsilon)}\right),
\qquad
s^{L1}_t=\min\left(1,\frac{c_t}{\max(u_t,\epsilon)}\right),
\qquad
s_t=\min(s^{cov}_t,s^{L1}_t),
\]

\[
w^{confidence}_t=w^{anchor}_t+s_t
(w^{proposal}_t-w^{anchor}_t).
\]

Thus confidence continuously bounds both the covariance norm and a
nondegenerate one-way L1 norm of the complete incremental move from the
hazard anchor toward replacement entry. The L1 channel is required because a
valid PSD covariance may have a large nullspace. At zero confidence, learned
hazard exits still execute, subject to the separately bound hard constraints
and turnover cap, while replacement entry is zero. If the exact-hold decision
also suppresses every learned hazard and the de-risk request is zero, the
hazard anchor is the carried feasible book exactly. An arbitrarily small
positive confidence can add only a marginal amount of replacement/new risk;
it cannot rotate the book inside a covariance nullspace.

The result-moving coefficient is frozen and identity-bound in the v5 active
risk protocol as
`maximum_confidence_incremental_one_way_turnover = 1.0`; therefore the L1
budget is exactly (c_t) one-way portfolio turnover.

V5 separates three meanings:

```text
signal_confidence / active_risk_scale
    budget for replacement entry or other new/enlarged active risk

learned exit hazard
    confidence-independent release of existing notional into cash

benchmark_derisk_request
    independent request to move existing deviations toward C1

risk-forced repair
    mandatory feasibility correction, separately accounted
```

Canonical `benchmark_derisk_request` is exactly zero. Therefore zero
confidence alone does not liquidate existing active positions, but it also
does not veto a learned exit. With no learned hazard sale and no forced repair,
the neutral action carries the current feasible book and produces zero
discretionary turnover.

## 6. Exact-hold output contract

V5 freezes three explicit actor outputs:

```text
exact_hold_logit
exact_hold_soft_probability
exact_hold_decision_st
```

`exact_hold_decision_st` is hard 0/1 in the forward pass and uses the declared
straight-through sigmoid gradient. The legacy field
`exact_hold_probability` is restricted to v2/v3/v4 compatibility. V5 ensemble
and execution consume the explicitly named hard decision.

## 7. Projection contract

V5 content-binds **two** post-ensemble hard-feasibility projection stages:

1. project the learned-hazard / explicit-de-risk anchor onto the affine,
   long-only/box, and linear PIT factor/sector sets, then benchmark-radially
   scale it only when needed for the independent 6% hard TE ceiling;
2. add the full replacement-entry proposal to that feasible anchor, project it
   onto the same linear sets, and apply the same independent 6% hard TE cap;
3. continuously interpolate from the feasible hazard anchor toward the
   feasible replacement proposal, capped by the confidence-dependent
   incremental covariance and L1 budgets;
4. turnover-limit the move from the repaired feasible carried book.

Because both confidence-interpolation endpoints are feasible and the hard
sets are convex, confidence scaling cannot invalidate factor, cap, simplex, or
TE constraints. The contract does **not** claim that either benchmark-radial
TE operation is the global Euclidean projection onto the joint covariance
ellipsoid intersection. Both projection applications, the confidence-budget
delta, and all cause-specific sale-age amounts retain separate diagnostics.

The execution boundary also binds the exact ordered asset axis. Current and
C1 weights, availability, and the age ledger travel in one
`M03RAssetAlignedBook`; its ordered IDs and SHA-256 must match the PIT risk
manifest. Each of the five ensemble members binds its seed, checkpoint,
model-state, asset-order, confidence-calibration, and data-manifest hashes.

Risk inputs are qualified once per content hash. Qualification clones the
manifest tensors, validates covariance PSD and all declared units,
normalization, estimation-window, missing-value, shrinkage, return-convention,
staleness, and infeasibility semantics, and precomputes a covariance factor.
It then issues a process-local, non-serializable capability bound to the exact
manifest and tensor object identities, digest, deterministic order, and tensor
mutation counters. The capability also stores canonical content fingerprints
for the manifest tensors and cached covariance factor. Every governed trust
boundary recomputes those fingerprints, so ordinary in-place writes, `.data`
writes, and shared NumPy-view writes fail closed even when PyTorch's mutation
counter does not change. This check is linear in the qualified tensor content;
the covariance eigendecomposition and full numerical qualification remain
one-time costs. Manual construction, `dataclasses.replace`, deserialization,
and process restart cannot reuse the capability; external/restored manifests
must qualify again.

The training objective cannot declare a separate factor inventory. Every
chronological objective row carries its exact ordered PIT risk-manifest
SHA-256. A typed objective-risk contract derived from those qualified manifests
binds the common risk-manifest schema, exposure names and order, families,
units, normalization IDs, and asymmetric lower/upper slabs. The objective
accepts bound policy and C1 weight rows, not caller-authored exposure tensors;
it derives every active exposure directly from those weights and the qualified
PIT loading matrix. It rejects reordered manifest rows, mismatched asset axes,
or any contract hash drift and evaluates its factor penalty against those exact
asymmetric slabs.

Execution returns deltas for:

```text
unavailable liquidation
risk-forced repair
learned-hazard release
entry reallocation
explicit benchmark de-risking
linear factor/box projection
tracking-error radial scaling
turnover truncation
```

These deltas must telescope exactly from the input book to executed weights.
The authoritative function also produces the final age ledger; net sells are
removed pro rata from age cohorts and net buys enter age zero. Executed
post-repair sells use the frozen attribution priority: learned hazard, explicit
benchmark de-risking, then feasibility projection. Only the first category is
a discretionary learned exit and can enter the learned early-exit penalty;
unavailability/risk repair are forced, benchmark de-risking and projection are
risk-layer turnover, and fold-end censoring is not a trade. Availability and
risk-repair turnover diagnostics are disjoint: repair is measured only from
the availability-repaired book. The receipt carries separate sold notional and
sale-age tensors for unavailability, risk repair, learned hazard, explicit
benchmark de-risking, and feasibility projection. Their sum must exactly
partition total cause-attributed sold cohort mass, so projection-only or forced
sales cannot be mislabeled as learned early exits.

## 8. Active-alpha inference contract

The evaluator runs three distinct regressions on the same PIT factor matrix:

```text
portfolio:  P - rf
benchmark:  C1 - rf
active:     P - C1
```

Portfolio and benchmark intercepts are diagnostics. Promotion factor evidence
is the lower confidence bound of the **active multifactor intercept**, stored
in daily and 252-times annualized units.

All three regressions use the same six-fold, effect-coded fixed-effect design.
The reported common intercept is the equal-weight mean of the six fold-specific
intercepts, rather than an intercept contaminated by level differences between
folds. HAC score products are formed inside each fold only: no lag pair may
cross a fold boundary. The moving-block bootstrap likewise samples circular
blocks independently within each fold before recomputing the deployed statistic.

The primary uncertainty contract is:

- circular moving-block bootstrap with 21-session blocks;
- sensitivity blocks fixed at 10 and 30 sessions;
- a complete 63-session fold block is rejected for primary inference;
- active-beta HAC lag fixed at 30;
- active-beta equivalence gate
  \(|\hat\beta_{active}|+z_{1-\alpha}SE_{HAC}\le0.10\).

The receipt binds common evaluator inputs separately from the candidate path:

```text
common_evaluator_inputs_sha256
    ISO score dates, fold IDs, C1/rf/market/factors, inference plan

candidate_policy_returns_sha256
    exact deployed chronological policy path

candidate_evaluator_receipt_sha256
    independently recomputed candidate metrics
```

Caller-authored checkpoint metrics cannot qualify a candidate. The current
chronological evaluator can reproduce return and factor evidence from exact
arrays, but it cannot yet reproduce holding survival, censoring, cause-typed
turnover, and projection distance from one authoritative execution ledger.
Consequently the v5 checkpoint-selection adapter is explicitly unavailable and
fails closed; self-hashing an aggregate metrics object cannot create evidence.

## 9. Selection and promotion

Checkpoint eligibility remains fail-closed. Ranking first uses the declared
block-bootstrap lower confidence bound of 20-bp net active return. Information
ratio, total Sharpe, drawdown, and cost robustness are secondary.

The active-beta eligibility inequality uses the one-sided equivalence upper
bound, not the point estimate:

\[
|\hat\beta_{active}-\beta^\star| + z_{1-\alpha}SE_{HAC} \le 0.10.
\]

Promotion requires, at minimum:

- positive 20-bp active-return lower confidence bound;
- positive multiplicity-adjusted active multifactor-alpha lower confidence bound;
- total Sharpe noninferior to C1 under the declared bound;
- 40-bp robustness;
- uncertainty-aware active-beta equivalence;
- exact null-control and signal-destruction gates;
- holding survival/censoring and empirical-capacity gates.

## 10. Launch blockers

M03R v5 is not authorized for the governed real-data/H100 experiment until all
of the following are receipt-complete:

1. one governed route for all eleven settings;
2. PIT data, benchmark, risk-free, factor, sector, universe, and asset-order manifests;
3. setting/fold/seed-specific confidence calibrators fit without outer access;
4. five-seed checkpoint/model-state identity binding;
5. cause-specific final age-ledger and turnover accounting;
6. empirical spread/impact and closed-loop 40-bp execution;
7. frozen multiplicity family and promotion plan;
8. CPU deterministic replay, CUDA one-rank/two-rank, restart, mixed-precision,
   and H100 memory/throughput receipts.

The aggregate route-status constructor recomputes the canonical blocker list
from the immutable route inventory and revalidated typed qualification
receipts. `launch_authorized` is derived from that list; a caller-authored
empty blocker tuple and matching self-hash cannot authorize launch.

Synthetic qualification proves mechanics only. It is not performance evidence
and does not authorize launch or promotion.
