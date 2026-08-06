# M03R v6 soft-persistence active-alpha RFC

**Protocol generation:** `prelockbox-hold30-active-alpha-m03r-v6`
**Schema version:** 6
**Design identity:** `daily_raw_pit300_hold30_m03r_v6`
**Canonical setting:** `M03R-soft-persistence-active-alpha-hold30`
**Status:** implementation qualification; governed PIT real-data/H100 launch blocked
**Supersedes:** immutable M03R v5 without modifying or relabeling v4/v5 artifacts

This protocol is for non-PHI scientific research and software qualification;
it is not a business-production, investment-advice, or deployment contract.

Confidence calibration targets whether a frozen, standardized unit-risk
30-session active proposal earns positive net active log return versus C1. It
must never target the final confidence-sized policy path, because that would
make confidence define its own calibration outcome.

Policy training and deployment calibration are separate stages. While policy
parameters remain trainable, the raw confidence head receives package-owned
binary-log-loss supervision from the standardized unit-risk outcome, consumes
a detached market representation, and cannot size the economic trading path.
After checkpoint freeze, deterministic inner-validation fitting binds the
calibrator; calibrated confidence may then size new or expanded active risk,
and no subsequent policy update is permitted.

Ordinary exits use the bounded continuous age-aware hazard. Exact full exit is
not attributed to the unreachable open endpoint of the tanh bound; it is a
separate learnable EXIT atom in a mutually exclusive HOLD / CONTINUOUS / EXIT
straight-through action. Exact HOLD remains optional and A11 removes only that
atom. Exact EXIT remains available at every position age.
This requirement applies to learned-hazard settings; A08 intentionally removes
the learned action head as the fixed-prior comparator.

This document and
`src/rl_quant/protocol/hold30_alpha_m03r_v6.py` define the v6 contract.
V6 changes the interpretation of Hold-30: 30 trading sessions is a soft
preference, never a trading prohibition or an evaluation target that must be
hit.

## Scientific question

V6 tests whether a cost-aware, active-alpha policy benefits from a soft prior
for persistent positions while retaining full daily discretion. The policy may
hold exactly, resize, or exit on every trading decision. Alpha reversal,
transaction costs, risk repair, unavailability, or a superior replacement can
justify an exit at any age.

The following are prohibited:

- a minimum holding period;
- masking or blocking sales before day 30;
- automatic expiry or forced sale on day 30;
- using realized holding duration as a promotion eligibility gate;
- requiring the learned policy to select the exact-hold action.

The optional exact-hold action remains an efficient atom for representing no
trade. It is supported by canonical M03R, but never required. A11 removes that
atom to measure its causal value.

The bounded learned hazard retains `-12` as exact zero release, while `+12`
remains an ordinary continuous partial-release endpoint. A separate learnable
EXIT atom provides reachable exact full discretionary exit at every age in
learned-hazard settings. Thus
a severe reversal can close even a young cohort, while a favorable position
may remain open indefinitely. Hard risk and availability repairs execute
before the discretionary action and ignore the persistence cost and HOLD atom.

## Temporal and economic semantics

| Quantity | Frozen value | Meaning |
|---|---:|---|
| High-resolution trainable raw context | 42 sessions | Recent raw intraday encoding |
| Learned temporal context | 252 sessions | Long learned cross-day context |
| Controlled rollout | 63 sessions | Optimization geometry |
| Economic origin | 30 post-fill returns | Return support for the economic objective, not a holding rule |
| Persistence preference | 30 sessions | Age scale in the soft early-exit penalty only |
| Age ledger | 61 bins, indexed 0–60 | Exact protocol-owned shape; bin 60 accumulates notional aged 60 sessions or more |
| Soft turnover reference | 1/30 one-way per session | Descriptive regularization reference, not a duration proxy, hard constraint, or promotion gate |
| Auxiliary horizons | 5, 21, 30, 63 | Residual-alpha representation targets |
| Evaluation warm-up / score | 63 / 63 sessions | Continuous-book evaluation geometry |

The 30-return economic origin, 30-session auxiliary label, and 30-session
persistence preference are separate content-bound quantities. None implies a
minimum holding duration.

## Soft-persistence objective

For policy-discretionary sales of notional `x_a` at age `a`, v6 freezes the
one-sided quadratic age weight

\[
q(a)=\max(0,1-a/30)^2.
\]

Let `N_valid` be the exact count of valid scored decision sessions represented
by the aggregated sale-age ledger. The NAV- and session-normalized early-exit
notional is

\[
E_{NAV/session}=\frac{1}{N_{valid}}\sum_a x_a q(a).
\]

There is deliberately no denominator involving total sold notional. A 1% NAV
young sale pays 1% of the full-NAV penalty, while adding mature sales has
exactly zero value and zero gradient in this term; mature sales cannot dilute
the cost of young exits.

For training completion fraction `u`, the 10% linear warmup is

\[
m(u)=\min(1,u/0.10).
\]

The total optimizer-step count that defines `u` is carried by a typed,
content-addressed v6 training-plan receipt. It cannot be changed as a free
runtime argument under the same plan identity. The valid decision-session
count is owned by the same typed cause inventory as the aggregated age ledger,
is derived by the sequence adapter, and must be positive; it cannot be supplied
independently to the objective.

The canonical loss contribution is

\[
L_{persist}=5\times 10^{-4}\,m(u)E_{NAV/session},
\]

corresponding to 5 bp per unit at age zero. Learned-hazard, explicit policy
de-risk, and policy-induced projection sales enter this loss. Pretrade
unavailability, current-book risk repair, corporate-action, other forced, and
terminal sales remain cause-typed telemetry and are excluded. The adapter
requires the cause tensors to be disjoint and to sum exactly to total executed
sales, so a policy cannot route a young exit through projection to avoid the
penalty.

Five basis points is the sole v6 coefficient identity. A 2-bp or 10-bp
coefficient is not a v6 inner-grid choice and cannot share a v6 receipt. Those
sensitivities are deferred to a new immutable generation (v7 or later), where
each must receive a separate non-promotable setting ID. Holding-duration
outcomes remain diagnostics, not promotion gates.

The generic design registry likewise carries no v6 `target_holding_days` or
`target_discretionary_turnover`. It binds
`holding_preference_horizon_sessions = 30` and
`soft_daily_one_way_discretionary_turnover_reference = 1/30` instead. The
63-session score tail is rollout geometry and is not coupled to an enforced
holding target.

The holding/economic gradient-norm ratio is reported separately with an
initial diagnostic target band of 5%–15%. It never rescales the loss or gates
promotion automatically. Inner-fold coefficient-response evidence must show a
smooth duration response without assuming that profitability is monotonic in
the holding penalty.

## Checkpoint eligibility and ranking

Hard eligibility is limited to complete evidence, positive 20-bp active
return, nonnegative 40-bp active return, the annual tracking-error ceiling,
the active-beta equivalence bound, exact 61-bin age-ledger validity/content
binding, and frozen projection/forced-turnover quality limits. Raw fold-censored
notional fraction remains content-bound telemetry; it is not an eligibility
threshold. Survival at 20 or 30 sessions, RMST60, and the precision implied by
their censoring pattern cannot make a checkpoint eligible or ineligible. A
profitable, otherwise valid RMST45 candidate remains eligible even when most
open notional is right-censored at a fold boundary.

This correction is bound by checkpoint-selection schema
`rl-quant.m03r-v6-soft-persistence-checkpoint-selection-contract-v2`; a v1
selection receipt containing a censoring threshold cannot identify this gate
contract.

## Generation-qualified numerical evaluator

`src/rl_quant/evaluation/hold30_alpha_m03r_v6.py` is the v6 public numerical
surface for already-produced chronological return arrays. It validates the
exact v6 protocol, design, and setting at entry and receipt boundaries. It is
not a data loader, production evaluator driver, checkpoint selector, or launch
authorization path.

Evaluation fails closed without both typed, content-addressed manifests:

- a point-in-time factor manifest whose factor set was defined without outer
  data; and
- an inference manifest bound to that exact factor manifest, bootstrap seed,
  replicate count, primary 21-session moving block, 10/30-session sensitivity
  blocks, 30-session primary HAC lag, and one-sided 5% tail.

The evaluator reports three separate fold-fixed-effect multifactor
regressions:

```text
portfolio excess return versus market and declared factors
C1 benchmark excess return versus market and declared factors
policy-minus-C1 active return versus market and declared factors
```

The primary uncertainty output is the 21-session within-fold circular
moving-block lower confidence bound of the **active** multifactor intercept;
10 and 30 sessions remain sensitivity blocks. The receipt binds chronology,
fold IDs, every numerical return array, both manifests, all regression output,
and bootstrap results. Its promotion flag remains false while the public
production evaluator driver, multiplicity-adjusted factor family, and outer
data-access receipts are unavailable.

Eligible checkpoints rank by 20-bp active-return bootstrap lower bound,
information ratio, total Sharpe, drawdown, discretionary turnover, cost, then
the weak one-sided score `-max(0, 25 - RMST60)`, followed by the earlier
checkpoint. Holding longer than 25 sessions receives no additional ranking
benefit, so a profitable 18- or 45-session policy is never displaced merely
for looking less like an exact 30-session schedule.

## V6 causal inventory

V6 uses new setting IDs so no v4/v5 result can be relabeled as v6.

| Index | Setting ID | Role |
|---:|---|---|
| 0 | `M00-absolute-return-v6` | Absolute-return baseline with soft persistence |
| 1 | `M01-benchmark-subtraction-v6` | Gradient-null benchmark-subtraction control |
| 2 | `M02-active-risk-no-alpha-heads-v6` | Active-risk controls without residual-alpha heads |
| 3 | `M03R-soft-persistence-active-alpha-hold30` | Sole promotion candidate |
| 4 | `A04-no-downside-score-adjustment-v6` | Removes downside score adjustment |
| 5 | `A05-fixed-te-floor-v6` | Restores the rejected 2% TE floor |
| 6 | `A06-sharpe-overlay-v6` | Adds a separate Sharpe overlay |
| 7 | `A07-direct-sharpe-v6` | Adds the direct two-pass Sharpe objective |
| 8 | `A08-fixed-exit-hazard-v6` | Freezes the soft 30-session hazard prior |
| 9 | `A09-no-long-context-v6` | Truncates learned context to 63 sessions |
| 10 | `A10-no-factor-neutral-projection-v6` | Removes factor/sector projection |
| 11 | `A11-no-exact-hold-atom` | Removes only the optional exact-hold action atom |

A04–A11 must each differ from the canonical setting in exactly one declared
causal field. A08 remains distinct from A11: A08 freezes the age-shaped exit
prior, while A11 removes the exact no-trade atom and leaves the learned hazard
intact.

### Generation-qualified route inventory

`src/rl_quant/training/hold30_alpha_m03r_v6_routes.py` binds every setting, in
protocol order, to its exact v6 objective, model, five-seed ensemble,
execution, and evaluator route IDs. No v5 route ID or artifact is accepted.

Every v6 route binds the sole persistence objective
`m03r-v6-persistence/proportional-nav-session-quadratic-one-sided-5bp/v2`.
The legacy generic early-exit term is explicitly inapplicable to all 12 v6
settings; it cannot be combined with, substituted for, or added to the v6
proportional persistence objective.

The route inventory is declarative and fail-closed. Each setting explicitly
reports these missing public production components:

```text
public all-setting training driver
public isolated confidence-head training step
public five-seed ensemble driver
public cause-typed execution adapter
public chronological evaluator adapter
public route receipt writer
```

Consequently, the aggregate route status always has
`launch_authorized = false` until a later generation-qualified implementation
and evidence path replaces those missing-component declarations. Unit-tested
lower-level primitives cannot self-attest a public production route.

The primary holding-mechanism contrast is canonical M03R minus A08. It reports
20/40-bp net active return, information ratio, total Sharpe, drawdown, RMST60,
survival at 10/20/30 sessions, discretionary early-exit mass and cost, plus
performance on predeclared reversal episodes. Canonical M03R is permitted—and
expected—to depart from A08's holding distribution when state information
justifies an earlier exit or a longer hold. A11 separately measures whether
the exact-hold atom adds value beyond the smooth hazard and transaction cost.

## Identity and governance

Every v6 artifact must bind at least:

```text
protocol generation and schema
design and exact v6 setting ID
soft-persistence contract SHA-256
canonical 5-bp soft-persistence coefficient
source archive and container image
PIT data, universe, benchmark, factor and sector manifests
seed/checkpoint ensemble manifest
per-seed confidence calibration manifests
projection/execution contract
inference contract
```

The v6 validator rejects v5 protocol identities. V4 and v5 source, payloads,
receipts, and setting identities remain immutable historical generations.

## Qualification and launch gate

Before a governed real-data/H100 run, the complete v6 production path must
demonstrate:

- differentiable soft-penalty behavior and zero penalty after age 30;
- no sell mask, forced expiry, or implicit duration eligibility check;
- forced-cause exclusion from the persistence loss;
- exact-hold supported-but-optional semantics and the A11 route;
- all 12 settings executable through one identity-bound driver;
- deterministic CPU/CUDA and two-rank parity, exact restart, and receipt replay;
- sealed PIT data, empirical execution, confidence calibration, factor bounds,
  checkpoint selection, and inference contracts.

Until those receipts exist, v6 is a prelaunch scientific architecture, not a
performance claim or an authorized governed H100 experiment.
