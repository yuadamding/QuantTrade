# M03R v6 soft-persistence active-alpha RFC

**Protocol generation:** `prelockbox-hold30-active-alpha-m03r-v6`
**Schema version:** 6
**Design identity:** `daily_raw_pit300_hold30_m03r_v6`
**Canonical setting:** `M03R-soft-persistence-active-alpha-hold30`
**Status:** implementation qualification; governed PIT real-data/H100 launch blocked
**Supersedes:** immutable M03R v5 without modifying or relabeling v4/v5 artifacts

This protocol is for non-PHI scientific research and software qualification;
it is not a business-production, investment-advice, or deployment contract.

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

The bounded learned hazard retains `-12` as exact zero release and v6 assigns
`+12` the symmetric meaning of exact full discretionary exit. Thus a severe
reversal can close even a young cohort, while a favorable position may remain
open indefinitely. Hard risk and availability repairs execute before the
discretionary hazard and ignore both the persistence cost and exact-hold atom.

## Temporal and economic semantics

| Quantity | Frozen value | Meaning |
|---|---:|---|
| High-resolution trainable raw context | 42 sessions | Recent raw intraday encoding |
| Learned temporal context | 252 sessions | Long learned cross-day context |
| Controlled rollout | 63 sessions | Optimization geometry |
| Economic origin | 30 post-fill returns | Return support for the economic objective, not a holding rule |
| Persistence preference | 30 sessions | Age scale in the soft early-exit penalty only |
| Auxiliary horizons | 5, 21, 30, 63 | Residual-alpha representation targets |
| Evaluation warm-up / score | 63 / 63 sessions | Continuous-book evaluation geometry |

The 30-return economic origin, 30-session auxiliary label, and 30-session
persistence preference are separate content-bound quantities. None implies a
minimum holding duration.

## Soft-persistence objective

For discretionary learned sales of notional `x_a` at age `a`, v6 freezes the
one-sided quadratic age weight

\[
q(a)=\max(0,1-a/30)^2.
\]

The normalized early-exit fraction is

\[
E=\frac{\sum_a x_a q(a)}{\sum_a x_a + 10^{-12}}.
\]

For training completion fraction `u`, the 10% linear warmup is

\[
m(u)=\min(1,u/0.10).
\]

The canonical loss contribution is

\[
L_{persist}=5\times 10^{-4}\,m(u)E,
\]

corresponding to 5 bp per unit at age zero. Only learned discretionary exits
enter this loss. Forced, unavailable, risk-repair, corporate-action, and
terminal sales remain cause-typed telemetry and are excluded.

The coefficient is selected only inside development from the frozen grid
`(2, 5, 10)` bp. The selected value and all input identities must be sealed
before any governed outer evaluation. Holding-duration outcomes remain
diagnostics, not promotion gates.

The holding/economic gradient-norm ratio is reported separately with an
initial diagnostic target band of 5%–15%. It never rescales the loss or gates
promotion automatically. Inner-fold coefficient-response evidence must show a
smooth duration response without assuming that profitability is monotonic in
the holding penalty.

## Checkpoint eligibility and ranking

Hard eligibility is limited to complete evidence, positive 20-bp active
return, nonnegative 40-bp active return, the annual tracking-error ceiling,
the active-beta equivalence bound, and frozen censoring/projection/forced-
turnover quality limits. Survival at 20 or 30 sessions and RMST60 cannot make a
checkpoint eligible or ineligible.

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
selected inner-development coefficient and selection receipt
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
