# M03R v7 active-alpha Hold-30 RFC

**Reserved protocol generation:** `prelockbox-hold30-active-alpha-m03r-v7`  
**Reserved schema version:** 7  
**Reserved design identity:** `daily_raw_pit300_hold30_m03r_v7`  
**Canonical setting:** `M03R-soft-persistence-active-alpha-hold30-v7`  
**Status:** scientific specification with generation-qualified protocol,
persistence-objective, route-inventory, and schedule-rendering primitives;
governed PIT real-data/H100 launch blocked  
**Predecessor:** [immutable M03R v6](prelockbox_hold30_active_alpha_m03r_v6.md)

**Companion documents:**
[canonical experiment](prelockbox_hold30_active_alpha_m03r_v7_experiment.md),
[revision/training guide](m03r_v7_revision_and_training_guide.md), and
[development-only seed-17 diagnostic](top2000_m03r_v7_seed17_diagnostic.md)

This protocol is for non-PHI scientific research and software qualification.
It is not a business-production, investment-advice, or deployment contract.

The generation-qualified protocol, objective, route inventory, and declarative
schedule reserve and validate the v7 identities above. No model checkpoint,
performance artifact, Kubernetes object, or promotion receipt may claim v7
until the remaining model, driver, evaluator, data, GPU, and receipt surfaces
are qualified. V6 artifacts and identities remain unchanged.

## Scientific proposition

V7 asks whether an age-aware, long-context policy can earn positive,
cost-adjusted active return relative to a point-in-time investable benchmark
while retaining acceptable total-portfolio Sharpe and a soft preference for
persistent positions.

Thirty trading sessions is an inductive bias, not a trading rule. The policy
may sell immediately, partially reduce a position, hold without trading, or
continue holding well beyond 30 sessions. Profitability and hard risk repair
always take precedence over the persistence preference.

Only the canonical setting is promotion eligible. Every other primary-panel
row is a causal or diagnostic ablation. An ablation that outperforms canonical
on outer data motivates a new protocol generation and a new untouched test; it
cannot replace canonical within v7.

## Shared canonical contract

Except for the single declared causal change in each setting, all primary-panel
settings share this contract:

| Quantity | Frozen value |
|---|---|
| Universe | Point-in-time active 300 |
| Decision frequency | Once per trading session |
| Primary benchmark | Frozen point-in-time C1 monthly equal-weight buy-and-drift benchmark |
| Recent high-resolution raw context | 42 sessions |
| Learned temporal context | 252 sessions |
| Controlled rollout | 63 sessions |
| Economic credit | Exactly 30 post-fill returns |
| Auxiliary residual horizons | 5, 21, 30, and 63 sessions |
| Training cost | 20 bp per unit one-way turnover |
| Validation cost ladder | 10, 20, and 40 bp |
| Annual tracking-error floor | None |
| Annual tracking-error ceiling | 6% |
| Active-market-beta target | 0 |
| Active-beta equivalence upper bound | 0.10 |
| Maximum stock weight | 1% |
| Holding-age state | Exactly 61 bins, ages 0 through 60; bin 60 accumulates older notional |
| Soft persistence | One-sided quadratic, 5 bp per unit NAV sold at age zero |
| Persistence warm-up | Linear over the first 10% of frozen optimizer updates |
| Exact HOLD action | Supported but optional |
| Exact EXIT action | Available at every age |
| Canonical Sharpe term | None |
| Promotion candidate | Canonical setting only |

The complete optimizer-step count is part of the content-bound training plan.
The warm-up cannot be changed through a free runtime argument under the same
plan identity.

### Correct persistence objective

For policy-discretionary sold notional `x[t,a]`, expressed as a fraction of
portfolio NAV, define

\[
q(a)=\max(0,1-a/30)^2
\]

and

\[
L_{persist}=c_{persist}\,m(u)\,
\frac{1}{T_{valid}}
\sum_{t=1}^{T_{valid}}\sum_{a=0}^{60}x_{t,a}^{disc}q(a).
\]

The canonical coefficient is

\[
c_{persist}=5\times10^{-4}.
\]

There is no denominator based on total sold notional. Consequently:

- a partial young sale pays a proportional cost;
- mature sales cannot dilute the cost of a young sale;
- sales at age 30 or older have zero persistence cost and zero gradient from
  this term;
- forced unavailability, hard risk repair, corporate-action, and terminal
  accounting exits are exempt;
- early discretionary exit remains possible at every age.

The zero- and 10-bp settings receive separate v7 setting identities. They are
not runtime coefficient choices under the canonical identity.

### Confidence and risk sizing

Canonical confidence controls initiation or enlargement of active risk within
a 0%–4% annualized budget. It does not automatically liquidate an existing
feasible active book. Ordinary exits come from the learned exit action;
constraint violations come from cause-typed hard repair.

Confidence calibration follows the frozen two-stage lifecycle described in
[the calibration protocol](m03r_confidence_calibration_protocol.md): train and
freeze the policy, fit the calibrator only from inner-validation standardized
unit-risk outcomes, freeze the calibrator, then perform validation/deployment
without further policy updates.

## Primary twelve-setting inventory

The exact order and stable setting IDs are part of the v7 identity.

| Index | Stable setting ID | Sole causal change from canonical | Scientific question |
|---:|---|---|---|
| 0 | `M03R-soft-persistence-active-alpha-hold30-v7` | None; canonical 5-bp persistence | Can the complete policy deliver active return, acceptable Sharpe, and freely revisable persistent positions? |
| 1 | `P00-no-soft-persistence-v7` | Persistence coefficient 5 bp to 0 bp | Does the explicit cost improve duration and net performance beyond transaction costs, age state, and the structural hazard prior? |
| 2 | `P10-soft-persistence-10bp-v7` | Persistence coefficient 5 bp to 10 bp | Does stronger encouragement help, or suppress valuable early exits and alpha? |
| 3 | `A08-fixed-exit-hazard-v7` | Freeze learned hazard to the structural 30-session prior | Does learned exit timing add value over a fixed persistence clock? |
| 4 | `A11-no-exact-hold-atom-v7` | Remove only the discrete exact-HOLD action | Does the no-trade atom add value beyond the continuous hazard, costs, and persistence term? |
| 5 | `A09-no-long-context-v7` | Learned context 252 to 63 sessions | Does long context improve alpha, regime recognition, and holding decisions? |
| 6 | `M02-active-risk-no-alpha-heads-v7` | Remove residual-return alpha heads | Do the 5/21/30/63 residual heads add stock-selection skill beyond the shared encoder and risk controls? |
| 7 | `A04-no-downside-score-adjustment-v7` | Use predicted mean only | Does downside-aware scoring improve Sharpe and drawdown or attenuate useful signals? |
| 8 | `A12-fixed-2pct-active-risk-budget-v7` | Replace calibrated 0%–4% confidence budget with fixed 2% | Does calibrated confidence sizing improve risk-adjusted return over constant midrange risk? |
| 9 | `A10-no-factor-neutral-projection-v7` | Disable factor/sector-neutral projection only | How much apparent return comes from systematic factor and sector tilts? |
| 10 | `A06-sharpe-overlay-v7` | Add separate total-risk/Sharpe overlay | Can total Sharpe improve without contaminating the alpha core? |
| 11 | `A07-direct-sharpe-v7` | Add full-effective-batch two-pass direct-Sharpe gradient | Does direct Sharpe optimization add value or unstable, selected noise? |

Every row retains the canonical five-basis-point persistence objective except
P00 and P10. Additional implementation semantics needed to keep each contrast
one-dimensional are:

- M02 retains a standalone confidence head and the same two-stage calibration
  lifecycle; it removes only residual-return alpha heads.
- A04 retains calibrated confidence sizing and removes only downside adjustment
  from the cross-sectional score.
- A12 continues computing confidence telemetry but does not let it size active
  risk; its fixed budget is exactly 2% annualized.
- A10 keeps long-only, asset-cap, tracking-error, active-beta, availability,
  and turnover constraints; it removes only declared factor/sector-neutral
  projection.
- A06 reports the unchanged alpha-core path and the overlaid final path
  separately.
- A07 uses exact full-effective-batch statistics through the frozen two-pass
  implementation and remains nonpromotable.

## Controls outside the primary panel

### M01 gradient-null qualification

`M01-benchmark-subtraction-v7` is a software-governance control, not one of the
360 primary training cells. With detached benchmark returns, its gradient and
parameter update must match the absolute-return control under identical input,
initialization, optimizer state, and minibatch order.

Run it for only 2–4 optimizer updates on the same two-H100 topology and bind:

```text
input/minibatch receipt
initial model and optimizer hashes
per-parameter gradient comparison
updated model and optimizer-state comparison
frozen numerical tolerances
```

Model-state and optimizer-state hashes should match when deterministic bitwise
execution is qualified. The complete setting receipt and checkpoint-package
hashes must differ because the setting identity and logged objective value are
different; equality of those complete artifact hashes is not a valid gate.

### A05 reserve diagnostic

`A05-fixed-te-floor-v7` is excluded from the primary panel because it compels
active variance when evidence is weak. It may run only after a predeclared,
inner-development-only reserve trigger establishes that canonical collapses to
near-zero active risk. The exact trigger must be frozen before any governed
data access. If activated, A05 becomes a registered adaptive trial and enters
the multiplicity family. Outer returns may never trigger it.

## Selection and promotion

Holding duration remains secondary. Checkpoint processing follows this order:

1. require complete fold/seed, chronology, manifest, and receipt coverage;
2. require positive 20-bp net active return;
3. require nonnegative 40-bp net active return;
4. enforce the tracking-error ceiling and active-beta equivalence bound;
5. enforce data-integrity, age-ledger, execution, and projection-quality gates;
6. maximize the 95% block-bootstrap lower confidence bound of 20-bp active return;
7. maximize information ratio;
8. maximize total-portfolio Sharpe;
9. minimize maximum drawdown;
10. minimize discretionary turnover and transaction cost;
11. use a one-sided persistence preference only as a late tie-breaker;
12. prefer the earlier checkpoint after exact ties.

Survival, sale age, RMST60, and raw fold censoring are diagnostics, not hard
eligibility or promotion gates. A materially stronger 18- or 45-session policy
must not lose merely because another checkpoint looks closer to 30 sessions.

Only index 0 can be promoted. Any ablation win is hypothesis-generating and
requires a later immutable generation and untouched evaluation period.

## Required evidence and launch blockers

V7 launch remains prohibited until all of the following are implemented and
bound to the exact source, data, container, and protocol identities:

- generation-qualified v7 protocol, objective, route, selection, evaluator,
  plan, and driver surfaces;
- executable one-causal-change paths for all twelve primary settings;
- proportional NAV/session persistence tests for 0-, 5-, and 10-bp settings;
- exact 61-bin, cause-typed chronological ledger and final-ledger receipts;
- public isolated confidence-head training and deterministic post-freeze
  calibration fitting;
- five-seed output-space ensemble construction and one chronological ensemble
  execution per setting/fold;
- point-in-time universe, market data, C1, risk-free, factor, sector, and
  execution manifests;
- empirical execution and the frozen 10/20/40-bp evaluation ladder;
- active-return, Sharpe-difference, active-beta, and active multifactor-alpha
  uncertainty with the sealed multiplicity family;
- deterministic CPU, one-H100, and two-H100 parity; rank invariance; exact
  restart; and mixed-precision tolerances;
- two-H100 throughput, peak-memory, and capacity qualification receipts;
- immutable Kubernetes plan/admitted-spec binding and ownership-scoped recovery;
- M01 short gradient-null qualification and the predeclared A05 reserve trigger;
- fail-closed outer-data access, checkpoint selection, evaluator, and promotion
  receipts.

Passing unit tests or publishing this RFC does not authorize training, GPU
allocation, outer-data access, or promotion.

## Required generation-qualified software surfaces

The table below is the canonical target boundary, not a claim that every row
currently exists. The enforced later-generation source/test registry is
`V3_LATER_GENERATION_SOURCE_TESTS` in
`rl_quant.workflows.hold30_alpha_prelockbox`; future contributors must update
that package-owned registry and its tests rather than relying on this prose as
an executable inventory. The executable TOP2000 compatibility modules have
separate development identities and do not satisfy a missing canonical v7
surface.

V7 should land through explicit public boundaries rather than adding v7 IDs to
v6 modules. The minimum source-to-blocking-test inventory is:

| Planned source | Required blocking test |
|---|---|
| `src/rl_quant/protocol/hold30_alpha_m03r_v7.py` | `tests/test_hold30_alpha_m03r_v7_protocol.py` |
| `src/rl_quant/training/hold30_alpha_m03r_v7.py` | `tests/test_hold30_alpha_m03r_v7_objective.py` |
| `src/rl_quant/training/hold30_alpha_m03r_v7_routes.py` | `tests/test_hold30_alpha_m03r_v7_routes.py` |
| `src/rl_quant/protocol/hold30_alpha_m03r_v7_schedule.py` | `tests/test_hold30_alpha_m03r_v7_schedule.py` |
| `src/rl_quant/training/hold30_alpha_m03r_v7_schedule.py` | `tests/test_hold30_alpha_m03r_v7_schedule.py` |
| `src/rl_quant/training/hold30_alpha_m03r_v7_selection.py` | `tests/test_hold30_alpha_m03r_v7_selection.py` |
| `src/rl_quant/training/hold30_alpha_m03r_v7_plan.py` | `tests/test_hold30_alpha_m03r_v7_plan.py` |
| `src/rl_quant/training/hold30_alpha_m03r_v7_driver.py` | `tests/test_hold30_alpha_m03r_v7_driver.py` |
| `src/rl_quant/evaluation/hold30_alpha_m03r_v7.py` | `tests/test_hold30_alpha_m03r_v7_evaluation.py` |

Shared v6 numerical primitives may be reused internally only when their
semantics are unchanged. Every v7 artifact-producing public boundary must
validate the v7 protocol, design, setting, persistence coefficient, fold/seed
inventory, and source identities.

Both v7 documents and every landed v7 source/test pair must be added to the
two existing Hold-30 qualification inventories. Until those registrations are
content-bound and tested, existing software receipts do not cover v7.
