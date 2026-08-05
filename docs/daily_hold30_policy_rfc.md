# RFC: Daily-decision policy with a soft 30-session holding target

**Status:** Implementation in progress; no production or large-GPU launch is authorized by this document

**Date:** 2026-08-04

**Mandate:** [ADR-0006](adr/0006-daily-decision-soft-30-session-holding.md)

**Experiment:** [Pre-lockbox Hold-30 alpha mechanism-8 v3](prelockbox_hold30_alpha_mech8_v3.md)

**Superseded implementation generation:**
[mechanism-8 v2](prelockbox_hold30_mech8_v2.md), retained as audit history

## Summary

QuantTrade will make one portfolio decision per eligible trading session,
carry positions continuously, and learn a soft holding-duration preference
centered near 30 trading sessions. The model may revise its view daily, but
ordinary one-day noise should not replace the book. Discretionary early exits
remain possible, and membership-, risk-, or availability-forced exits remain
immediate.

The redesign has five coupled parts:

1. carry portfolio, model, age, and sleeve state across intra-sweep credit and
   loader boundaries;
2. replace inferred holding duration with fill-time age/cohort accounting;
3. budget actual executed discretionary turnover rather than a scalar gate;
4. make per-stock entry and age-aware exit decisions separately from total
   risky exposure; and
5. train on one-session benchmark-relative log utility while keeping duration
   penalties separate from economic P&L.

The target is a soft duration distribution, not a hard 30-session lock. A
planted adverse reversal must still be able to trigger a prompt early exit.

V3 adds the scientific objective contract without changing these economic
mechanics: C1 is the training benchmark, canonical m03 learns 30-session alpha
mean and downside uncertainty under a 2%/4%/6% annual tracking-error band and
beta 1.0 +/- 0.1, and only m03 may promote. C1 anchors actions and active
training; PIT market data is restricted to beta objective/checkpoint/evaluation;
PIT risk-free/CASH is restricted to accounting, a06/a07 total-excess-Sharpe,
and evaluation; declared factors are evaluator-only. Every one of these
artifacts has `policy_feature_access=false`.

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** in this RFC are
normative.

## Evidence and source baseline

The supplied review used `bf067153d3a7de735c453ac9cbacceb28cd601a9` as its
source/code-review baseline. At authoring time this checkout was
`91fafca60cb8339c9fc3bed487dd14298046f3a7`; the relevant legacy daily-policy,
training, dataset, design, and portfolio-environment files were byte-identical
between those commits. The review's behavioral findings therefore apply even
though `bf067153d3` is not the current checkout head.

The current code establishes useful but incomplete semantics:

- [the daily policy](../src/rl_quant/models/daily_policy.py) emits an absolute
  target and scalar gate, while
  [the direct rollout](../src/rl_quant/training/daily_policy.py) performs the
  blend against a separately drifted future-pretrade book;
- [the TOP2000 design](../src/rl_quant/training/designs.py) proves a 252-session
  episode and stride 21; the reviewed external runner used about 21 newly
  scored sessions, but no current package caller certifies that score-tail
  argument until R0's missing driver is bound;
- every direct sampled suffix still starts from cash and has no age state; and
- [the separate PPO workflow](../src/rl_quant/workflows/top2000_ppo.py) has
  63-session samples but resets from cash and terminates each sampled book.

The legacy TOP2000 action budget is especially easy to misread. It compares
the scored mean scalar gate with `12 / 252`; under the reviewed historical
21-session suffix, its penalty kink is about one summed gate unit. It does not
count 12
trades, does not measure executed turnover, and does not establish a holding
duration.

The authoritative legacy Phase-1 driver and launcher also remain outside the
repository worktree. Before another scientific run, their package-owned
replacement MUST be committed and bound to the same source receipt as the
model, accounting, and evaluator code.

The 2026 S0–S7 evaluation informed this RFC, its variant set, and its initial
threshold ranges; it is therefore consumed rather than independent evidence.
After this generation is frozen, it MUST NOT be reread for further
implementation debugging, threshold adaptation, checkpoint selection, or model
choice.

## Goals

- One causal ensemble portfolio decision, action build, and execution per
  eligible trading session.
- Continuous economic state across sessions and intra-sweep credit/replay
  boundaries.
- A measurable soft holding distribution centered near 30 trading sessions.
- Low response to isolated one-session score noise.
- Legal and prompt discretionary exits for sufficiently strong reversals.
- Immediate, separately accounted membership, risk, and availability exits.
- One authoritative execution, reward, turnover, and age-accounting contract.
- A compact shared raw-stock model whose capacity is credible for the number
  of unique decision dates.
- Reconstructable training and evaluation evidence.

## Non-goals

- A hard minimum holding period.
- One decision every 30 days.
- Intraday portfolio replacement.
- An overlapping 30-day production-return reward.
- Reusing 2026 as another lockbox.
- Expanding the old learning-rate, GAE, or turnover-cap grid.
- Deciding PPO versus direct optimization before holding mechanics work.
- Treating repeated sweeps, environments, GPUs, or minibatches as new market
  observations.

## Required invariants

Every conforming implementation MUST satisfy these invariants:

1. `intra_sweep_graph_boundary != economic_terminal`; a complete-update
   replay restart is explicitly declared and is not represented as economic
   continuation.
2. `requested_action != executed_action` whenever a constraint changes intent;
   both are retained.
3. `economic_return != learning_utility`; only economic return updates equity.
4. `startup`, `discretionary`, `membership_forced`,
   `availability_forced`, `risk_forced`, and `terminal` turnover are disjoint
   and reconcile to total turnover.
5. Position age begins at legal fill, not at signal time.
6. Same-name buys and sells are netted before age state changes.
7. Forced exits do not incur discretionary early-exit penalties.
8. No scored or future observation enters policy state before its legal time.
9. A continuing-wealth path never liquidates merely because scoring ends.
10. A deterministic restart from a receipt reproduces model, optimizer,
    portfolio, age, sleeve, pending intent/order metadata, RNG, and
    dataset-cursor state.

## Decision and execution chronology

State is indexed immediately after legal close-`t` fills and transaction cost.
Define three different books:

- `w_decision[t]`: the post-fill book and age summaries visible to the actor at
  decision `t`;
- `w_exec_pre[t+1]`: that old book after the legally subsequent `t -> t+1`
  holding return and corporate actions, immediately before fills; and
- `w_repaired[t+1]`: the fill-time book after mandatory membership,
  availability, and risk repair, used only by the execution adapter.

Execution additionally retains four distinct action/accounting stages:

- `i_raw[t]`: unbounded member outputs and their one aggregated raw intent;
- `delta_constructed[t+1]`: the variant builder's pre-cost net order after
  fill-time remasking, capacity, exposure, and 10% turnover construction;
- `delta_filled[t+1]`: actual pre-cost order notionals after the safety
  constraint layer, with
  `w_pre_cost[t+1] = w_repaired[t+1] + delta_filled[t+1]`; and
- `w_decision[t+1]`: the normalized post-cost economic book.

Cost financing and the resulting normalization from `w_pre_cost` to
`w_decision` are economic accounting, not an additional trade. Discretionary
turnover is `0.5 * ||delta_filled||_1`; requested-to-executed distance is
`0.5 * ||delta_constructed-delta_filled||_1`. Neither metric compares a
pre-cost book with a post-cost book.

The normative cycle is:

```text
legal observations and w_decision[t]
-> evaluate each frozen ensemble member once
-> aggregate one pending intent carrying decision timestamp t
-> old cohorts earn the t -> t+1 return and advance one age
-> form w_exec_pre[t+1]
-> apply fill-time membership and availability repair, then risk repair
-> apply and cross-net the pending discretionary intent
-> charge fills/cost; net purchases enter age 0
-> expose the post-fill book as w_decision[t+1]
```

`exec_delay = 1` remains part of the initial contract. Fill-time availability
or return information MAY change execution but MUST NOT enter the actor input
for decision `t`. The initial implementation therefore needs a new fill-time
availability series or a declared conservative proxy; current decision-time
availability cannot silently be relabeled fill-time data.

Let `M_decision[t]` and `M_fill[t+1]` be point-in-time active-300 membership at
the decision and legal fill events, and let `T_decision[t]` and `T_fill[t+1]`
be point-in-time tradability/availability. Define

```text
A_decision[t] = M_decision[t] ∩ T_decision[t]
A_fill[t+1]   = M_fill[t+1] ∩ T_fill[t+1]
A_trade       = A_decision[t] ∩ A_fill[t+1]
```

Only `A_trade` can receive a discretionary buy. A held name removed from
`M_fill` is liquidated at the first legal fill as `membership_forced`, pays the
common economic cost, contributes a competing-risk exit, and receives no
discretionary early-exit penalty. It is not grandfathered. A name newly added
only at fill has no legally observed policy score and waits until the next
decision. C1 executes its point-in-time reconstitution at the effective fill,
selling deletions and buying additions under the same cost/constraint ledger;
corporate-action handling follows its own point-in-time schedule.

The calendar economic ledger records the return earned by the old book during
`t -> t+1`, the mandatory fills, the pending discretionary fill, and their
costs in that chronological order. Credit from intent `t` reaches its first
post-fill holding return on the next cycle; the direct optimizer obtains that
credit through the continuous trajectory rather than moving future return into
the decision observation.

The net row indexed by decision `t` is the old-book/cash return over
`t -> t+1` minus mandatory and pending-intent fill costs at `t+1`. New fills do
not earn that already-completed return. This preserves the no-lookahead
distinction between decision-visible and future execution books, but it is an
intentional reindexing of the legacy `daily_raw` loss/score row. The optional
R0 fixture must retain the legacy post-fill-return mapping; H0–H3 use only this
new chronology and receive a new result identity.

There is one ensemble decision/action build/execution per eligible session.
Five frozen seed members therefore perform five member forwards but cannot
create five portfolio decisions or a post-fill second look. Tests MUST prove
that no `t+1` return or repair input enters the decision-`t` member forward.

## State ownership and exact resume

ADR-0004 assigns domain mutation and reward to the environment, not all runtime
state. Ownership is explicit:

| Owner | State |
|---|---|
| Environment/shared economic ledger | cash/risky weights, equity, costs, availability/risk repairs, age/cohort state, turnover causes, H3 sleeve books |
| Policy/coordinator | causal token cache or temporal model state and pending member outputs |
| Runtime/checkpoint | optimizer, pending aggregated intent and decision/fill metadata, RNG streams, dataset cursor, update/sweep position |

One resume receipt binds all three owners. A delayed-fill checkpoint is
incomplete without the pending intent, decision timestamp, scheduled fill
timestamp, mask/axis identity, and inputs needed for fill-time repair.

Economic, policy, and runtime state carry through the canonical chronology and
every origin-indexed credit replay. Restoring a recorded origin boundary applies
`stopgrad` to graph state but reproduces the same numeric state; it does not
re-endow or mutate the canonical book. The current temporal encoder is a
Transformer, not a recurrent module with a hidden-state API. At a parameter
update the implementation MUST recompute the permitted causal context under
current parameters or use a proven cache-refresh rule; reusing stale
parameter-dependent keys/values is prohibited.

## Position-age and cohort accounting

### Economic age ledger

For each asset `i`, retain value-notional bins

$$
m_{i,t}^{(a)}, \qquad a \in \{0,1,\ldots,59,60+\}.
$$

Sixty-one bins are used because a terminal `30+` bin cannot resolve a required
20–40-session median-sale-age gate or 60-session survival. The additional state
is small relative to raw second-bar observations.

For the `t -> t+1` cycle, apply return/corporate-action drift to cohorts held
after the close-`t` fill, advance those surviving cohorts by one age, and only
then process close-`t+1` repairs and pending fills. New close-`t+1` purchases
enter age zero and are not advanced for a return they did not earn.

Within the close-`t+1` fill, the order is membership-forced sale,
availability-forced sale, risk-forced sale, then the net discretionary order.
For H2, let
`r_i^(a) = q_i^(a) * m_i^(a)` be proposed cohort release. If the actual net
discretionary sale `x_i` is no greater than `sum_a r_i^(a)`, allocate it across
bins in proportion to `r_i^(a)`. If `x_i` is larger, remove all proposed
release first and remove the residual pro rata from the remaining cohorts.
Same-name purchases cancel proposed release before this attribution; only a net
increment enters age zero. Membership, availability, and risk sales have their
own frozen proportional attribution and never enter the discretionary
early-sale term.

Ledger risky mass MUST reconcile with executed risky holdings within the
declared float tolerance at every decision. Identifier changes carry the age
ledger. Splits preserve age. Cash consideration, delisting, and unavailability
use the forced-exit path rather than silently deleting mass.

### Return-neutral retention ledger

Value-weighted survival can be biased upward when winners grow. Each net
purchase cohort therefore receives **entry-notional units** equal to its
portfolio weight at legal fill. Returns never change those units. A sale removes
the same cohort fraction from entry-notional units that it removes from that
cohort's current economic value; identifier changes and splits preserve them.
Report both:

- discretionary survival, with forced exits treated as competing-risk
  censoring; and
- all-cause survival, in which forced exits count as exits.

Only cohorts purchased during the declared scored interval enter outer
holding-survival gates. Pre-existing endowed cohorts remain useful state but do
not masquerade as experimental entries.

## Target learned action

The preferred H2 actor emits three intent objects:

```text
z[i]   persistent entry/attractiveness score
h[i]   per-stock exit-hazard residual
u_g    risky-exposure residual
```

The deployed hazard residual is `h=clip(raw_h,-12,12)`, not a tanh asymptote;
both endpoints must be attainable at finite raw output. Its derivative is one
strictly inside the interval and zero on/outside either boundary.

The entry score is market-only,
`z[i] = f_entry(e[i,t], c[t])`, so the supervised C5 control has an
unambiguous state contract. The hazard and exposure heads additionally consume
`w_decision[t]`, compact position-age summaries, fraction of notional younger
than 10/20/30 sessions, and fraction at least 30 sessions old. The actor never
receives `w_exec_pre[t+1]`, `w_repaired[t+1]`, or fill-time masks.

### Forced repair precedes discretion

Let `w_repaired[t+1]` be the future drifted portfolio after fill-time
membership, availability, and risk repairs. The execution adapter applies the already
pending decision-`t` intent to this book; the actor does not observe it. Repairs
do not consume the 10% discretionary turnover limit and do not incur an
early-exit penalty.

### Age-aware release

The reference prior for age `a` is

$$
\beta(a) = -2.00 + \frac{\min(a,60)-30}{4}.
$$

For a bounded learned residual, define the attainable hold baseline

$$
p_{\min}(a)=
\sigma\left(\operatorname{clip}(\beta(a)-12,-20,20)\right)
$$

and the normalized release hazard

$$
q_{i,t}^{(a)} =
\frac{
\sigma\left(\operatorname{clip}
(\beta(a)+\operatorname{clip}(h_{i,t},-12,12),-20,20)\right)
-p_{\min}(a)}{1-p_{\min}(a)}.
$$

The pending decision-`t` residual is evaluated against the post-return,
fill-time age bins at `t+1`; those bins are execution-adapter state, not actor
input.

Without same-name reacquisition, zero residual gives a **gross release clock**
with approximately 30.4-session mean and 31-session median. It is not a claim
about executed sale age: if the same asset remains attractive, cross-netting
preserves the cohort rather than manufacturing a sell/rebuy age reset. Actual
duration must pass the holding gates. The bounded value `h=-12` gives
`q=0` exactly for every age and is the finite, state-independent hold action.
The value `h=+12` permits a young position to be released promptly on a strong
reversal. The proposed release is

$$
s_{i,t}=\sum_a q_{i,t}^{(a)}m_{i,t}^{(a)}.
$$

Together with `u_g=0`, `h=-12` is an exact hold without relying on a
coincidental same-name reacquisition. Entry scores are irrelevant when no mass
is released and exposure is unchanged. A proposed release and same-name
reacquisition still cross-net before cost or age mutation.

### Entry and exposure

Center `z` over `A_trade`, clip it to `[-2,2]`, and tilt the frozen
point-in-time benchmark direction `b` over that same mask:

$$
p_i \propto b_i
\exp\left(\operatorname{clip}(z_i-\overline z_{\mathcal A},-2,2)\right).
$$

Separate total risky exposure from stock selection, but keep the mechanism
screen close to C1 so market timing cannot substitute for stock selection. Let
`g_b` be C1 risky exposure at the decision event. At fill time the risk engine
must supply an absolute risky-gross ceiling `g_risk_max` and absolute per-name
ceilings `cap_i_risk`; an explicit no-buy repair is represented by a zero
ceiling. Define

```text
cap_i = min(0.01, cap_i_risk)
g_hard_max = min(1, g_risk_max, sum_i cap_i)
g_band_min = min(max(0, g_b - 0.02), g_hard_max)
g_band_max = min(g_b + 0.02, g_hard_max)
g_min = min(g_repaired, g_band_min)
g_max = max(g_repaired, g_band_max)
```

and then

$$
g^*=\operatorname{clip}
\left(g_{\mathrm{repaired}}+0.10\tanh(u_g),g_{\min},g_{\max}\right).
$$

The minimal envelope `[g_min,g_max]` makes `h=-12,u_g=0` an exact hold even when
the repaired book is temporarily outside the C1 band. A nonzero exposure action
may move such a book toward the band but not farther away. H0/H1 desired
absolute targets use `[g_band_min,g_band_max]` before scalar-gate interpolation,
so their zero gate is likewise hold-neutral. Availability, risk, or name-
capacity shortfall is reported separately; discretion cannot reverse a
mandatory de-risking or fill an unavailable/zero-cap asset.

The deterministic forward builder is:

1. Let `r_i = sum_a q_i^(a) m_i^(a)`, `v_i = w_repaired,i - r_i`, and
   `g_v = sum_i v_i`.
2. If `g* >= g_v`, set `B = g* - g_v`, incremental capacity
   `c_i = max(0, cap_i - v_i)`, and
   `P={i: p_i>0 and c_i>0}`. Let
   `B_eff=min(B,sum_{i in P} c_i)`. If `B_eff=0`, allocate no buy. If
   `B_eff=sum_{i in P} c_i`, allocate every `c_i` in `P`. Otherwise choose the
   smallest `alpha>0` satisfying
   `sum_{i in P} min(c_i,alpha*p_i)=B_eff`, and set
   `d_i=v_i+min(c_i,alpha*p_i)` on `P` and `d_i=v_i` elsewhere. Any unfilled
   `B-B_eff` remains cash.
3. If `g* < g_v`, set `D = g_v - g*` and
   `d_i = v_i - D*v_i/g_v`; when `g_v = 0`, `d_i = 0`.
4. Set `d_cash = 1 - sum_i d_i` and `delta = d - w_repaired` over cash plus
   risky coordinates.
5. Let `tau = 0.5*sum_j abs(delta_j)`. Use `rho = 1` when
   `tau <= 0.10`, otherwise `rho = 0.10/tau`, and request
   `delta_constructed = rho*delta`.

Both endpoints are feasible, so interpolation preserves the simplex and
per-name cap. The equations already create one asset-level net delta; they do
not execute gross release and reacquisition as separate trades. The age-ledger
attribution rule above maps an actual net sell back to proposed cohorts.

For the interior water-fill branch, bisection starts at `lo=0` and
`hi=max_{i in P}(c_i/p_i)`, which saturates every reachable coordinate. Run 100
monotone iterations and return the smallest upper bracket whose mass is at
least `B_eff`. The reference uses float64 sums, absolute mass tolerance
`1e-12`, ascending stable asset ID for exact ties, and cash for infeasible
residual. Its custom backward is zero through the zero/full-cap branch; on the
interior branch it holds the converged active set fixed, uses the implicit
derivative on unsaturated coordinates, and assigns zero derivative to a
coordinate exactly on a cap/tie. `clip`/`min` use zero derivative on the
saturated branch, and `abs(0)` uses derivative zero. These piecewise rules,
plus finite-difference tests away from kinks, are part of the source receipt;
substituting a different solver changes result identity.

Apply the pre-cost `delta_filled` once and charge cost once. Hard projection
remains a safety correction, not the routine action builder. Its
`delta_constructed`-to-`delta_filled` distance and binding causes are mandatory
telemetry; post-cost normalization is stored separately and is never included
in that distance.

During differentiable training, the safety projection MUST be the identity
within `1e-6` one-way distance; a larger change invalidates the transition
rather than using a straight-through estimator. Exogenous availability/risk
masks have zero derivative through removed coordinates and preserve the normal
gradient through surviving coordinates. The analytic quotient rules for
proportional cohort removal use zero output/gradient when their denominator is
zero.

## H1 transitional action

H0/H1 receive decision-time target logits and one scalar adjustment gate. At
fill, remask logits to `A_trade` risky names plus cash, center and clip them to
`[-8,8]`, apply the temperature-0.5 softmax, and retain its risky proportions.
Clamp its desired risky mass to the fill-time risk-aware
`[g_band_min,g_band_max]`, use the same capped water-fill to construct feasible
target `d`, and form

$$
\widetilde w = (1-\eta)w_{\mathrm{repaired}}+\eta d.
$$

Apply the common 10% one-way delta scale to
`widetilde_w - w_repaired`. If the softmax has zero eligible risky mass, use the
C1 direction before water-fill. This exact adapter, including the H2
water-fill/backward and fill-time remask rules, is common to H0 and H1.

H1 retains this absolute target-softmax and scalar adjustment gate
as the common scalar-gate reference. Because H0 and H1 already share corrected
state carry and credit length, `H1 - H0` isolates the package of slow-gate
initialization, entropy removal, and actual-turnover budgeting. H1 initializes

$$
\eta^*=1-e^{-1/30}=0.0327838995,
\qquad \operatorname{logit}(\eta^*)=-3.3844844191.
$$

That initialization is valid only with benchmark endowment or continuous
carry. It MUST NOT be combined with a repeated cash start. Gate entropy is zero
for H1. H1 is an ineligible transitional mechanism control and is not the
target production action.

## H3 structural 30-sleeve comparator

H3 divides the initial portfolio into 30 equal-NAV sleeve accounts with fixed
phases. Sleeve NAVs drift independently and are never recapitalized back to
`1/30`; capital cannot be transferred from a nonmaturing sleeve. Exactly one
sleeve matures per session and reallocates only its own current risky proceeds
and cash. The other 29 sleeves buy-and-drift.

Membership, availability, and aggregate-cap/risk repairs remain immediate. An aggregate
name-cap excess is removed pro rata across sleeves holding that asset and is
labeled risk-forced. Proceeds stay as cash in the affected sleeve until that
sleeve matures. At maturity, it centers/clips its score as H2 does and forms
the same benchmark tilt `p`. Let
`V_s` be that sleeve's current risky-plus-cash NAV and let capacity after the
29 locked sleeves also respect the fill-time risk/name ceilings. The same
frozen water-fill allocates the feasible part of `V_s` by `p`; insufficient
capacity remains sleeve cash. This uses only the maturing sleeve's NAV and has
no exposure head. Aggregate all sleeve orders and cross-net by asset before
cost, then apply the common 10% discretionary one-way cap. If that cap scales a
maturity order, the unexecuted residual remains in its original sleeve and the
event is labeled `maturity_cap_censored`; it is not transferred or silently
treated as a completed 30-session renewal. After the partial fill, the entire
resulting sleeve book starts its next fixed 30-session phase immediately; there
is no next-day retry and no residual sub-sleeve. Unexecuted asset cohorts retain
their economic ages even though the sleeve review phase restarts. Any such event
fails H3's clean structural-duration gate but does not relax the common
portfolio constraint. Same-asset renewal also preserves economic cohort age.

H3 has no ordinary signal-driven pre-maturity exit and therefore does not fully
satisfy ADR-0006. It is an ineligible structural comparator, not an operational
fallback. It reports exact phase spacing, sleeve review age, and asset-level
sale age separately. A later sleeve policy with a signal-emergency exit would
be a new registered mechanism.

## Economic reward and learning objective

Let `R_p,t^net` be the economic policy return after the common execution cost,
and let `R_b,t^net` be the realizable point-in-time benchmark return after its
own common execution cost. The daily learning utility is

$$
u_t = \log(1+R_{p,t}^{\mathrm{net}})
      -\log(1+R_{b,t}^{\mathrm{net}}).
$$

Benchmark subtraction is an accounting/reporting normalization for this direct
optimizer: because C1 is parameter-independent, it does not change the policy
gradient by itself. The simple-to-log transformation, endowment/action
semantics, constraints, and duration terms are the result-moving pieces.
Economic equity still records only `R_p,t^net`; the study MUST NOT claim that
subtracting C1 alone teaches alpha or removes market-exposure incentives.
The common C1 ±2-percentage-point desired-exposure band is the explicit v1
control for that confound.

Define executed discretionary one-way turnover

$$
\tau_t^{\mathrm{disc}}=
\frac12\left\lVert\Delta w_t^{\mathrm{filled,disc}}\right\rVert_1.
$$

Define age-weighted discretionary early-sale mass

$$
e_t=\sum_i\sum_{a=0}^{29}
x_{i,t}^{\mathrm{sell,disc},a}\frac{30-a}{30}.
$$

For H1/H2 on a canonical sequence of `T` loss-bearing anchor decisions, report
the calendar-row diagnostic

$$
J_{\mathrm{calendar}}=\frac1T\sum_t u_t
-\lambda_{\mathrm{turn}}
\left[\frac1T\sum_t\tau_t^{\mathrm{disc}}-\frac1{30}\right]_+^2
-\lambda_{\mathrm{early}}\frac1T\sum_t e_t.
$$

The v1 experiment freezes `lambda_turn = 1.0`. H2 additionally freezes
`lambda_early = 0.002`, equivalent to a 20 bp learning charge for one complete
portfolio-equivalent age-zero discretionary sale. Those terms are not added
to the economic cost ledger. H0 retains legacy gate-budget semantics as a
mechanism control; H3 obtains duration structurally and has neither penalty.

The optimizer does not differentiate this calendar diagnostic end to end. It
uses the origin-indexed finite-credit vector-Jacobian surrogate specified below;
documentation and telemetry MUST NOT call that surrogate the full derivative
of `J_calendar`.

Daily one-session return remains the production control reward. V1 has no
auxiliary return head: adding 5/21/30/63-session labels would require a new
registered representation ablation with frozen targets, coefficients, masking,
and purge. Overlapping 30-session returns MUST NOT replace daily economic
utility or be counted repeatedly as wealth. Reuse of daily utility rows in
origin-indexed VJPs is training credit only; the canonical economic ledger
books each row once.

## Model architecture

H0–H3 share the same compact representation and direct trajectory optimizer so
the first experiment isolates holding mechanics. V1 consumes raw-second-sourced
OHLCV deterministically resampled to 60-second bars over the nominal 23,400-
second U.S. regular-session capacity (390 slots), grouped in 300-second blocks.
On audited early-close dates the tensor retains 390 slots and masks the
post-close suffix. It uses 42 recent raw-bar days, a 63-session causal policy
context, and `raw_norm="level"`, which is causal per-stock/day normalization
with no fitted state. Any additional fitted scaler or normalization buffer is
fit on fold-training data only. Exact calendar, resampler, missing-bar mask,
feature ordering, and normalization hashes are manifest fields; calling this
configuration native one-second input is prohibited.

$$
e_{i,t}=E_\theta(x^{\mathrm{60s}}_{i,t}),
\qquad
c_t=C_\phi(\{e_{j,t}:j\in\mathcal A_t\}),
$$

`E` is one shared causal raw-bar encoder and `C` is permutation invariant
over the eligible cross-section. Variant-specific heads then consume that
common state:

- H0/H1 emit absolute-target logits and one scalar gate;
- H2 emits shared per-stock entry scores and age-aware hazard residuals plus a
  portfolio risky-exposure residual; and
- H3 emits scores for the day's maturing sleeve.

Only H2 requires the full age summaries as actor inputs, although the shared
environment tracks age for every variant. V1 has no critic. The experiment
binds exact counts and caps the complete trainable model at 7 million unique
parameters, with no more than 5 million on the actor path.

PPO, direct-versus-PPO, and factor-covariance exploration are later registered
ablations. They MUST use the same economic state, age primitive, action builder,
and deployed deterministic action. More GPUs do not justify more parameters.

## Stateful chronological training

Training uses deterministic chronological sweeps through each permitted
outer-training block, not randomly overlapping cash-start windows. Let
`A={63,...,a_k-32}` be the anchor decisions and `T=|A|`. A complete optimizer
update has one canonical pass and a batchable set of origin-indexed credit
replays, all with parameters held fixed:

1. Pass A starts from the contemporaneous benchmark endowment, age `0..29`
   initialization, equity one, and zero temporal state. It runs the deployed
   action under `no_grad` through one continuous chronology, records every
   state needed for replay, and computes the complete-anchor mean turnover or
   gate needed by the nonlinear sequence penalty.
2. The first 63 decisions are causal warm-up. Loss-bearing **anchor** decisions
   end 31 state positions before the training boundary. The remaining support
   comprises 30 ordinary deployed decisions plus one terminal observation;
   buys, sells, hazards, exposure changes, and H3 maturities are not masked or
   changed merely because the sweep is ending.
3. For each `t in A`, restore the exact Pass-A state immediately before
   decision `t` with `stopgrad`. Retain autograd only for the policy/model
   computation whose action origin is `t`. Replay ordinary decisions
   `t+1,...,t+30` with the exact raw intents and next policy/cache states
   recorded by Pass A, all under `stopgrad`, and rerun the common economic
   builder on the differentiable replay state. This avoids 30 redundant model
   forwards without changing deployed numeric behavior. Process the terminal
   state at `t+31` without creating another decision. Economic/age state remains
   differentiable with respect to origin `t` throughout the replay.
4. Let `u_tilde[t,r]` be utility row `r` in that replay for
   `r in {t,...,t+30}`. Row `t` contains the origin fill cost; rows
   `t+1,...,t+30` contain exactly 30 post-fill holding returns. Let
   `tau_tilde[t]`, `e_tilde[t]`, `gate_tilde[t]`, and `entropy_tilde[t]` be
   origin-`t` action quantities. No support-origin penalty is included in this
   replay; that date receives its own origin replay if it belongs to `A`.
5. Pass A supplies

   ```text
   mu_tau = mean_A(tau_disc)
   c_tau  = 2*lambda_turn*max(mu_tau - 1/30, 0)
   mu_gate = mean_A(gate)
   c_gate  = 1e-3 * I(mu_gate > 12/252)
   ```

   With inapplicable coefficients set to zero, the maximization direction is

   $$
   \widehat g(\theta)=\frac1T\sum_{t\in A}\nabla_\theta\left[
   \sum_{r=t}^{t+30}\widetilde u_{t,r}(\theta)
   -c_\tau\widetilde\tau_t(\theta)
   -\lambda_{\mathrm{early}}\widetilde e_t(\theta)
   -c_{\mathrm{gate}}\widetilde g_t(\theta)
   +\lambda_{\mathrm{gate\ entropy}}\widetilde H_t(\theta)
   \right].
   $$

   H1/H2 use `c_tau`; only H2 uses `lambda_early`; H0 uses `c_gate` and
   `lambda_gate_entropy=1e-5`; H3 uses only utility. This is the exact gradient
   of the declared local stop-gradient surrogate at unchanged Pass-A
   parameters, not the full untruncated derivative of chronological wealth.
6. Batch up to 32 independent origin windows for GPU efficiency; batching may
   not connect their graphs or alter the formula. Every replayed numeric action,
   state, and utility must match Pass A. Accumulate with the single denominator
   `T`, clip once, and call `optimizer.step()` exactly once after all anchors.

There is no value function or trajectory bootstrap. Restoring a Pass-A state
for overlapping gradient replay does not mutate the canonical economic ledger
and cannot liquidate or re-endow it. Parameters never change in the middle of
an update, so no replay state is generated by stale within-update parameters.
Validation runs only after the completed step.

A new complete optimizer update deliberately replays the same permitted
history from the declared endowment. This preregistered replay restart is not
described as continuous economic operation or independent market evidence.
The paired origin-window schedule is identical across H0–H3 and seeds, and the
artifact records the repeated row count and exact `[t,t+30]` utility mask of
every anchor.

The current stride-21 no-duplicate dataset builder cannot create the anchor,
31-position lifecycle support, and origin-window credit roles by a
configuration-only change. The stateful iterator and replay-boundary snapshot
builder MUST land first.

## Evaluation initialization and terminal semantics

Each validation or outer evaluation begins with 63 strictly earlier causal
warm-up decisions. At warm-up start all variants and controls receive:

- the same contemporaneous point-in-time benchmark holdings;
- equity one;
- deterministic age initialization that divides each risky holding evenly
  across ages `0..29`;
- zero model temporal state; and
- no synthetic initialization trade or cost.

Holdings, age, sleeve state, and model state carry into the scored interval.
Policy and benchmark wealth are separately rebased to one at the score
boundary. Warm-up actions and costs affect the carried state but not the scored
return endpoint.

After 63 scored decisions, retain 31 sealed support state positions: the first
30 contain ordinary follow-through decisions and the last is a terminal
observation that processes its inbound return/fill but creates no new decision,
outbound return, or pending order. They are
excluded from training, checkpoint selection, and economic-return scoring.
They resolve holding outcomes for cohorts whose **originating intent decision**
is in the scored interval, including the last scored intent that fills at the
first support timestamp. That decision-origin rule, rather than fill-date
membership, controls estimand inclusion. Support actions and holding evidence
remain hidden until the one-shot outer reveal. Cohorts originated by support
decisions are tracked for state conservation but excluded from outer holding
estimands. Sixty-session survival is right-censored under the frozen procedure.

The first support timestamp completes the pending fill from the last
scored decision. Its fill and cost belong to that last scored net row; its
outbound market return and newly created follow-through decision do not enter
economic scoring. The terminal support observation cannot create an unfillable
intent beyond the bound axis.

Continuing wealth is primary. Optional liquidation is secondary, costed, and
reported separately. Scoring, support, credit replay, and data-loader
exhaustion MUST NOT themselves liquidate the book.

If the final eligible decision leaves an intent whose legal fill lies beyond
the bound data axis, retain and censor that pending intent; do not synthesize a
fill. The optional liquidation diagnostic first cancels the unfilled intent and
then liquidates the current executed book, reporting both operations.

## Seed ensemble

Every fold's five selected same-window seeds produce one causal economic trace.
Each seed sees the same ensemble holdings and age summaries and retains its own
model temporal state. Aggregate output intent, then construct and execute one
portfolio:

- H0/H1: over eligible risky coordinates plus cash, center each seed's target
  logits, clip to `[-8,8]`, average, apply the frozen temperature-0.5 masked
  softmax once, and take the median scalar gate probability;
- H2: center and clip each entry-score vector to `[-2,2]`, average it, take the
  per-stock median hazard residual in `[-12,12]` and median exposure residual,
  then apply the age prior and portfolio builder once; and
- H3: center and clip each entry-score vector to `[-2,2]`, average it for the
  maturing sleeve, and construct that sleeve once with no exposure output.

Averaging absolute weights or independently executed returns is prohibited.
For seed-sign diagnostics only, each seed also executes an independent causal
portfolio from the common endowment using its own state; it does not receive
the ensemble portfolio. Fold checkpoints from different training cutoffs are
never mixed. Final same-window refits use the identical output-space rule.

## Required telemetry and artifacts

Every training, validation, and outer trace retains:

- raw member and aggregate intent, fill-time repaired book,
  `delta_constructed`, `delta_filled`, pre-cost book, post-cost book, and the
  correctly staged requested-to-executed distance;
- entry scores, hazard residuals/fractions, exposure output, and constraint
  binding causes;
- economic and return-neutral cohort ledgers or sufficient exact
  reconstruction events;
- anchor/origin-window/support masks, utility-row mask, and cohort-censor
  events;
- proposed and actual net buy/sell mass by cause;
- startup, entry, exit, resize, membership-forced, availability-forced,
  risk-forced, and terminal turnover and cost;
- notional-weighted current age and discretionary-sale-age distribution;
- young-sale fractions below 10, 20, and 30 sessions;
- return-neutral survival at 5, 10, 20, 30, and 60 sessions;
- portfolio overlap at lags 5, 10, 20, and 30;
- P&L by position-age bucket;
- approximate turnover-implied horizon, labeled as an approximation;
- H3 sleeve ID, phase, holdings, maturity, `maturity_cap_censored`, and
  cross-net savings; and
- objective terms, gradients, checkpoint rule, stop reason, unique decision
  dates, and exact-resume state.

All result-moving tensors and ledgers are content-addressed. Aggregate metrics
without reconstructable traces are insufficient evidence of holding duration.

## Blocking software tests

The next large run is blocked until all of the following pass:

1. A planted 30–40-session signal produces persistent entries near the target
   duration.
2. An isolated one-session score perturbation causes no material rotation.
3. A sufficiently strong reversal exits promptly despite the early-exit term.
4. Unavailable and membership-deleted assets exit at the first legal execution
   with distinct causes and no early penalty.
5. Initial endowment/startup is not discretionary turnover.
6. Batched independent one-anchor/30-support credit replay matches the
   canonical chronology and gives every anchor exactly 30 post-fill return
   transitions.
7. An intra-sweep graph, replay, loader, or scoring boundary causes no economic
   reset or liquidation.
8. Age-ledger mass and return-neutral cohort units conserve through drift,
   actions, constraints, and corporate actions.
9. Same-name cross-netting cannot reset age.
10. In-memory, streaming, and exact-resume paths produce identical actions,
    state, losses, and gradients within tolerance.
11. No ordinary H3 sleeve is discretionarily rebuilt before its scheduled
    30-transition maturity; forced repairs, cap-censored maturities, and
    same-asset renewals are traced separately.
12. Null data converges toward low discretionary trading.
13. Frozen temporal-return and cross-sectional-return transformations eliminate
    active edge after full retraining and closed-loop evaluation.
14. Observation/execution chronology is demonstrably free of lookahead.

Synthetic thresholds and statistical qualification are frozen in the linked
experiment specification; passing unit tests alone does not authorize scale-up.

## File-level implementation map

| Area | Required change |
|---|---|
| `src/rl_quant/envs/portfolio.py` | Own the age/cohort ledger, turnover causes, configurable endowment, continuous state, optional liquidation, and benchmark-relative utility |
| `src/rl_quant/models/daily_policy.py` | Add age inputs, entry scores, per-stock hazard residuals, and separate risky-exposure output |
| `src/rl_quant/training/daily_policy.py` | Consume the shared accounting primitive; implement canonical sweeps, batched independent origin windows with 30 support decisions, and duration losses |
| `src/rl_quant/training/designs.py` | Add explicit holding, scored-tail, discretionary-turnover, continuation, and terminal fields; deprecate legacy action-count semantics |
| `src/rl_quant/datasets/daily.py` | Build legal stateful sequences, split support, anchor/credit masks, and cohort-censor metadata |
| Production runtime | Move the authoritative Phase-1 driver/launcher into the package and bind it to the executed source receipt |
| Evaluation | Add 63-session warm-up, 30 sealed follow-through decisions plus a terminal observation, continuing wealth, age/survival endpoints, and matched controls |
| Tests | Add every blocker above plus direct/environment trace parity required by [ADR-0004](adr/0004-env-execution-owns-reward.md) |

The age/cohort implementation MUST be one shared, differentiable accounting
primitive. The environment is authoritative. Any temporary direct rollout
adapter must demonstrate exact trace parity rather than maintain a second
interpretation.

## Delivery sequence

1. Package the authoritative runtime and freeze source/economic types.
2. Implement the shared age/cohort ledger and cause-typed turnover.
3. Add benchmark endowment, continuing terminals, canonical state carry, and
   batched one-anchor/30-support origin replay.
4. Add H0/H1 compatibility builders and create R0 only after its complete
   legacy source/configuration/data fixture is bound.
5. Add the H2 hazard/entry/exposure builder and H3 sleeves.
6. Add telemetry, reconstruction artifacts, and all blocker tests.
7. Run systems-only CPU/small-universe profiling without scientific outcomes.
8. Hash the H0–H3 freeze manifest and trial inventory.
9. Run the pre-2026 study once and reveal outer evidence only after receipts are
   complete.
10. Register any later optimizer ablation or untouched lockbox separately.

## Alternatives rejected

- `label_horizon_days = 30` alone: labels do not create portfolio persistence.
- `max_actions_per_day = 252 / 30`: the legacy field penalizes mean gate, not
  actions, turnover, or age.
- `gate_init_bias = -3.38` with cash starts: it creates a slow deployment
  artifact.
- A global gate as the target architecture: it cannot retain most names while
  exiting a small subset cleanly.
- A hard 30-session lock: it prevents legitimate risk and signal exits.
- Overlapping 30-session production rewards: they double-count future returns.
- A terminal `30+` age bin: it cannot resolve the declared sale-age and
  60-session survival endpoints.
- Random overlapping cash-start episodes: they teach deployment and destroy
  position lifecycle evidence.
- More model parameters or GPUs before mechanism validation: compute does not
  add independent market dates.

## Compatibility and result identity

This RFC changes action semantics, portfolio state, training geometry,
learning loss, evaluation carry, telemetry, and ensemble construction. It is a
new result generation. An old checkpoint cannot be relabeled Hold-30, and a
compatibility flag cannot make old evidence conforming.

Every run MUST bind repository URL, commit and tree SHA, clean/patch state,
retained source archive, dependency lock, container image, Python/PyTorch/CUDA/
NCCL/driver versions, design hash, dataset/universe/corporate-action hashes,
fold identities, seed schedule, and evaluator/action receipts. The exact
executed source archive must be retained, not only its digest.

## Implementation completion criteria

This RFC may move from Proposed to Implemented only when:

- every file-level change has landed in the repository-owned runtime;
- current daily and PPO paths either use the shared state contract or are
  explicitly barred from Hold-30 result generation;
- every blocking software test passes in CI;
- a deterministic receipt reproduces a stopped-and-resumed trace;
- the experiment freeze renderer validates all split, state, label, control,
  and artifact invariants in non-executable dry-run mode; and
- documentation no longer describes a scalar-gate or label-only setting as a
  holding-duration implementation.

Scientific qualification remains governed by the H0–H3 experiment and a later
untouched lockbox. RFC implementation is necessary but not evidence of alpha.
