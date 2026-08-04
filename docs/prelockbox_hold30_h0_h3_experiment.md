# Experiment specification: Pre-lockbox Hold-30 H0–H3 mechanism ablation

**Status:** Proposed and launch-blocked until the implementation RFC is
`Implemented` and every freeze and software gate is complete

**Date:** 2026-08-04

**Protocol generation:** `prelockbox-hold30-h0-h3-v1`

**Mandate:** [ADR-0006](adr/0006-daily-decision-soft-30-session-holding.md)

**Implementation RFC:** [Daily-decision Hold-30 redesign](daily_hold30_policy_rfc.md)

**Supersedes:** unexecuted draft generation `prelockbox-a0-a5-v1`

## Purpose

This experiment asks whether a daily-decision portfolio can carry positions
continuously, exhibit a soft holding duration near 30 trading sessions, and
produce stable net active returns. It isolates holding mechanics before
comparing optimizers or scaling the stock universe.

The experiment compares four mechanisms under one corrected economic-state,
reward, representation, optimizer, split, and execution contract:

- H0: ported scalar-gate baseline;
- H1: minimal slow-hold correction;
- H2: preferred per-stock age-aware hazard policy; and
- H3: structural staggered 30-sleeve comparator.

This specification uses **variant** for H0–H3 collectively and reserves
**candidate** for H2, the only mandate-conforming mechanism eligible to scale.

A planned legacy trace fixture is named R0 and remains blocked on complete
source binding. If created, it is operational evidence only, not a scientific
candidate, and does not share an outer-result family with H0–H3.

This is not another learning-rate, GAE, turnover-cap, architecture, or GPU
search. A null result is valid and stops scale-up.

A non-executable dry-run render is required while the linked RFC remains
`Proposed` so schemas, split arithmetic, and artifact invariants can satisfy the
RFC's implementation criteria. Approving, freezing, signing, or launching an
executable experiment manifest is prohibited until the RFC is `Implemented`.
RFC implementation is a launch prerequisite, not evidence that H2 is
scientifically qualified.

## Scientific questions

The four planned contrasts are:

1. `H1 - H0`: under the shared corrected carry/credit contract, do slow-gate
   initialization, entropy removal, and actual-turnover budgeting remove the
   legacy daily-adjustment behavior?
2. `H2 - H1`: does the H2 package—per-stock age-aware exits, age inputs,
   score-based entries, and separate exposure control—outperform H1's
   portfolio-wide gate?
3. `H3 - H1`: does structural 30-sleeve holding outperform the minimally
   corrected scalar gate?
4. `H2 - H3`: does the complete learned H2 action package add value over an
   ineligible structural duration control?

The first question is a mechanism-package contrast, not a single-coefficient
claim: H1 changes gate initialization, gate entropy, and budget semantics.
Corrected carry and credit are common to H0–H3 and therefore are not identified
by `H1 - H0`. `H2 - H1` and `H2 - H3` are also mechanism-package contrasts;
they do not isolate the hazard from exposure or action geometry. H3 is a
different structural mechanism and cannot qualify for scale-up.

## Evidence boundary

- The reviewed S0–S7 2026 panel informed this v1 variant set, cost ladder,
  and broad threshold ranges. It is therefore consumed, not independent
  evidence.
- After this specification is frozen, no further 2026 observation, result,
  action, checkpoint, or diagnostic may enter implementation debugging,
  feature choice, threshold adaptation, model selection, checkpoint selection,
  or this experiment's result generation.
- Every training, validation, scored, purge, warm-up, label-support, and
  follow-through position MUST be before 2026-01-01.
- Outer evidence is revealed once, only after all variants, controls,
  negative branches, artifacts, and receipts are complete.
- A later confirmatory evaluation requires a separately registered untouched
  or prospective lockbox. This protocol cannot authorize reuse of 2026.
- The independent market sample is the unique scored date. Seeds, sweeps,
  environments, minibatches, and GPUs do not multiply it.

## Freeze checklist

Launch is prohibited until an immutable manifest supplies and hashes every row:

| Binding | Required value |
|---|---|
| Protocol | Generation ID, rendered specification SHA-256, trial inventory, approval state, and RFC status |
| Repository | URL, commit SHA, tree SHA, and clean-worktree flag or dirty-patch SHA-256 |
| Source | Retained exact executed-tree archive, archive SHA-256, and patch against the bound tree when dirty |
| Runtime | Dependency-lock hash, image digest, Python, PyTorch, CUDA, NCCL, driver, and compiler versions |
| Data | Raw snapshot, bars, decision axis, returns, cash series, corporate actions, and their hashes |
| Universe | Point-in-time events, identifier history, eligibility implementation, ordered action axis, and hashes |
| Benchmark | C1 membership, weights, repairs, rebalance events, costs, cash yield, and trace hash |
| Splits | Materialized date/position arrays for all six folds, support, purge, embargo, warm-up, and follow-through |
| Model | Layer graph, parameter counts, normalization state, action adapter, and initialization |
| Training | Optimizer, date/sweep schedule, checkpoint rule, loss normalization, seed-specific RNG streams |
| Execution | Observation cutoff, delay, legal fill time, constraints, netting, cost, and precision/tolerance fields |
| State | Endowment, age/cohort initialization, pending-intent queue, owner boundaries, graph-detach semantics, terminal behavior, and resume schema |
| Controls | Algorithms, matching fields/tolerances, replicate seeds, solver code, and fail-closed behavior |
| Statistics | Estimands, block bootstrap, WRC, SPA, max-T, multiplicity, and RNG source |
| Artifacts | Expected path inventory, schema versions, content hashes, and receipt graph |
| Compute | Systems-only placement and mechanically equivalent sharding; no scientific field inferred from GPU count |

The authoritative Phase-1 driver and launcher MUST be package-owned and present
in the retained source archive. A digest of an external controller script is
not a substitute for version-controlled executed source.

The manifest derives independent deterministic RNG streams for parameter
initialization, chronological sweep order, any action noise, minibatch order,
controls, and bootstrap sampling. Paired variants reuse applicable stream
identities; an undocumented global RNG is prohibited.

## Dataset and point-in-time universe

The initial study uses exactly 300 active U.S. common stocks per monthly
reconstitution. At each event, rank eligible securities by median dollar
volume over the prior 60 completed sessions ending five sessions before the
effective date. Require:

- at least 126 completed sessions;
- an as-traded close of at least USD 5 at the ranking timestamp;
- at least 90% required-bar coverage over the ranking window; and
- point-in-time tradability, identifier, listing, corporate-action, and
  delisting data.

If fewer than 300 securities qualify, that decision axis is invalid. It is not
padded with future members. Future-back-adjusted prices and future-ranked
TOP2000 membership are prohibited.

The model action axis is the stable ordered union of all securities selected at
least once by the frozen pre-2026 monthly process. Inactive or not-yet-eligible
coordinates are masked. No security selected from a 2026-derived ranking may
enter the union.

News is disabled. The study uses the same raw-market observation contract for
all H variants. Normalization is fit from each fold's permitted training data
only.

Let `N = len(session_positions)` after every point-in-time and data-quality
rule. A position normally supports one daily decision and its outbound return;
when designated as the terminal observation of a support interval, it supports
neither. This generation requires `N >= 1811`. Every support position after the
last scored block must remain strictly pre-2026. Failure is closed: do not
reduce folds, holding horizon, warm-up, or support after seeing any result.

## Six-fold walk-forward geometry

Use six expanding outer folds. Freeze:

| Field | Value |
|---|---:|
| Decisions per session | 1 |
| Outer scored decisions | 63 |
| Inner validation decisions | 63 |
| Causal warm-up | 63 unscored decisions |
| T+1/holding/C5-label support | 31 state positions: 30 decisions plus one terminal observation |
| Validation-to-score separation | 31 positions |
| Trailing holding follow-through | 30 sealed decisions plus one terminal observation |
| Embargo | 5 positions after folds 0–4 |
| Outer folds | 6 |
| Paired seeds | `17, 29, 43, 71, 101` |
| Training window | Expanding |

The 31 support positions consist of 30 executable decisions followed by one
terminal observation. For a last scored intent at `d`, the legal fill is at
`d+1`; holding returns run from `d+1 -> d+2` through `d+30 -> d+31`; the state
at `d+31` processes its inbound return/fill but creates no decision, outbound
return, or pending fill. H0–H3 have no auxiliary return heads in v1.

Set

```text
initial_train_size = N - 1339
fold_advance = 224
```

and require `initial_train_size >= 472`. The initial 63 training decisions are
causal warm-up; at least 378 loss-bearing anchor decisions follow; and 31
support positions close the last anchor's lifecycle without changing deployed
behavior. Thus `N >= 1339 + 472 = 1811`.

For fold `k in {0,...,5}`, let `a_k = initial_train_size + 224*k`. Materialize
these half-open position intervals:

| Interval | Positions | Length | Permitted use |
|---|---|---:|---|
| expanding training | `[0, a_k)` | `a_k` | warm-up; anchors `[63,a_k-31)`; 30 ordinary support decisions; terminal observation `a_k-1` |
| train/validation separation | `[a_k, a_k+31)` | 31 | purge and causal validation warm-up only; no H/C5 training loss |
| inner validation | `[a_k+31, a_k+94)` | 63 | checkpoint selection |
| validation holding support / score separation | `[a_k+94, a_k+125)` | 31 | 30 ordinary decisions plus terminal observation for validation holding/C5 labels; also replayed causally for outer warm-up; never outer economic score |
| outer score | `[a_k+125, a_k+188)` | 63 | primary economic evidence |
| outer holding support | `[a_k+188, a_k+219)` | 31 | 30 ordinary decisions plus terminal observation; sealed holding evidence only |
| embargo | `[a_k+219, a_k+224)` | 5 | folds 0–4 only |

Terminal-observation status is trace-role-specific. The same underlying date
may later be replayed as an ordinary causal warm-up decision for a distinct
validation or outer trace; no economic state is shared across those separately
endowed evaluation traces.

Five complete advances use `5 * 224` positions; the final fold uses 219, so
`5*224 + 219 = 1339`. The renderer MUST verify exact non-overlap, chronology,
support, and pre-2026 bounds from materialized arrays rather than trust this
arithmetic alone.

Later expanding folds may train on earlier outer periods only according to this
predeclared schedule. No intermediate outer result is exposed before every fold
and variant is complete.

### Evaluation warm-up and follow-through

Immediately before every validation or outer scored block, replay the frozen
checkpoint for 63 strictly earlier decisions. At warm-up start, endow all
variants and controls except cash-only C0 with:

- the contemporaneous C1 benchmark holdings;
- equity one;
- zero model temporal state;
- every endowed risky position split evenly over ages `0..29`; and
- no synthetic transaction or cost.

H3 maps the same endowment evenly to its 30 sleeve phases. Carry holdings,
cash, neural state, age/cohort state, and sleeve state through warm-up, score,
and follow-through. Rebase policy and benchmark wealth separately to one at the
score boundary.

The support interval contains 30 ordinary follow-through decisions and one
terminal observation. It is excluded from training, checkpoint selection, and
economic-return scoring. Holding estimands include cohorts whose originating
intent decision is one of the 63 scored decisions, including a last-scored
intent that legally fills at the first support timestamp. Cohorts originated by
support decisions remain in conservation state but are excluded from outer
holding estimands. Support observations, actions, and metrics remain hidden
until the one-shot reveal. Sixty-session survival uses the frozen
right-censoring procedure.

The first support timestamp closes the last scored net row under the RFC
chronology: include the inbound old-book/cash return, every mandatory
membership/availability/risk repair and its cost, and the pending last-scored
discretionary fill/cost. Exclude the new support decision and its outbound
return. The last support position processes its inbound return/fill, is
otherwise observation-only, and cannot create a pending intent beyond the
bound axis. No later support-timestamp economic field enters the scored
endpoint.

## Common economic contract

H0–H3 and controls share:

- one ensemble portfolio decision/action build/execution per session, with one
  forward per frozen member and no post-fill second inference;
- the RFC's distinct decision-visible, future-pretrade, and fill-time-repaired
  books plus a receipt-bound pending T+1 intent;
- point-in-time membership `M` and tradability `T` masks with
  `A_decision=M_decision∩T_decision`, `A_fill=M_fill∩T_fill`, and
  discretionary-buy mask `A_trade=A_decision∩A_fill`;
- membership deletions liquidated at the first legal fill as
  `membership_forced`, costed and competing-risk counted but exempt from the
  discretionary early-exit loss; fill-only additions wait for a legally
  observed next-decision score;
- C1 reconstituted at the effective fill under the common point-in-time
  membership, cost, and constraint ledger;
- one environment-owned economic, cost, constraint, turnover, age, and cohort
  state implementation;
- long-only weights, gross risky exposure at most 1.0, per-stock weight at most
  1%, and fill-time risk-engine gross/name ceilings that discretion cannot
  reverse;
- H0–H2 nonneutral exposure targets within two percentage points of C1,
  clipped by fill-time risk/availability/capacity, with the repaired exposure
  admitted only to preserve an exact hold and no move farther from the band;
- discretionary one-way turnover at most 10% per decision;
- identical initial endowment and warm-up dates;
- a required point-in-time cash-return series;
- continuing wealth as the primary terminal convention; and
- 10, 20, and 40 bp all-in linear one-way cost rungs, with 20 bp primary.

An intent whose legal fill is beyond the bound axis is retained as censored and
unfilled. Continuing wealth does not charge it. The optional liquidation
diagnostic cancels it first and then liquidates only the executed book.

The v1 screen uses `cost_t = rung * executed_one_way_turnover_t` under the
common fill chronology. Availability/risk repairs incur the same economic cost
but have their own turnover cause. A later full-size run additionally requires
the second-bar latency, participation, partial-fill, spread, and nonlinear-
impact simulator described by the RFC; systems work cannot retroactively
reinterpret this v1 cost ladder.

Every trace persists, in order, raw member/ensemble intent, the fill-time
repaired book, the variant-constructed pre-cost delta, the safety-constrained
filled delta, the pre-cost book, cost financing, and the normalized post-cost
book. Discretionary turnover is half the L1 norm of the filled pre-cost delta;
requested-to-executed distance compares constructed and filled deltas. The
10% action-builder scale is not safety projection, and post-cost normalization
is neither turnover nor projection distance.

All simplex, sleeve, economic-age, and entry-notional ledger reconciliations
use absolute tolerance `1e-6` in portfolio-weight units. Reported wealth and
statistical aggregation use float64; tolerance failure is never rounded away.

The benchmark C1 is monthly point-in-time equal weight over the active 300 and
buy-and-drifts between reconstitutions except for mandatory repairs. Scheduled
and forced benchmark trades use the common cost model. Freeze separately:

1. the benchmark economic-return trace;
2. its complete post-event portfolio anchor, including cash; and
3. the point-in-time membership/availability event schedule.

Daily rebalanced point-in-time equal weight is C2. C0 remains cash-only and
uses the same cash-return series.

The primary 20-bp run creates the canonical closed-loop action and state trace.
After it is sealed, 10/40-bp primary stresses reprice that exact requested-
intent trace under their cost rung; the actor still receives the canonical
20-bp state, so stress costs cannot change its decisions. Optional cost-specific
closed-loop reruns are diagnostic, separately hashed, and cannot affect gates.

Economic equity records portfolio net return only. The common learning utility
for every H variant is

$$
u_t=\log(1+R_{p,t}^{\mathrm{net}})
    -\log(1+R_{C1,t}^{\mathrm{net}}).
$$

Because C1 is parameter-independent, subtracting its log return does not change
the direct policy gradient. It normalizes accounting and inference; the log
transform, action/endowment contract, constraints, and duration terms are the
result-moving mechanisms. No report may claim that benchmark subtraction alone
teaches alpha or removes market-exposure incentives.
The common C1 ±2-percentage-point desired-exposure band is the explicit v1
control for that confound.

All H variants use direct differentiable trajectory optimization and the
same compact shared raw-bar stock encoder and pooled market context. V1 uses
raw-second-sourced OHLCV resampled to 60-second bars, as frozen below. This
holds reward, representation, and optimizer fixed while holding mechanics
change.

## Stateful training contract

For every fold, let `A_k={63,...,a_k-32}` and `T_k=|A_k|`. One optimizer update
uses a canonical chronological pass plus origin-indexed finite-credit replays
of `[0,a_k)`, with parameters fixed until the update ends. Partition the
canonical role map as:

```text
[0, 63)             causal warm-up; no anchor loss
[63, a_k-31)        loss-bearing anchor decisions
[a_k-31, a_k-1)     30 ordinary deployed support decisions; no own anchor loss
{a_k-1}             terminal observation; no decision or outbound return
```

There is no terminal entry mask or buy-disabled wind-down. Support decisions
use the same deployed action as any other date; new support-origin cohorts are
right-censored and are not loss-bearing anchors.

Each update executes:

```text
theta = current parameters

Pass A (canonical, no_grad):
  reset benchmark endowment, age 0..29, sleeves, equity, and temporal state
  run one continuous deployed-policy chronology through the terminal state
  record every replay boundary, anchor mean, and economic/action-state hash

Pass B (origin-indexed gradient replay):
  batch at most 32 independent anchors for throughput
  for each origin t in the batch:
    restore the exact Pass-A state before t with state stop-gradient
    retain autograd only for the policy/model action originated at t
    reuse Pass-A raw intents and next policy/cache states for t+1,...,t+30
    stop-gradient those support records and rerun only the economic builder
    process terminal state t+31 without creating a decision
    keep economic/age propagation differentiable with respect to origin t
    verify every replayed numeric action, state, and utility against Pass A
  accumulate all origin contributions with denominator T_k
  clip the accumulated finite-credit surrogate gradient once
  optimizer.step() exactly once
```

For origin `t`, define `u_tilde[t,r]` as the replayed daily utility row for
`r=t,...,t+30`. Row `t` contains the origin fill cost; rows `t+1,...,t+30`
contain exactly 30 post-fill holding returns. Define
`tau_tilde[t]`, `e_tilde[t]`, `gate_tilde[t]`, and `entropy_tilde[t]` from the
origin action only. A support-origin penalty is excluded from this window and
is included only in that date's own replay when the date belongs to `A_k`.

For H1/H2, Pass A computes `mu_tau` over anchors and freezes

```text
c_tau = 2 * lambda_turn * max(mu_tau - 1/30, 0)
```

H0 freezes `c_gate=1e-3*I(mean_A(gate)>12/252)`. With inapplicable coefficients
zero, the exact maximization direction for the declared local surrogate is

$$
\widehat g(\theta)=\frac1{T_k}\sum_{t\in A_k}\nabla_\theta\left[
\sum_{r=t}^{t+30}\widetilde u_{t,r}(\theta)
-c_\tau\widetilde\tau_t(\theta)
-\lambda_{\mathrm{early}}\widetilde e_t(\theta)
-c_{\mathrm{gate}}\widetilde g_t(\theta)
+\lambda_{\mathrm{gate\ entropy}}\widetilde H_t(\theta)
\right].
$$

H1/H2 use `c_tau`; only H2 uses `lambda_early=0.002`; H0 uses
`c_gate` and `lambda_gate_entropy=1e-5`; H3 uses utility only. This is the exact
gradient at unchanged Pass-A parameters of the origin-indexed stop-gradient
surrogate, not the full derivative of chronological wealth.

There is no critic, GAE, or value bootstrap. A credit-replay boundary restores
canonical numeric state but never mutates, re-endows, or liquidates the Pass-A
economic path. A new optimizer update deliberately starts a new deterministic
replay from the declared endowment; that restart is not called independent
evidence or live economic continuation. Validation occurs only after the one
optimizer step and cannot mutate the next update.

The origin/support schedule is identical across variants and paired seeds. The
manifest reports every repeated row and exact `[t,t+30]` utility mask; replays
and GPUs never multiply the independent market sample.

For H1/H2, the canonical sequence separately reports the **per-anchor
calendar-row** diagnostic

$$
J_{\mathrm{calendar}}=\overline u
-\lambda_{\mathrm{turn}}
 [\overline{\tau^{\mathrm{disc}}}-1/30]_+^2
-\lambda_{\mathrm{early}}\overline e.
$$

This diagnostic is not differentiated end to end. The formula for
`g_hat(theta)`, the common `T_k` denominator, and the frozen Pass-A coefficients
are result identity. A per-window penalty, mid-update parameter change, or
different origin weighting is prohibited.

## Variant and reproduction matrix

| ID | Holding/action mechanism | Duration terms | Eligibility |
|---|---|---|---|
| R0 | Planned legacy 21-session/cash-start/scalar-gate fixture, blocked until one combined source/config/data archive is bound | Exact retained legacy objective | Operational compatibility only; no outer score |
| H0 | Absolute target-softmax plus one portfolio scalar gate under common continuous state | Legacy gate budget and entropy | Mechanism control; not eligible for scale-up |
| H1 | H0 actor with slow gate initialization and actual executed turnover budget | Excess-turnover penalty | Transitional mechanism control; not eligible for scale-up |
| H2 | Per-stock entry score and age-aware exit hazard plus risky-exposure head | Excess-turnover and early-exit penalties | Preferred learned candidate |
| H3 | Thirty staggered, daily-maturing sleeves | Structural duration; no duration penalty | Ineligible structural control |

R0 does not yet exist as a certifiable reproduction. `bf067153d3` omits the
external Phase-1 driver and launcher, and the historical training bundle is not
currently mapped to an exact Git tree plus dirty patch. R0 may be created only
after source, driver, launcher, configuration, dependency, and fixture-data
identity are retained together. It may verify named legacy semantics; it MUST
NOT claim reproduction of original S0–S7 training, read H outer dates, enter
planned contrasts, or be reported as Hold-30 evidence.

The core learned inventory contains `4 variants * 6 folds * 5 seeds = 120`
training runs. Controls and registered negative/positive branches add runs
listed in the manifest. The receipt gate derives expected completion from that
inventory rather than a hard-coded job count.

### H0: ported scalar-gate baseline

H0 uses the common benchmark-relative log utility, compact encoder, stateful
training, endowment, warm-up, and evaluation contract. It retains:

- a newly generated absolute target-softmax every decision;
- one portfolio-wide scalar gate;
- gate bias `2.0`, giving a nominal initial interpolation probability near
  88.1% before state-dependent head output and the common turnover cap;
- a linear legacy gate penalty
  `1e-3 * [mean(gate) - 12/252]_+`; and
- gate-entropy bonus coefficient `1e-5`.

H0/H1 use the RFC's exact fill-time remask, centered/clipped `[-8,8]`
target logits, temperature-0.5 softmax,
C1 ±2-percentage-point exposure clamp, capped water-fill, scalar interpolation,
and 10% turnover scaling. Thus the planned contrast does not hide a different
constraint adapter.

H0 is deliberately not an exact reproduction; the still-blocked R0 fixture has
that role. H0 permits the
planned H1 contrast under a common state/economic contract.

### H1: minimal slow-hold correction

H1 differs from H0 only in the holding controls:

- initialize `eta = 1 - exp(-1/30) = 0.0327838995` with logit
  `-3.3844844191`;
- set gate-entropy coefficient to zero;
- remove the mean-gate budget; and
- add
  `lambda_turn * [mean(tau_disc) - 1/30]_+^2` with
  `lambda_turn = 1.0`.

`tau_disc` is `0.5*||delta_filled_disc||_1` after forced repair and before cost
normalization. Startup, membership-forced, availability-forced, risk-forced,
and terminal turnover are excluded. The common endowment and carry are
mandatory; a low gate with a cash start is not H1.

### H2: preferred age-aware policy

H2 implements the RFC's per-stock entry score `z_i`, exit-hazard residual
`h_i`, and risky-exposure residual `u_g` with the shared 61-bin age/cohort
ledger.

For age bin `a`, freeze

$$
\beta(a)=-2.00+\frac{\min(a,60)-30}{4},
\qquad
p_{\min}(a)=\sigma(\operatorname{clip}(\beta(a)-12,-20,20)),
$$

$$
q_i^{(a)}=
\frac{
\sigma\left(
\operatorname{clip}(\beta(a)+\operatorname{clip}(h_i,-12,12),-20,20)
\right)-p_{\min}(a)}{1-p_{\min}(a)}.
$$

Without same-name reacquisition, the zero-residual **gross release clock** has
approximately 30.4-session mean and 31-session median. It is not the executed
sale distribution: same-name reacquisition cross-nets to zero and preserves
cohort age. `h=-12` gives an exact zero-release hold action for every age.
Proposed release is

$$
s_i=\sum_a q_i^{(a)}m_i^{(a)}.
$$

Entry proportions use the C1 anchor:

$$
p_i\propto b_i
\exp(\operatorname{clip}(z_i-\overline z_{\mathcal A},-2,2)).
$$

Desired exposure is

$$
g^*=\operatorname{clip}
(g_{\mathrm{repaired}}+0.10\tanh(u_g),g_{\min},g_{\max}),
$$

where fill-time risk outputs define

```text
cap_i = min(0.01, cap_i_risk)
g_hard_max = min(1, g_risk_max, sum_i cap_i)
g_band_min = min(max(0, g_C1 - 0.02), g_hard_max)
g_band_max = min(g_C1 + 0.02, g_hard_max)
g_min = min(g_repaired, g_band_min)
g_max = max(g_repaired, g_band_max)
```

The H2 envelope makes `h=-12,u_g=0` exactly hold-neutral; a repaired book
outside the C1 band may move toward but not farther away from it. H0/H1 target
exposure uses `[g_band_min,g_band_max]` before gate interpolation, so gate zero
is also hold-neutral. A point-in-time availability/risk/capacity shortfall stays
cash and is labeled; discretion cannot reverse the repair or relax a name cap.

Use the RFC's exact survivor, exposure, capacity-aware water-fill, net-delta,
and turnover-scaling equations, including its frozen forward/backward rules.
Net same-name orders before updating age. Attribute an actual net sell first to
the proposed `q*m` cohort releases in proportion to those releases; only a sale
beyond proposed release removes remaining cohorts pro rata. Only an actual net
buy enters age zero.

H2 uses the H1 turnover term and

$$
\lambda_{\mathrm{early}}
\overline{
\sum_i\sum_{a<30}
x_{i}^{\mathrm{sell,disc},a}\frac{30-a}{30}}
$$

with `lambda_early = 0.002`. This is a learning-only 20 bp charge per complete
portfolio-equivalent age-zero sale. It is never booked as economic cost.

### H3: staggered 30-sleeve comparator

H3 divides initial portfolio NAV equally across 30 deterministic sleeve
accounts. Sleeve NAVs drift and are never recapitalized. One sleeve matures per
session and reallocates only its own current risky proceeds and cash; the other
29 cannot fund it. Forced-exit proceeds remain cash in their originating sleeve
until maturity. Aggregate name/risk-cap excess is removed pro rata from sleeves
holding the asset and labeled risk-forced. At maturity, allocate over aggregate residual name capacity
after locked sleeves and all fill-time risk ceilings: center/clip entry scores to `[-2,2]`, form the RFC's C1
benchmark tilt `p`, and use its frozen water-fill for
`min(current sleeve NAV, total residual capacity)`. Leave infeasible mass as
sleeve cash. There is no exposure head and no transfer from a locked sleeve.

Aggregate sleeve orders, cross-net same-name flow, apply the common 10%
discretionary one-way cap, and execute once. If the cap scales a maturity order,
leave the residual in its original sleeve and record `maturity_cap_censored`;
do not transfer it or pretend it completed a 30-session renewal. The entire
post-fill sleeve book immediately starts its next fixed 30-session phase; there
is no next-day retry or residual sub-sleeve, and unexecuted cohorts retain their
economic ages. That event fails H3's clean structural-duration gate. Same-asset
renewal preserves economic cohort age. H3 has no turnover or early-exit learning penalty and no
ordinary signal-driven pre-maturity exit, so it cannot satisfy ADR-0006 and
cannot qualify. Report phase spacing, sleeve review age, asset-level sale age,
forced repairs, cap censoring, and cross-net savings separately.

## Common model and optimizer

All H variants use the same compact raw-market representation:

- raw-second-sourced OHLCV resampled deterministically to 60-second bars;
- the nominal 23,400-second regular-session capacity, 390 slots, and 300-second
  encoder blocks; early closes retain the grid and mask the post-close suffix;
- 42 recent raw-bar days and a 63-session causal policy context;
- causal per-stock/day `raw_norm="level"` with no fitted state; any additional
  fitted scaler/buffer uses fold-training data only;
- one shared causal per-stock raw-bar encoder;
- one permutation-invariant market-context encoder;
- width-64 stock state and width-128 causal/context blocks;
- two layers, four heads, width-256 feed-forward sublayers;
- 63-session causal token context and zero dropout;
- at most 5 million actor-path parameters; and
- at most 7 million total unique trainable parameters.

The manifest binds the exchange calendar and timezone, 09:30–16:00 regular-
session grid, OHLCV aggregation rules, missing-bar mask/fill policy, raw feature
order, resampler implementation hash, and fitted normalization buffers. V1 is
not described as a native one-second model.

The manifest records exact shared, head-exclusive, actor-path, and total counts
without double-counting. H0/H1 use a target and scalar-gate head; H2 uses entry,
hazard, and exposure heads; H3 uses the entry head and sleeve state. No variant
may be enlarged after outer access.

H2 parameterizes `h = clip(raw_h,-12,12)`, so both exact-hold and
maximum-release endpoints are attainable at finite raw output. Its entry,
raw-hazard, and exposure final weights use gain `1e-3` and zero bias, giving
zero residual intent at initialization; zero hazard residual invokes the age
prior and is not the exact hold action. The clamp derivative is one strictly
inside `(-12,12)` and zero on/outside either boundary. H3 uses the same entry-head
initialization. H0/H1 target-logit
heads use zero bias and gain `1e-3`; their gate biases are the variant values
below. The initialization algorithm and every tensor shape are manifest fields.

Freeze the training budget:

| Field | H0–H3 |
|---|---:|
| Optimizer | AdamW, direct differentiable trajectory objective |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Adam epsilon | `1e-5` |
| Maximum updates | 128 |
| Passes per update | 1 canonical no-grad sweep plus frozen origin-indexed credit replays |
| Credit replay | one attached anchor plus 30 support decisions and one terminal state; batch at most 32 independent origins |
| Initial causal warm-up | 63 decisions, loss masked |
| Validation cadence | every 8 updates |
| Minimum updates | 32 unless numerical failure |
| Validation patience | 4 validations |
| Gradient norm | at most `0.5` |
| Precision | bfloat16 encoder; float32 action/loss/accounting; float64 final aggregation |

Validate the deterministic deployed action. Select the earliest checkpoint
within one basis point of maximum inner-validation continuing 20-bp active log
wealth. Break ties by lower discretionary turnover, then lexical checkpoint
ID. Retain initial, selected, and final checkpoints and one stop reason.

Non-finite loss, gradient, intent, executed action, ledger, or state invalidates
the attempt. A failed run is retained in the append-only registry and is not
silently replaced with a new seed or budget.

## Exact seed ensemble

For each fold, the five selected same-window seeds construct one causal
portfolio/state/age trace. Each seed receives the shared ensemble holdings and
age summaries, while retaining its own causal token state.

- H0/H1: center each seed's logits over eligible risky coordinates plus cash,
  clip to `[-8,8]`, average, apply one masked softmax at temperature 0.5, and
  take the median scalar gate probability. H0/H1 have no exposure output.
- H2: center and clip each entry-score vector to `[-2,2]`, average it, take the
  per-stock median hazard residual clipped to `[-12,12]` and median unbounded
  exposure residual, then apply the deterministic builder once.
- H3: center and clip each entry-score vector to `[-2,2]`, average it for the
  maturing sleeve, and build once with no exposure output.

Qualification uses this exact trace. A median of independently executed daily
returns and an average of absolute portfolio weights are not realizable
ensembles and are prohibited. Individual seed traces remain required
robustness diagnostics. For those diagnostics, every seed executes its own
portfolio from the common endowment and receives its own holdings/age state; it
does not receive the ensemble state. Fold checkpoints from different training
cutoffs are never mixed.

## Controls

Evaluate these controls on every fold and cost rung:

| ID | Control |
|---|---|
| C0 | Cash plus the frozen point-in-time cash-return series |
| C1 | Monthly PIT equal weight, buy-and-drift between events, with mandatory repairs |
| C2 | Daily rebalanced PIT equal weight under the common execution contract |
| C3 | Initial-universe hold-until-forced-exit; forced proceeds remain cash |
| C4 | Frozen 21-session momentum ranker through the canonical H2 builder |
| C5 | Frozen supervised 30-session excess-return ranker through the canonical H2 builder |
| C6 | 64 deterministic time-permuted empirical-action diagnostics; noncausal and nondeployable |
| C7 | 64 deterministic random controls using the learned variant's same age/holding mechanism |
| C8 | 64 feasible random controls matched to the learned variant's risk, turnover, and holding profile |

C4 is the within-date z-score of trailing 21-session split-adjusted log return,
winsorized to `[-2,2]`, using only closes legally available at decision time.
Missing/unavailable names are masked; if fewer than two valid names exist or
cross-sectional variance is zero, every valid score is zero. Stable asset ID
breaks exact ranks. C4 sets H2 hazard and exposure residuals to zero.

C5 predicts the 30-session buy-and-hold stock log return beginning at the first
legal T+1 fill, minus C1 log return over the identical interval, including
delisting/availability outcomes. For each training date, a hash-derived
schedule selects 1,024 eligible unordered stock pairs. C5 minimizes pairwise
logistic loss on the sign of their label difference. It uses the common encoder
and H2's market-only `f_entry(e_i,c)` score head; no holdings, age, cash, or
pending-intent state enters that head. It sets hazard/exposure residuals to zero
and uses AdamW with
learning rate `3e-4`, weight decay `1e-4`, epsilon `1e-5`, 128-update ceiling,
validation every eight updates, and four-validation patience. A labeled date
is eligible only when its complete 31-position execution/return support lies
inside its permitted training or validation role; no label crosses a split.
All five seeds validate at the same update
indices. At each index construct the exact five-seed H2-builder ensemble from
the five same-index checkpoints; select one shared update by the common
earliest-within-one-basis-point rule, and retain that update from every seed.
There is no search over combinations of seed checkpoints. Ensemble patience
stops all five seeds together. Pair schedules, labels, masks, normalization,
and checkpoints are hashed before outer access.

One C5 update uses exactly 16 distinct training dates from a hash-derived
cyclic permutation. A date is not repeated until that fold's permitted dates
are exhausted and the next hash-derived permutation begins. Process four
four-date microbatches, accumulate the pairwise loss as total valid-pair loss
divided by total valid pairs, clip the accumulated gradient at 0.5, and call
`optimizer.step()` once. Invalid pairs are masked, never replaced after their
outcomes are known. The manifest reports unique dates and completed cycles.

C4 and the five-seed C5 score ensemble each produce exactly one canonical trace
in the WRC/SPA family; there are no learned-variant-builder-specific C4/C5 traces.
C7/C8 are generated separately for each H1/H2 learned variant and fold using
that variant's holding mechanism.

The sealed evaluator generates one bank of 8,192 closed-loop random traces per
H1/H2 variant/fold. Replicate `r` uses a counter-based RNG keyed by
`SHA256(protocol || variant || fold || r)` and cannot read returns. Per-stock
latents follow
`x_t=rho*x_{t-1}+sqrt(1-rho^2)*epsilon_t`. Hash-derived choices come from the
following frozen arrays:

```text
rho                 = [0.90, 0.95, 0.98, 0.99]
score_scale         = [0.25, 0.50, 1.00, 2.00]
H1 gate_mean        = [0.01, 0.02, 1/30, 0.05, 0.08]
H1 gate_logit_noise = [0.10, 0.30, 0.60]
H2 hazard_center    = [-4, -2, 0, 2, 4]
H2 hazard_scale     = [0.50, 1.00, 2.00, 4.00]
H2 exposure_scale   = [0.00, 0.25, 0.50, 1.00]
```

H1 maps the stock latent to centered target logits and an independent scalar
AR(1) latent to `logit(gate_mean)+gate_logit_noise*y_t`. H2 maps independent
latents to centered entry scores,
`clip(hazard_center+hazard_scale*y_i,t,-12,12)`, and
`exposure_scale*v_t`. Each trace then uses the exact learned-variant builder,
constraints, costs, state, and chronology. C7 is the first 64 replicate IDs.

C8 is selected from IDs 64–8191 only after the canonical 20-bp H trace is
sealed.
Filter by every tolerance below, then rank feasible controls by the sum of
squared tolerance-normalized gaps, with stable replicate ID as tie-break; take
the first 64. The selection implementation has an explicit field allowlist and
MUST NOT read mean return, terminal wealth, active return, information ratio,
drawdown, p-values, or performance rank. It may read realized return/covariance
data only to compute the named beta and tracking-error gaps. The learned-variant
summary, bank identity, every rejected reason, ranking distance, and selected
IDs are retained. Fewer than 64 feasible controls fails the matched-control
gate; no tolerance, bank size, or generator field may be changed after reveal.

Cross-fold paths are deterministic. C7 aggregate path `r` concatenates bank ID
`r` from each of the six folds. For C8, let `L_f[j]` be fold `f`'s `j`-th
selected control after the frozen distance/ID ordering; aggregate path `j`
concatenates `L_0[j],...,L_5[j]`. Common bank IDs across folds are not required,
and controls are never re-ranked by return. Persist and hash the complete
`(aggregate_path, fold, bank_id, distance_rank)` mapping before computing the
64 aggregate control wealth values used by the gate.

C8 matches:

- discretionary turnover within 5% relative;
- risky exposure within one percentage point;
- beta within 0.05;
- tracking error and HHI within 10% relative;
- return-neutral entry-notional-unit-weighted median discretionary sale age
  within three sessions;
- 30-session discretionary survival within five percentage points; and
- point-in-time sector/liquidity allocation within the frozen tolerance when
  those classifications are available.

Before C8 generation, each H1/H2 variant/fold must contain at least 0.10
portfolio-equivalent discretionary sold notional and nonzero return-neutral
cohort mass at risk at age 30. Otherwise fold-level median sale age or `S(30)`
is not estimable and the learned variant's matched-control gate fails. No dimension
is dropped or imputed. Every accepted C8 replicate must satisfy the same
estimability condition.

The filter/ranker may use realized return/covariance data only for the named
beta and tracking-error distances. It MUST NOT optimize mean return, wealth,
return rank, or outperformance. Failure to produce all 64 controls under frozen
tolerances fails that variant's matched-control gate.

C6 permutes an already realized empirical intent sequence across timestamps and
is deliberately noncausal. It is only an action-timing diagnostic; it is not a
substitute for the signal-destruction retraining branches below.

### Signal-destruction datasets

Before any run, separate ordinary risky return from cash return and mandatory
corporate-action/delisting adjustment. Mandatory events, membership,
availability, cash, costs, and observation tensors remain at their legal dates.
For fold `k`, materialize exactly three disjoint outbound-return row domains:

```text
D_train      = [0, a_k)
D_validation = [a_k, a_k+125)
D_outer      = [a_k+125, a_k+218)
```

The evaluation warm-up reuses the already transformed rows from the preceding
domain; support is never transformed as a standalone 31-position domain. The
terminal state at `a_k+218` has no outbound row. Create two hash-bound
transformed datasets independently inside each of these three domains:

- `N_time`: build a one-to-one destination/source-date matching within the
  role. An edge is legal only when the dates are separated by at least 31
  positions and the source has an ordinary return for every destination-active,
  nonmandatory coordinate. Sort adjacency by
  `SHA256(transform_seed || destination || source)` and run deterministic
  Hopcroft–Karp; absence of a perfect matching fails the branch. Assign the
  matched source's complete ordinary return vector. This preserves vector-level
  cross-sectional dependence while breaking its feature timestamp.
- `N_xs`: at each date, apply a hash-derived nonidentity cyclic permutation to
  ordinary returns across coordinates that are eligible for ordinary return
  assignment. Coordinates with mandatory delisting/corporate-action outcomes
  stay fixed; fewer than two permutable coordinates fails the transform.

Recompute C1, labels, portfolio drift, actions, costs, and every endpoint on the
transformed market. Retrain H1/H2 from initialization with the paired seed and
unchanged budget. Transform seeds, the three materialized domains, source row
maps, terminal masks, and output hashes are frozen in the manifest. No
transformed outer result is exposed early, and no source/destination edge
crosses a domain. These are the null/shuffle branches referenced by the
falsification gates.

## Primary and secondary endpoints

For H1/H2, the primary economic endpoint is the exact seed ensemble's
aggregate **continuing 20-bp active log wealth versus C1** over all six outer
scored blocks.

Required economic endpoints include:

- continuing and separately liquidated active wealth at 10, 20, and 40 bp;
- mean active return, volatility, information ratio, and drawdown;
- beta, tracking error, active share, HHI, effective holdings, and top-10/top-50
  mass;
- risky exposure, cash, requested-to-executed distance, and binding rates;
- startup, discretionary entry/exit/resize, membership-forced,
  availability-forced, risk-forced, and terminal turnover/cost;
- active P&L by fold, seed, calendar segment, and first 10 versus remaining
  scored decisions; and
- initialization and terminal-liquidation diagnostics.

Required holding endpoints include:

- notional-weighted current age;
- return-neutral entry-notional-unit-weighted median and quantiles of
  discretionary sale age (primary), plus economic-value-weighted sale age as a
  secondary diagnostic;
- discretionary sold-notional fractions younger than 10, 20, and 30 sessions;
- return-neutral cohort survival at 5, 10, 20, 30, and 60 sessions;
- all-cause survival and competing-risk forced-exit incidence;
- entry, exit, resize, membership-forced, availability-forced, risk-forced,
  startup, and terminal turnover;
- portfolio overlap at lags 5, 10, 20, and 30;
- P&L contribution by age bucket;
- `1 / mean(discretionary turnover)`, labeled only as an approximate horizon;
- anchor/origin-window/support role and censor counts;
- age/cohort reconciliation error; and
- H3 sleeve maturity and cross-net savings.

Sale-age gates exclude endowed/startup, membership-forced,
availability-forced, risk-forced, and terminal transactions.

Each net purchase cohort receives entry-notional units equal to its portfolio
weight at legal fill. Returns never change those units; a sale removes the same
fraction of units that it removes from that cohort's current economic value.
Survival uses these units and a frozen weighted product-limit estimator. At
each integer age `j`, let `n_j^disc` be units at risk after that
session's forced repairs and before discretionary sales, and let `d_j` be
discretionary units sold. Then

$$
S_{\mathrm{disc}}(k)=
\prod_{j=0}^{k-1}\left(1-\frac{d_j}{n_j^{\mathrm{disc}}}\right).
$$

For all-cause survival, let `n_j^all` be units at risk before forced repair and
`f_j` forced-exit units; replace the factor with
`1 - (f_j + d_j) / n_j^all`, with discretionary sales computed after forced
mass is removed. Administrative end-of-support is right-censoring.
Undefined horizons with no at-risk cohort are reported `NA`, never imputed.
The primary sale-age median is the weighted empirical median of actual
discretionary sold return-neutral entry-notional units, not a turnover-implied
duration. The separately named economic-value-weighted median cannot satisfy
the holding gate or C8 match.

Compute each fold's risk/event table first. The primary pooled holding estimate
sums at-risk and event units across the six folds at the same integer age and
then applies the product; a cohort never crosses a fold boundary. Report every
fold estimate and censor count alongside the pooled value. Only score-origin
cohorts enter these tables; support-origin entries are ignored, while their
state remains in conservation checks.

## Statistical plan

For variant `a`, fold `f`, scored date `t`, and cost rung `c`, define the
exact ensemble active log return

$$
e_{a,f,t,c}=
\log(1+R_{a,f,t,c}^{\mathrm{net}})
-\log(1+R_{C1,f,t,c}^{\mathrm{net}}).
$$

Define

$$
E_{a,f,c}=\sum_t e_{a,f,t,c},
\qquad
H_{a,c}=\sum_f E_{a,f,c}.
$$

Individual seed quantities are reported but seeds are paired optimization
replicates, not independent market samples.

Use 10,000 joint within-fold moving-block bootstrap replicates. The primary
block length is 10 decisions. Independently recompute the active-return lower
bound, every planned-contrast interval, Holm decisions, White, SPA, and max-T
p-values at block lengths 5 and 30. Blocks never cross fold boundaries. H2 must
retain its qualifying lower bound and every applicable p-value decision at all
three lengths; a positive point estimate alone is not a sensitivity pass.

Apply Holm correction to the four planned contrasts:

```text
H1 - H0
H2 - H1
H3 - H1
H2 - H3
```

The White Reality Check and Hansen SPA family is exactly H0–H3 and C2–C5 in
20-bp active return versus C1. Use the exact H ensembles and the predeclared C
traces. C0 and C6–C8 replicates are not family members.

For the learned-variant family, compute simultaneous one-sided max-T adjusted
p-values over exactly H1 and H2. Freeze null-centering, studentization, block
resampling, tie behavior, and RNG in the inference-plan artifact. White, SPA,
max-T, and bootstrap procedures operate jointly across aligned learned-variant
series so pairing is preserved.

The append-only registry supplies the broader trial count for deflated-Sharpe
and selection-bias diagnostics. The named family is not permission to omit
earlier related experiments.

## Conjunctive pre-lockbox gates

Only H2 is eligible for scale-up and it must pass every applicable gate. H0 is
the ported scalar-gate mechanism control, H1 is an ineligible transitional
scalar-gate control, and H3 is an ineligible structural comparator because it
lacks an ordinary signal-driven pre-maturity exit. Apply the same diagnostics
to H1 where defined, but a passing H1 cannot satisfy the strategy mandate or
substitute for H2.

### Operational and holding gates

For H1 and H2, pool the six exact-ensemble scored traces and report every item
below, using sealed support only for holding outcomes whose intent originated
in the scored block. H2 must pass all nine gates; H1 is diagnostic only:

1. exactly one ensemble portfolio decision/action build/execution occurs per
   scored session, with each member evaluated once;
2. no internal reset or liquidation occurs during warm-up, score, or support;
3. mean steady-state discretionary one-way turnover is in `[0.02, 0.05]`;
4. at least one complete portfolio-equivalent of discretionary sold notional is
   observed before applying sale-age gates;
5. return-neutral entry-notional-unit-weighted median discretionary sale age is
   in `[20, 40]` sessions;
6. discretionary return-neutral survival satisfies `S(10) >= 0.80` and
   `S(30) >= 0.35`;
7. less-than-10-session discretionary sells are at most 15% of discretionary
   sold notional;
8. all economic, age, cohort, turnover, censor, and sleeve ledgers reconcile
   within the frozen tolerance; and
9. exact resume reproduces the uninterrupted trace.

Report `S(30) - S_H0(30)` with its paired interval as a planned mechanism
diagnostic, but do not make `S_H0 + 0.10` a gate: a no-trade H0 can make that
bound mathematically unattainable even when absolute Hold-30 gates pass.

H3 instead has structural integrity gates: each uncensored sleeve phase is
exactly 30 transitions; no ordinary pre-maturity discretionary rebuild occurs;
sleeve NAV is never recapitalized from another sleeve; no
`maturity_cap_censored` event occurs in a clean structural trace; and all forced
repairs, residual capacity, same-name renewal, and cross-net savings reconcile.
H3's asset-level age/survival metrics are reported but do not qualify it.

### Economic and statistical gates

1. **Fold sign:** at least four of six `E[a,f,20bp]` are positive.
2. **Seed sign:** at least four of five individual seeds have positive aggregate
   20-bp active log wealth.
3. **Cost robustness:** `H[a,40bp] > 0`.
4. **No dominant fold:** if `P_f=max(E[a,f,20bp],0)`, require
   `max(P_f)/sum(P_f) <= 0.50`; zero positive sum fails.
5. **Initialization robustness:** a diagnostic starting directly from the C1
   benchmark at score time, with zero temporal state, remains positive.
6. **Terminal robustness:** separately liquidated aggregate 20-bp active log
   wealth remains positive.
7. **Matched controls:** `H[a,20bp]` strictly exceeds the 61st value after
   sorting the 64 aggregate C8 controls in ascending order; ties fail.
8. **Execution alignment:** on every transition, including a date with prior
   mandatory repair, H2 constructed-delta-to-filled-delta one-way distance is
   at most `1e-6`; any larger safety-projection correction fails. Report the
   same staged diagnostic for H1. Construction consumes repaired state and
   current hard ceilings, so there is no forced-date exception.
9. **Uncertainty:** the one-sided 95% moving-block-bootstrap lower bound for
   mean exact-ensemble 20-bp active return is positive under the primary block
   length, and the lower bound remains positive at lengths 5 and 30.
10. **Family tests:** one-sided White Reality Check and Hansen SPA p-values are
    both at most 0.10 at block lengths 5, 10, and 30.
11. **Candidate adjustment:** H2's max-T adjusted one-sided p-value from the
    frozen H1/H2 family is at most 0.10 at block lengths 5, 10, and 30.
12. **Provenance:** every source, data, fold, checkpoint, action, holding, and
    evaluator artifact validates against its receipt.

### Synthetic and falsification gates

1. A planted 30–40-session signal produces positive ensemble active wealth,
   at least four positive seeds, and median sale age 20–40; the frozen
   `N_time` and `N_xs` transformations remove the edge after full retraining.
2. An isolated one-session score perturbation changes one-way allocation by at
   most 0.5% and reduces `S(30)` by at most two percentage points.
3. Both H1 and H2, when trained on a planted strong reversal affecting at most
   5% of portfolio notional, liquidate at least 80% within three legal
   executions. The separate H2 adapter unit fixture uses five named age-zero
   positions at 1% each, assigns those held names zero C1 entry weight while
   keeping them legally sellable, injects `h=+12`, has no competing
   discretionary order, and has sufficient capacity so neither reacquisition
   nor the 10% portfolio turnover cap can make the assertion impossible.
4. Forced unavailability and point-in-time membership deletion each liquidate
   100% at the first legal execution, use distinct cause codes, and incur zero
   discretionary early-exit penalty.
5. H1/H2 retrained on `N_time` and `N_xs` converge to at most 0.5% median
   discretionary one-way turnover and do not meet the four-of-six fold-sign
   threshold. H3 reports the corresponding diagnostic after sleeve
   cross-netting.
6. Streaming, in-memory, origin-window, canonical-pass, and exact-resume
   traces match within the frozen numeric tolerance.
7. Age/cohort conservation and same-name netting tests pass.
8. Every uncensored H3 sleeve phase spans exactly 30 trading-return transitions
   and has no ordinary discretionary rebuild before maturity; any maturity cap
   censor is surfaced and fails H3's clean structural gate.
9. Observation/execution lookahead tests pass.

No operational, holding, synthetic, economic, or statistical gate can be
waived after outer access.

## Selection rule

If H2 fails any gate, stop. Do not select a least-negative variant, widen its
grid, substitute H0/H1/H3, or open a new lockbox. There is no within-protocol
ranking rule because H2 is the sole mandate-conforming candidate.

A passing H2 fixes the final same-window five-seed refit and output-space
ensemble rule. It does not authorize an architecture or optimizer search.

## Required software blockers

The versioned workflow's current 378/575/772 three-fold schedule does not
establish the required `N >= 1811` pre-2026 PIT-300 axis. A longer
point-in-time raw/bar/universe snapshot must be acquired or rebuilt and its
coverage receipt validated before this is a runnable configuration change.

Before rendering the run manifest, CI MUST cover:

- fill-time age-bin progression and 61-bin terminal behavior;
- economic-value and return-neutral cohort conservation;
- same-name cross-netting before age update;
- startup, discretionary, membership, availability, risk, and terminal cause
  typing;
- strong reversal and forced unavailability;
- persistent signal and isolated-noise invariance;
- canonical-state and numeric-economic carry across overlapping credit replay;
- exactly 30 post-fill return transitions of credit for every anchor;
- no liquidation/reset at intra-sweep replay, loader, or score boundaries;
- ordinary unmasked support actions, terminal observation, and right-censoring;
- fill-time risk/name ceilings cannot be reversed by discretion;
- exact H2 `h=-12,u_g=0` zero-delta action on arbitrary feasible repaired
  books and strong young-position reversal;
- H3 phase, maturity, cap censoring, cross-netting, and forced exits;
- H2 water-fill/custom-backward finite differences and cohort-release
  attribution, including zero/full-cap and zero-probability branches;
- training-time safety-projection identity within `1e-6`, with fail-closed
  behavior rather than a straight-through gradient;
- raw-intent, constructed-delta, filled-delta, pre-cost, and post-cost stage
  accounting, with cost normalization excluded from turnover/projection;
- canonical/replay forward equality, anchor-normalized gradient accumulation,
  and one-step-per-update equivalence;
- direct/environment action, reward, cost, and ledger trace parity;
- causal Transformer cache or recomputation equivalence;
- streaming/in-memory and stopped/resumed equivalence;
- deterministic `N_time`/`N_xs` transforms and C8 bank/filter reconstruction;
- cost-ladder replay from the sealed 20-bp action trace; and
- fail-closed split renderer at `N < 1811` or any post-2025 position.

## Artifact contract

Use a content-addressed root such as:

```text
artifacts/prelockbox-hold30-h0-h3-v1/<manifest-sha256>/
```

Retain at least:

```text
protocol/
  rendered-specification.md
  run-manifest.json
  trial-inventory.jsonl
  inference-plan.json
source/
  git.json
  executed-tree.tar.zst
  dirty.patch
  dependencies.lock
  runtime.json
data/
  data-manifest.json
  universe-events.parquet
  decision-axis.parquet
  folds.json
  benchmark-trace.parquet
  null-transform-manifest.json
runs/<variant>/<fold>/<seed>/
  config.json
  checkpoints/
  optimizer-telemetry.parquet
  state-resume-receipt.json
  pending-intent.parquet
  raw-intent.parquet
  action-stages.parquet
  anchor-credit-map.parquet
  turnover-ledger.parquet
  age-ledger/
  cohort-ledger/
  censor-mask.parquet
  stop.json
ensembles/<variant>/<fold>/
  output-aggregation.parquet
  action-trace.parquet
  economic-ledger.parquet
  holding-ledger.parquet
  sleeve-trace.parquet
controls/<variant>/<fold>/
  c6/
  c7/
  c8/
  c8-bank-manifest.json
evaluation/
  access-marker.json
  panel.parquet
  holding-panel.parquet
  statistics.json
  gate-results.json
  evaluation-receipt.json
receipts/
  SHA256SUMS
  completion.json
```

`dirty.patch` is omitted only when the manifest says `clean-worktree=true`.
Every decision row includes causal timestamp, fill timestamp, asset IDs,
scores, hazards, exposure, raw/constructed/filled deltas, repaired/pre-cost/
post-cost books, costs, turnover causes, age/cohort events, anchor/support/
censor state, and constraint reasons. H3 rows also include sleeve ID, phase,
and maturity-cap censoring.

Completion requires the exact expected path set, schema validation, SHA-256
validation, and source-to-checkpoint-to-action-to-evaluation receipt closure.
A summary file is not a substitute for missing action or holding traces.

## Invalidation and stop rules

Invalidate the affected generation before outer reveal if any of these occurs:

- source, dependency, image, data, universe, fold, action, or evaluator hash
  differs from the frozen manifest;
- the authoritative runtime is external to the retained source archive;
- any scored or support position reaches 2026;
- split geometry, C5 label horizon, cost, constraint, seed, update budget,
  checkpoint rule, holding threshold, or statistical method changes;
- an economic reset/liquidation occurs at an intra-sweep credit-replay,
  scoring, or loader boundary; the declared complete-update replay restart is
  the only training re-endowment;
- sealed support holding evidence is exposed before the one-shot reveal;
- any variant sees a different anchor/credit schedule without a preregistered
  mechanically necessary reason;
- fewer than 64 required C7/C8 controls are produced;
- a non-finite state or unreconciled ledger is skipped rather than failed;
- a failed attempt is replaced without an append-only registry entry; or
- any further 2026 access after the v1 freeze influences debugging or selection.

Operational interruption may resume only from a receipt-complete checkpoint
that reproduces the uninterrupted state trace. Otherwise restart that run under
the same frozen identity and record both attempts.

## Scale-up and later lockbox

Only a passing H2 mechanism may be scaled from the PIT 300 universe.
Scale-up first repeats the software and systems gates, then freezes a new
generation for the full point-in-time 1,000–2,000-stock scope. Capacity may increase only
after the compact model demonstrates stable active evidence; GPU availability
is not a scientific reason to enlarge it.

Train the final confirmatory research candidate as five same-window
pre-lockbox refits and use the same output-space ensemble and one shared
economic/age state. Register
the final data cutoff, prospective or untouched lockbox, execution simulator,
controls, and promotion rule before access.

Passing this pre-lockbox study is not promotion. It establishes only that the
holding mechanism behaves as intended and has enough development evidence to
justify one new confirmatory evaluation.

## Interpretation rules

- Low turnover alone is not proof of 30-session holding.
- A correct age distribution without positive active return is an operational
  success and a scientific failure.
- Positive active return without passing age, null, provenance, and
  multiplicity gates is not qualifying evidence.
- H3 economically outperforming is evidence about the sleeve control, but H3
  cannot be selected because it lacks ordinary signal-driven early exits. It
  does not validate H2's learned hazard.
- H2 winning means the age-aware learned mechanism merits a later optimizer
  ablation; it does not prove PPO is needed.
- More repeated sweeps or GPUs never increase the number of unique scored
  market dates.
