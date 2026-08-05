# RFC and experiment specification: Pre-lockbox Hold-30 alpha mechanism-8 v3

**Status:** Components implemented and locally CPU integration-qualified;
end-to-end training and launch blocked

**Date:** 2026-08-04

**Protocol generation:** `prelockbox-hold30-alpha-mech8-v3`

**Base design:** `daily_raw_pit300_hold30_alpha_v3`

**Supersedes before launch:**
[`prelockbox-hold30-mech8-v2`](prelockbox_hold30_mech8_v2.md)

**Mandate:** [ADR-0006](adr/0006-daily-decision-soft-30-session-holding.md)

**Scientific objective decision:**
[ADR-0007](adr/0007-benchmark-relative-hold30-alpha-objective.md)

## Decision

V3 keeps the carried-book, fill-time age, one-daily-decision, and soft
30-session holding mechanics developed under v2. It changes the scientific
screen from a holding-mechanism comparison into a staged alpha-objective
comparison. The screen asks whether the age-aware mechanism can produce
economically material, benchmark-relative stock selection at a declared active
risk level.

The only promotion candidate is canonical `m03`. The other seven settings are
controls or attribution diagnostics. A higher observed return from an
ineligible row cannot promote that row or be used to redesign `m03` after
outer access.

V2 was superseded before any v2 scientific launch. Its source, tests, receipt
primitives, accounting, age ledger, two-rank mechanics, and qualification work
remain reusable implementation history. They acquire no v3 identity merely by
being copied. Every artifact-producing v3 path MUST carry the v3 generation
and one of the eight disjoint v3 setting IDs; every v2 generation or setting
ID MUST fail closed.

## Precedence and evidence boundary

For v3, precedence is:

1. this document for generation identity, objectives, setting inventory,
   active-risk controls, costs, checkpoint selection, and data roles;
2. the [Hold-30 policy RFC](daily_hold30_policy_rfc.md) for economic state,
   action, execution, age, turnover, and continuation semantics; and
3. the [H0–H3 v1 specification](prelockbox_hold30_h0_h3_experiment.md) for
   unchanged point-in-time universe, fold, chronology, seed, control,
   inference, and receipt rules.

The 2026 S0–S7 evidence is consumed. V3 development, threshold selection,
checkpoint selection, and qualification use only the frozen pre-2026
walk-forward roles. No v3 artifact authorizes rereading 2026.

## Current implementation boundary

The version-controlled v3 protocol, typed data contract, setting-specific
model and action surfaces, objective/checkpoint logic, and sealed-evaluator
components are implemented. Their deterministic component and cross-component
contracts pass the local CPU test suite. This is **component-integration
qualification**, not end-to-end training qualification or scientific
evidence.

The repository contains a deliberately non-authorizing, one-update synthetic
driver that connects policy, delayed action, age accounting, objective,
optimizer, validation telemetry, and content-bound checkpoint artifacts for
all eight qualification-only routes. A06 exercises separate core-only and
executed-overlay economic streams, disjoint optimizers, and parent-linked
post-update state receipts that reconstruct from the saved checkpoint. The
driver cannot execute the frozen 240-trial real-data study or issue launch
authority. The separately content-addressed pilot plan freezes the resolved
numeric profile and checkpoint thresholds. It freezes the A06 total-Sharpe
variance floor at `1e-6`: the same 10-bp daily-volatility floor as A07,
negligible at ordinary equity volatility but stabilizing near-zero
denominators. This was a prelaunch choice, not an outcome-tuned value.
The resulting pilot scientific-profile receipt is
`7cb98970c93bc4e8cd59c49cc09b1b7883025ff700acec3783453585c7084752`.
Production remains blocked by typed global-path/real-data bindings,
an immutable image, full-model GPU/H100 parity and capacity, and executable
approval. CPU distinct-shard parity is qualified only for the frozen
one-path-per-rank geometry, and exact all-eight restart/resume parity is
qualified. No production
checkpoint family or v3 performance result exists.

## Common design

The machine-readable source of truth is
`rl_quant.protocol.hold30_alpha_v3`. Freeze:

| Field | Value |
|---|---:|
| Decisions | One per eligible trading session |
| Soft holding target | 30 trading sessions |
| Controlled/scored tail | 63 sessions |
| BPTT/credit span | 63 sessions |
| Alpha horizons | `5, 21, 30, 63` sessions |
| Primary alpha horizon | 30 sessions |
| Training benchmark | C1 monthly PIT active-300 equal weight, buy-and-drift |
| Training execution cost | 20 bp |
| Validation cost rungs | 10, 20, and 40 bp |
| Annual tracking-error floor/target/ceiling | 2% / 4% / 6% |
| Market beta target/band | 1.0 / `[0.9, 1.1]` |
| Outer folds | 6 |
| Paired seeds | `17, 29, 43, 71, 101` |

The multi-horizon outputs are auxiliary causal representation targets. The
economic reward remains one-session net log return or one-session active net
log return, according to the setting. Forward-horizon labels MUST be
split-local and censored when their complete T+1 fill-through-horizon support
does not remain inside the permitted training or inner-validation role.
Overlapping 30-session returns MUST NOT be added as daily economic P&L.

## Exact eight-setting inventory

Stable IDs are artifact identity. Aliases, renumbering, or substitutions are
not compatible reruns.

| Index | Stable setting ID | Frozen objective/change | Eligible |
|---:|---|---|:---:|
| 0 | `hold30a-m00-legacy-absolute` | Legacy scalar gate; absolute net log-return control | No |
| 1 | `hold30a-m01-persistent-absolute` | Age-aware Hold-30; absolute net log return | No |
| 2 | `hold30a-m02-active-te` | C1-relative mean objective; annual TE 2%/4%/6% | No |
| 3 | `hold30a-m03-alpha-core` | Canonical 30d alpha mean and downside-uncertainty heads; TE band; beta target | **Yes** |
| 4 | `hold30a-a04-no-uncertainty` | m03 without downside/uncertainty heads | No |
| 5 | `hold30a-a05-no-te-floor` | m03 without the 2% TE floor; 4% target and 6% ceiling remain | No |
| 6 | `hold30a-a06-sharpe-overlay` | m03 plus a separate total-risk/Sharpe overlay | No |
| 7 | `hold30a-a07-direct-sharpe` | m03 plus a direct two-pass Sharpe-gradient term | No |

Rows a04–a07 are attribution diagnostics relative to m03. A06 and a07 are
explicitly ineligible because they introduce total-return risk/Sharpe terms
whose contribution must be separated from stock-selection alpha. M00 is the
only legacy scalar-gate row; it is a control, never a candidate.

Supervised residual-alpha heads are absent from m00, m01, and m02 and present
in m03 and a04–a07. A04 retains the supervised alpha-mean head but removes the
downside/uncertainty head. This distinction is frozen so m02 cannot silently
receive the m03 prediction mechanism and erase the m03-minus-m02 attribution.

## Objective contract

Let (R_{p,t}^{net,20}) be the portfolio's one-session return after the common
20-bp execution model and (R_{C1,t}^{net,20}) the exact same-interval C1 net
return. Define active log return:

\[
a_t = \log(1 + R_{p,t}^{net,20})
      - \log(1 + R_{C1,t}^{net,20}).
\]

- M00 and m01 optimize `absolute-net-log-return`.
- M02 optimizes the chronological mean of (a_t), with the TE band.
- M03 optimizes C1-relative alpha mean with the 30-session alpha-mean and
  downside-uncertainty heads, TE band, and beta target.
- A04 uses the m03 portfolio/action/risk contract but removes the
  downside/uncertainty heads.
- A05 keeps the m03 objective but removes only the TE floor.
- A06 keeps m03 intact, then applies a separately parameterized total-risk and
  Sharpe overlay. Its overlay parameters, gradients, checkpoint state, and
  attribution trace MUST be separate from the alpha core. Total Sharpe uses
  portfolio return in excess of the explicit PIT risk-free/CASH series.
- A07 adds the declared direct two-pass Sharpe-gradient term to m03. Pass one
  materializes the chronological return moments without updating parameters;
  pass two applies the gradient using those frozen moments. A one-pass batch
  approximation is not equivalent. Its Sharpe term also uses the explicit PIT
  risk-free/CASH series.

The implementation additionally fails closed on the following training
bindings:

- every objective batch carries exactly one typed `train` or `validation`
  label domain; outer-role labels are never accepted by the optimizer;
- Pass A and Pass B carry identical source-row, batch-row, immutable-return,
  target, actor-output, and model-evaluation-point identities;
- caller-supplied distributed moments are prohibited until those same
  identities are bound across ranks;
- A07 and total-Sharpe reporting use simple policy return minus the exact PIT
  risk-free return, while TE and IR continue to use C1-relative log returns;
- auxiliary-horizon weights and scales are manifest fields; the weights sum
  to one and give the 30-session target the unique largest weight; and
- executable penalties are strictly positive, and learned uncertainty is
  constrained by explicit manifest-owned log-scale bounds.

Tracking error is the annualized standard deviation of daily active return,
using `sqrt(252)`. The 2% floor prevents a nominal alpha model from collapsing
back into the near-C1 behavior seen in S0–S7; the 6% ceiling bounds active risk;
4% is the target. The floor is a training/qualification design field, not
permission to manufacture turnover. M03 also targets realized beta 1.0 with a
hard acceptable band of 0.9–1.1. Beta uses the receipt-bound PIT cap-weight
market series; TE remains C1-active. Both use only causally available
training-role returns during optimization and inner-validation-role returns
during selection.

The exact penalty/constraint estimator, denominator, warm-up, and gradient
normalization MUST be bound in `training-plan.json` and pass one-rank versus
two-rank parity before software qualification. They MUST NOT be inferred from
GPU count or tuned after outer access.

## Checkpoint contract

Freeze:

| Field | Value |
|---|---:|
| Maximum updates | 128 |
| Emit checkpoint | Initial, every 8 updates, and final |
| Validate | Every 8 updates |
| Minimum updates | 32, absent numerical failure |
| Patience | 4 validations |
| Retained checkpoints | Initial, selected, final |

Selection evaluates the deterministic deployed ensemble. Each candidate update
bundles the same update index from all six folds and all five seeds (30
checkpoints per setting). One update index is selected for the setting;
per-fold, per-seed, or mixed-update selection is prohibited.

Checkpoint eligibility is tested in this exact order:

1. complete predeclared validation coverage;
2. active results available at both 20 and 40 bp;
3. annual tracking error in `[0.02, 0.06]`;
4. market beta in `[0.9, 1.1]`;
5. median discretionary sold-notional age in `[20, 40]` sessions;
6. projection distance at or below its separately frozen maximum; and
7. forced-turnover fraction at or below its separately frozen maximum.

Rank eligible checkpoints lexicographically using the 20-bp rung only:
median active return across folds and seeds (descending), active information
ratio across folds and seeds (descending), total Sharpe across folds and seeds
(descending), maximum drawdown across folds and seeds (ascending), then
turnover and cost across folds and seeds (ascending). Break exact ties by
earlier update and lexical checkpoint ID. The 40-bp rung is an availability and
robustness eligibility input; it is never pooled with 20 bp for ranking.

The canonical protocol template deliberately retains `null` for the two
implementation thresholds and for objective coefficients. It is a
launch-incapable audit baseline, not the pilot plan. The separately
content-addressed pilot decision resolves every applicable field without
mutating that template:

| Pilot field | Frozen value | Calibration |
|---|---:|---|
| Auxiliary horizon weights `(5,21,30,63)` | `(0.10,0.20,0.50,0.20)` | 30 sessions is uniquely primary |
| Auxiliary horizon scales | `0.02*sqrt(H)` | unstandardized cumulative active-log labels under a 2% daily prior |
| Active log-scale bounds | `(log(0.5),log(1.5))` | maps the 4% scale exactly to 2%--6% |
| Uncertainty log-scale bounds | `(log(0.01),0)` | 1%--100% positive 30-session scale |
| Downside `kappa` | `0.25` | downside-adjusted stock score |
| TE floor / ceiling lambdas | `0.25 / 0.50` | 2-point breach costs 1 / 2 reference bp per day |
| Beta lambda | `0.01` | 0.10 miss costs 1 reference bp per day |
| Turnover lambda / target | `0.25 / 1/30` | 2-point excess costs 1 reference bp per day |
| Early-exit lambda | `0.002` | 5% fully weighted young-sale mass costs 1 bp per day |
| Auxiliary-alpha / uncertainty lambdas | `1e-4 / 5e-5` | representation terms remain subordinate to active P&L |
| Projection-distance maximum | `0.01` | mean requested-to-constructed one-way NAV distance |
| Forced-turnover fraction maximum | `0.10` | forced / (forced + discretionary), with zero denominator mapped to zero |
| A06 total-excess / Sharpe lambdas | `1.0 / 5e-5` | separate overlay-only objective |
| A06 volatility target / lambda | `1.0 / 0.01` | benchmark volatility ratio; 0.10 miss costs 1 bp per day |
| A06 log-drawdown limit / lambda | `-log(0.85) / 0.04` | 5-point breach costs 1 bp per day |
| A06 exposure step | `0.05` | tanh-bounded overlay changes risky exposure by at most 5 points before the common envelope |
| A06 total-Sharpe epsilon | `1e-6` | 10-bp daily-volatility variance floor |
| A07 direct-Sharpe lambda / epsilon | `5e-5 / 1e-6` | diagnostic two-pass term with the same variance floor |

A04 omits only uncertainty fields, a05 omits only the TE-floor lambda, a06
adds only its disjoint overlay fields, and a07 adds only its direct-Sharpe
fields. Every other inapplicable optional field remains exact `null`; it is not
an unresolved choice. A06 additionally binds `alpha-core-only` and
`a06-overlay-only` parameter selectors, stops gradients in both directions,
and binds an immutable optimizer-spec receipt distinct from mutable optimizer
state.

Action geometry remains frozen by the higher-precedence Hold-30 RFC and the
source/RFC hashes in the manifest: hazard residual `[-12,12]`, age cap 60,
entry-score clip `[-2,2]`, 1% risky-name cap, 10% discretionary one-way cap,
C1 risky-exposure band `+/-2` percentage points, and legacy-control
temperature `0.5`. These are not tuned by the pilot profile. The alpha action's
learned risk scale is the pilot-bound 2%--6% range above; only a06 may apply the
additional 5-point total-risk step.

An executable manifest MUST accept the typed resolved training-plan receipt,
invoke the same setting-specific config validator used by training, and match
its content hash. A bare `training_plan_sha256` is insufficient. Prohibited
ablation terms must be absent or exact zero. Executability requires the exact
typed pilot plan and scientific-profile receipt
`7cb98970c93bc4e8cd59c49cc09b1b7883025ff700acec3783453585c7084752`.
The canonical unresolved template remains appropriate for non-authorizing
software qualification only. No outer return can change a coefficient, action
bound, threshold, update, cost rung, or stop decision.

## Data contract

V3 consumes the existing C1 monthly point-in-time active-300 equal-weight
buy-and-drift trace as the sole training benchmark. The C1 receipt MUST bind:

- membership and eligibility events;
- exact monthly rebalance decisions;
- mandatory repairs and cause-typed trades;
- weights, cash, costs, returns, and continuing wealth; and
- the decision/fill axis and all source hashes.

The evaluator additionally requires all of the following real,
point-in-time, receipt-bound artifacts:

1. a risk-free return series;
2. a cap-weight market return series; and
3. a declared factor set, factor returns, and point-in-time exposures.

Their roles are distinct and fail closed:

- C1 is the action anchor and active training benchmark.
- The PIT cap-weight market is available only to the beta objective, beta
  checkpoint eligibility, and sealed beta evaluation.
- The PIT risk-free/CASH series is available only to portfolio accounting,
  the a06/a07 total-excess-Sharpe objective, 20-bp total-Sharpe checkpoint
  ranking, and sealed evaluation.
- Declared factor returns and exposures are evaluator-only.

Every artifact has `policy_feature_access=false`: none may enter actor inputs,
representation normalization, or alpha prediction features. Factors cannot
enter any training loss or checkpoint choice. Market and risk-free use beyond
the exact objective/checkpoint roles above is prohibited.

There are no fake, zero, equal-weight, constant, forward-filled-from-the-
future, or library-default substitutes. A missing artifact, missing
point-in-time receipt, incomplete fold coverage, undeclared factor, or digest
mismatch fails data qualification. The evaluator must report the declared
factor names and source lineage; it cannot silently add a factor after reveal.

Residual alpha labels for horizons 5, 21, 30, and 63 are built independently
inside each permitted split. A label beginning at decision `t` uses the first
legal T+1 fill and is valid only when the full outcome interval and required C1
support remain inside that same split role. Training cannot see
inner-validation or outer-label statistics. Inner validation cannot see outer
labels. Risk-free, cap-market, and factor artifacts are not used to
residualize these labels; residual alpha is defined only against C1.

## Freeze and receipt contract

The V3 manifest has schema version 3 and binds:

- the exact generation, ordered setting inventory, common design, and
  checkpoint contract;
- this RFC, ADR-0007, and the superseded v2 specification hashes;
- repository commit/tree, clean or dirty-patch state, retained source archive,
  dependency lock, and container digest;
- the C1, PIT market, PIT risk-free/CASH, and evaluator-only factor receipts,
  including their exact usage allowlists and `policy_feature_access=false`;
- exact decision axis, split arrays, training/evaluation plans, trial inventory,
  and recovery policy; and
- component, software, data, capacity, and explicit executable-approval
  receipts.

The typed training plan is the authoritative checkpoint contract for a
rendered manifest. The rendered common design, top-level manifest contract,
and embedded training-plan contract MUST be exactly equal and bind the same
digest. A future executable render substitutes the resolved contract into the
design; it MUST NOT retain the protocol template's unresolved `null`
thresholds beside resolved training-plan values.

The inventory remains exactly `8 * 6 * 5 = 240` training trials. Two ranks or
two H100s are one distributed trial, not two seeds or two observations. A
renderer is always launch-incapable: even an executable-state payload records
`render_grants_launch_authority=false` and requires a separately issued
approval receipt.

## Qualification gates

V3 advances only in order:

1. **Protocol:** schemas and tests freeze the eight settings, design, data
   roles, checkpoint rule, and v2 rejection. **Locally qualified.**
2. **Component:** alpha labels/censoring, objectives, uncertainty/downside
   heads, TE/beta estimators, overlays, accounting, and gradients pass focused
   tests. The implemented protocol/data/model/action/objective/evaluator
   integration is **locally CPU-qualified**, including the synthetic A06
   three-stream and receipt-chain contract.
3. **Software:** deterministic synthetic receipt closure, exact all-eight
   two-update restart/resume, and CPU two-rank distinct-shard parity pass for
   the frozen one-path-per-rank geometry. Production remains **blocked** on
   typed file-backed real-data/global-path binding and GPU/H100 parity.
4. **Data:** real pre-2026 PIT active-300/C1, risk-free, cap-market, and declared
   factor artifacts pass without substitutes.
5. **Capacity:** the approved two-H100 worker and bounded admission/recovery
   checks pass independently of scientific outcomes.
6. **Scientific:** one frozen 240-trial execution and sealed evaluator complete
   without selective retries or missing terminal receipts.

This document and local tests establish implemented components, local CPU
component integration, exact restart, and bounded CPU distributed parity.
They do not establish a production v3 training runtime, real-data
qualification, GPU/H100 parity, launch authorization, completed training, or
investment performance.

## Migration from v2

V2 is retained unchanged as an audit record and marked
`superseded-before-launch`. Reusable implementation pieces must be ported under
explicit v3 tests and new receipts. In particular:

- v2 checkpoints and result rows are never valid v3 inputs;
- v2 setting IDs are never aliases for similarly placed v3 rows;
- v2 Stage-1 code may be reused only if its exact v3 source/data/fold receipt
  closes;
- v2 age/accounting/action code may be reused after parity tests; and
- no v2 Job is resumed, renamed, or reinterpreted as v3.

Until the remaining gates pass,
`prelockbox-hold30-alpha-mech8-v3` is **launch blocked**.
