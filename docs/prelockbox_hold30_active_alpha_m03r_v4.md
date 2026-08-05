# M03R active-alpha Hold-30 protocol and experiment specification

**Protocol generation:** `prelockbox-hold30-active-alpha-m03r-v4`

**Base design:** `daily_raw_pit300_hold30_m03r`

**Sole promotion candidate:** `M03R-active-alpha-hold30`

**Typed source of truth:** `rl_quant.protocol.hold30_alpha_m03r`
**Status:** version-controlled redesign; real-data launch remains fail-closed until
all required bindings and qualifications are receipt-complete

## 1. Decision and lineage

M03R is a new immutable protocol generation. It does not modify, alias, or
reinterpret the identities in `prelockbox-hold30-alpha-mech8-v3`. V3 artifacts
remain V3 audit records. A checkpoint, receipt, result row, or source bundle from
V3 cannot be relabeled as M03R.

The legacy E0–E7 recurrent-PPO panel is operationally independent of this RFC.
It may finish as a development-only sensitivity panel, but none of its settings
implements M03R and none is promotion eligible under this generation.

This revision makes the no-tracking-error-floor design canonical, constrains
active rather than total beta, restores a learned 252-session slow context, and
separates all temporal quantities that V3 could describe ambiguously.

## 2. Scientific proposition

M03R tests whether a direct differentiable, age-aware policy can generate
positive net active return relative to the frozen C1 benchmark while:

- normally carrying positions for about 30 trading sessions;
- using no compulsory active-risk floor when signal confidence is weak;
- keeping annual tracking error at or below 6%;
- keeping active market beta near zero, with absolute active beta no greater
  than 0.10;
- respecting point-in-time factor and sector exposure bounds;
- retaining an investor-facing total Sharpe that is acceptable relative to C1;
- surviving realistic cost and execution sensitivity tests.

The policy's primary economic quantity is net active log return:

\[
a_t = \log(1+r^{P,\mathrm{net}}_t)
      - \log(1+r^{C1,\mathrm{net}}_t).
\]

Until positive, uncertainty-adjusted multifactor regression evidence exists,
receipts and reports must call this quantity `active_return_vs_C1`, not
unqualified `alpha`.

## 3. Explicit temporal contract

The following fields are distinct and carry their units in their names:

| Typed field | Frozen value | Meaning |
|---|---:|---|
| `decisions_per_trading_session` | 1 | Exactly one portfolio decision per session |
| `fast_raw_context_trading_sessions` | 42 | High-resolution fast raw-data branch |
| `slow_raw_context_trading_sessions` | 252 | Learned slow raw-data branch |
| `rollout_trading_sessions` | 63 | Stateful optimizer/TBPTT rollout interval |
| `economic_origin_post_fill_return_count` | 30 | Returns read by one primary economic origin |
| `maximum_auxiliary_label_horizon_trading_sessions` | 63 | Longest auxiliary residual label |
| `target_holding_trading_sessions` | 30 | Soft persistence target |
| `evaluation_warmup_trading_sessions` | 63 | Unscored chronological carry-in |
| `evaluation_score_trading_sessions` | 63 | Scored outer interval |

An economic origin therefore contains one decision state followed by exactly
30 post-fill return transitions: 31 state rows in total. A 63-session rollout
does not give the 30-return loss permission to read 63 future returns.

The auxiliary horizons are `(5, 21, 30, 63)` trading sessions, with 30 sessions
primary. Each loss term must declare the precise return indices it reads. The
generic names `horizon` and `credit_span` are forbidden in M03R manifests when
they could refer to more than one row in this table.

Computational truncation does not terminate the economic portfolio. Holdings,
cash, age cohorts, and recurrent state carry across rollout boundaries; the
autograd graph may detach at a declared boundary without liquidating or
reinitializing the book.

## 4. Model and information flow

The canonical model uses direct differentiable trajectory optimization rather
than PPO. Its frozen capacity ceiling is 7,000,000 trainable parameters. Initial
searches should remain between 1,000,000 and 3,000,000 parameters.

### 4.1 Two-speed raw context

The model must contain:

- a 42-session fast raw branch for recent dynamics;
- a 252-session slow raw branch for momentum, regime, volatility, and recovery
  dynamics that cannot be represented by a 42- or 63-session view;
- a permutation-equivariant cross-sectional market-context branch;
- a portfolio branch containing current holdings, cash, availability, and
  age-cohort summaries.

The slow branch receives raw daily observations or learned pooled raw-bar tokens.
It does not receive hand-engineered momentum or factor identities.

### 4.2 Heads

The canonical policy has separate heads for:

- entry/active-alpha strength;
- learned age-aware exit hazard;
- predictive uncertainty/downside;
- auxiliary residual returns at 5, 21, 30, and 63 sessions.

Learned-hazard settings also have a deterministic exact-hold branch. Its
forward decision is hard (`hold` or `release-eligible`), so the hold branch
produces exactly zero discretionary release; training uses an explicitly named
straight-through sigmoid gradient surrogate. `A08` disables this learned
branch together with the learned hazard and uses only the frozen structural
prior. A continuous probability multiplied into the hazard is not sufficient
to claim an exact-hold action.

The benchmark-relative target is constructed from C1 and model scores. Zero
scores represent zero active tilt rather than an accidental equal-weight target.
The exact neutral trading action is to carry the current feasible book.

### 4.3 Holding prior versus learned exits

The 30-session structural prior and the learned exit policy are separate causal
objects. The canonical exit hazard is learned and age aware. `A08` freezes that
hazard to its 30-session prior, allowing the incremental value of learned exits
to be measured.

Forced unavailability, risk, and data-integrity exits are exempt from the early
exit penalty and must be recorded under separate cause codes.

## 5. Canonical active-risk contract

### 5.1 Tracking error

The canonical annual tracking-error contract is:

```text
floor = 0.00
ceiling = 0.06
confidence-preferred range = [0.00, 0.04]
preferred_annual_te = 0.04 * calibrated_confidence
calibrated_confidence in [0, 1]
```

There is no lower tracking-error penalty. In a null or low-confidence regime,
copying C1 and avoiding unnecessary turnover is valid behavior. The old fixed
2% floor is retained only in `A05-fixed-te-floor`.

The canonical loss may penalize only ceiling violations:

\[
P_{\mathrm{TE}} =
\left[\max(0,\widehat\sigma_{\mathrm{TE}}-0.06)\right]^2.
\]

A confidence-scaled preferred value may size active exposure within the
feasible range, but it is not a variance floor.

### 5.2 Active beta

The constrained quantity is active market beta:

\[
\beta_{\mathrm{active}}
=
\frac{\operatorname{Cov}
\left(r^P_t-r^{C1}_t,\ r^M_t-r^f_t\right)}
{\operatorname{Var}(r^M_t-r^f_t)}.
\]

The target is zero and the hard eligibility maximum is
`abs(active_market_beta) <= 0.10`. Total portfolio beta remains a secondary
diagnostic, not a substitute for active-beta neutrality.

Every evaluation receipt must contain:

```text
portfolio_market_beta
benchmark_market_beta
active_market_beta
active_beta_standard_error
active_beta_constraint_satisfied
```

### 5.3 Turnover and concentration

The soft target for daily one-way discretionary turnover is `1/30`. It is not a
claim that every asset is held for exactly 30 sessions. Startup, forced,
unavailable, projection-only, and terminal transactions must remain separately
typed. Maximum post-projection asset weight is 1%.

Holding qualification uses notional-weighted survival and competing-risk
statistics, not turnover-implied holding time alone.

## 6. Factor and sector projection

The actor does not receive factor or sector identities as predictive inputs.
The post-ensemble risk layer projects requested C1-relative active weights onto
predeclared point-in-time exposure constraints.

The frozen optimization semantics are:

\[
\min_{\Delta w_t}
\left\|\Delta w_t-\Delta w_t^{\mathrm{requested}}\right\|_2^2
\]

subject to:

\[
\mathbf 1^\top \Delta w_t=0,
\qquad
|B_t^\top\Delta w_t|\leq\epsilon_B,
\qquad
0\leq w^{C1}_{i,t}+\Delta w_{i,t}\leq0.01,
\]

plus the canonical tracking-error and active-beta constraints.

The exposure families are frozen as:

```text
market
sector
size
momentum
value
volatility
liquidity
```

All loadings must be point in time. The projection runs once after seed-score or
seed-intent ensembling. Solver ties use lexical point-in-time asset ID. An
infeasible projection fails closed and produces no performance artifact.

Numerical exposure bands are result-moving and dataset-dependent. They must not
come from silent code defaults. Before launch, the following typed binding must
exist and be content addressed:

```text
schema = rl-quant.m03r-factor-sector-exposure-bounds-v1
factor_sector_exposure_bounds_manifest_sha256 = <64 lowercase hex>
```

That manifest must bind every exposure name, unit, normalization rule, bound,
estimation window, missing-value rule, and infeasibility rule. Absence or hash
mismatch blocks launch.

## 7. Experiment inventory

M03R freezes eleven mechanism rows. Only the canonical row is promotion
eligible. A04–A10 each change exactly one causal field relative to M03R.

| Index | Stable setting ID | Causal purpose | Promotion |
|---:|---|---|---|
| 0 | `M00-absolute-return` | Hold-30 architecture with absolute net-return objective | No |
| 1 | `M01-benchmark-subtraction` | M00 plus C1 subtraction only | No |
| 2 | `M02-active-risk-no-alpha-heads` | M01 plus active-risk projection, without residual-alpha heads | No |
| 3 | `M03R-active-alpha-hold30` | Full canonical candidate | **Sole candidate** |
| 4 | `A04-no-uncertainty-scaling` | Disable uncertainty-based sizing only | No |
| 5 | `A05-fixed-te-floor` | Restore the rejected 2% annual TE floor only | No |
| 6 | `A06-sharpe-overlay` | Add a separate total-risk/Sharpe overlay only | No |
| 7 | `A07-direct-sharpe` | Add the direct two-pass Sharpe-gradient term only | No |
| 8 | `A08-fixed-exit-hazard` | Freeze exits to the 30-session prior only | No |
| 9 | `A09-no-long-context` | Replace the 252-session slow context with 63 sessions only | No |
| 10 | `A10-no-factor-neutral-projection` | Disable factor/sector projection only | No |

The scientific contrasts are:

```text
M01 - M00: benchmark-relative versus absolute reward
M02 - M01: explicit active-risk control
M03R - M02: residual-alpha and uncertainty structure
M03R - A04: uncertainty-scaled sizing
M03R - A05: harm or benefit of compulsory active variance
A06 - M03R: separate investor-facing Sharpe overlay
A07 - M03R: direct Sharpe-gradient ablation
M03R - A08: value of learned exit timing over a structural prior
M03R - A09: value of 252-session learned context
M03R - A10: value and attribution effect of factor neutrality
```

Seeds are algorithmic replications on shared market histories; they are not
independent economic observations. Statistical inference must resample the
deployed chronological return stream, not count fold-by-seed cells as
independent dates.

## 8. Economic objective

The canonical additive objective is active net log return with explicit risk,
turnover, holding, forced-exit, and auxiliary terms:

\[
\begin{aligned}
\mathcal L ={}&
-\frac1T\sum_t a_t
+\lambda_{\mathrm{TE}}
  [\max(0,\sigma_{\mathrm{TE}}-0.06)]^2\\
&+\lambda_\beta\beta_{\mathrm{active}}^2
+\lambda_F\|B_t^\top(w_t-w_t^{C1})\|_2^2\\
&+\lambda_{\mathrm{turn}}
  [\max(0,\tau_t^{\mathrm{disc}}-1/30)]^2
+\lambda_{\mathrm{early}}E_t
+\lambda_{\mathrm{forced}}F_t
+\mathcal L_{\mathrm{aux}}.
\end{aligned}
\]

There is no TE-floor term. The economic objective reads exactly 30 post-fill
returns per origin. Overlapping 30-day forward returns are not inserted as a
daily production reward; multi-horizon residual labels remain auxiliary.

The canonical candidate has no Sharpe term. `A06` and `A07` retain Sharpe as
separate causal ablations so improved headline Sharpe cannot be confused with
stock-selection skill.

## 9. Validation and selection direction

Training uses 20 basis points per unit of one-way turnover. Deterministic
validation uses 10, 20, and 40 basis points.

Checkpoint eligibility must precede ranking. At minimum, it requires complete
coverage, data integrity, mechanism integrity, active-beta compliance,
tracking-error ceiling compliance, factor/sector projection compliance, holding
survival evidence, and valid 20/40-bp active returns.

The local selection primitive currently qualifies identity, exact fold/seed
inventory, gate resolution, receipt integrity, and deterministic ordering. It
does not authorize a trainer to self-report the aggregate metrics it ranks.
Before real-data launch, those metrics must be recomputed from the bound
chronological validation arrays by an independent evaluator and supplied in a
verified evaluator receipt. A caller-authored scalar record, even when
self-hashed, is not sufficient selection evidence.

Among eligible checkpoints, the primary ranking statistic is the predeclared
moving-block-bootstrap lower confidence bound of pooled 20-bp active return.
Information ratio, total Sharpe, drawdown, forced turnover, and cost robustness
are secondary. A numerically larger point estimate cannot override a worse
primary confidence bound.

The exact inferential contract is a required content-addressed launch binding.
It must freeze the factor model, block rule, replicate count, random seed,
multiplicity family, tie behavior, missing-data behavior, and outer-fold pooling
before concealed outer access.

Multiplicity membership comes from an append-only, content-addressed trial
registry rather than a manually supplied lower bound. Every candidate whose
design was influenced by observed development evidence records:

```text
trial_id
parent_generation
scientific_change
data_role
outer_data_accessed
selection_influence
promotion_family_membership
```

Seeds are retained as algorithmic robustness evidence, while uncertainty is
computed from the single chronological deployed ensemble stream. Fold-by-seed
cells are never counted as independent market observations.

### 9.1 Finite-control diagnostic

When the compatibility diagnostic uses exactly 64 controls, it must count ties
against the candidate:

\[
k_{\geq}=\#\{T_b\geq T_{\mathrm{candidate}}\},
\qquad
p_{64}=\frac{1+k_{\geq}}{65}.
\]

A nominal 5% pass therefore requires `count_control_ge_candidate <= 2`. The
previous “exceed the 61st of 64” wording is not equivalent because it can allow
three controls to equal or exceed the candidate, yielding `4/65`, about 0.0615.
Every receipt must store `count_control_ge_candidate`, the tie rule `>=`, and the
computed finite-control value.

Controls selected after reading realized tracking error or beta are a
**matched-placebo diagnostic**, not exact randomization inference unless
exchangeability under that conditioning procedure is established in the frozen
inferential contract. Confirmatory inference should instead use a fully
pre-generated control family, complete signal-destruction retraining, or a
predeclared max-statistic/SPA/Reality-Check family containing every adaptively
considered candidate.

### 9.2 Cost-repricing labels

Repricing one frozen requested-intent trace at 40 basis points is a useful
first-order sensitivity, but it is not a closed-loop 40-bp backtest because
costs change cash, holdings, and future policy state. These quantities must be
named separately:

```text
fixed_intent_repricing_40bp
closed_loop_execution_40bp
```

The fixed-intent result is mandatory for checkpoint screening. A receipt-complete
closed-loop result is mandatory before promotion.

### 9.3 Promotion direction

The numerical block/bootstrap and multiplicity choices remain launch bindings,
but the direction of every primary gate is frozen now. Promotion requires:

- the 95% lower confidence bound of 20-bp net active return above zero;
- the multiplicity-adjusted 95% lower confidence bound of the declared
  multifactor alpha above zero;
- the 95% lower confidence bound of total Sharpe minus C1 Sharpe above `-0.10`;
- nonnegative closed-loop 40-bp active return;
- the predeclared null-control family passing its exact frozen rule;
- no signal-destruction control passing the candidate gates;
- holding-survival, censoring, projection, execution-capacity, and source/data
  integrity gates passing.

No point estimate can override a failed confidence-bound or integrity gate.

## 10. Required null and causal checks

The following checks block scale-up:

1. With residual labels permuted or set to zero, M03R approaches C1, low
   turnover, near-zero active beta, and near-zero tracking error. It is not
   penalized for tracking error below 2%.
2. A planted 30–40-session signal produces persistent positions.
3. A one-session perturbation does not cause material portfolio replacement.
4. A strong reversal permits an economically justified early exit.
5. `A08` separates a fixed holding prior from learned exit skill.
6. `A09` tests whether the 252-session branch adds information.
7. Factor, sector, asset-order, and time-label destruction remove the claimed
   effect.
8. Startup, forced, unavailable, projection, discretionary, and terminal
   turnover reconcile exactly.
9. Holdings and cohort ages cross TBPTT boundaries exactly while gradients
   detach.
10. One-rank and two-rank economic ledgers and gradients agree within the frozen
    tolerance.

## 11. Holding telemetry

Receipts must include notional-weighted:

- current age distribution;
- survival at 5, 10, 20, 30, and 60 sessions;
- restricted mean holding time through 60 sessions;
- median discretionary sale age and the sold mass supporting it;
- learned, forced, unavailable, risk, and right-censored competing risks;
- entry, exit, resize, startup, projection, forced, and terminal turnover;
- P&L and costs by entry-score, uncertainty, liquidity, and position-age bucket.

A sale-age median cannot pass a gate when insufficient notional has been sold or
when censoring makes the estimate unidentified.

## 12. Required launch bindings

The typed payload names these bindings as mandatory:

```text
point_in_time_factor_manifest_sha256
point_in_time_sector_manifest_sha256
factor_sector_exposure_bounds_manifest_sha256
m03r_risk_manifest_sha256
m03r_seed_checkpoint_ensemble_manifest_sha256
confidence_calibration_manifest_sha256
projection_execution_contract_sha256
inference_contract_sha256
source_archive_sha256
container_image_digest
data_manifest_sha256
```

Production receipts additionally bind the clean Git tree, dependency lock,
universe, benchmark, risk-free series, execution model, fold arrays, seed,
setting ID, qualification results, and admitted Kubernetes object.

The canonical deployed decision aggregates exactly five distinct seeded
members in ascending integer-seed order using
`five-seed-m03r-output-ensemble-v1`. It then applies the content-bound hard risk
projection exactly once. The seed/checkpoint hashes, cash asset ID, maximum
one-way turnover, PIT exposure/covariance as-of identity, Dykstra tolerance,
and iteration ceiling are receipt fields; none may be inferred from mutable
runtime defaults.

Branch names and mutable tags are not sufficient source identities. Any missing,
malformed, stale, or mismatched binding blocks launch and artifact publication.

## 13. Qualification boundary

This RFC and its typed protocol define the scientific design. They do not by
themselves establish real-data, CUDA, two-H100, execution-realism, or performance
qualification.

The present code is intentionally not launch authority. In addition to the
external bindings below, these integration gaps remain explicit blockers:

- the production chronological trainer does not yet route through the
  setting-bound M03R objective and post-seed projection;
- one governed, comparable execution route does not yet cover every causal
  setting (notably M00--M02, A05--A07, and A10);
- the confidence-calibration payload is not yet loaded and applied by the
  model; a syntactically valid manifest digest alone is insufficient;
- seed/checkpoint membership, cash identity, turnover, solver numerics, and
  decision-date/asset-axis identities are not yet verified through one
  immutable execution receipt;
- checkpoint ranking still needs evaluator-derived per-cell evidence rather
  than trainer-authored aggregate scalars; and
- the implemented two-stage linear projection plus benchmark-ray tracking-risk
  reduction has not yet qualified as the joint minimum-L2 constrained solve
  specified by this RFC.

These are fail-closed gaps, not optional follow-up polish. No M03R Job should be
created until they are resolved or a new immutable protocol generation changes
the corresponding scientific contract.

Before a governed real-data launch, the implementation must demonstrate:

- deterministic CPU replay;
- CUDA single-rank equivalence;
- CUDA two-rank/NCCL equivalence;
- interruption and exact checkpoint restart;
- projection reference, gradient, permutation, and active-set tests;
- FP32 and declared mixed-precision tolerance;
- content-bound PIT data, benchmark, factor, and sector inputs;
- evaluator-derived checkpoint metrics rather than trainer-authored scalars;
- identical trainer and evaluator economic ledgers;
- H100 capacity and throughput receipts;
- no concealed outer-data read before eligibility is sealed.

Synthetic qualification establishes mechanism correctness only. It is not
performance evidence.

## 14. Immutability rule

Any result-moving change—including an objective coefficient, temporal field,
factor family, exposure bound, benchmark, selection rule, cost convention,
execution timing, setting switch, or promotion gate—requires a new protocol
generation. It must not silently change the meaning of
`prelockbox-hold30-active-alpha-m03r-v4` or any stable setting ID listed here.
