# TOP2000 M03R-v13 context-matched direct-h3 research

## Purpose and authority

V13 is the fresh predictive-only successor to the completed negative V12 run
and its a08 inference audit. It tests whether the daily-aggregate encoder can
learn a three-session factor-residual score after removing the proven
training/evaluation context mismatch and the disconnected rank head.

V13 cannot reuse V12 model or optimizer state, run an economic optimizer, or
access 2026 outcomes. The TOP2000 universe remains future-selected, so every
result is development-only, nonreportable, and nonpromotable. Source code and
local tests do not authorize a package or remote launch.

## Scientific settings

| Index | Setting | Sole difference |
| ---: | --- | --- |
| P0 | `V13-P0-direct-rank-gaussian-economic-h3` | Rank-Gaussian, robust regression, and distributional losses all train the h3 economic mean/scale used by the future sleeve. |
| P1 | `V13-P1-direct-economic-h3-control` | Removes rank loss and renormalizes robust/distributional weights to 0.60/0.40. |

There is no separate rank-score head. The exact h3 economic mean is both the
rank-loss input and the execution score. Only the three-session target is
constructed; 5-, 21-, 30-, and 63-session losses cannot dilute the selected
horizon or consume its return support.

## Corrected fold and sampling geometry

The exact 1001-state pre-2026 cache remains immutable. The study uses 2022 as
the 252-session observation burn-in and six expanding folds. Qualification
starts are `469, 562, 655, 748, 841, 934`; each fold has 63 qualification
origins and a 30-session purge. The final h3 qualification/support boundary is
state 1001, so the final fold reaches the 2025-12-29 cache boundary without
opening 2026.

Every training origin has local position at least 251. Qualification origins
have local positions 311 through 373. This replaces V12's 0%–16.6% full-context
training fractions with 100% full-context training and qualification.

Training uses eight deterministic epochs. Eligible origin counts are
`184, 277, 370, 463, 556, 649`; each origin appears exactly once in each epoch.
Updates contain at most 64 adjacent origins and total `24, 40, 48, 64, 72, 88`
per fold. P0 and P1 share the same episode, global-origin, and complementary
two-rank shard schedule. Uneven final batches retain every origin rather than
silently dropping one for equal rank cardinality.

Each rank scales its local mean loss by `local_origin_count / global_origin_count`
before backward. Gradients are then explicitly summed across the two-rank NCCL
group before clipping and optimizer mutation. This makes odd final shards an
exact global-origin mean rather than an equal-rank mean, and every step receipt
records local/global counts, the weight, and completed synchronization.

## Causal target/action separation

Future label-path availability may determine whether an h3 target is
diagnosable. It cannot shape the action universe. V13 therefore publishes two
operators per origin:

```text
target operator = future h3 label-path validity + origin risk weights
action operator = decision-origin availability + origin risk weights
```

The target operator constructs the factor-residual label. A future sleeve must
use only the action operator, then perform fill-time availability repair. Both
operators bind the same point-in-time exposure row, asset axis, exposure order,
weights, and source receipt.

## Deterministic qualification sleeve

V13 freezes a small, nonlearned rank sleeve at 25 bp one-way active mass. This
choice is explicitly part of the new development identity and was informed by
the completed V12 a08 post-hoc audit; it is not confirmatory evidence. The
sleeve applies the action operator, ranks the remaining score, anchors to C1,
performs the common risk projection, and evaluates 0/10/20/40-bp costs.

Chronology is exact and causal:

```text
decision at t -> fill at t+1 -> turnover -> return from t+1 to t+2 -> drift
```

There are 63 actions and 63 earned returns. Target-path availability can affect
IC/spread diagnostics only; it cannot affect the action mask. The trace records
signal-residualization retention, risk-projection retention, action/operator
identities, requested and executed weights, scale quantiles, turnover, and the
complete cost ladder.

## Current implementation boundary

The repository currently contains and locally verifies:

- the immutable two-setting, h3-only research protocol;
- the full-context six-fold geometry through the pre-2026 cache boundary;
- setting-neutral, every-origin-once-per-epoch two-rank schedules;
- one direct h3 mean and scale output with no disconnected rank head;
- the rank/economic objective over that same mean;
- separate target and action residual operators;
- a setting-neutral paired-input binding and executable two-rank fold-update
  runtime;
- fail-closed optimizer mutation with exact before/after state identities;
- immutable, no-follow, byte- and semantic-hash-bound checkpoint publication
  and strict reload; and
- worker/panel plans that bind the variable, full-epoch fold update counts;
- exact checkpoint-to-qualification-tail replay;
- the causal fixed-rank sleeve described above; and
- one joint six-fold 10/21/30-session moving-block bootstrap and fail-closed
  selection receipt;
- a common-parameter artifact that binds semantic bytes, serialized bytes, and
  architecture shape/dtype identity across P0/P1; and
- a content-bound two-worker/four-H100 package-plan contract that remains
  launch-disabled until its builder, entrypoint, and lifecycle are complete.

The completed local surface has focused regression, Ruff, and strict-mypy
coverage. It also includes common initial-state publication and a real-cache
preflight over every scheduled origin. Package construction, workflow receipts,
and Kubernetes lifecycle remain to be qualified under the V13 identity. Until
those surfaces and their exact tests exist, no GPU launch is authorized.

The local real-data preflight used the exact existing pre-2026 cache and risk
artifacts. It qualified all 716 unique scheduled origins (`251` through `996`),
with minimum target/action support of 1,715/1,723 assets and minimum residual
degrees of freedom of 1,701/1,709. Target and action masks differed at all 716
origins, so the causal separation is result-material rather than cosmetic. The
immutable receipt SHA-256 is
`b3cff7583d4e6f04cb6881923b3ebea516599f4df30c544da4c5a301acc1ad1d`
and its file SHA-256 is
`0137bbb6c299c25634dc9677d0277a86a6a2685a78d9c89903c44e8627057475`.
This is structural evidence only; it does not inspect 2026 or authorize a GPU
launch.

## Advancement rule

A V13 setting must satisfy the unchanged predictive and economic gates before
any separate economic generation can be designed:

```text
mean Spearman IC >= 0.020
positive mean-IC folds >= 4/6
positive median-IC folds >= 4/6
folds with positive-date IC fraction > 0.50 >= 4/6
positive spread folds >= 4/6
21-session block-bootstrap spread LCB > 0
gross active-return LCB > 0
10-bp net active-return LCB > 0
aggregate break-even one-way cost >= 10 bp
median signal-residualization retention >= 0.50; every fold >= 0.20
median requested-to-executed retention >= 0.50; every fold >= 0.20
```

Failure stops the daily h3 path. It does not authorize lowering a threshold,
opening 2026, or adding economic settings. The next representation question
would require a separately materialized and validated ordered five-minute
sequence; the current cache contains daily OHLCV aggregated from five-minute
source bars and is not an intraday token sequence.

## Local verification

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v13_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v13_*.py
```
