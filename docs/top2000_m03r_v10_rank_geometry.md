# TOP2000 M03R-v10 rank-geometry research

This is a non-PHI, development-only research generation. It is not a trading
system, a business-production surface, reportable performance, or promotion
evidence.

## Why this generation exists

The completed v9 A04 predictive panel rejected all six eligible
setting-horizon candidates. Factor-residual ranking nevertheless produced a
positive top-minus-bottom spread in five of six folds at both 21 and 30
sessions, while its broad mean rank IC remained approximately zero and its
simple sleeve lost money. Benchmark-relative labels were materially worse.

V10 therefore does not change holding behavior, costs, risk limits, the
economic policy, or the predictive gate. It asks one narrower question:

> Can a loss aligned with broad cross-sectional ranks convert the residual
> predictor's extreme-decile separation into general IC without disconnecting
> the economically scaled mean and uncertainty tensors used by execution?

## Frozen initial panel

All settings use factor-residual targets and a fresh random initialization.
No v9 model or optimizer state may be reused.

| Index | Setting | Sole change |
| ---: | --- | --- |
| P0 | `V10-P0-factor-residual-standardized-listwise-control` | Reproduce the standardized-return listwise geometry in the fresh generation. |
| P1 | `V10-P1-factor-residual-rank-gaussian-correlation` | Replace only the ranking component with date-balanced correlation against average-tie Gaussian rank scores. |
| P2 | `V10-P2-factor-residual-rank-gaussian-21-30-only` | Starting from P1, give predictive-loss weight only to the eligible 21- and 30-session horizons. |

P1 and P2 use rank scores only inside the ranking component. The mean head
continues to predict factor-residual cumulative return in economic units, and
the scale head continues to represent uncertainty in those same units.
Robust-regression and distributional losses must not silently consume rank
scores.

The P2 horizon weights are the normalized v9 21/30 weights:

```text
5 sessions    0
21 sessions   7/15
30 sessions   8/15
63 sessions   0
```

## Bound limits and stop rule

```text
optimizer updates             exactly 64
early stopping                disabled
qualification evaluations     update 64 only
folds                          6 chronological folds
seed                           17
workers                        3
H100s per worker               2
maximum requested H100s        6
economic optimizer updates     0
2026 access                    forbidden
future-selected universe       true
reportable/promotable          false/false
```

The complete v9 gate is unchanged, including mean Spearman IC at least 0.020,
four positive-IC folds, positive spread and sleeve performance, and break-even
one-way cost at least 10 bp. If no setting-horizon pair passes, stop again.
Do not lower the gate, extend training automatically, or launch economic
policy optimization.

## Implementation state

The immutable protocol, complete predictive loss, typed training batch,
mutation receipt, update-64 checkpoint, worker plan, fold schedule, and
untouched-tail diagnostics are implemented. The rank
objective uses deterministic average ranks for ties, maps them to centered
Gaussian scores, standardizes predictions cross-sectionally, and optimizes a
date-balanced correlation surrogate. Huber and distributional terms continue
to consume factor-residual returns in economic units. Focused contract,
isolation, horizon-support, and gradient tests are present.

The v10 batch explicitly imports the audited v9 P0 architecture while binding
a separate v10 scientific setting and protocol. The optimizer step rejects
qualification access, update 64 continuation, and any alternate v9 setting.
The checkpoint publishes only at update 64, is evaluation-only, forbids v9
model/optimizer reuse, and cannot overwrite an existing path.

The worker remains exactly three Indexed completions, two H100s each, six
folds, and zero economic updates. The fold schedule wraps the audited v9
chronological geometry under a distinct v10 receipt and uses a new
setting-bound episode schedule. Qualification diagnostics now record the mean,
population standard deviation, median, and positive-date fraction of IC plus
prediction dispersion, target dispersion, predicted scale, and decile spread.

The simple-sleeve qualification wrapper and six-fold panel selection are now
implemented. The unchanged audited v9 sleeve executes only under its original
runtime protocol and P0 architecture identity; a separate v10 wrapper binds
that exact trace hash, arrays, risk state, source receipt, v10 setting, and v10
protocol. The fold result pairs diagnostics, trace, and sleeve evidence before
the six-fold gate is evaluated. Passing permits only minting a distinct future
economic generation; it never authorizes an economic Job directly.

The workflow entrypoint, package, and Seadragon lifecycle are not yet
v10-qualified. Consequently, no local package or remote GPU launch is
authorized at this point.

## Implementation map

- `protocol/hold30_alpha_m03r_v10_top2000_dev.py` — fresh identity, settings,
  predecessor evidence, resource ceiling, gates, and access prohibitions.
- `training/top2000_m03r_v10_rank_objective.py` — listwise control and
  rank-Gaussian correlation objectives.
- `training/top2000_m03r_v10_pretraining_step.py` — typed v10-over-v9 batch
  boundary and sole optimizer mutation receipt.
- `training/top2000_m03r_v10_checkpoint.py` — exclusive evaluation-only v10
  checkpoint with explicit imported-architecture and no-reuse fields.
- `training/top2000_m03r_v10_predictive_worker.py` — exact three-worker,
  six-H100 predictive-only panel plan.
- `training/top2000_m03r_v10_fold.py` — distinct deterministic episode
  schedule, optimizer-only training split, and update-64 tail opening.
- `training/top2000_m03r_v10_diagnostics.py` — fold-level IC stability,
  dispersion, scale, and spread evidence.
- `training/top2000_m03r_v10_selection.py` — exact imported-sleeve lineage,
  fold evidence, unchanged predictive gates, and LCB-based horizon selection.
- `tests/test_hold30_alpha_m03r_v10_top2000_dev_protocol.py` — immutable
  identity and fail-closed gate tests.
- `tests/test_top2000_m03r_v10_rank_objective.py` — tie handling, gradient,
  horizon-support, and malformed-input tests.
- `tests/test_top2000_m03r_v10_pretraining_step.py` and
  `tests/test_top2000_m03r_v10_checkpoint.py` — mutation, cursor, architecture,
  immutable checkpoint, external-hash, and no-clobber tests.
- `tests/test_top2000_m03r_v10_predictive_worker.py`,
  `tests/test_top2000_m03r_v10_fold.py`, and
  `tests/test_top2000_m03r_v10_diagnostics.py` — resource ceiling, schedule,
  split routing, stability, and collapsed-prediction tests.
- `tests/test_top2000_m03r_v10_selection.py` — sleeve lineage, six-fold gates,
  horizon mismatch, failed IC, and 30-session tie-break tests.

The next implementation boundary is the package-owned worker workflow and
immutable terminal artifact. No v9 training, trace, or qualification receipt
may be relabeled.
