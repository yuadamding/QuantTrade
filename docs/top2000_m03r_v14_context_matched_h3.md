# TOP2000 M03R-v14 executable-score-aligned h3 research

## Purpose and evidence boundary

V14 is a fresh predictive-only correction of the reviewed V13 design. It asks
whether the existing daily-aggregate encoder can learn usable three-session
factor-residual alpha after making the trained, diagnosed, and traded score
the same causal tensor.

V14 cannot reuse V13 model or optimizer state, run an economic optimizer, or
access 2026 outcomes. The TOP2000 universe remains future-selected, so all
results are development-only, nonreportable, and nonpromotable. Source and
tests do not authorize a remote run; a fresh source archive, real-data
preflight, package, execution authorization, and lifecycle receipts are still
required.

## Scientific settings

| Index | Setting | Sole difference |
| ---: | --- | --- |
| P0 | `V14-P0-action-projected-rank-h3` | Scale-invariant rank, robust regression, and detached uncertainty calibration. |
| P1 | `V14-P1-action-projected-no-rank-h3` | Removes rank loss; robust regression and detached uncertainty calibration remain. |

Both settings share exact initial parameter bytes, data, causal masks, origin
schedule, two-rank shards, optimizer geometry, checkpoint selection, and
qualification. Only the ranking objective differs.

## One executable score

For decision origin \(t\), define origin-action support \(A_t\) and complete
label support \(F_{t,3}\):

\[
A_t=\text{available at }t\cap\{w_t^{\mathrm{reg}}>0\},
\qquad
L_{t,3}=A_t\cap F_{t,3}.
\]

The action and label operators are distinct:

\[
s_t=P_t^A f_\theta(x_{\le t}),
\qquad
y_{t,3}=P_{t,3}^L R_{t,3}.
\]

The exact executable score \(s_t\) is used for rank loss, robust regression,
qualification IC and decile spread, fixed-rank sleeve construction,
signal-retention attribution, and checkpoint/batch identity. The raw mean is
diagnostic only.

A future-valid but origin-unavailable asset enters neither residual regression
nor loss. An origin-valid asset with an incomplete future path may remain
actionable but does not enter the label or diagnostic reduction.

## Numerics and fit quality

V14 precomputes a small QR-derived residual coefficient map once per operator.
Application remains differentiable with respect to the score and uses tensor
version checks instead of repeating QR and full content validation in the hot
path.

The h3 mean head starts at a Xavier gain of `0.025`, calibrated to the
approximately `0.02*sqrt(3)` return scale rather than the previous much wider
initial distribution. Rank normalization is dimensionless:

```text
economic score / h3 scale
-> cross-sectional centering
-> detached RMS with 0.05 floor
-> rank-Gaussian correlation
```

Uncertainty calibration cannot update the mean or shared encoder: the scale
head consumes a detached hidden state and its likelihood uses a detached mean.
This preserves scale evidence without allowing a currently unused uncertainty
objective to reshape the action score.

Each update records raw/executable/target RMS, raw-to-executable retention,
loss components, valid observations, gradient norms, and clipping status.
Gradients are reduced across ranks before float64 norm calculation and
fail-closed clipping. Model parameters and Adam state are verified finite
after mutation and before any receipt or checkpoint can be published.

## Sampling and qualification

V14 retains V13's corrected context geometry:

- one h3 target only;
- 252-session history for every training and qualification origin;
- every eligible origin once per deterministic epoch;
- paired setting-neutral schedules and complementary two-rank shards;
- one action followed by one earned post-fill return;
- six chronological 63-origin qualification tails; and
- no 2026 access.

The small 25-bp fixed-rank sleeve is a score-ordering diagnostic, not a final
portfolio policy. Economic policy training remains blocked until the frozen
projected-score, spread, gross/net return, cost, and retention gates pass.

## Current implementation status

The local V14 source includes protocol, policy, objective, causal batch
construction, reusable residual operators, optimizer mutation, checkpoint,
qualification, selection, package/workflow, and research lifecycle surfaces.
The focused V11-V14 regression run on 2026-08-14 passed `253` tests with one
environment-dependent skip. The focused V14 suite passed `53` tests; Ruff
passed on all V14 source and tests. After repairing the explicit legacy
Hold-30 source-to-test inventory for the landed V11-V14 protocols, the full
repository suite passed `1,708` tests with `7` skips.

These local results establish implementation consistency only. No V14
real-data structural receipt, source package, GPU result, or predictive result
is claimed by this document.

## Advancement rule

The existing predictive gate remains unchanged. A setting must independently
satisfy, among the other frozen breadth and retention rules:

```text
mean action-projected Spearman IC >= 0.020
positive mean-IC folds >= 4/6
positive spread folds >= 4/6
spread block-bootstrap LCB > 0
gross active-return LCB > 0
10-bp net active-return LCB > 0
aggregate break-even one-way cost >= 10 bp
median signal retention >= 0.50; every fold >= 0.20
median requested-to-executed retention >= 0.50; every fold >= 0.20
```

Failure does not authorize a lower threshold or economic training. It directs
the next study toward longer holding-aligned targets and then genuinely
ordered five-minute inputs, rather than another h3 loss-weight sweep.

## Local verification

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v14_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v14_*.py

conda run -n quanttrade ruff check \
  src/rl_quant/protocol/hold30_alpha_m03r_v14_top2000_dev.py \
  src/rl_quant/training/top2000_m03r_v14_*.py \
  src/rl_quant/workflows/top2000_m03r_v14_*.py \
  tests/test_hold30_alpha_m03r_v14_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v14_*.py
```
