# TOP2000 M03R-v16 holding-aligned selection research

## Decision

V16 is the fresh predictive-only successor to the completed V15 h3 screen.
V15's best mean projected IC was `0.01013`, its best break-even cost was only
`1.84 bp`, and both 10-bp net lower bounds were negative. Further h3 loss or
learning-rate tuning is prohibited.

The new question is whether the existing daily representation contains a
slower selection signal aligned with the intended approximately 30-session
holding behavior. V16 does not reuse V15 model or optimizer state, run RL or
an economic optimizer, or access 2026 outcomes. The future-selected TOP2000
surface remains development-only, nonreportable, and nonpromotable.

## Scientific settings

| Index | Setting | Primary selection target |
| ---: | --- | --- |
| P0 | `V16-P0-h21-selection-h3-timing` | cumulative 21-session factor-residual log return |
| P1 | `V16-P1-h30-selection-h3-timing` | cumulative 30-session factor-residual log return |
| P2 | `V16-P2-survival30-selection-h3-timing` | normalized survival-weighted 1–30-session factor-residual log return |

All settings share one architecture, initial parameter bytes, optimizer
geometry, date schedule, risk source, action projection, timing target, and
qualification rules. The sole scientific difference is the primary selection
target.

## Selection and timing separation

The policy emits two distributions:

```text
long-horizon selection mean and scale
h3 timing mean and scale
```

The selection score answers which stocks deserve capital. The h3 score is an
auxiliary timing output for a later entry/exit controller; it cannot determine
the V16 selected portfolio or qualify an economic generation by itself.

The survival target uses the structural daily hazard `1/30`:

\[
S(k)=(29/30)^{k-1},
\qquad
y^{\mathrm{hold}}_t=
\frac{\sum_{k=1}^{30}S(k)r^{\mathrm{res}}_{t+k}}
{\sum_{k=1}^{30}S(k)}.
\]

Every weight through day 30 is positive. This is a soft value horizon, not a
minimum hold or forced expiry.

The existing daily cache supports only the frozen development convention:

```text
observe through close t
→ fill at close t+1
→ earn the declared post-fill return horizon
```

V16 is therefore a next-close-fill diagnostic, not a live execution claim. A
next-open or causal VWAP contract requires the later ordered intraday surface
and must receive a fresh protocol identity.

## Objective and calibration

The score stage trains the exact action-projected means:

```text
0.85 × robust selection loss
0.15 × robust h3 timing loss
```

No ranking term is included in the first target screen because V15's paired
Huber control exceeded the corrected rank setting. This prevents another
rank-loss study from confounding the target comparison.

Uncertainty calibration is a separate stage. After the training-only selected
mean checkpoint is frozen, only the selection and timing scale heads may be
fit from training/inner-validation residuals. Scale likelihood cannot reshape
the encoder or either mean.

## Context and split geometry

V16 uses the exact 1,001-state pre-2026 cache and maximum 30-session target
support. Each episode contains 345 states:

```text
252 causal context states
63 paired optimizer/qualification origins
30 future return transitions
```

Full 63-origin blocks occupy local positions 251–313 in both training and
qualification. All eligible optimizer origins appear once per epoch. A
31-session optimizer-to-validation embargo prevents any h30 target from
touching the training-only validation slice. Six 63-origin outer tails remain
qualification-only, and the final fold consumes the last admissible pre-2026
target row.

Checkpoint search is limited to a maximum of 24 score epochs with a minimum of
four and patience four, using only chronological inner-validation evidence.
The outer qualification tails may not select an epoch, target, or setting.

## Qualification path

The final implementation must retain two distinct deterministic diagnostics:

1. a small fixed-rank ordering sleeve for projected IC and tail attribution;
2. an overlapping cohort sleeve matched to the setting's declared selection
   horizon, with causal carry, fill repair, absolute and incremental costs.

The cost ladder is frozen at `0, 1, 2, 3, 5, 10, 20, 40 bp`. A setting cannot
advance unless its own selection target independently satisfies the unchanged
IC, fold-breadth, spread-LCB, gross-LCB, 10-bp net-LCB, and 10-bp break-even
gates. No passing setting means no RL or economic training.

## Current implementation boundary

The repository currently implements the immutable protocol, origin-aligned
fold/update schedule, one common no-clobber initial state, separate
selection/timing raw outputs, action-projected causal batch construction,
score-stage mutation, frozen-mean scale calibration, no-clobber epoch
checkpoints, and training-only checkpoint selection. Full state hashes occur at
artifact boundaries rather than every optimizer update. Package-owned
structural/static/capacity gates, cohort qualification, worker/lifecycle
integration, and remote execution remain intentionally unimplemented and
unauthorized.

If all three corrected daily settings fail, stop daily target/loss tuning and
move to an ordered five-minute stock-day encoder under a new protocol. Quotes
and trades follow only after the five-minute control establishes incremental
projected signal.

## Local verification

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v16_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v16_*.py

conda run -n quanttrade ruff check \
  src/rl_quant/protocol/hold30_alpha_m03r_v16_top2000_dev.py \
  src/rl_quant/training/top2000_m03r_v16_*.py \
  tests/test_hold30_alpha_m03r_v16_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v16_*.py
```
