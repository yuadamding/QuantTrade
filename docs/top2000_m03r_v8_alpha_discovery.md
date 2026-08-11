# TOP2000 M03R-v8 alpha-discovery development stage

**Status:** protocol, policy/pretraining, risk projection, chronological action,
structural sentinel, exact split/schedule, two-rank pretraining, immutable
checkpoint/resume, package, admission, and predictive worker boundaries
implemented; the first development panel did not pass predictive qualification,
and its objective-integrity correction requires fresh source-homogeneous evidence
**Research use:** non-PHI, development-only, nonreportable, and nonpromotable
**Protocol SHA-256:**
`e0633d5ef168c2317dd0e883bd7ff9d818c9c5e33ed3c459d81c4720148369c8`

## Why v8 is a new generation

The completed v7 Phase-0 audit found distinct checkpoint, model, optimizer,
alpha-prediction, and raw-action identities, but many causal settings still
produced identical requested and executed economic books. Canonical requested
active L1 norm was approximately 1.98 in folds 1--3, while projection retained
only 3.8%, 2.4%, and 0.9% of the requested signal. Canonical 21- and 30-session
rank IC was nonpositive on average, and its aggregate break-even one-way cost
was only 1.2 bp.

Changing that economic action is result-moving and therefore must not be
backfilled into an immutable v7 identity. The v8 TOP2000 generation remains a
future-selected mechanism-development surface; it cannot support a promotion
or investment-performance claim.

## Implemented local boundaries

| Surface | Module | Implemented contract |
|---|---|---|
| Frozen discovery protocol | `rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev` | raw 42/252-session context; training-fold-only 5/21/30/63 residual-alpha pretraining; 2-bp soft persistence; eight causal settings; 0/10/20/40-bp evaluation; nonreportable identities |
| V8 policy adapter | `rl_quant.training.top2000_m03r_v8_policy` | reviewed v7 encoder/alpha core under a disjoint v8 identity; four-horizon distributional scale head; exact 1.0/1.5 three-way-action temperatures; fixed-hazard source only for setting 6 |
| Causal target binding | `rl_quant.training.top2000_m03r_v8_pretraining_runtime` | differentiable single-path predictions; benchmark-relative future log returns; split-bounded 5/21/30/63 targets; CASH/availability masks; no outer-score route |
| Alpha loss and qualification | `rl_quant.training.top2000_m03r_v8_alpha_pretraining` | linear-memory date-balanced listwise ranking, Huber, and Gaussian losses; frozen 10%/35%/40%/15% horizon weighting; per-fold IC/spread evidence; exact six-fold 21/30-session qualification gate |
| Pretraining optimizer | `rl_quant.training.top2000_m03r_v8_pretraining_optimizer` | disjoint encoder/prediction-head AdamW groups at 2e-5/1e-4; unrelated action/risk heads excluded; 64-update ceiling and four-check patience frozen |
| Pretraining step and resume | `rl_quant.training.top2000_m03r_v8_pretraining_step`, `top2000_m03r_v8_pretraining_checkpoint` | equal two-rank loss averaging after date-balanced, frozen horizon weighting; gradient average; nonfinite rejection before AdamW mutation; earliest-tie early stopping; semantic state hashes; RNG-bound immutable checkpoint validation |
| Fold and worker geometry | `rl_quant.training.top2000_m03r_v8_plan`, `top2000_m03r_v8_pretraining_fold`, `rl_quant.workflows.top2000_m03r_v8_pretraining` | 63 inner-validation origins plus terminal support entirely inside each training fold; paired 378-state schedule; exact H100/NCCL startup; rank model/optimizer/RNG equality; four-update qualification and six-fold worker modes |
| Incremental active proposal | `rl_quant.execution.cost_aware_active_policy` | learned-exit/de-risk anchor first; cost and uncertainty gate; lower retention than entry hurdle; confidence-bounded new/reallocation risk; long-only capacity; exact zero-sum buys and sells |
| Qualified risk projector | `rl_quant.execution.top2000_m03r_v8_projection` | content-bound nonzero factor/sector slabs; active beta, TE, caps, availability, and gross limits; 1.0x/1.5x radial feasibility with one-time PSD qualification |
| Chronological action | `rl_quant.training.top2000_m03r_v8_runtime` | repaired, raw hazard, qualified hazard, gated proposal, projected, and executed books remain separate; policy setting and delayed-fill builder are bound together |
| Eight-row sentinel | `rl_quant.training.top2000_m03r_v8_sentinel` | CPU-only stage-local causality check; observed uniqueness 5 causal inputs, 7 gated proposals, 8 projected books, and 8 executed books |
| Focused regressions | `tests/test_cost_aware_active_policy.py` and `tests/test_top2000_m03r_v8_*.py` | gradients, chronology, target bounds, action temperature, projection tamper rejection, distinct-policy stages, and fail-closed inputs |
| Protocol regressions | `tests/test_hold30_alpha_m03r_v8_top2000_dev_protocol.py` | disjoint identity, exact shared contract, one-field ablations, research gates, hashes, and resolution failures |

The incremental 2% one-way value is a per-decision proposal ceiling, not a
tracking-error budget, turnover target, or performance claim. The final
factor, sector, active-beta, 6% TE, 1% stock-cap, availability, and benchmark-
anchoring projector is implemented and mandatory. Unlike v7's exact-zero
factor projection, v8 requires nonzero content-bound factor/sector slabs, so
the 1.5x row changes a real feasible region.

## Incremental action semantics

The action boundary receives a feasible `hazard_anchor_weights` book after
learned exits, forced repairs, and explicit de-risking. For each risky asset it
forms benchmark-relative expected alpha and a hysteretic hurdle:

```text
hurdle = multiplier(held) * (one_way_cost + uncertainty_multiplier * uncertainty)
gate = sigmoid((abs(centered_alpha) - hurdle) / temperature)
```

The retention multiplier is 0.5 and the new-entry multiplier is 1.0. Positive
and negative gated strengths define capacity-bounded buy and sell directions.
The two sides receive the same notional, so the incremental delta sums exactly
to zero. Its one-way size cannot exceed:

```text
signal_confidence * 2%
```

Consequences:

- zero confidence returns the hazard anchor exactly;
- confidence cannot reverse or suppress an exit already present in the anchor;
- transaction cost and uncertainty shrink proposed reallocation;
- extreme scores retain their relative cross-sectional ordering instead of
  being clipped into a common replacement book;
- factor/beta/TE projection remains a later, independently governed stage.

## Frozen eight-setting panel

| Index | Stable setting ID | Sole change from setting 0 |
|---:|---|---|
| 0 | `V8-0-pretrained-alpha-costgate-top2000-dev-v1` | Reference: pretrained alpha, ranking loss, cost gate, learned hazard, canonical factor bounds. |
| 1 | `V8-1-no-alpha-pretraining-top2000-dev-v1` | Joint random initialization instead of training-fold pretraining. |
| 2 | `V8-2-no-ranking-loss-top2000-dev-v1` | Ranking-loss weight 0.50 to 0. |
| 3 | `V8-3-softer-exact-hold-top2000-dev-v1` | Raise the three-way straight-through action temperature from 1.0 to 1.5; deterministic labels for fixed logits remain unchanged while the training surrogate is softened. |
| 4 | `V8-4-no-cost-gate-top2000-dev-v1` | Disable the cost/uncertainty gate. |
| 5 | `V8-5-strong-cost-gate-top2000-dev-v1` | Stronger cost/uncertainty gate. |
| 6 | `V8-6-fixed-exit-hazard-top2000-dev-v1` | Freeze exit hazard to the structural 30-session prior. |
| 7 | `V8-7-relaxed-factor-bounds-top2000-dev-v1` | Relax factor/sector bounds to 1.5x while retaining beta and TE controls. |

Direct-Sharpe optimization is intentionally absent. This panel first asks
whether predictive cross-sectional alpha can survive cost-aware execution.

## Predictive qualification and objective-integrity correction

The first source-homogeneous development panel completed its operational
worker geometry but did not pass the frozen scientific gate. Its best mean
21-session rank IC was approximately 0.0142; the gate remains mean rank IC at
least 0.02 with positive IC in at least four of six folds at 21 or 30 sessions.
The threshold must not be weakened to convert that result into a pass.

Post-run source review found that the loss averaged the four horizons equally
even though the frozen protocol assigns weights `(0.10, 0.35, 0.40, 0.15)`.
That defect gave only 50% of objective weight to the 21/30-session
qualification horizons instead of the intended 75%. The implementation now:

1. averages supported dates independently within each horizon;
2. applies the frozen horizon weights to listwise, Huber, and distributional
   components;
3. renormalizes only across horizons genuinely absent from a bounded batch;
4. regression-tests the combined loss against separately evaluated horizons.

Artifacts produced by the earlier source bytes remain immutable failed
development evidence. They must not be resumed, relabeled, or reused as
scientifically qualified. Any retry must use a fresh source archive, package
plan, run identity, suspended live-admission binding, and terminal/cleanup
receipts. The corrected retry must still pass the original gate before any
economic policy optimization.

After predictive qualification, the remaining scientific sequence is:

1. materialize content-bound training-fold risk slabs/covariance from the
   pre-2026 development data and qualify them with the implemented projector;
2. freeze the confidence-calibration handoff and use the separately
   receipt-gated economic optimizer/worker;
3. begin economic optimization with 64 updates and allow 128 only through an
   explicit curve-based continuation receipt; never double every setting
   automatically.

Repository documentation does not authorize a Kubernetes mutation or 2026
outcome access; remote research execution still requires the approved
receipt-gated Seadragon lifecycle.

## Local verification

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest \
  tests/test_cost_aware_active_policy.py \
  tests/test_hold30_alpha_m03r_v8_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v8_alpha_pretraining.py \
  tests/test_top2000_m03r_v8_policy.py \
  tests/test_top2000_m03r_v8_pretraining_optimizer.py \
  tests/test_top2000_m03r_v8_pretraining_step.py \
  tests/test_top2000_m03r_v8_pretraining_checkpoint.py \
  tests/test_top2000_m03r_v8_plan.py \
  tests/test_top2000_m03r_v8_pretraining_fold.py \
  tests/test_top2000_m03r_v8_pretraining_worker.py \
  tests/test_top2000_m03r_v8_pretraining_runtime.py \
  tests/test_top2000_m03r_v8_projection.py \
  tests/test_top2000_m03r_v8_runtime.py \
  tests/test_top2000_m03r_v8_sentinel.py -q

conda run -n quanttrade ruff check \
  src/rl_quant/execution/cost_aware_active_policy.py \
  src/rl_quant/execution/top2000_m03r_v8_projection.py \
  src/rl_quant/protocol/hold30_alpha_m03r_v8_top2000_dev.py \
  src/rl_quant/training/top2000_m03r_v8_alpha_pretraining.py \
  src/rl_quant/training/top2000_m03r_v8_policy.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_optimizer.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_step.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_checkpoint.py \
  src/rl_quant/training/top2000_m03r_v8_plan.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_fold.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_runtime.py \
  src/rl_quant/training/top2000_m03r_v8_runtime.py \
  src/rl_quant/training/top2000_m03r_v8_sentinel.py \
  src/rl_quant/workflows/top2000_m03r_v8_pretraining.py \
  tests/test_cost_aware_active_policy.py tests/test_top2000_m03r_v8_*.py

PYTHONPATH=src conda run -n quanttrade mypy --strict \
  src/rl_quant/execution/cost_aware_active_policy.py \
  src/rl_quant/execution/top2000_m03r_v8_projection.py \
  src/rl_quant/protocol/hold30_alpha_m03r_v8_top2000_dev.py \
  src/rl_quant/training/top2000_m03r_v8_alpha_pretraining.py \
  src/rl_quant/training/top2000_m03r_v8_policy.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_optimizer.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_step.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_checkpoint.py \
  src/rl_quant/training/top2000_m03r_v8_plan.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_fold.py \
  src/rl_quant/training/top2000_m03r_v8_pretraining_runtime.py \
  src/rl_quant/training/top2000_m03r_v8_runtime.py \
  src/rl_quant/training/top2000_m03r_v8_sentinel.py \
  src/rl_quant/workflows/top2000_m03r_v8_pretraining.py
```
