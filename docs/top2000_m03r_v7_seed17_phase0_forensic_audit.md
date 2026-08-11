# TOP2000 M03R-v7 seed-17 Phase-0 forensic audit

This guide documents the inference-only audit that must precede any M03R-v8
training allocation. It applies to the completed twelve-setting, six-fold,
seed-17 TOP2000 diagnostic. All outputs remain development-only,
nonreportable, and nonpromotable because the universe was selected using
future data.

## Why a new replay artifact is required

The completed fold-execution receipts correctly retain the compact mechanism
screen evidence:

- 20-bp policy and C1 net returns;
- active log returns;
- total, discretionary, and forced turnover;
- discretionary sale-age and terminal-age arrays.

They do not retain benchmark turnover, gross returns, residual-head prediction
arrays, or the requested/post-hazard/post-projection/executed weight books.
Those values must not be inferred from rounded benchmark tables. The Phase-0
workflow therefore performs a deterministic, no-gradient replay of each
frozen final-update checkpoint and requires the replayed compact arrays to
match the original receipt hashes before publishing any new diagnostic.

The replay does not train a model, select a checkpoint, open 2026 outcomes, or
change any existing training artifact.

## Implemented surfaces

| Requested audit | Package surface | Exact output |
| --- | --- | --- |
| `m03r_v7_trace_audit.py` | `rl_quant.evaluation.m03r_v7_trace_audit` | model/optimizer/calibration identity; requested action; all weight-stage, gross/net-return, and turnover hashes; pairwise trace comparisons |
| `m03r_cost_ladder_evaluator.py` | `rl_quant.evaluation.m03r_cost_ladder_evaluator` | policy and C1 gross return/cost; 0/10/20/40-bp active return and IR; break-even one-way cost |
| `m03r_alpha_head_diagnostics.py` | `rl_quant.evaluation.m03r_alpha_head_diagnostics` | date-balanced Pearson/rank IC, IC IR, decile spread, top-quintile precision, direction, calibration error, dispersion, and optional governed breakdowns |
| `m03r_projection_attribution.py` | `rl_quant.evaluation.m03r_projection_attribution` | requested-to-executed active-book norms, stage distances, binding frequency, signal retention, optional covariance TE, and optional score attribution |
| `m03r_setting9_risk_audit.py` | `rl_quant.evaluation.m03r_setting9_risk_audit` | initialization/startup-turnover checks, ex-ante/realized TE availability, common-control inventory, and fail-closed causal status |
| Frozen setting runner | `rl_quant.workflows.top2000_m03r_v7_forensic_audit` | six immutable fold bundles plus one setting receipt; cache validation/load occurs once per setting |
| Indexed worker | `rl_quant.workflows.top2000_m03r_v7_forensic_audit_worker` | explicit local-completion-to-scientific-setting map, one-H100 startup proof, and one no-clobber completion receipt |

The fold runner requires explicit cache, training-plan, evaluation-source, and
training-source hashes. A completed retry validates the existing receipt and
array file and performs no replay. A partial or conflicting output is a hard
no-clobber failure.

Panel aggregation consumes the trace-array inventory from
`forensic_trace.arrays`; `array_sha256` belongs to other receipt families and
is not an alias. Aggregation-only recovery must bind all completed worker,
terminal, and exact-cleanup receipts, write to a disjoint continuation path,
and never replay a validated GPU worker merely to repair a panel writer.

## Weight-stage semantics

The existing runtime retains these exact economic books:

1. `decision_weights`: the carried post-cost book visible to the actor;
2. `requested_weights`: fill-time repaired book plus the actor's age-aware
   requested delta;
3. `post_hazard_weights`: equal to the built request because the retained
   action already includes exact-HOLD and hazard decisions;
4. `post_projection_weights`: repaired book plus the factor-projected
   constructed delta;
5. `executed_weights`: the post-cost filled book.

There is no retained pre-exact-HOLD target book in the frozen runtime. The
projection receipt marks that field unavailable rather than inventing it.
Likewise, ex-ante tracking error is unavailable unless a separately bound
date-specific covariance tensor is supplied.

## One-setting invocation

Run only from a reviewed immutable evaluation bundle against the exact
source-homogeneous training tree. The paths below are placeholders, not a
launch authorization:

```bash
python -m rl_quant.workflows.top2000_m03r_v7_forensic_audit \
  --setting-root /approved/output/completion-00-setting-00 \
  --cache-path /approved/package/cache.pt \
  --cache-sha256 <64-hex> \
  --training-plan-file-sha256 <64-hex> \
  --evaluation-source-inventory-sha256 <64-hex> \
  --source-training-archive-sha256 <64-hex> \
  --all-folds \
  --output-root /approved/audit/setting-00 \
  --device cuda:0
```

`--fold-index 0` remains available for a bounded diagnostic, but the production
audit worker uses `--all-folds`: one GPU owns one setting and evaluates folds
`0..5` serially after one verified-cache load. This avoids six redundant
multi-gigabyte cache loads per setting.

Before execution, bind the exact 72 setting/fold inputs and output paths. Do
not discover them by directory scanning. A single-GPU replay is sufficient;
the original two-H100 training geometry is historical evidence and must not
be repeated for inference.

## Interpretation gates

- A causal setting pair with identical executed-weight hashes fails the
  distinct-policy gate unless a predeclared mathematical equivalence explains
  it.
- Gross active return is policy gross minus C1 gross. Net active return uses
  the separate policy and C1 turnover paths; assuming zero C1 cost is not an
  exact decomposition.
- Confidence-decile diagnostics are unavailable for the old development route
  when only its uncalibrated scalar sizing output exists.
- Setting 9 is not a clean factor-neutrality ablation if disabling its route
  also removes the effective active-beta/TE control surface or if ex-ante risk
  evidence is unavailable.
- No v8 setting should be frozen until all 72 replays match their original
  compact hashes and the pairwise distinctness, cost, alpha-head, projection,
  and setting-9 reports have been reviewed.

## Local verification

```bash
.venv/bin/pytest -q \
  tests/test_m03r_v7_phase0_audit.py \
  tests/test_top2000_m03r_v7_dev_validation.py

.venv/bin/ruff check \
  src/rl_quant/evaluation/m03r_v7_trace_audit.py \
  src/rl_quant/evaluation/m03r_cost_ladder_evaluator.py \
  src/rl_quant/evaluation/m03r_alpha_head_diagnostics.py \
  src/rl_quant/evaluation/m03r_projection_attribution.py \
  src/rl_quant/evaluation/m03r_setting9_risk_audit.py \
  src/rl_quant/workflows/top2000_m03r_v7_forensic_audit.py \
  tests/test_m03r_v7_phase0_audit.py
```

These checks validate software behavior only. They do not establish that the
72 frozen checkpoints have been replayed or that a new model has positive
alpha.

## Completed A07 audit and gate decision

The exact A07 workers completed and validated all twelve settings and 72 fold
artifacts. Both GPU Jobs were exact-cleaned before panel aggregation. The
original panel writer then failed on a schema lookup: it requested the legacy
scalar `forensic_trace.array_sha256`, while the trace contract intentionally
publishes the per-array map at `forensic_trace.arrays`.

The immutable aggregation-only continuation v3 fixed that lookup without
replaying a worker or touching Kubernetes. It binds the original supervisor
failure, both prior safe continuation failures, the package, publication, and
process identities, both phase terminal and cleanup receipts, and all 72 fold
receipt hashes. Its outputs are:

- success file SHA-256:
  `76bbe89b0b51d8f355b6011648e5de51d937aa16f45129847b603fc99a7c1105`;
- panel file SHA-256:
  `fd30e16dd44bf01b4fc43f47183d5b7745e5d78ff99ca19111ca293ff94ccc97`;
- setting count: 12;
- fold count: 72;
- GPU worker replays: 0;
- retraining or checkpoint selection: none.

The scientific decision is **no-go for a broad v8 launch**. All checkpoints
and aggregate raw-action traces are distinct, but multiple causal settings
produce byte-identical requested and executed economic books. Canonical
projection retention falls to 0.038, 0.024, and 0.009 in folds 1–3, and the
canonical 21/30-session alpha-head ICs are nonpositive on average. The exact
cost ladder finds only +0.038% mean gross active return for canonical and a
1.2-bp aggregate break-even one-way cost; its 20-bp net active result is
-0.613%.

The next implementation stage must therefore change the active-policy
interface before retraining: operate on bounded incremental active weights,
pretrain and validate the 21/30-session alpha representation, introduce a
cost/uncertainty no-trade band, and preserve active-beta/TE controls during any
factor-bound relaxation. The 30-session holding prior does not need to be
strengthened. The disjoint local implementation contract is documented in the
[M03R-v8 alpha-discovery stage](top2000_m03r_v8_alpha_discovery.md); its
existence does not authorize training.
