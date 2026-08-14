# TOP2000 M03R-v11 a15 post-hoc inference audit

## Purpose and evidence status

This is a separate, immutable, inference-only development study of the exact
M03R-v11 a15 update-64 checkpoint bytes. It was introduced because the a15
result has an important mismatch:

- P0 has positive mean gross active return and positive gross-return block
  lower bounds, but its broad rank IC and top-minus-bottom uncertainty do not
  qualify;
- P0 remains negative at the 10-bp net-return lower bound and frequently uses
  a large part of the action budget;
- P1 has the strongest broad IC, but its prediction magnitude collapsed and
  its deterministic sleeve is uneconomic;
- P1 and P2 produce nearly identical economic traces, so repeating the P2
  horizon-weight ablation would add little information.

The audit asks whether P0 contains a real, target-blind tail signal that can
survive deterministic controls and lower action caps. It also records the
missing action, probability, scale, and P&L attribution needed to diagnose the
P1 scale collapse.

This audit is post-hoc exploratory evidence. It is development-only,
nonreportable, and nonpromotable. It cannot train a model, select a checkpoint,
mint an economic generation, or access 2026 outcomes.

## Frozen contract

The corrected protocol is
`top2000-dev-m03r-v11-a15-posthoc-inference-audit-v2`. It binds:

```text
source checkpoints       exact v11 update-64 files, reloaded by file hash
settings                 P0 and P1 only
horizons                 21 and 30 sessions
action-cap ladder        200, 150, 100, and 50 bp one-way per decision
negative controls        zero signal, sign-flipped signal, shuffled signal
cost ladder              0, 10, 20, and 40 bp one-way
quantile curves          deciles and vigintiles
bootstrap blocks         10, 21, and 30 sessions
bootstrap replicates     10,000 common, fold-bounded draws
risk identity            exact semantic inputs plus explicit numeric-state hash
economic updates         0
2026 access              forbidden
```

The shuffle is deterministic and binds the protocol shuffle seed, setting,
fold, horizon, origin, and variant. It permutes only factor-qualified risky
assets, then reapplies the exact residual operator. Outcomes and targets never
enter action construction.

## Implemented surfaces

- `rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit` freezes the
  audit identity and variants.
- `rl_quant.training.top2000_m03r_v11_a15_inference_audit_runtime` replays the
  chronological action path and records exact arrays and hashes.
- `rl_quant.training.top2000_m03r_v11_a15_inference_audit_fold` rebuilds the
  untouched v11 qualification context and consumes a strictly reloaded
  checkpoint.
- `rl_quant.training.top2000_m03r_v11_a15_inference_audit_plan` binds the exact
  completed a15 package, authorization, two worker terminals, twelve fold
  terminals, twenty-four horizon checkpoints, terminal evidence, and cleanup
  receipt.
- `rl_quant.training.top2000_m03r_v11_a15_inference_audit_package` authorizes
  only two one-H100 inference workers and rejects training, checkpoint
  selection, economic generation, and 2026 access.
- `rl_quant.evaluation.top2000_m03r_v11_a15_inference_audit` produces fold and
  six-fold panel evidence with fold-bounded moving-block inference.
- `rl_quant.workflows.top2000_m03r_v11_a15_inference_audit` revalidates the
  live parent evidence, reloads each exact checkpoint from the read-only
  parent output, and publishes no-clobber tensor and panel artifacts.
- `rl_quant.workflows.top2000_m03r_v11_a15_inference_audit_static_validate`
  is the zero-GPU same-image import, source-inventory, package, and parent
  lineage gate.

The fold adapter validates the worker, cache, checkpoint, horizon, asset axis,
risk source, qualification origins, and residual-operator lineage before
replay. The derived factor-plus-diagonal tensors are recomputed from exact
cache, exposure, projector, estimator, origin, and asset-axis identities. Their
new numeric byte hash and the parent numeric byte hash are both retained, but
cross-node floating-point bytes are not treated as a portable semantic
identity. Publication uses a separate source-only package and read-only
mounts of the exact parent package and output. This avoids copying the cache,
risk surface, and checkpoint files while preserving their hashes and lifecycle
lineage. Remote execution still requires fresh static, capacity, admission,
activation, terminal, and cleanup receipts.

The first capacity attempt, `a03`, is immutable failed development evidence.
It reproduced the exact qualification source-array and residual-operator roots
but rejected a cross-node recomputation because the derived risk tensor byte
hash differed. The Job reached the H100 startup guard, published no audit
cursor, preserved terminal and cleanup evidence, and was exact-cleaned. Its
model or output state is never resumed. The corrected `a04` identity replaced
only that invalid byte-equality gate, but its zero-GPU static Job then failed
before publication because Kubernetes auto-created an absent PVC `subPath` as
`root:root` mode `0700`; the non-root validator could not write its terminal.
That Job and its UID-owned Pod were terminal-captured and exact-cleaned. The
fresh `a05` package keeps the scientific v2 protocol unchanged and adds one
fail-closed operational invariant: before activation, the pure-file preparer
must derive the host output path from the rendered writable PVC mount, create
every task-owned component without following symlinks, prove the controller
UID/GID matches the Pod security context, and require the final phase root to
be empty and worker-owned. Neither failed attempt's model or output state is
resumed, and no source, model, checkpoint, risk-input, or residual-operator
binding is weakened.

## Diagnostics

Each replay records:

```text
pretrade, anchor, requested, and projected weights
factor-feasible signal and selected predictive scale
entry/exit probabilities and buy/sell gates
requested and allowed incremental turnover
requested-to-executed retention
policy and C1 gross returns and turnover
carry, anchor-repair, and alpha-signal active-return attribution
```

Each fold then reports:

```text
complete decile and vigintile target-return curves
signal-to-requested-action Spearman correlation
selected-scale quantiles
Brier score and ECE for return exceeding the bound 10-bp cost
action-cap hit fraction and gate occupancy
gross active return and incremental turnover paths
```

Panel evidence concatenates the ordered out-of-sample fold paths while keeping
bootstrap blocks inside fold boundaries. Break-even cost is computed from the
aggregate gross-return and incremental-turnover sums; it is never an average
of fold ratios.

## Decision rule

The audit does not retroactively change the frozen v11 predictive gate. Its
outcome can support only the following research decision:

- if P0 remains positive while the zero, shuffled, and sign-flipped controls
  remove or reverse that performance, its quantile curve is coherent, and a
  predeclared lower cap produces a positive 10-bp block lower bound, design a
  fresh tail-alpha predictive generation;
- otherwise, do not start economic training. Use a fresh predictive identity
  with separate rank and economically calibrated magnitude heads;
- if that corrected representation also fails, prioritize learned intraday
  sequence representation over more holding-period or Sharpe-objective tuning.

No scientifically meaningful untouched pre-2026 chronology remains after the
completed v7 and v11 development studies. Therefore this audit cannot become
confirmation evidence, and 2026 remains governed by its separate frozen
retrospective boundary.

## Completed result and successor decision

The receipt-bound a05 audit completed both workers and all 28 panel reports,
then exact-cleaned its Job and UID-owned Pods. It performed no training or
checkpoint selection, opened no 2026 outcomes, and authorized no economic
generation.

P0's original 200-bp-cap path retained positive annualized mean gross active
return at both horizons: 1.3221% at 21 sessions and 1.3361% at 30 sessions.
The corresponding 10-bp net means were 0.7762% and 0.7736%, with aggregate
break-even one-way costs of 24.22 and 23.75 bp. Shuffled, sign-flipped, and
zero-signal controls all had negative gross and 10-bp net means. This supports
a genuine directional P0 effect.

The frozen advancement rule nevertheless failed:

- every original P0 cap from 50 through 200 bp was hit on every scored date;
- no cap produced a positive 21-session-block lower confidence bound at
  10 bp—the best was still negative;
- decile and vigintile spread lower bounds remained negative; and
- P1 used effectively none of any cap and remained economically negative,
  confirming that its improved rank geometry did not preserve usable scale.

The authorized successor is therefore not economic training and not a tuned
continuation of v11. It is the fresh predictive-only
`top2000-dev-hold30-active-alpha-m03r-v12-rank-scale-decoupled-v1` generation:
one dedicated rank-score head, separate economic mean/scale heads, a bounded
rank-gradient contribution to the shared encoder, and a nonsaturating
turnover-utilization map. The v11 checkpoint and optimizer states are not
resumed.

## Local verification

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_top2000_m03r_v11_a15_inference_audit.py

PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v11_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v11_*.py \
  tests/test_cost_aware_active_policy_v3.py

# Current combined v11/a15 scoped result is reported from the exact rerun.

conda run -n quanttrade ruff check \
  src/rl_quant/protocol/hold30_alpha_m03r_v11_a15_inference_audit.py \
  src/rl_quant/training/top2000_m03r_v11_a15_inference_audit_runtime.py \
  src/rl_quant/training/top2000_m03r_v11_a15_inference_audit_fold.py \
  src/rl_quant/training/top2000_m03r_v11_a15_inference_audit_plan.py \
  src/rl_quant/training/top2000_m03r_v11_a15_inference_audit_package.py \
  src/rl_quant/evaluation/top2000_m03r_v11_a15_inference_audit.py \
  src/rl_quant/workflows/top2000_m03r_v11_a15_inference_audit.py \
  src/rl_quant/workflows/top2000_m03r_v11_a15_inference_audit_static_validate.py \
  src/rl_quant/workflows/top2000_m03r_v11_a15_inference_audit_package_builder.py \
  tests/test_top2000_m03r_v11_a15_inference_audit.py
```
