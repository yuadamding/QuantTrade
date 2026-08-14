# TOP2000 M03R-v12 rank/scale-decoupled predictive research

## Purpose and authority

V12 is the fresh predictive-only successor to the completed v11 a15 inference
audit. It tests whether cross-sectional rank learning and economic
mean/uncertainty calibration can coexist without one objective starving the
other or forcing the deterministic sleeve to its turnover cap.

This is future-selected TOP2000 mechanism research. It is development-only,
nonreportable, and nonpromotable. The protocol authorizes exactly 64 predictive
updates, zero economic updates, and no access to 2026 outcomes. V11 model and
optimizer state cannot be resumed. Documentation and local tests do not
authorize a remote launch.

## Scientific settings

All settings share the same raw encoder architecture, factor-qualified v11
residual operator, horizons, paired episode schedule, two-rank shards, source
data, fold geometry, and packaged initial parameter bytes.

| Index | Setting | Sole scientific difference |
| ---: | --- | --- |
| P0 | separate listwise rank + economic scale | Standardized-return listwise loss acts on the dedicated rank-score head |
| P1 | separate rank-Gaussian + economic scale | Rank-Gaussian correlation acts on the dedicated rank-score head |
| P2 | economic-scale no-rank control | Rank loss is zero; robust economic mean and distributional scale remain |

P0/P1 use component weights `(0.25, 0.45, 0.30)` for rank, robust economic
regression, and distributional calibration. P2 renormalizes the active
economic terms to `(0.0, 0.60, 0.40)`. All settings use the same
`(0.10, 0.10, 0.30, 0.35, 0.15)` weights over the supervised
`(3, 5, 21, 30, 63)`-session heads.

## Result-moving corrections

### Separate output heads

`Top2000M03RV12PredictivePolicy` emits:

```text
rank_score_by_horizon          ranking objective only
economic_mean_by_horizon       robust regression and execution only
economic_log_scale_by_horizon  distributional calibration and execution only
```

The rank tensor cannot alias either economic tensor. Checkpoint identity must
bind all three head states and the selected economic horizon. The model
constructor requires the argument

```python
selected_horizon_sessions=3
```

and the v12 protocol admits no alternative selected horizon. This is a real
3-session mean/scale/rank head and a 3-session factor-residual target—not a
renamed 21/30-session output or a relaxed checkpoint expectation. The 5-, 21-,
30-, and 63-session heads remain joint auxiliary supervision only.

### Bounded shared-encoder rank gradient

V11 normalized a collapsed prediction by approximately its own standard
deviation. With nearly zero dispersion, the rank gradient could exceed the
economic gradients by orders of magnitude; one global norm clip then scaled
down the mean and uncertainty heads as collateral damage.

V12 floors rank-score variance inside the square root, separates optimizer
groups, and computes rank and economic component gradients independently. The
distributed, averaged rank contribution to the shared encoder is capped at
25% of the economic encoder-gradient norm. Economic-head gradients are clipped
independently at 1.0; rank-head gradients are clipped independently at 0.25.
The no-rank control leaves the rank head without a gradient or AdamW mutation.

### Nonsaturating predictive sleeve

V11's P0 path hit every tested action cap on every scored date. V12 retains the
magnitude-preserving v3 proximal allocator but scales its permitted turnover by

```text
tanh(RMS(relu(abs(mean) - one_way_cost) / scale) / 2)
```

over currently tradable risky assets. Zero economic edge produces zero
utilization and turnover. Utilization grows monotonically but remains strictly
below one for finite signals. This is a predeclared mapping, not a cap chosen
from v11 outcomes.

## Qualified implementation boundary

The current source includes:

- the immutable v12 protocol and three settings;
- the separate-head policy and exact head identity;
- constructor-bound 3-session selection, common initial parameter bytes, and
  immutable write/reload/evaluate checkpoints;
- corrected target-batch integration over the frozen v11 residual operator;
- the decoupled rank/economic objective;
- three optimizer groups and the component-aware gradient mutation boundary;
- the v4 nonsaturating action proposal and hazard-free 3-session simple-sleeve
  runtime;
- the three-worker, two-H100-per-worker Kubernetes render surface;
- create-once, suspended-admission binding, attach-only lifecycle supervision,
  exact-clean recovery, and zero-GPU static validation; and
- focused protocol, numerical, optimizer, runtime, lifecycle, and action
  regressions.

The real-data structural preflight enumerated all 168 distinct fold-0 training
origins across all five supervised horizons. All 840 residual operators had
the expected rank 14; the minimum factor-qualified fraction was
`0.9881586737714624`, and the minimum residual degrees of freedom was 1641.
This preflight read frozen date, availability, and risk tensors only. It did
not read future-return targets, the qualification tail, or 2026 outcomes.

The immutable executed attempt is `a05`. Its package plan binds
the exact tested source archive, unchanged pre-2026 cache and risk tensors,
common initial parameter bytes, worker entrypoint, and the structural-preflight
receipt. The package authorized three workers with two H100 requests each,
exactly 64 predictive updates, and no economic updates. Its external lifecycle
receipts establish successful same-image capacity, admission, startup, all
eighteen fold executions, terminal completion, and exact Job/Pod cleanup.

## Completed a05 result

All three workers completed their six folds at exactly 64 predictive updates.
Each setting evaluated only the constructor-bound 3-session candidate. No
setting passed, no horizon was selected, and no economic generation may be
minted.

| Setting | Mean rank IC | Positive mean/median/date-fraction folds | Positive-spread folds | Annualized gross active | Annualized net active at 10 bp | Gross / net 21-session block LCB | Spread LCB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 separate listwise rank | -0.004447 | 1 / 1 / 1 of 6 | 5 of 6 | -0.1450% | -0.1680% | -0.4621% / -0.4837% | -0.0635% |
| P1 separate rank-Gaussian | -0.005680 | 0 / 0 / 0 of 6 | 5 of 6 | -0.1484% | -0.1714% | -0.4647% / -0.4871% | -0.0242% |
| P2 no-rank control | -0.004867 | 1 / 1 / 1 of 6 | 5 of 6 | -0.1517% | -0.1747% | -0.4671% / -0.4894% | -0.0279% |

All settings passed the prediction-dispersion and prediction/target-dispersion
ratio checks, and all retained a median and minimum-fold median projection
ratio of 1.0. The failure therefore is not attributable to projection erasing
the requested sleeve. The 3-session target has negative broad rank IC,
negative gross economics, negative 10-bp net economics, and no positive
break-even cost. Five positive point-estimate spread folds do not rescue the
result because every spread lower bound is negative.

The result rejects the hypothesis that merely shortening the selected horizon
to three sessions repairs the v11 predictive interface. It also leaves no
evidence for preferring listwise, rank-Gaussian, or no-rank training at this
horizon: their economics are uniformly negative and tightly clustered. The
frozen stop rule applies—do not start the economic panel, tune a gate from this
outcome, or access 2026 results.

## Historical coverage correction

The first post-run lifecycle process captured the completed Job and three
successful Pods, exact-cleaned them, and then failed coverage validation
because its validator retained a stale `{21, 30}` horizon inventory. The worker
and protocol artifacts correctly contained only horizon 3. The corrected
validator derives its inventory from the immutable protocol and has a
regression rejecting any extra horizon.

No Job was recreated and no training was repeated. A CPU-only historical
continuation revalidated the preserved Job, Pods, all worker/fold artifacts,
the exact historical failure, and the cleanup receipt. The first continuation
`c01` failed safely because it compared JSON-array cleanup fields with typed
Python tuples; its immutable failure receipt is retained. Continuation `c02`
uses canonical JSON comparison and passed.

Durable c02 evidence:

```text
continuation bundle archive SHA-256  b6fb4e568dfc266f8d10e55d2cd53c20402ad0a13a9196659cebd7aa88dd6e00
requalification plan file SHA-256    0598031272c6d67659d97004b8a1764ce4b033033e29ba03814236d2f83dfab7
coverage file SHA-256                48c43f40b2a5f59c572961e3db9e1e86830388d2e8e1c995a2301d9e65c79c92
coverage semantic SHA-256            c7154c1cab010471ecc3191f243432730e04380a0e0a7c6e9bb42913a54cb00e
historical receipt file SHA-256      c31244bbf2720c2cfc3469fc0e689ae9464be91ec587c95d61884e5a382043f0
historical receipt SHA-256           f34ce9042d39a19dc58d8816c6c16c89c6473b04aef52aa4ff3c821a27255298
```

This correction qualifies coverage of already-completed research only. It is
not new training, a result-moving scientific change, H100 capacity evidence,
economic authorization, or permission to open 2026 outcomes.

## Local verification

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v12_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v12_*.py \
  tests/test_cost_aware_active_policy_v4.py

PYTHONPATH=src conda run -n quanttrade python -m mypy --strict \
  src/rl_quant/protocol/hold30_alpha_m03r_v12_top2000_dev.py \
  src/rl_quant/execution/cost_aware_active_policy_v4.py \
  src/rl_quant/training/top2000_m03r_v12_*.py \
  src/rl_quant/workflows/top2000_m03r_v12_*.py

conda run -n quanttrade ruff check \
  src/rl_quant/protocol/hold30_alpha_m03r_v12_top2000_dev.py \
  src/rl_quant/training/top2000_m03r_v12_*.py \
  src/rl_quant/execution/cost_aware_active_policy_v4.py \
  tests/test_hold30_alpha_m03r_v12_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v12_*.py \
  tests/test_cost_aware_active_policy_v4.py
```
