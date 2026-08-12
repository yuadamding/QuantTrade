# TOP2000 M03R-v9 predictive stage

This document records the development-only successor to the failed M03R-v8
predictive attempt. It is a research contract, not a business-production,
reportable-performance, or promotion surface.

## Decision and generation boundary

The v8 evidence remains immutable failed development evidence. V9 does not
resume a v8 model or optimizer, reuse a v8 package identity, lower the 0.020
rank-IC threshold, open the 2026-YTD retrospective, or authorize economic
policy optimization.

The v9 generation is:

```text
top2000-dev-hold30-active-alpha-m03r-v9-predictive-v1
```

It contains three seed-17 settings, six chronological folds, exactly 64
predictive optimizer updates, one qualification at update 64, two H100s per
worker, and at most three workers (six requested H100s). Economic optimizer
updates are exactly zero.

| Index | Setting | Sole predictive change |
| ---: | --- | --- |
| P0 | `V9-P0-factor-residual-ranked` | factor-residual labels with normalized rank, Huber, and distributional losses |
| P1 | `V9-P1-factor-residual-no-ranking` | remove ranking and renormalize the remaining weights to 0.60/0.40 |
| P2 | `V9-P2-benchmark-relative-ranked` | benchmark-relative rather than factor-residual labels |

## Implemented predictive boundary

V9 uses one typed four-horizon alpha distribution from pretraining through
the future economic boundary. A checkpoint binds one eligible 21- or
30-session horizon for checkpoint selection, qualification, and execution;
the three values must be identical. The old `alpha_downside_30d` route is not
part of the v9 policy contract.

The listwise objective standardizes cumulative predicted and realized returns
by `0.02 * sqrt(horizon)`. Active component weights always sum to one:
canonical uses 0.50/0.30/0.20, while no-ranking uses 0/0.60/0.40. The
qualification tail is not used for early stopping or checkpoint selection.

Factor-residual settings require decision-origin exposures with an exact
asset-axis identity. The exposure surface must contain the same named
sector, active-beta, and style/risk families used by the projector. Every
exposure carries an availability time no later than its decision origin.
Factors are target/evaluation inputs only; actor inputs remain raw market
states.

Each setting-horizon candidate must independently pass all frozen gates:

```text
mean Spearman IC >= 0.020
positive-IC folds >= 4/6
mean top-minus-bottom spread > 0
positive-spread folds >= 4/6
simple-sleeve gross active return > 0
simple-sleeve 10-bp net active return > 0
gross-positive folds >= 4/6
break-even one-way cost >= 10 bp
```

When both horizons pass for one setting, the higher mean 10-bp sleeve active
return lower bound wins; an exact tie selects 30 sessions. No passing
setting-horizon pair means stop—never start the economic panel.

## Implementation and launch record

## Completed A04 result and stop decision

The source-corrected A04 run completed all three workers and all eighteen
folds on 2026-08-12. It had no failed fold, worker restart, or supervisor
error. The exact Job and its UID-owned Pods were subsequently deleted with
UID/resourceVersion preconditions and two independent absence observations.

```text
run ID                    qt-m03r-v9-predictive-s17-20260811-a04
Job                       qt-m03r-v9-pred-s17-a04
Job UID                   bcc52d1e-8280-4ee6-8e56-c3fc21f570db
workers/folds complete    3/3 and 18/18
predictive gates passed   0/3
selected horizons         none
economic panel launched   no
completion coverage SHA   2acebe99ceec46315b45ac3c6b62e4bdac003ea9bc72ed9db17eef62f34858ee
terminal evidence SHA     bbb337415ee905be0e63965364b87851c64daf9c1ae2e9fed4e5e31a7151ba2b
cleanup receipt file SHA  68913de7aa8ee4290471363bbd113db88682911d40b6d9909ddacd2619a9898d
```

The two eligible horizons produced the following exact six-fold aggregate
results. Returns are decimal annualized active returns; break-even cost is a
one-way basis-point estimate.

| Setting | Horizon | Mean IC | IC-positive folds | Mean decile spread | Spread-positive folds | Gross active | Net active, 10 bp | Gross-positive folds | Break-even bp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 factor-residual ranked | 21 | -0.002375 | 3 | 0.005058 | 5 | -0.001946 | -0.002168 | 3 | -115.42 |
| P0 factor-residual ranked | 30 | 0.000093 | 3 | 0.005630 | 5 | -0.001816 | -0.001982 | 3 | -378.09 |
| P1 factor-residual no-ranking | 21 | -0.001082 | 2 | -0.000853 | 2 | -0.002456 | -0.002420 | 2 | unavailable |
| P1 factor-residual no-ranking | 30 | 0.003653 | 4 | -0.000690 | 3 | -0.002456 | -0.002420 | 2 | unavailable |
| P2 benchmark-relative ranked | 21 | -0.019854 | 2 | -0.005233 | 2 | -0.001815 | -0.001928 | 1 | -354.74 |
| P2 benchmark-relative ranked | 30 | -0.024755 | 1 | -0.010544 | 2 | -0.001092 | -0.001178 | 3 | -1031.64 |

The frozen 0.020 mean-rank-IC and sleeve gates remain unchanged. No
setting-horizon pair qualified, so the v9 economic panel is permanently
blocked for this evidence chain. The result supports three narrower
conclusions:

1. factor residualization materially improves the target relative to the
   benchmark-relative route;
2. ranking contributes a repeatable extreme-decile spread, but the current
   softmax listwise geometry does not create broad cross-sectional IC; and
3. no trained predictor survives the deterministic sleeve at positive gross
   or 10-bp net active return.

The next authorized work is a fresh representation/label research generation,
not longer v9 training, a lower gate, persistence tuning, or economic policy
optimization. Its first bounded question is whether rank-aligned
cross-sectional target geometry can convert P0's positive decile spread into
broad IC while preserving the economically scaled mean and uncertainty used
by execution. The benchmark-relative route is retired from that next panel.

The historical risk-source data boundary is now materialized locally. The
source is the monthly Polygon provider-as-of overview archive plus the exact
SHA-verified pre-2026 daily cache. SIC codes map to declared major-division
sector one-hots; missing or unavailable SIC remains an explicit unknown
sector and is never future-filled. Active beta, 63-session log return,
volatility, and log-volume controls use only past cache rows. Sector,
active-beta, and style availability are separately recorded for every origin.

The frozen development artifact has:

```text
date range                         2022-01-03 through 2025-12-29
exposure shape                     [1001, 1999, 16] float32
availability shape                 [1001, 1999, 3] int64
asset-axis SHA-256                 94d4367c9e2959b3822463a636793e032a051db5051ac4f29f0adeb223321116
artifact SHA-256                   5c051895f7608b6870c543bb387e79f7eb4259418ac8e36dfa14602e0898fb59
manifest file SHA-256              8f82d99c139bac255cb3cb7dac848e94414fd667164816a17aa4e0cace69c748
materialization receipt SHA-256    f9f4cad011dd6cd65392080f4e4143fcf5687fee85f469805c496f5d8988038c
exposure receipt SHA-256           83c01bc85b4dc1b06f4c6b86a26520526b18521abd296b6c5b79d492227c443c
2026 delta rows used               0
```

The separately validated Polygon delta reaches 2026-08-11 and is bound as a
source input, but no post-2025 observation contributes to the predictive
surface. The artifact is still future-selected TOP2000 development evidence;
it is nonreportable and nonpromotable.

The immutable materialized artifact continues to record its original two
pre-projector blockers:

```text
missing-projector-manifest
target-projector-exposure-name-mismatch
```

Those original bytes were not rewritten. A separate projector/source binding
now closes only those two known blockers after proving byte-for-byte exposure
name, family, order, asset-axis, materialization, artifact-file, and external
manifest equality. The bound readiness authorizes only the predictive worker;
economic authorization remains false.

```text
projector semantic SHA-256            7a7b20b8c17445d810487ec5418d786bf22d81f2a12ab38702480ff449bd68e2
projector/source binding SHA-256      e555b3e75476b46e77786ee5ce50658a2608dde58e0a8e865a59f583a8114c2e
projector manifest file SHA-256       93e76644dc560b91b37e7422f06281a3744085dff7d8321c55767e016d0a3608
bound blocker codes                   none
predictive worker authorized          true
economic panel authorized             false
```

The projector freezes nonzero sector/style slabs, beta and 6% TE limits, a
252-session past-only factor-plus-diagonal covariance estimator, 21-session
risk refreshes, shrinkage, specific-risk floor, and PSD repair. Full hashes,
causality, and CPU-to-device transfers run once per fold. The hot path checks
only the asset identities, projector manifest, and tensor version counters.

The deterministic simple sleeve is implemented. It uses only the selected
mean and selected scale, removes signal components in the same exposure-null
space used for target residualization, disables the learned hazard, carries
the book chronologically, charges 10 bp in its action hurdle, and publishes
gross-return, turnover, requested/projected book, signal-retention, and
projection-retention arrays. It performs zero economic optimizer updates.

Cost-aware action v2 separately records learned-exit proceeds and explicit
de-risking, redeploys only their difference, rejects same-step repurchase,
uses frozen entry/expansion/retention/exit probability thresholds, and runs
exactly eight factor-plus-diagonal proximal iterations. Its one-way turnover
definition is `0.5 * abs(delta).sum()`.

The worker plan is frozen at exactly three Indexed completions, two H100s per
completion, six requested H100s maximum, six folds, seed 17, 64 updates, and
qualification only after update 64. The update checkpoint binds the selected
horizon, mean-head hash, scale-head hash, distribution contract, asset axis,
risk binding, source array, and plan. Loading is evaluation-only; v8 model or
optimizer state cannot enter it.

The source-homogeneous package builder, deterministic relocation-safe transfer
archive, exact two-rank worker CLI, capacity-only startup mode, suspended
Kubernetes renderers, pure-file launch preparation, one-attempt create
operator, and detached attach-only supervisors are implemented and locally
validated. Both 21- and 30-session update-64 checkpoints are published before
qualification is opened; their common causal risk state is fully qualified and
transferred once per fold.

The live lifecycle is deliberately split:

1. a pure-file preparer renders one capacity or predictive manifest and binds
   every operator/supervisor configuration;
2. the mutation-owning operator performs one server dry-run, two exact-name
   absence reads, exactly one suspended create request, full-window ambiguity
   reconciliation, two stable UID/spec reads, and zero-Pod binding;
3. a detached attach-only capacity supervisor activates once, proves two H100
   80GB ranks and NCCL startup, captures terminal Job/Pod/log evidence, issues
   one UID/resourceVersion-conditioned delete, proves absence twice, and only
   then publishes the capacity qualification;
4. a separately bound predictive supervisor repeats the lifecycle for three
   Indexed completions and validates all six folds plus both horizon
   qualifications for every worker;
5. ambiguous activation or delete outcomes publish attach-required evidence
   and are never converted into success or retried.

The lifecycle-complete `20260811-a03` package is now frozen and independently
validated locally. Remaining before GPU mutation is to verify the approved
connection, run the current-session mutation preflight, and stage exactly that
archive. The economic worker remains explicitly fail-closed; a later economic
generation must be minted separately.

The first local `20260811-a01` candidate remains immutable but was superseded
before transfer after the repository-wide Hold-30 inventory gate exposed a
missing later-generation registration. No remote or Kubernetes state ever
consumed it. The corrected locally sealed `20260811-a02` candidate is not a
launch receipt and is now superseded because it predates the receipt-gated
lifecycle/operator source. It must remain immutable and must not be transferred
or launched. It records:

```text
transfer archive SHA-256       9b6f20428585ac0da6c18c5ff2c1d277ca830aa8574246f4a29c268c7aebabce
package-build file SHA-256     2cd0de58c2a4ce0d2337ba1cf94a0c951eaf97f3cd103c85bba5f5029f32875d
package-build receipt SHA-256  b9c5e4d69f3860cf22f8c295433eb0804653316ad59f911baf91ae990fef0571
package-plan semantic SHA-256  68f0080891a1e5be9532f3d2e79f94da817aa28fa53a3626a489b25c8727d9e4
package-plan file SHA-256      a6010b176b82e89666de5cea14d90fc83ddc9fdee6cc68d653df47dfba59ac0e
source archive SHA-256         47c6676cc710552f234867595e7ac7b01c9bc46769a772cc099349c220140c22
source manifest SHA-256        a223a48f98bf38aedd5f94df4d0b6922761475a92e40228161a3c8afb42a347b
worker source SHA-256          2ab76d6a0e81d053bb2eb920b3c3f727306d761c6e3c8a686db1e25427826144
source files                   230
archive regular files          240
economic panel authorized      false
outer evaluation authorized    false
```

The archive was validated without extraction, then its package source was
imported from the sealed tree and compared byte-for-byte with all 230 current
runtime-source files. This is local implementation evidence only: it proves no
Seadragon admission, allocation, startup, training, or scientific result.

The launchable local candidate is `20260811-a03`. It reuses the exact validated
development cache and risk artifacts from a02, but mints a new source archive,
package plan, receipt, and transfer identity containing the receipt-gated
lifecycle/operator/preparer bytes:

```text
transfer archive SHA-256       627d5dfd3cb4a3aa253a10151ac842d6ea0a7097b8cf86290b9affd2763613f6
package-build file SHA-256     95f118a909ddd582e1106e4752f593322516e64fbca7f7f66e17cac1d5e1c8eb
package-build receipt SHA-256  fe09fc3581c17d0eea9d8dc9788ad6ef66073536502183e3fdbdd43d8d8f3f24
package-plan semantic SHA-256  d7c671f0aed2599809c966b6e7bdfef6b0036b0229617a53bdcfebf0f3572d51
package-plan file SHA-256      42c1ee6124b65b54efed7268d5c931c53578575dc988240fe128716cf078e0d9
source archive SHA-256         b343e4fd6e881ae273c89d24404b86eb1f1d2b9307f1bc58d8f3aabfc177dba7
source manifest SHA-256        678e745f0c22649ef429babb8f91e3e18a4d7b82c38a2a47fc3668de6e0a253d
execution manifest SHA-256     3855651cb8bf88918e24aa379f7994104066ba389de482627b5404409f9688ac
worker source SHA-256          2ab76d6a0e81d053bb2eb920b3c3f727306d761c6e3c8a686db1e25427826144
source files                   233
archive regular files          243
economic panel authorized      false
outer evaluation authorized    false
```

All 233 source-manifest hashes match the current runtime-source files. The
transfer archive has only regular files and directories, no links, devices,
absolute paths, or traversal members, and every receipt-bound member hash was
validated without extraction. Packaged imports resolve to the sealed a03 tree.
This remains prelaunch evidence, not a GPU startup or training result.

## Implementation map

- `protocol/hold30_alpha_m03r_v9_top2000_dev.py` — immutable generation,
  settings, resources, loss/gate thresholds, and horizon binding.
- `training/top2000_m03r_v9_policy.py` — shared mean/scale distribution and
  alpha-head identities.
- `training/top2000_m03r_v9_alpha_pretraining.py` — normalized/scaled loss and
  date-balanced fold diagnostics.
- `training/top2000_m03r_v9_pretraining_runtime.py` — point-in-time
  factor-residual target construction with exact availability and axis checks.
- `training/top2000_m03r_v9_selection.py` — immutable sleeve evidence,
  six-fold qualification, and horizon selection.
- `training/top2000_m03r_v9_risk_source.py` — prelaunch source-readiness gate.
- `training/top2000_m03r_v9_risk_materialization.py` — no-clobber historical
  SIC-sector plus past-only beta/style tensor materialization and external
  hash-bound artifact loading.
- `training/top2000_m03r_v9_projection.py` — immutable projector/source
  binding, causal factor-plus-diagonal device state, exposure-null signal
  transform, exact asset-axis enforcement, and radial safety projection.
- `execution/cost_aware_active_policy_v2.py` — exit-proceeds replacement,
  same-step repurchase guard, probability gates, and fixed-iteration portfolio
  cost/risk optimization.
- `training/top2000_m03r_v9_runtime.py` — deterministic no-hazard simple sleeve
  that produces the exact tradeability-gate arrays.
- `training/top2000_m03r_v9_pretraining_optimizer.py` and
  `training/top2000_m03r_v9_pretraining_step.py` — exact predictive parameter
  partition and sole one-/two-rank mutation boundary.
- `training/top2000_m03r_v9_checkpoint.py` — immutable update-64 checkpoint and
  evaluation-only loader.
- `training/top2000_m03r_v9_predictive_worker.py` — three-setting/six-H100
  worker plan and direct Indexed completion mapping.
- `training/top2000_m03r_v9_fold.py` — deterministic 64-update folds, untouched
  update-64 qualification, common once-per-fold device risk, and simple sleeve.
- `workflows/top2000_m03r_v9_predictive.py` — exact two-H100/NCCL capacity and
  predictive worker entrypoint with immutable terminal evidence.
- `training/top2000_m03r_v9_package.py` and
  `workflows/top2000_m03r_v9_package_builder.py` — content-bound package plan,
  relocation-safe validation, and deterministic safe transfer archive.
- `training/top2000_m03r_v9_kubernetes.py` — capacity/predictive suspended-Job
  rendering, strict admitted-spec adapter, and content-bound capacity
  qualification under the six-H100 request ceiling.
- `workflows/top2000_m03r_v9_seadragon_prepare.py` — pure-file rendering and
  exact create/attach configuration; it performs no Kubernetes calls.
- `training/top2000_m03r_v9_seadragon_operator.py` — the sole one-attempt
  server-dry-run/create/reconciliation/binding surface.
- `training/top2000_m03r_v9_seadragon_lifecycle.py` — detached capacity and
  predictive attach-only supervision, terminal evidence, semantic worker
  coverage, and one-delete/two-absence cleanup.
- `training/top2000_m03r_v9_economic_worker.py` — intentional fail-closed guard
  against premature economic training.

Focused verification:

```bash
uv run pytest \
  tests/test_hold30_alpha_m03r_v9_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v9_policy.py \
  tests/test_top2000_m03r_v9_alpha_pretraining.py \
  tests/test_top2000_m03r_v9_pretraining_runtime.py \
  tests/test_top2000_m03r_v9_risk_materialization.py \
  tests/test_top2000_m03r_v9_selection.py \
  tests/test_top2000_m03r_v9_risk_source.py \
  tests/test_top2000_m03r_v9_projection.py \
  tests/test_cost_aware_active_policy_v2.py \
  tests/test_top2000_m03r_v9_pretraining_step.py \
  tests/test_top2000_m03r_v9_checkpoint.py \
  tests/test_top2000_m03r_v9_predictive_worker.py \
  tests/test_top2000_m03r_v9_package.py \
  tests/test_top2000_m03r_v9_package_builder.py \
  tests/test_top2000_m03r_v9_fold.py \
  tests/test_top2000_m03r_v9_kubernetes.py \
  tests/test_top2000_m03r_v9_seadragon_operator.py \
  tests/test_top2000_m03r_v9_seadragon_lifecycle.py \
  tests/test_top2000_m03r_v9_seadragon_prepare.py -q
```

No Kubernetes Job should be created from mutable source bytes. Only the frozen,
independently audited package archive may enter the receipt-gated live launch
lifecycle. The eight-setting economic panel remains prohibited until a
setting-horizon pair passes all six-fold predictive and tradeability gates and
a new economic protocol is frozen.
