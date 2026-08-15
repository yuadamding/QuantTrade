# TOP2000 M03R-v15 corrected executable-score h3 research

## Purpose and evidence boundary

V15 is the fresh predictive-only successor to the completed, failed v14 h3
study. It corrects result-moving provenance, rank-gradient, ablation, capacity,
and fit-selection defects without reusing v14 model or optimizer state.

V15 cannot run an economic optimizer or access 2026 outcomes. The TOP2000
universe is future-selected, so any result remains development-only,
nonreportable, and nonpromotable. Repository source, local tests, package
qualification, and capacity qualification do not establish a predictive
performance result.

## Scientific settings

| Index | Setting | Sole difference |
| ---: | --- | --- |
| P0 | `V15-P0-corrected-action-projected-rank-h3` | Adds corrected rank-Gaussian correlation with coefficient `0.25`. |
| P1 | `V15-P1-paired-action-projected-huber-h3` | Omits only the rank term. |

Both settings use robust and scale coefficients `0.45` and `0.30`. The
coefficients are intentionally not renormalized when rank loss is removed, so
the robust, scale, input, initialization, and schedule gradients remain paired.

## One causal executable score

For origin (t), define action support (A_t) and label support (L_{t,3}):

\[
A_t=\text{available at }t\cap\{w_t^{\mathrm{reg}}>0\},
\qquad
L_{t,3}=A_t\cap F_{t,3}.
\]

The model produces a raw mean, but every scientific boundary consumes:

\[
s_t=P_t^A f_\theta(x_{\le t}).
\]

The exact (s_t) tensor drives robust and rank loss, inner validation,
qualification IC and spread, and the fixed-rank sleeve. The raw output type
does not expose rank or execution aliases. Built batches fail when the target
mask is not a subset of action support or when the executable score cannot be
recomputed from the raw mean and bound action operator.

Future label availability may remove a label. It never adds an asset to the
decision universe or changes the origin action operator.

## Correct rank and optimizer geometry

Rank normalization is dimensionless:

```text
executable mean / (0.02 * sqrt(3))
-> cross-sectional centering
-> differentiable RMS with a 0.05 floor
-> rank-Gaussian correlation
```

The RMS is not detached. Above the floor, a perfectly aligned prediction has
zero rank loss, near-zero gradient norm, and no radial amplitude incentive.
Below the floor, a finite anti-collapse gradient remains.

Encoder, mean-head, and scale-head gradients are reduced across ranks and
clipped separately. Bias and normalization parameters receive zero weight
decay. Scale consumes detached hidden state and a detached mean, so its
likelihood cannot update the encoder or mean and cannot shrink their update
through shared clipping. Each fold uses a deterministic 5% linear warm-up and
cosine decay to 10% of the base rate; actual group rates are bound to every
update receipt.

## Sampling and checkpoint selection

V15 uses:

- h3 supervision only;
- at least 252 causal sessions for every optimizer origin;
- every optimizer origin once per deterministic epoch;
- identical setting-neutral schedules and complementary rank shards;
- right-aligned training windows that concentrate local positions near the
  qualification range;
- a 32-origin chronological inner-validation slice per fold;
- an explicit four-session gap between optimizer and validation origins;
- a separate 30-session purge before the outer qualification tail; and
- six untouched 63-origin outer qualification tails.

All eight predeclared epochs run. After each epoch, the training-only slice is
evaluated on the action-projected score. Checkpoint selection maximizes mean
projected IC, then projected decile spread, then minimizes robust loss. Exact
ties select the earlier epoch. The selected state is written immutably,
destroyed, strictly reloaded, and only then evaluated on the outer tail.

## Package-owned structural and capacity gates

Package construction copies exact source, cache, risk, and projector files,
loads those package-owned bytes, runs the real-data structural sweep, and only
then seals the package. The preflight binds source and operator hashes, cache
and asset axis, risk artifact and semantic receipts, exposures, projector
file and semantic identity, every training/inner-validation/qualification
origin, and both operator roots.

The two-H100 capacity job must execute one disposable exact-shape update,
including forward, backward, NCCL reduction, separate clipping, and optimizer
mutation. It also executes one qualification risk projection, proves rank
state equality, records peak CUDA memory, publishes no scientific checkpoint,
and is terminally cleaned before predictive activation.

## Qualification interpretation

The 25-bp fixed-rank sleeve is a score-ordering diagnostic under the existing
next-close-fill daily contract. It is not a realistic h3 cohort strategy or a
Hold-30 policy. Weighted raw-to-executable projection retention remains
attribution telemetry, not an advancement gate, because the model is trained
through the executable projection and raw null-space magnitude is not
scientifically identifiable.

The frozen predictive gate still requires:

```text
mean action-projected Spearman IC >= 0.020
positive mean-IC folds >= 4/6
positive spread folds >= 4/6
spread block-bootstrap LCB > 0
gross active-return LCB > 0
10-bp net active-return LCB > 0
aggregate break-even one-way cost >= 10 bp
median requested-to-executed retention >= 0.50
minimum fold requested-to-executed retention >= 0.20
```

Failure does not authorize a lower threshold or economic training. It ends the
daily h3 loss study and directs the next generation toward a 21/30-session or
survival-weighted selection target, followed by genuinely ordered five-minute
inputs if the corrected daily control remains weak.

## Completed result

The local source implements the protocol, causal batch, objective, optimizer,
training-only checkpoint selection, immutable checkpoint reload, package-owned
preflight, exact disposable capacity update, qualification, lifecycle, and
selection boundaries. The focused suite currently passes `63` tests, with one
CUDA-only device-identity regression skipped when CUDA is unavailable, and
Ruff passes on all v15 source and tests.

The sealed A03 package passed its zero-GPU static gate but failed its disposable
capacity update before optimizer mutation: the built-batch validator compared
a CUDA objective-valid mask directly with the immutable CPU operator mask.
Both ranks failed at the same boundary, no scientific checkpoint was written,
and the exact Job and Pods were terminally cleaned. A03 remains immutable
failed development evidence. A04 reconciles the complete mask through a
device-independent tensor identity and was rebuilt from fresh source bytes;
no A03 model or optimizer state was resumed.

The immutable A04 package completed its package-owned real-data preflight,
same-image zero-GPU static gate, disposable exact-shape two-H100 capacity gate,
both predictive settings, and all twelve folds. Exact terminal coverage and
cleanup receipts validate. Neither setting passed:

| Setting | Mean projected IC | Annualized gross active | Annualized net active at 10 bp | Break-even cost |
| --- | ---: | ---: | ---: | ---: |
| P0 corrected rank h3 | 0.00789 | 0.0022% | -0.0711% | 0.30 bp |
| P1 paired Huber h3 | 0.01013 | 0.0129% | -0.0573% | 1.84 bp |

P0 had four positive-mean-IC folds but its primary 21-session gross and net
LCBs were negative. P1 had four positive-mean-IC folds and five folds with a
positive date-IC fraction, but only three positive-spread folds; its gross,
net, and spread primary LCBs were negative. Risk projection retention was 1.0
for both settings, so the failure is weak information and tradeability rather
than final projection loss.

No economic optimizer ran, no horizon was selected, and no 2026 outcomes were
opened. V15 therefore closes the daily h3 loss-tuning line and directs the
fresh successor toward longer holding-aligned selection targets with h3 used
only for timing.

The durable A04 qualification identities are:

```text
package archive SHA-256       a11703bb9a8ff4e937dbad0550f91938f683b68698b790afc25a9f78f46c24a6
package-plan file SHA-256     7a4fbf5605b261a0e80e28c70ff432fdca6c39223e3b98a3df8fd73e10444926
source archive SHA-256        4ee21cd809add15460296b601f18e51dc4d38bae8faebcaf119ab5a1e8d040a8
structural-preflight SHA-256  88ce2a000217f45e5912d9030e41da7270d409db7568cd0e537270ba9b006d1a
static-gate file SHA-256      2305698b7617d03a366181cfefb3273053d67248c69f0d84548493c3b643d35a
capacity file SHA-256         397d61513086dbc2837e36fe34c613f692bba64eb1ff844d4aea39aaba2b7eee
capacity terminal receipt     05b62145c2091867931749055d344a91b81ebe127b44125151f56cbf3883cfa9
```

Transient Job names, UIDs, Pod names, and live status do not belong in this
document; establish those only from the exact external lifecycle receipts.

## Local verification

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_hold30_alpha_m03r_v15_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v15_*.py

conda run -n quanttrade ruff check \
  src/rl_quant/protocol/hold30_alpha_m03r_v15_top2000_dev.py \
  src/rl_quant/training/top2000_m03r_v15_*.py \
  src/rl_quant/workflows/top2000_m03r_v15_*.py \
  tests/test_hold30_alpha_m03r_v15_top2000_dev_protocol.py \
  tests/test_top2000_m03r_v15_*.py
```
