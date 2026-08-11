# TOP2000 M03R-v7 seed-17: twelve-setting performance benchmark

**Status:** completed one-seed development validation
**Research use:** non-PHI, development-only, nonreportable, and nonpromotable
**Training run:** `qt-m03r-v7-t2k12-s17-20260808-a05q2`
**Source commit:** `2adaccdaa4e8fa14b3eea49b0a4c3ca9fb813151`
**Validation span:** 2023-08-18 through 2025-09-25
**Evidence:** six chronological 63-session validation folds per setting, seed 17

## Executive result

None of the twelve settings produced a positive mean annualized active return
across the six validation folds. The panel therefore does not demonstrate
active alpha and supplies no promotion evidence.

- Setting 3, fixed exit hazard, had the least-negative mean active return at
  -0.248% annualized, but did so with only 0.131% mean tracking error. It was
  primarily a very low-active-risk result.
- Setting 11, direct Sharpe, had the best mean information ratio at -0.198 and
  the most positive folds at three of six. It remained negative on average and
  is a diagnostic, nonpromotable objective.
- Setting 9, no factor-neutral projection, failed the risk benchmark: -17.058%
  mean annualized active return and 14.940% mean tracking error, far above the
  6% ceiling.
- Removing the exact-HOLD action or the residual alpha heads materially
  worsened active return, turnover, and holding behavior.
- The 0-bp, 5-bp, and 10-bp soft-persistence settings were very similar. This
  run offers little evidence that persistence strength itself changed
  validation performance materially.

## Completed Phase-0 forensic result

The post-training Phase-0 audit completed on all 72 frozen setting/fold
checkpoints. The successful aggregation-only continuation did not retrain a
model, select a checkpoint, replay a GPU worker, or access Kubernetes. Its
immutable panel file SHA-256 is
`fd30e16dd44bf01b4fc43f47183d5b7745e5d78ff99ca19111ca293ff94ccc97`.

The audit changes the interpretation in two important ways:

1. Several settings have a small positive gross active point estimate, but no
   setting survives even 10 bp of one-way cost on an across-fold mean basis.
   The best aggregate break-even cost is only about 5.6 bp for setting 11.
2. The distinct-policy gate fails. Every checkpoint, model state, optimizer
   state, and aggregate requested-action trace is distinct, but 45 setting
   pairs share an exactly identical executed-weight trace in at least one
   fold. The equality begins in the requested economic book, not in artifact
   reuse.

### Exact frozen-path cost ladder

The values below are unweighted means of the six fold-level, full-precision
receipts. Break-even cost is computed from mean gross active return divided by
mean incremental policy-minus-C1 turnover. `None` means gross active return is
nonpositive or incremental turnover is nonpositive, so no positive break-even
cost exists.

| Setting | Gross active | Net @10 bp | Net @20 bp | Net @40 bp | Incremental cost @20 bp | Break-even | Gross-positive folds | Net-20-positive folds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.038% | -0.288% | -0.613% | -1.264% | +0.651% | 1.2 bp | 2/6 | 1/6 |
| 1 | +0.100% | -0.227% | -0.553% | -1.206% | +0.653% | 3.1 bp | 3/6 | 2/6 |
| 2 | +0.032% | -0.305% | -0.642% | -1.316% | +0.674% | 1.0 bp | 2/6 | 1/6 |
| 3 | -0.068% | -0.159% | -0.250% | -0.431% | +0.182% | None | 3/6 | 2/6 |
| 4 | -0.794% | -1.652% | -2.511% | -4.228% | +1.717% | None | 2/6 | 1/6 |
| 5 | +0.174% | -0.381% | -0.936% | -2.046% | +1.110% | 3.1 bp | 3/6 | 1/6 |
| 6 | -0.780% | -1.619% | -2.458% | -4.135% | +1.678% | None | 2/6 | 1/6 |
| 7 | +0.152% | -0.403% | -0.958% | -2.067% | +1.109% | 2.7 bp | 3/6 | 2/6 |
| 8 | +0.162% | -0.406% | -0.974% | -2.110% | +1.136% | 2.9 bp | 3/6 | 1/6 |
| 9 | -18.550% | -18.465% | -18.381% | -18.211% | -0.170% | None | 1/6 | 1/6 |
| 10 | +0.147% | -0.189% | -0.524% | -1.195% | +0.671% | 4.4 bp | 3/6 | 2/6 |
| 11 | +0.188% | -0.150% | -0.488% | -1.164% | +0.676% | 5.6 bp | 3/6 | 3/6 |

The fixed-hazard setting's relatively favorable 20-bp result is entirely a
low-turnover effect: its gross active result is negative. Settings 5, 7, 8,
10, and 11 have positive gross point estimates, but their edge is too small to
survive the frozen cost ladder.

### Predictive-alpha gate

The canonical setting fails the proposed residual-head development gate:

| Horizon | Mean Spearman IC | Positive-IC folds | Mean top-minus-bottom decile residual return |
|---:|---:|---:|---:|
| 5 | +0.0022 | 3/6 | -0.0008 |
| 21 | -0.0137 | 2/6 | -0.0060 |
| 30 | -0.0027 | 3/6 | -0.0020 |
| 63 | -0.0031 | 2/6 | -0.0032 |

No valid setting reaches mean rank IC 0.02 at either 21 or 30 sessions. Setting
9 has the largest 21-session mean IC (+0.0097), but it is below the gate and
its execution route is not a valid causal risk-control ablation.

### Economic-action and projection collapse

The evidence rules out checkpoint reuse: all 72 checkpoint-file, model-state,
alpha-optimizer-state, and aggregate requested-action hashes are distinct.
Nevertheless:

- fold 2 has one identical executed-weight group containing settings 0, 1, 2,
  4, 5, 6, 7, 8, 10, and 11;
- fold 3 has one identical group containing settings 0, 1, 2, 5, 6, 7, 8,
  10, and 11;
- fold 4 has an identical group containing settings 0, 1, 2, 10, and 11,
  plus a separate group containing settings 4, 5, and 6;
- settings 4 and 6 are also identical in folds 1 and 5.

These equalities already exist in the requested-weight arrays. The raw action
receipts differ, so the collapse occurs when raw intent is converted into the
economic requested book. Projection then amplifies the problem. For canonical
setting 0, the mean projection-retention ratio across folds is 0.340; in folds
1, 2, and 3 it is only 0.038, 0.024, and 0.009. The requested active L1 norm in
those folds is approximately 1.98, near the maximum separation between two
long-only books, while the executed active L1 norm falls to 0.088, 0.059, and
0.033.

This is a failed Phase-0 gate. A new broad v8 training launch is not authorized
until the active-policy adapter avoids saturated common books and the risk
layer preserves a governed, predictive fraction of requested stock-selection
signal.

### Setting 9 resolution

Setting 9 is confirmed not to be a clean factor-neutrality experiment. Its
route disables factor/sector projection, active-beta control, and tracking-
error control together. Startup turnover is excluded from the reported mean,
and ex-ante TE evidence is unavailable. The setting is excluded from causal
interpretation; a future relaxation experiment must retain benchmark
anchoring, active-beta control, and the 6% TE ceiling.

## Benchmarks and metric definitions

Two benchmarks are used:

1. **Economic benchmark — C1:** active return is policy net log return minus
   the matching C1 benchmark net log return. Portfolio Sharpe is not a
   substitute for active performance.
2. **Experimental benchmark — setting 0:** every causal setting is compared
   descriptively with the canonical 5-bp M03R setting.

The table reports an unweighted mean of the six fold-level estimates. It does
not pool folds, treat them as independent market histories, or provide a
bootstrap confidence interval.

- **Mean active:** mean annualized active log return at the 20-bp validation
  cost, shown as percent per year.
- **Delta vs S0:** mean active-return difference from canonical setting 0,
  shown in annualized basis points.
- **Positive folds:** folds with annualized active return above zero.
- **IR:** mean annualized information ratio.
- **TE:** mean annualized tracking error.
- **Turnover:** mean total one-way turnover per trading session as percent of
  portfolio NAV.
- **Sale age:** mean notional-weighted discretionary sale age in sessions.
- **Young-sale share:** discretionary sold notional before age 30 divided by
  all discretionary sold notional.

## Explicit setting definitions

| Index | Stable setting ID | Exact causal change from setting 0 |
|---:|---|---|
| 0 | `M03R-soft-persistence-active-alpha-hold30-v7` | Canonical: quadratic one-sided soft persistence, 5 bp at age zero. |
| 1 | `P00-no-soft-persistence-v7` | Persistence coefficient changes from 5 bp to 0 bp. |
| 2 | `P10-soft-persistence-10bp-v7` | Persistence coefficient changes from 5 bp to 10 bp. |
| 3 | `A08-fixed-exit-hazard-v7` | Learned exit hazard is frozen to the structural approximately 30-session prior. |
| 4 | `A11-no-exact-hold-atom-v7` | Removes only the discrete exact-HOLD action. |
| 5 | `A09-no-long-context-v7` | Learned temporal context is reduced from 252 to 63 sessions. |
| 6 | `M02-active-risk-no-alpha-heads-v7` | Removes the residual-return alpha heads at 5, 21, 30, and 63 sessions. |
| 7 | `A04-no-downside-score-adjustment-v7` | Uses predicted mean only and removes downside-aware scoring. |
| 8 | `A12-fixed-2pct-active-risk-budget-v7` | Replaces confidence-calibrated 0%-4% active risk with a fixed 2% budget. |
| 9 | `A10-no-factor-neutral-projection-v7` | Disables factor- and sector-neutral projection. |
| 10 | `A06-sharpe-overlay-v7` | Adds a separately optimized total-risk/Sharpe overlay. |
| 11 | `A07-direct-sharpe-v7` | Adds the full-batch, two-pass direct-Sharpe gradient. |

## Aggregate performance benchmark

| Setting | Mean active | Delta vs S0 | Positive folds | Mean IR | Mean policy Sharpe | Mean TE | Daily turnover | Sale age | Young-sale share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.560% | 0.0 bp | 1/6 | -1.006 | 1.086 | 0.750% | 1.629% | 49.8 | 25.6% |
| 1 | -0.502% | +5.8 bp | 2/6 | -0.315 | 1.090 | 0.729% | 1.634% | 49.6 | 25.7% |
| 2 | -0.591% | -3.1 bp | 1/6 | -1.251 | 1.085 | 0.739% | 1.676% | 49.1 | 25.2% |
| 3 | **-0.248%** | **+31.2 bp** | 2/6 | -1.289 | 1.097 | **0.131%** | **0.699%** | 51.1 | **12.6%** |
| 4 | -2.309% | -174.8 bp | 1/6 | -1.709 | 1.027 | 2.150% | 3.745% | 38.5 | 46.5% |
| 5 | -0.774% | -21.4 bp | 1/6 | -1.027 | 1.073 | 1.363% | 2.541% | 45.1 | 38.3% |
| 6 | -2.254% | -169.4 bp | 1/6 | -1.324 | 1.029 | 2.159% | 3.667% | 39.5 | 47.5% |
| 7 | -0.796% | -23.6 bp | 2/6 | -1.015 | 1.073 | 1.362% | 2.539% | 45.1 | 38.3% |
| 8 | -0.814% | -25.4 bp | 1/6 | -1.331 | 1.071 | 1.352% | 2.592% | 44.3 | 37.5% |
| 9 | **-17.058%** | **-1,649.8 bp** | 2/6 | -1.086 | -0.331 (1/6 available) | **14.940%** | 0.002% | N/A | N/A |
| 10 | -0.469% | +9.1 bp | 2/6 | -1.087 | 1.091 | 0.787% | 1.670% | 50.0 | 25.4% |
| 11 | -0.438% | +12.3 bp | **3/6** | **-0.198** | **1.093** | 0.758% | 1.679% | 48.4 | 26.6% |

## Fold-level annualized active return

Values are annualized active log returns. They are shown to expose instability
that an across-fold average would otherwise conceal.

| Setting | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.050% | -2.674% | +1.155% | -1.720% | -0.061% | -0.011% |
| 1 | +0.293% | -2.674% | +1.155% | -1.720% | -0.061% | -0.005% |
| 2 | -0.231% | -2.673% | +1.155% | -1.720% | -0.061% | -0.016% |
| 3 | -0.253% | -0.488% | -0.046% | -0.722% | +0.005% | +0.018% |
| 4 | -0.386% | -2.685% | +1.155% | -1.720% | -1.355% | -8.861% |
| 5 | -0.052% | -2.666% | +1.155% | -1.720% | -1.355% | -0.006% |
| 6 | -0.057% | -2.685% | +1.155% | -1.720% | -1.355% | -8.861% |
| 7 | -0.053% | -2.676% | +1.155% | -1.720% | -1.483% | +0.002% |
| 8 | -0.266% | -2.674% | +1.155% | -1.720% | -1.355% | -0.024% |
| 9 | +11.718% | -27.659% | -9.238% | -28.842% | +0.160% | -48.489% |
| 10 | +0.610% | -2.680% | +1.155% | -1.720% | -0.061% | -0.121% |
| 11 | +0.219% | -2.674% | +1.155% | -1.720% | -0.061% | +0.454% |

## Interpretation by mechanism

### Persistence and exits

Settings 0, 1, and 2 produced almost the same turnover, holding age, young-sale
share, and fold returns. The current one-seed panel does not establish a useful
incremental effect from either 5-bp or 10-bp persistence.

The fixed-hazard setting reduced turnover and young exits substantially, but
also reduced active risk almost to zero. Its better active-return point
estimate should therefore be interpreted as benchmark-like restraint, not as
evidence that fixed exits generate alpha.

Removing the exact-HOLD atom shortened discretionary sale age by about 11
sessions, raised young-sale share from 25.6% to 46.5%, more than doubled daily
turnover, and reduced mean active return by about 175 annualized basis points.
This is the strongest evidence in the panel that the exact-HOLD action changes
behavior usefully.

### Alpha and context

Removing residual alpha heads reduced mean active return by about 169
annualized basis points and produced high turnover and young-sale activity.
This supports keeping the heads, although the canonical policy itself still
did not establish positive alpha.

Reducing temporal context, removing downside-aware scoring, and replacing
confidence sizing with a fixed risk budget each produced a moderate
deterioration relative to canonical. Their effects are similar enough that one
seed cannot distinguish them reliably.

### Risk controls and Sharpe

Removing factor-neutral projection was the clear failure. Mean tracking error
rose to 14.940%, its maximum fold tracking error reached about 19.0%, and most
policy Sharpe estimates were unavailable. Raw unconstrained tilts are not a
viable substitute for stock-selection alpha in this panel.

The separate Sharpe overlay and direct-Sharpe objective slightly improved the
mean active-return point estimate relative to canonical. Direct Sharpe also
had the best mean IR and latest-fold result. Neither achieved positive mean
active return, and both remain diagnostic settings that cannot be promoted
from this panel.

## Execution benchmark

All twelve settings completed six folds and 64 optimizer updates per fold on
two H100 80-GB ranks. No completion receipt reported an allocator OOM or
allocator retry.

- Peak allocated memory was approximately 53.2-53.5 GiB per rank.
- Peak reserved memory was approximately 72.6-76.6 GiB per rank.
- Most settings completed in approximately 8.3-9.4 hours of rank wall time.
- Setting 10 was the runtime outlier at approximately 15.9 hours because of
  its separately optimized overlay.

These measurements establish execution capacity only. They do not improve the
scientific interpretation of the negative active-return results.

## Evidence limitations

- The static TOP2000 universe is future-selected, so the evidence is affected
  by survivorship and selection look-ahead.
- There is one algorithmic seed, not the required five-seed ensemble.
- The six fold values are chronological validation diagnostics, not six
  independent investment histories.
- The completed Phase-0 audit provides a governed frozen-path cost ladder, but
  no bootstrap lower confidence bounds, multifactor alpha, or beta-equivalence
  family exists for this development panel.
- No 2026-YTD evaluation result exists in this report.
- Only setting 0 is conceptually canonical, but even setting 0 is not promotion
  eligible under this reduced TOP2000 seed-17 generation.

The metrics above were read from 72 fold-execution receipts and checked against
their setting-level completion-receipt SHA-256 inventories. The immutable phase
receipt for the completed run is
`8459c53d3d6a28f9ca7cdd276d40f54a95fd5ffa405ff6b13f1da044569d9631`.
