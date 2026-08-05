# Hold-30 alpha v3 evaluation tranche

**Protocol:** `prelockbox-hold30-alpha-mech8-v3`

**Status:** implemented diagnostics; promotion and launch blocked

The package evaluator is
`rl_quant.evaluation.hold30_alpha_evaluation`. It rejects the superseded v2
generation and accepts only the eight stable `hold30a-*` identities. The sole
candidate is `hold30a-m03-alpha-core`. The
`hold30a-a06-sharpe-overlay` return stream is evaluated and receipt-bound
separately; it can neither overwrite nor promote as the alpha core.

All eight setting streams are mandatory. The family receipt refuses a partial
panel or any endpoint reused across settings, and reports exact 20-bp
contrasts for m01−m00 (persistence), m02−m01 (active objective), m03−m02
(alpha heads), m03−A04 (uncertainty), m03−A05 (TE floor), A06−m03 (separate
Sharpe overlay), and A07−m03 (direct Sharpe term). These are diagnostics; only
m03 is the predeclared promotion candidate.

## Typed evidence

Each setting/fold panel contains exactly 63 daily observations and binds the
10/20/40-bp policy, C1, and active-log arrays to the endpoint receipt. It also
binds the exact point-in-time risk-free series, declared factor returns,
portfolio weights, cross-sectional scores/outcomes/masks, uncertainty, and
age-P&L arrays. It additionally binds a typed v3 data receipt plus the exact
point-in-time cap-market artifact identifier, artifact SHA-256, aligned daily
array, and provenance receipt. Each endpoint also content-binds the strictly
increasing source-row indices, ordered score dates, and exact policy and C1
weight tensors used for active share. The data-binding CASH digest must equal
the exact PIT risk-free return tensor digest, and uncertainty inputs must be
finite and nonnegative. Six fold panels must have disjoint dates and
distinct endpoint, data-binding, risk-free, factor, and cross-sectional
receipts.

Scientific data roles remain disjoint: the point-in-time cap-market series may
serve the beta objective, checkpoint eligibility, and evaluator; the
point-in-time risk-free/CASH series may serve economic accounting, A06/A07
total-excess Sharpe terms, 20-bp total-Sharpe checkpoint ranking, and
evaluation; declared factors are evaluator-only.
None of market, risk-free, CASH-yield, or factor artifacts is actor-visible.
Formal beta uses only the separately bound cap-market excess series
(`market total return − PIT risk-free`). There is no factor named `MKT`
fallback. Declared factor conventions are frozen per factor and identical
across folds: zero-investment and excess-return factors enter as supplied;
total-return factors are converted to excess returns by subtracting the exact
PIT risk-free series before regression. The formal multifactor regression
always contains the explicit cap-market excess column followed by the declared
factor columns, so market beta cannot be absorbed into reported alpha.

The evaluator recomputes, rather than accepts pass flags for:

- total net return, volatility, Sharpe, Sortino, Calmar, drawdown, and downside
  deviation at every cost rung;
- active log return, tracking error, information ratio, relative drawdown, hit
  rate, active share, and correlation versus C1;
- market-only and declared multifactor alpha/loadings, residual volatility,
  residual Sharpe, and Newey-West sensitivity at lags 10, 21, 30, and 42;
- optional deterministic within-fold moving-block alpha intervals from a
  manifest-bound seed/replicate/block plan;
- Pearson and rank IC at 5/21/30/63 sessions, score-decile returns,
  uncertainty-bucket IC/returns, and alpha P&L decay by position age; and
- five individual-seed, C1-initialized, and 64 C8 daily paths from typed
  arrays with their run, endpoint, selection, and cross-fold mapping receipts.

C6 ownership is exhaustive: one 64-replay receipt for each of the eight
settings and six folds. Only the 63 outer-score rows are permuted; every other
row is fixed. The terminal training inventory is exactly 8×6×5 with no
duplicate artifact graph or selective retry.

## Point gates and controls

At the primary 20-bp rung, alpha core reports these frozen point gates:

- pooled information ratio greater than 0.5;
- annual tracking error in `[0.02, 0.06]`;
- market beta in `[0.9, 1.1]`;
- policy net Sharpe minus C1 net Sharpe at least `-0.10`; and
- active total strictly above the 61st of 64 controls matched on turnover,
  risky exposure, median sale age, and 30-session survival.

The control evaluator also reports the 64-control IR distribution, but does
not silently turn that diagnostic into a second promotion threshold.

## Fail-closed issues

The scientific request does not yet freeze:

1. the exact confirmatory factor-alpha hypothesis family;
2. its multiplicity procedure and family alpha; or
3. the moving-block replicate count, block lengths, seed, and interval alpha.

These choices can change the factor-alpha conclusion. The evaluator therefore
reports them as promotion blockers and always emits
`promotion_authorized=false` and `launch_authorized=false`. A future
content-addressed decision must freeze them before code may implement a
confirmatory promotion rule.

Artifact inventories hash live file bytes and, when declared, independently
verify the parsed JSON payload/self-hash. Pretty printing and a terminal
newline do not invalidate a correct payload, while duplicate JSON keys do.
All SHA-256 bindings recursively named in the v3 manifest must resolve to a
live byte or verified payload digest.

The one-shot lockbox marker uses exclusive creation, `fsync` on the file and
parent directory, and rejects an existing different reveal. It explicitly
records that the historical 2026 S0–S7 evidence is consumed and unused by v3.
Publishing this marker records consumption; it does not authorize launch or
promotion.
