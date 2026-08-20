# Alpha-learning foundation

This document describes the reusable local implementation that follows the
v16 daily-selection closeout. It is an implementation boundary, not a claim
that a reportable dataset exists or that any candidate has predictive alpha.

The dependency order is fixed:

```text
point-in-time economic data
→ post-fill residual targets
→ predictive comparison
→ signal-only attribution
→ deterministic translation and execution
→ sequential RL uplift
→ prospective confirmation
```

No downstream layer may compensate for a failed upstream gate.

## Implemented source surfaces

### Point-in-time data authority

`rl_quant.alpha.contracts` and `rl_quant.alpha.accounting` provide:

- permanent security and issuer identities;
- causal ticker and universe membership histories;
- separate observable, decision, fill, mark, and terminal states;
- corporate-action and terminal-disposition records;
- causal cash returns;
- exact no-follow dataset manifests and byte inventories;
- economic positions that book splits, dividends, mergers, spin-offs,
  tenders, delistings, and worthless outcomes exactly once;
- post-fill total returns that reject missing economic paths and never replace
  delistings or missing marks with zero returns.

The implementation does not include a vendor adapter. A source adapter may be
added only when its approved inputs, timestamps, and independent reconciliation
authority are available.

### Targets and residualization

`rl_quant.alpha.targets` freezes:

- one decision/fill convention per experiment;
- one primary horizon and a sorted auxiliary horizon inventory;
- targets that begin at the actual fill;
- simple or log economic total-return semantics;
- terminal outcomes without a future-survival condition;
- origin-available weighted QR residual operators;
- one permanent asset axis for scores and targets.

Exact total losses remain valid simple returns of `-1`. They are not converted
to fabricated finite log returns.

### Signal attribution

`rl_quant.alpha.attribution` enforces the period identity:

```text
active net
= signal gross - signal cost
 + repair gross - repair cost
 + benchmark cost advantage
 + other active return
```

Signal break-even uses only signal gross return and signal-created turnover.
Repair profits and benchmark-cost advantages cannot satisfy the promotion
gate. Promotion requires positive lower confidence bounds for both signal net
return and factor-adjusted signal alpha, plus absolute signal break-even of at
least `max(10 bp, 2 × modeled median one-way cost)`.

### Experiment and trial identity

`rl_quant.alpha.experiment` content-addresses the dataset, universe, target,
decision/fill rules, modalities, horizons, model, optimizer, folds, seed, and
parent trial ledger. Predictive specifications cannot authorize economic
optimization, RL, or prospective access. Every result-moving architecture,
modality, target, loss, seed, risk, portfolio, cost, threshold, and historical
generation choice has an explicit trial record.

### Ordered representation and objective

`rl_quant.models.alpha_hierarchical` implements:

- ordered raw five-minute tokens with fixed training-only normalization;
- causal within-day attention;
- causal cross-day attention;
- a fixed-size market-latent bottleneck with linear stock-count complexity;
- ordered downside, median, and upside quantiles;
- positive predictive scale;
- a hierarchical raw-interval → stock-day → cross-day → market path.

`rl_quant.training.alpha_supervised` implements date-balanced robust mean,
tail-pair rank, quantile, calibration, and residual self-supervised losses.
The same executable score is supplied explicitly for ranking.

### Evaluation and economics

`rl_quant.evaluation.alpha_panel` provides paired model/baseline rank IC,
tail spreads, fold-cluster/non-wrapping-block inference, and concentration
gates. Model and baseline metrics use identical date/asset support.

`rl_quant.execution.age_aware_no_trade` provides a soft quadratic young-sale
friction that reaches zero at the preferred age. It is never a sell mask or a
mandatory holding period. Forced exits are exempt.

`rl_quant.execution.impact_model` separates half-spread, nonlinear volatility
impact, linear participation impact, delay, and fees. Capacity evaluation
reports clipping and lost notional at declared capital levels.

## Current authorization state

The source and golden tests authorize only local software development.

```text
real PIT dataset materialized:       no
independent data reconciliation:     no
five-minute discovery panel run:     no
predictive candidate confirmed:      no
economic optimization authorized:   no
RL authorized:                       no
prospective access authorized:       no
```

The next external dependency is an approved, point-in-time source adapter and
an independently reconciled `PITAlphaDatasetV1` artifact. Until that exists,
model training would test fixtures rather than real alpha.

## Blocking invariants covered by tests

The focused alpha tests require:

- ticker changes preserve permanent identity;
- future membership cannot alter a current action;
- delisting losses and mixed merger consideration are booked;
- post-fill targets wait for their complete economic endpoint;
- future survival cannot enter target support;
- origin factor exposures were available by the decision;
- score residuals are weighted-orthogonal to declared exposures;
- signal, repair, benchmark, and other P&L reconcile exactly;
- repair or benchmark profits cannot promote a negative signal;
- changing future intervals cannot change earlier intraday or cross-day tokens;
- invalid padding payloads are ignored;
- stock permutation permutes stock outputs and leaves market latents unchanged;
- date loss weights do not depend on eligible asset counts;
- aligned tail ranks beat reversed ranks;
- holding friction remains soft;
- costs increase with participation and capital;
- paired panel evidence beats its baseline on common support.
