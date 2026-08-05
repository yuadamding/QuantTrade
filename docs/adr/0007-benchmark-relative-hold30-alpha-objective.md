# ADR-0007: Benchmark-relative alpha is the Hold-30 promotion objective

**Status:** Accepted; components locally CPU integration-qualified; end-to-end
training and launch blocked

**Date:** 2026-08-04

## Context

The sealed S0–S7 review found that learned daily returns were almost perfectly
correlated with equal weight and that apparent turnover-cap differences were
dominated by cash deployment and terminal liquidation. Absolute-return
optimization can be satisfied by broad market exposure and therefore does not
establish state-dependent stock selection.

ADR-0006 corrected the economic holding mechanism: one daily decision,
positions carried across sessions, fill-time age, soft 30-session duration,
and legal early exits. Holding mechanics alone cannot create alpha. The next
pre-lockbox generation needs a promotion objective that distinguishes C1-like
market exposure from active stock selection and prevents a learned policy from
collapsing invisibly back to C1.

## Decision

`prelockbox-hold30-alpha-mech8-v3` uses the existing C1 monthly point-in-time
active-300 equal-weight buy-and-drift portfolio as its training benchmark.

The canonical promotion candidate is
`hold30a-m03-alpha-core`. It combines:

- a 30-session alpha-mean head;
- a downside-uncertainty head;
- one-session net active log-return utility versus C1;
- an annual tracking-error floor/target/ceiling of 2%/4%/6%; and
- a market-beta target of 1.0 with an allowed range of 0.9–1.1.

Auxiliary alpha horizons are exactly 5, 21, 30, and 63 sessions. Labels are
built and censored inside each permitted split. Training uses 20-bp costs;
validation retains 10-, 20-, and 40-bp rungs, with only 20 bp used for
checkpoint selection.

The eight-setting inventory includes absolute-return and active-mean controls,
two canonical component removals, and two ineligible Sharpe diagnostics. M03
is the sole promotion candidate. A total-risk/Sharpe overlay and a direct
two-pass Sharpe-gradient term are attribution experiments, not alternative
promotion routes.

Real point-in-time risk-free, cap-weight market, and declared factor artifacts
are mandatory. Their allowlists differ: C1 is the action anchor and active
training benchmark; the market series is beta-objective, beta-checkpoint, and
beta-evaluation data; risk-free/CASH is portfolio-accounting,
a06/a07-total-excess-Sharpe, 20-bp total-Sharpe checkpoint-ranking, and
evaluation data; factors are evaluator-only.
All have `policy_feature_access=false`. Missing inputs fail qualification; the
implementation cannot substitute zeros, equal weight, constants, synthetic
traces, or defaults.

V2 is superseded before launch. Its implementation history remains auditable
and reusable only under new v3 identities and receipts. V3 artifact producers
must reject v2 generation and setting IDs.

## Consequences

- The [v3 RFC and experiment specification](../prelockbox_hold30_alpha_mech8_v3.md)
  is the current pre-lockbox protocol.
- The [v2 specification](../prelockbox_hold30_mech8_v2.md) remains an audit
  record and cannot authorize a Job, checkpoint family, or result.
- A policy that tracks C1 too closely fails the canonical TE floor even if its
  nominal return is positive.
- A policy that takes excessive active risk or leaves the beta band fails the
  canonical risk contract even if its return is high.
- Market and risk-free artifacts are restricted to their named objective,
  checkpoint, accounting, and evaluation roles; factors support sealed
  attribution only. None are policy features.
- No 2026 evidence may be used to tune the objective, bands, heads, costs, or
  checkpoint rule.
- A hash-only training plan cannot make v3 executable. The manifest must bind
  a typed plan whose setting-specific objective config validator confirms all
  required coefficients and A06/A07 routing while rejecting prohibited
  ablation terms.
- The typed plan's checkpoint contract is authoritative at render time. Its
  resolved value must equal the contract embedded in both the common design
  and top-level manifest; stale unresolved thresholds cannot coexist with
  resolved values.

## Implementation status

The protocol and typed data contracts, setting-specific model/action and
objective surfaces, checkpoint logic, sealed-evaluator components, and a
non-authorizing one-update synthetic artifact driver are implemented. The
driver closes policy/action/accounting/objective/checkpoint receipts for all
eight qualification-only routes, including A06's core-intact three-stream
update and parent-linked optimizer-state receipts. This does not qualify the
production 240-trial study.

A separately content-addressed pilot profile freezes the result-moving numeric
coefficients and checkpoint thresholds without changing the canonical
unresolved template. Before launch, that pilot profile freezes A06's
total-Sharpe variance floor at `1e-6`, matching A07's 10-bp daily-volatility
floor. The floor is negligible at ordinary equity volatility, stabilizes
near-zero denominators, and was not chosen from observed outcomes. Typed
real-data/global-path bindings, an immutable training image, GPU/H100 parity
and capacity, launch approval, completed training, and investment performance
are all unqualified. Full-policy CPU two-rank parity is qualified only for one
path per rank, and exact all-eight restart/resume parity is qualified. No executable
receipt hash is recorded here; executable receipts must be reissued only after
the remaining contracts are resolved.

## Revisit conditions

Changing the training benchmark, alpha horizons, primary horizon, TE band,
beta target, risk-artifact roles, promotion candidate, or checkpoint-selection
cost requires a new protocol generation and ADR amendment. An observed outer
result is never sufficient justification for an in-generation change.
