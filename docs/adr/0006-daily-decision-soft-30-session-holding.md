# ADR-0006: One daily decision with a soft 30-session holding target

**Status:** Accepted; mechanism-8 v2 implementation/qualification in progress and launch blocked

**Date:** 2026-08-04

## Context

QuantTrade is intended to make one portfolio decision per trading session and
carry the resulting positions across sessions. The desired holding behavior is
approximately monthly, defined here as **30 trading sessions**. It is not 30
calendar days and it is not the roughly 21 sessions in an average calendar
month.

The existing `daily_raw` path already drifts holdings through realized returns,
supports a no-trade scalar-gate action, and separates decision-visible state
from delayed execution. It does not implement the intended holding behavior:

- the reviewed historical external TOP2000 run contract used an approximately
  21-session controlled cash-start suffix; current versioned configuration
  proves stride 21 but cannot certify that score-tail caller without the
  missing driver/R0 source binding;
- its actor proposes a fresh absolute portfolio with one portfolio-wide gate;
- its gate bias starts at `2`, a nominal 88.1% interpolation probability before
  state-dependent output and execution constraints;
- its legacy action budget penalizes mean gate rather than executed turnover;
- it has no position-age state, age-dependent exit semantics, or holding-age
  telemetry; and
- every independently sampled training window is still an economic cash reset,
  even though the legacy loop does carry weights across its internal BPTT
  detach boundaries.

The separate generic PPO workflow does not close this gap: its sampled
episodes also begin from cash and terminate the portfolio at the episode end.
Changing a label horizon or a legacy action-count setting would therefore not
make either path a 30-session holding strategy.

## Decision

### Strategy mandate

The target policy has all of the following semantics:

1. Exactly one ensemble portfolio decision, action build, and execution is made
   for each eligible trading session. Each frozen ensemble member is evaluated
   at most once for that decision.
2. Positions, cash, portfolio constraints, model state, and holding-age state
   carry across trading sessions.
3. Ordinary signal noise should preserve the current portfolio rather than
   replace it daily.
4. The desired holding duration is soft and centered near 30 trading sessions.
5. Early exits remain legal when expected benefit is sufficiently large, a
   risk rule binds, or an asset becomes unavailable.
6. An intra-sweep graph, credit-window, or loader boundary is not an economic
   portfolio boundary. It must not cause a cash reset, synthetic startup trade,
   or liquidation. A preregistered complete-optimizer-update replay may restart
   from its declared benchmark endowment, but it is a new deterministic
   training sweep rather than a continuation of the prior economic path.

There is no hard 30-session lock. Risk and availability controls remain
authoritative, and the learned policy may pay a predeclared early-exit penalty
when evidence supports a discretionary early sale.

### State and action contract

The target learned action is an entry score, a per-stock age-aware
exit/reduction hazard, and a separate risky-exposure control. The neutral action
is to retain the feasible pretrade portfolio. A single portfolio-wide scalar
gate remains only as a reproduction or transitional comparator.

The environment owns an age ledger measured from legal fill time. Buys and
sells for the same asset are netted before the ledger is updated, so a wash-like
sell-and-rebuy cannot reset age. Startup, discretionary, membership-forced,
availability-forced, risk-forced, and terminal turnover are distinct accounting
causes.

The authoritative implementation uses enough age resolution to measure the
declared endpoints. The initial contract uses bins `0..59` and `60+`, plus a
return-neutral cohort-retention record for survival analysis. Forced exits are
reported separately as competing risks.

### Objective contract

Economic equity is updated only by realized one-session portfolio return after
execution cost. The learning utility uses daily benchmark-relative log return.
Duration and stability terms are learning-only penalties and may not be booked
as additional economic transaction costs.

The design budgets actual executed **discretionary** one-way turnover and may
penalize discretionary sold notional younger than 30 sessions. Startup,
membership-forced, availability-forced, risk-forced, and terminal transactions
are exempt from those learning penalties and remain visible in their own
ledgers.

### Evidence contract

The 2026 S0–S7 evaluation informed this change and the initial v1 protocol, so
it is consumed. After v1 is frozen, it cannot be reread or used for further
training, debugging, threshold adaptation, checkpoint selection, or model
choice. The Hold-30 mechanism must first pass the pre-2026 H0–H3 protocol and
then face a separately registered untouched or prospective lockbox.

## Consequences

- The [Hold-30 RFC](../daily_hold30_policy_rfc.md) is the proposed
  implementation contract.
- The [H0–H3 experiment specification](../prelockbox_hold30_h0_h3_experiment.md)
  remains the normative mechanics base. The
  [alpha mechanism-8 v3 specification](../prelockbox_hold30_alpha_mech8_v3.md)
  is the current pre-lockbox generation. The
  [mechanism-8 v2 delta](../prelockbox_hold30_mech8_v2.md) was superseded
  before launch and remains implementation-history evidence only.
- H1's scalar gate is an ineligible transitional mechanism control. Only H2
  implements the target learned action contract and may qualify for scale-up.
- The earlier A0–A5 draft is superseded before execution. Its optimizer and
  architecture questions may be registered later, after holding mechanics are
  validated, but that protocol must not be launched unchanged.
- H3's fixed sleeves are an ineligible structural comparator: without an
  ordinary signal-driven pre-maturity exit, they do not satisfy this mandate.
- Current `daily_raw` and generic PPO behavior is nonconforming until portfolio,
  age, and model state cross graph chunks without economic resets and the
  blocking tests pass.
- A setting such as `max_actions_per_day = 252 / 30`, a 30-day label alone, or
  a small global gate is not sufficient evidence of 30-session holdings.
- [ADR-0004](0004-env-execution-owns-reward.md) remains authoritative: age,
  turnover cause, execution, and reward must converge on one environment-owned
  accounting implementation, with exact parity for any temporary direct
  differentiable adapter.

## Revisit conditions

Changing 30 trading sessions to a calendar-month convention, imposing a hard
minimum holding period, permitting intraday portfolio decisions, or removing
ordinary early exits changes the strategy mandate and requires a new ADR and a
new experiment generation.
