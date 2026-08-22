# Massive PIT adaptive-alpha V1

`massive-pit-adaptive-alpha-v1` is a fresh scientific generation. It does not
modify or reinterpret any Hold-30 protocol or artifact.

The objective is to learn the conditional cross-sectional term structure of
factor-residual stock returns. The portfolio is reconsidered once per session
after the delayed-data cutoff and trades at the next-session 09:35–09:45 ET
qualifying-trade VWAP. Position duration is completely endogenous.

## Immutable boundary

```text
dataset:             MassiveStocksDeveloperPITV1
context universe:    PIT-1500
action universe:     PIT-500
decision:            close + 60 minutes
fill:                next session 09:35–09:45 ET VWAP
controller:          daily cost-aware receding-horizon optimization
canonical cost:      20 bp one-way
cost ladder:         10 / 20 / 40 bp one-way
RL:                  blocked pending supervised and compiler promotion
```

The seven non-overlapping target buckets are:

```text
B01 B02_05 B06_10 B11_21 B22_42 B43_63 B64_126
```

No bucket is primary. Each produces a mean residual return, q10, median, q90,
predictive scale, and positive-return probability. Training-only robust scales
give every bucket equal scientific weight.

## No-duration rule

The adaptive generation contains no minimum or preferred duration, position-age
input, persistence coefficient, young-sale penalty, fixed exit hazard, duration
reward, duration checkpoint score, or duration promotion gate. Position age may
be computed only after execution as descriptive telemetry.

The package enforces both a configuration-key firewall and an AST import
firewall. New adaptive modules cannot import historical Hold-30 or age-aware
execution implementations.

## Data authority before training

Massive Stocks Developer supplies delayed bars and trades, but final next-day
files do not prove exactly what was visible at the prior 17:00 ET decision.
Historical performance therefore remains blocked until actual delayed
WebSocket captures reconcile with finalized files at both event and
model-feature level.

The first milestone is limited to:

1. secret-free entitlement authority;
2. immutable source-object receipts;
3. condition and correction authorities;
4. causal trade replay;
5. delayed-stream/final-file parity.

It authorizes no model training, portfolio optimization, historical lockbox,
prospective access, or RL.

The source milestone is evidence-derived rather than label-derived:

- normalized trade events bind the entitlement, session, condition,
  correction, source-object, permanent-identity, and ticker-history receipts;
- replay re-resolves the condition and correction rules and rechecks the
  Developer delay, session boundary, source bytes, and identity;
- open/close, high/low, and volume eligibility remain separate through bar
  reconstruction;
- delayed capture completeness is derived from authentication, subscription,
  heartbeat coverage, disconnect inventory, raw-capture identity, and the
  subscribed ticker set;
- delayed/final parity is computed from typed capture, source, replay, and
  feature artifacts, with one actual row required per canary kind and a
  minimum six-symbol-day/two-session coverage contract;
- aggregate reconciliation is nonempty, source-homogeneous, unadjusted, and
  bound to a finite immutable tolerance specification;
- source publication and reload traverse directories and open files relative
  to no-follow directory descriptors.

These contracts can authorize historical as-of replay only after real canary
artifacts satisfy the coverage gate. They still cannot authorize predictive
training; PIT identity, both PIT universes, economic accounting, targets, and
repeatable tensors remain separate blocking authorities.

## Discovery family

`AD00`–`AD11` are cumulative registered neural settings. `AD00` is a daily
single-21-session diagnostic, `AD01` introduces the term structure, and later
rows add rank, distributional, market-context, intraday, tape, restricted
expert, router, pretraining, and long-context components. Only `AD11` is
initially promotion eligible. Elastic net, histogram gradient boosting,
momentum/reversal, and bars-only models remain mandatory external baselines.

## Evidence boundary

Documentation and local tests are design evidence only. A reportable result
requires the exact Massive entitlement, source, delayed replay, PIT identity,
dual-universe, economic, target, tensor, package, lifecycle, training, compiler,
evaluation, and cleanup receipts.
