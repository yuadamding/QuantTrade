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
- compact delayed-WebSocket, REST, and flat-file records pass through distinct
  canonicalizers, preserving the raw payload hash and timestamp units;
- delayed events use the later of a conservative clock-error upper bound on
  actual local receipt and the qualified SIP-plus-delay time, while finalized
  rows use only the qualified delay rule;
- every replay and capture lifecycle binds one protocol-derived decision clock
  equal to the authority-resolved regular close plus exactly 60 minutes;
- replay re-resolves the condition and correction rules and rechecks the
  Developer delay, session boundary, source bytes, and identity;
- open/close, high/low, and volume eligibility remain separate through bar
  reconstruction;
- committed recorder JSONL derives authentication, successful subscription,
  transport heartbeat, disconnect, trade, clock, recorder-image, and
  subscribed-ticker evidence without persisting auth requests or credentials;
- the capture domain begins at the Eastern source-calendar-day boundary and
  continues through the protocol decision, preserving premarket events and
  correction ancestry; one multi-ticker capture is extracted independently
  for each security-session;
- delayed/final parity requires committed source bundles, extraction receipts,
  and one source-byte-bound feature materializer that parity reruns itself;
  arbitrary feature mappings are not an evidence surface;
- finalized daily gzip files are scanned as streams, require the exact
  `us_stocks_sip/trades_v1` identity and schema, and retain original physical
  line numbers, raw-row hashes, and tape IDs;
- ticker-change canaries resolve both adjacent records inside the supplied PIT
  universe authority rather than trusting a caller-provided authority digest;
- transport success and HTTP 200 are not runtime entitlement evidence: every
  required surface also needs bound, typed semantic evidence for its schema,
  requested date, and returned inventory;
- aggregate reconciliation is nonempty, source-homogeneous, unadjusted, and
  bound to a finite immutable tolerance specification;
- source publication and reload traverse directories and open files relative
  to no-follow directory descriptors and bind the final inode and ctime;
- the Stocks condition authority binds the exact stocks/trade query and rejects
  non-stock condition rows;
- aggregate receipts sort intervals canonically before hashing.

Historical as-of replay remains fail-closed for the current runtime evidence:
`runtime_entitlement_qualified` and `canonical_source_parsers_qualified` are
both required. Synthetic committed-parser tests do not qualify the real
entitlement or real source inventory; the real finalized-file and delayed-
capture rehearsal must be sealed. Even after that closure, replay
parity cannot authorize predictive training; PIT identity, both PIT universes,
economic accounting, targets, and repeatable tensors remain separate blockers.

## Discovery family

`AD00`–`AD11` are cumulative registered neural settings. `AD00` is a daily
single-21-session diagnostic, `AD01` introduces the term structure, and later
rows add rank, distributional, market-context, intraday, tape, restricted
expert, router, pretraining, and long-context components. Only `AD11` is
initially promotion eligible. Elastic net, histogram gradient boosting,
momentum/reversal, and bars-only models remain mandatory external baselines.

## Current implementation boundary

The package now includes a create-only
`MassiveAdaptiveDecisionTensorV1` model-input commitment. It constructs one
deterministic permanent-security axis from committed Feature V3 rows,
separates context membership from the package-owned action mask, preserves
bars and tape missingness independently, carries the committed source
staleness channel, and binds exact float32 per-decision array hashes. A generic
reload has no tensors and no model-input authority; package promotion rebuilds
all arrays from the exact Feature V3 and adaptive-origin inventories.

`build_massive_adaptive_alpha_training_batch_v2()` remains the nonauthorizing
component adapter that maps bare target rows from each changing PIT action set
onto that stable context axis. It
derives the action-mask benchmark and B01 benchmark return internally and
leaves context-only securities explicitly target-missing. This removes the V1
assumption that context and action security axes are identical.

The authorizing training boundary is separate. A
`MassiveAdaptiveContextOriginAuthorityV1` proves the exact PIT-1500 feature
support, and `MassiveAdaptiveDecisionRootV1` reconciles it with the PIT-500
action origin, decision timestamp, decision clock, session authority, and
Feature V3 inventory. `MassiveAdaptiveSplitPlanV1` freezes 126-session inner
and outer purges plus a 126-session outer-to-lockbox embargo. A package-derived
`MassiveAdaptiveWindowPlanV1` owns the tensor and candidate indices; callers
cannot provide an origin index or free split receipt. The historical builder
must reopen the create-only archive freeze and reproduce its candidate
inventory; the direct calendar builder is engineering-only and cannot qualify
training.

`MassiveAdaptiveTrainingAuthorityV1` then requires one replayed
`MassiveAdaptiveSourceTargetsV1` for every eligible window origin through a
create-only `MassiveAdaptiveTargetArchiveV1`. Each per-decision
`MassiveAdaptiveTargetRootV1` reexecutes the complete target path from the live
decision clock, session calendar, action identity, daily-input, fill, terminal,
and economic-coverage authorities and reconciles it with the exact
dual-universe decision root. Generic archive reload has neither runtime target
objects nor training authority. The synthetic binder is explicitly
nonqualifying.

The full decision chronology and the target-bearing origin inventory are
separate commitments. Context-only, purge, and not-yet-mature rows may remain
in the decision tensor without requiring target artifacts. The window plan
derives the eligible origin subset, the target archive contains exactly that
subset, and both the full and origin-only decision-root inventories are bound
through the training authority and checkpoint. A 505-date regression fixture
exercises a 504-session validation context spanning fit and purge dates while
opening a target for only the validation origin. The decision root also
records an explicit package-derived PIT-500 action-identity qualification;
binding the action identity without qualifying its reconstructed membership is
not sufficient for training qualification.

The package-owned supervised trainer reopens the decision tensor, rebuilds all
decision and target roots plus the split/window plans, performs the model
forward itself, and publishes a create-only exact-resume checkpoint. The
training authority and checkpoint bind the target-archive, target-root, and
experiment-source inventories plus the separate full-chronology and
target-origin decision-root inventories in addition to the model, optimizer,
scheduler, gradient-scaler placeholder, CPU/CUDA/data-order RNG state,
epoch/update/window cursor, window permutation, and loss trace. A generic
checkpoint reload has no runtime state or training authority. Substituting an
independently valid identity, daily-input, fill, terminal, or coverage receipt
after archive commitment is rejected.

The synthetic qualification path is deliberately named a canary and cannot
promote itself. The historical entry point requires qualified context roots,
rebuilds them from the live PIT-1500 identity authority and decision-clock
inventory, and also requires the canonical AD11 model specification and frozen
training configuration.
Neither path authorizes profitability reporting, lockbox access, or RL.

This closes source-to-optimizer and exact-restart wiring. It does **not** yet
close checkpoint-to-forecast replay, compiler-input materialization, or the
chronological compiler/economic simulator. Intraday path tensors are also not
yet materialized by the V1 tensor artifact. Those deterministic profitability
boundaries remain prerequisites for an H100 historical launch and for any RL
work.

## Evidence boundary

Documentation and local tests are design evidence only. A reportable result
requires the exact Massive entitlement, source, delayed replay, PIT identity,
dual-universe, economic, target, tensor, package, lifecycle, training, compiler,
evaluation, and cleanup receipts.
