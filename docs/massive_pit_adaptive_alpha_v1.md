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

## Endogenous holding duration

> **Endogenous holding duration.** The adaptive RL system has no mandatory,
> preferred, minimum, maximum, or rewarded holding period. At every decision
> date, all positions are reconsidered using current causal forecasts,
> uncertainty, portfolio risk, liquidity, trading costs, and alternative
> investment opportunities. A position persists only while retaining it remains
> economically optimal. Realized holding duration is an output of the learned
> policy and deterministic feasibility compiler, not an input constraint or
> evaluation target.

The adaptive generation therefore contains no entry clock, persistence
coefficient, young-sale penalty, fixed exit schedule, duration reward,
duration checkpoint score, or duration promotion gate. Post-experiment
duration diagnostics may describe behavior, but they cannot affect training,
checkpoint or seed selection, policy freezing, outer pass/fail, or
profitability authorization.

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

`MassiveAdaptiveForecastArchiveV1` closes same-plan checkpoint-to-forecast
replay for every package-selected training origin. It runs the exact promoted checkpoint in
deterministic evaluation mode over the promoted decision tensor and window
plan, commits every distribution, routing, context, score, validity, and
positive-probability array by content hash, and binds both full-chronology and
origin-only decision-root inventories. Generic reloads expose metadata only.
`MassiveAdaptiveForecastReplayAuthorityV1` reruns the checkpoint/tensor/root
calculation before exposing runtime forecast authority; it cannot authorize
profitability, lockbox access, or RL. The current normalization identity is the
already committed Feature V3 representation and the current forecast outputs
remain explicitly uncalibrated, so calibration is still a downstream
training-only artifact rather than an implied profitability qualification.

Role-separated inference is a distinct V2 boundary.
`MassiveAdaptiveInferencePlanV1` derives every decision in one complete
inner-validation chronology from a replayed tensor and decision-root
inventory. Unlike the supervised window plan, it opens no target archive and
does not require a 126-session target to mature. It binds causal model context
and the next exchange-session schedule identity, but deliberately does not
claim that a qualifying fill exists.

`MassiveAdaptiveForecastEligibilityAuthorityV2` reconciles the replayed
training checkpoint and its training window with a disjoint inference tensor,
root inventory, and plan. It requires the same fold, split family, model
specification, protocol, and source semantics while explicitly rejecting the
old equality between training and inference tensors, roots, and origins. V1
currently authorizes only the legal training-to-inner-validation transition;
outer-test and lockbox roles remain unavailable until checkpoint selection is
frozen.

`MassiveAdaptiveForecastArchiveV2` and its replay authority then rerun the
training checkpoint over all target-free inner-validation origins, persist the
same distribution, routing, score, context, validity, and array receipts, and
rebuild compatibility plus inference before restoring runtime forecasts. A
generic V2 reload remains nonauthorizing. Raw V2 forecasts retain the explicit
uncalibrated identity and cannot authorize profitability, lockbox access, or
RL.

The first deterministic inner-validation profitability kernel now continues
from V2 forecasts through a training-only calibration, causal compiler-input
authority, the existing no-duration compiler, decision-close share intents,
next-session source VWAP execution, and a continuous cash/share book marked at
the next qualified close. The compiler-input authority derives its 63-session
risk/liquidity state without opening the future fill; realized fill price and
volume enter only the execution transition. Strategy and equal-weight
buy-and-drift shadow books use the same fill and cost kernel. Profit traces
reconcile requested, filled, and unfilled shares, costs, book receipts, net
wealth, benchmark wealth, and active log return. The composite development
authority grants no reporting, outer-test, lockbox, or RL capability and
reexecutes the entire trace before accepting it.

Corporate, terminal, successor, and cash-return transitions are now bound to
the complete raw-provider V6 economic archive rather than a caller-supplied
resolved event list. Each fill-date close resolves revisions and interaction
order from that archive. Events between the prior close and morning fill
repair both the carried holdings and pending target; events after the fill are
applied to the executed book before the close mark. Splits, cash
distributions, worthless dispositions, successor holdings, and causal cash
accruals are therefore part of the same replayed share ledger used by both the
strategy and benchmark. Omitting the provider archive keeps the engineering
path nonqualifying, and changing the archive invalidates trace replay.

Inner-validation checkpoint choice is derived from a frozen-action 10/20/40
basis-point ladder. The selection value itself remains nonauthorizing. A
separate create-only selection authority reopens the committed choice and
recomputes the winner from the exact candidate inventory before restoring
runtime selection state. Consequently, an in-memory selection object cannot
open the outer chronology. The outer inference plan additionally requires the
replayed selection authority, its selected training checkpoint, a qualified
outer decision tensor, and qualified decision roots before it will enumerate
any outer-test date. A separate create-only outer forecast archive then binds
that selection authority, the selected checkpoint, the outer plan, the outer
tensor, and both training and outer root inventories. Promotion reruns every
target-free outer forecast exactly; the archive still grants no profitability
reporting, lockbox, or RL authority.

Outer economic traces reuse the same compiler, next-morning execution,
corporate/terminal-event transition, and continuous cash/share book as inner
validation. Each fold is evaluated at $10M under a frozen-action 10/20/40
basis-point ladder. Four-fold evidence applies a deterministic fold-cluster,
nonwrapping 63-session moving-block bootstrap with 2,000 replicates to both
primary net return and benchmark-relative active log return. The evidence
must also retain nonnegative 40-bp mean performance, positive primary profit
in at least three folds, and a monotone cost ladder. The value artifact itself
is nonauthorizing; a create-only evidence authority reopens it and recomputes
all four folds before it can authorize a development outer conclusion. Final
profitability reporting, lockbox access, and RL remain false.

RL-training forecasts have a separate leakage boundary. The frozen V1
prequential artifact remains readable for earlier synthetic canaries, but it
uses inner-validation forecast blocks and is not the historical RL-fit source.
`MassiveAdaptiveRLFitInferencePlanV1` is the production-facing correction. It
derives an expanding fit-only prefix of exactly `126 * (fold_index + 1)`
sessions from the split plan, partitions that prefix into 21- or 63-session
blocks, and accepts no caller dates or inference role. Every row is target-free
and binds only its causal model context and following exchange-session
identity.

`MassiveAdaptiveRLFitForecastArchiveV1` binds one of those blocks to a replayed
training checkpoint and training window. Both the supervised training cutoff
and the maximum target-maturity cutoff must precede the first forecast. A
generic reload exposes metadata only; promotion reconstructs compatibility and
reruns every committed output array. RL permission remains downstream of the
checkpoint-choice, calibration, complete-block, and chronology composite
authority.

The adaptive RL-facing surface is a bounded compiler-control action. Seven
controls rescale the forecast buckets, uncertainty and portfolio-risk controls
are bidirectional within frozen ranges, and a bidirectional trade-cost control
scales only the compiler's soft entry, exit, and replacement hurdles from 0.5
to 2.0 times their source-derived values. The hard turnover ceiling is never
changed. Security, issuer, tracking-error, active-beta, and ADV limits cannot
be loosened, and the controller never emits security weights. The unique zero
action returns the original compiler inputs and configuration unchanged.

The corrected engineering path adds checkpoint/fold-bound calibration,
all-cash initialization, one shared buy-and-drift benchmark authority,
fill-start/close event snapshots, and compiler inputs derived from the current
three-book state. `MassiveAdaptiveEconomicStepV1` is the sole prepare/settle
kernel for both deterministic and policy-controlled transitions. Strategy,
neutral compiler, and benchmark books pass through the same order, fill,
capacity, cost, event, and next-close accounting. The zero action is tested
through the complete transition: decision, orders, fills, costs, posttrade
books, wealth, reward, and semantic receipts are identical.

`MassiveAdaptiveProfitabilityEnvV1` carries those three books continuously and
reports unpenalized strategy and neutral active log returns plus the
basis-point-scaled incremental strategy-minus-neutral reward. Its observation
is a fixed 90-value summary of the seven forecast buckets, compiler risk and
liquidity inputs, current books, recent realized economics, and the previous
bounded action. True episode termination includes a conservative liquidation
adjustment; a rollout boundary preserves the exact books and chronology.

The first PPO canary uses separate two-layer actor and critic networks. All ten
controls use transformed-Normal distributions over the open interval from -1
to 1, so no post-sampling clip invalidates the PPO log probability. The trainer
implements GAE, clipped actor and value losses, and update-boundary checkpoints
containing model, optimizer, RNG, economic environment, and chronology state.
A split-run regression reproduces the next actions, transitions, losses, model
tensors, and checkpoint receipt exactly.
The create-only durable checkpoint authority publishes only safe tensor and
primitive state, strips runtime state on a generic reload, and restores the
actor, critic, both optimizers, every RNG, all three books, and the chronology
cursor only after its causal training-forecast authority is replayed.

`MassiveAdaptiveRLTrainingForecastAuthorityV2` closes the historical
prequential selection boundary over the fit-only archives. It requires the
complete package-derived 126/252/378/504-session prefix and admits rolling 21-
or 63-session blocks only when the
supervised training cutoff, 126-session target-maturity cutoff, training-loss
checkpoint-choice cutoff, and checkpoint-bound calibration cutoff all precede
the first forecast in every block. Checkpoint choice uses final training loss
only; missing, duplicated, reordered, validation-role, or caller-dated blocks
fail closed. Synthetic or otherwise unqualified sources continue to withhold
RL-training authority. V1 remains the compatibility surface for earlier
synthetic artifacts and is not the historical launch authority.

`MassiveAdaptivePrequentialPPORunnerV1` consumes that complete block inventory
in its committed order. It carries actor, critic, optimizer, and RNG state
across every block. Distinct forecast archives and calibrations may replace the
model state at a source-authorized refit boundary while the strategy, neutral,
and benchmark books remain continuous; only nonconsecutive economic episodes
restart from the registered cash state. Its
durable checkpoint binds the current block, within-block cursor,
completed-block inventory, calibration, environment source inventory,
transition inventory, and nested PPO checkpoint. Generic reopening strips
runtime state, and promotion restores the runner and reproduces every state
receipt before training may continue.

The immutable experiment configuration is available through a package-owned
command:

```bash
quanttrade-adaptive-rl manifest \
  --experiment-id <registered-id> \
  --output <create-only-manifest.json> \
  --block-sessions 63 \
  --seed 17

quanttrade-adaptive-rl validate \
  --manifest <create-only-manifest.json>
```

The V2 manifest freezes the four folds, causal block size, elapsed-session
checkpoint schedule `(126, 252, 378, 504)`, one canonical seed, PPO
configuration, $10 million capital,
10/20/40-bp ladder, two-percent participation limit, the complete registered
constant comparator inventory and fit-selected FC06,
observation/action/reward identities, all-cash initialization, shared
benchmark, drawdown cap, and outer gates. It contains no outcomes and is
nonauthorizing by itself. Fold-local update indices are replay-derived from the
block size: `(2)`, `(2,4)`, `(2,4,6)`, and `(2,4,6,8)` for 63-session blocks,
or `(6)`, `(6,12)`, `(6,12,18)`, and `(6,12,18,24)` for 21-session blocks.
Callers cannot register a global update-index tuple.

FC00--FC05 and FC07--FC12 are the immutable symmetric constant-action fitting
grid. FC06 is not a caller-supplied extra action: it is the package-derived
winner of that complete grid on the RL-fit chronology only. The selection
binds one common fit-origin
inventory and economic context across every grid member. A create-only fit
authority executes every registered action over the complete prequential tape,
carries all three books through forecast refits, and promotes only after
rerunning every transition. FC06 selection consumes only those replayed
candidates. A package-owned FC06
evaluator resolves the selected action from the sealed grid and generates its
inner-validation transitions itself; the registry-aware PPO-candidate builder
does not accept an independently assembled fixed-control trace. V1 does not
yet contain a fitted and replay-authorized contextual baseline; a label alone
would not be a comparator.

`run_massive_adaptive_rl_training_workflow_v1()` owns every authorized block
and publishes each scheduled update twice from the same state: an exact
prequential-runner resume authority and the policy-checkpoint authority used
by deterministic evaluation. The same training workflow executes the complete
registered constant grid
over that complete fit tape and publishes the replayed FC06 selection.
`run_massive_adaptive_rl_validation_workflow_v1()` then reloads every
registered policy candidate, regenerates its primary actions, publishes exact
frozen-target 10/40-bp stresses, and regenerates FC06 on the identical shared
validation context. Neither workflow accepts caller actions, transitions,
returns, or P&L arrays.

There is intentionally no historical `run` subcommand yet. The repository
still lacks one persisted composite loader that can reopen the complete live
forecast, calibration, decision-root, fill, event, and identity bundle from a
source root. Adding a launch command that bypasses that authority would
silently restore caller control over economic sources. Until that loader and a
positive real-source canary exist, the command surface remains manifest and
validation only, and profitability reporting remains false.

The RL chronology is opened in two stages. PPO fitting and policy selection
first bind fit, validation, and outer date inventories directly from the split
plan, without opening a selected-checkpoint outer forecast. Only after policy
selection may the actual outer inference plan bind to those precommitted fold
dates. The three origin inventories are pairwise disjoint.

Prequential forecast refits do not end the economic episode. A dedicated
continuity authority permits a forecast archive and calibration to change only
when the previous block's next-session close is the next block's first
decision close and execution, identity, accounting, benchmark, capital, cost,
and participation semantics match exactly. Strategy, neutral, and benchmark
books, high-water marks, trailing returns, and the previous action carry into
the next archive. The internal boundary is non-liquidating and PPO bootstraps
its value estimate from the first causal observation under the next refit.
Nonconsecutive blocks remain separate economic episodes.

All PPO checkpoint candidates are evaluated against one committed validation
context receipt. Forecasts, calibration, chronology, roots, fills, events,
identity, compiler configuration, capital, participation, initialization, and
benchmark semantics cannot vary by candidate; only the attached actor state
may differ. The 10/20/40-bp rungs share that context while excluding only the
cost value from the shared-context identity.

Inner-validation policy economics are derived from complete environment
transitions rather than caller return arrays. A create-only policy-selection
authority binds the 10/20/40-bp frozen-target ladder, terminal liquidation,
drawdown, active wealth, incremental strategy-minus-neutral wealth, and the
selected PPO update. Static compiler controls are selected separately using
training-only traces; the PPO candidate must then beat that committed fixed
control on inner validation. The selected model state is published in a
separate create-only frozen-policy artifact before any outer date is opened.
The immutable constant grid is symmetric in uncertainty aversion, portfolio
risk aversion, and soft trading aggressiveness and also includes registered
short/long-horizon combinations. FC06 is selected only after every registered
constant traverses the same complete RL-fit economic tape. This is a stronger
static baseline, but it is not yet a fit-optimized continuous 10-dimensional
constant or a contextual linear controller; claims requiring those contrasts
remain unauthorized.

Validation evaluation is checkpoint-owned: it reloads the actor, reconstructs
every observation, emits the registered deterministic bounded action, and
records the distribution parameters, action, compiler control, decision,
execution, next book, and reward. Its create-only trace authority promotes
only after rerunning the actor and complete chronology. The cost-ladder
authority runs the actor and compiler once at 20 bp and replays that exact
target-weight inventory at 10 and 40 bp through the same fill, event, and book
kernel.

Each outer plan binds one fold's selected supervised checkpoint, calibration,
target-free outer forecast archive, selected frozen RL policy, compiler
configuration, all-cash initialization, and shared benchmark. Four-fold RL
evidence uses the same 126-session, nonwrapping fold-cluster bootstrap for
strategy active wealth, strategy-minus-neutral wealth, and PPO-minus-fixed-
control wealth. It also requires at least three positive folds for every
load-bearing contrast, a monotone frozen-target cost ladder, nonnegative mean
40-bp return, and no fold drawdown above 25 percent. Generic evidence reloads
remain nonauthorizing and final reporting and lockbox access remain false.

Outer evaluation is likewise actor-owned. A create-only rollout authority
loads the selected frozen model state, forbids updates, regenerates every
outer action, and reproduces its economic trace. A second authority replays
the primary rollout's target weights at 10 and 40 bp without reevaluating the
policy or compiler. Checkpoint-authenticated outer evidence accepts a fold
only when both the frozen-policy rollout authority and frozen-target cost
ladder authority match its policy, traces, and target inventory. The selected
FC06 comparator has a separate create-only outer authority: it reopens the
fit and selection authorities, recovers the registered constant action, and
reruns it on the same sealed outer environment. V3 evidence admits the
PPO-minus-FC06 contrast only when both outer authorities share the exact
environment source inventory, chronology, forecast, calibration, and capital.
The create-only outer-access commitment freezes the selected PPO policy, fit
authority, FC06 selection, selected action, compiler, benchmark, capital, cost
ladder, and chronology before the package-owned outer forecast may be
materialized. The resulting gated forecast archive binds that commitment as
its entitlement. Outer-plan V3 can then be built only from the comparator-bound
V2 plan, replay-authorized commitment, and replay-authorized gated forecast.
The static comparator receives its own frozen-target 10/20/40-bp outer ladder.
V4 adds a paired 40-bp PPO-minus-FC06 gate.

The durable V4 evidence record binds each fold's V2 plan, outer-access
commitment, gated forecast archive, rebuilt V3 plan, PPO ladder, and static
ladder. Generic reload remains nonauthorizing. Promotion validates the
authorized commitment and gated forecast, rebuilds V3 from those dependencies,
and rejects even a self-consistent V3 object whose receipt strings were
invented and rehashed. It then rebuilds every fold and the aggregate evidence
before restoring a development conclusion. Profitability reporting and
lockbox access remain false.

The V2 manifest uses one canonical predeclared seed. It permits neither
validation-selected seeds nor seed ensembling, and seeds are never treated as
additional market observations. A later multi-seed protocol requires a new,
explicit selection or shared-book ensemble authority.

Every position is reconsidered from its current net economics at every
decision. The canonical adaptive observation, action, reward, compiler,
environment, and checkpoint-selection surfaces contain no entry-clock state,
release schedule, persistence incentive, or selection threshold based on time
in the book. Forecast-model refits preserve the economic books because a refit
is not a market event, but they confer no requirement that any position remain.

Trade-cost control is strictly an economic replacement-hurdle control. It may
make economically justified trading more or less aggressive by scaling soft
cost estimates, while the registered hard turnover and liquidity limits remain
fixed. It is tuned against realized net profitability and may never be
calibrated, described, or reported as a proxy for a target holding interval.
Likewise, the seven forecast horizons describe return information available to
the optimizer; they do not define or schedule how long a resulting position
should remain in the portfolio.

This closes the minimum package-owned library path from causal
prequential forecasts through bounded PPO, exact durable resume,
inner-validation policy selection, a frozen fold-bound policy, and paired
four-fold outer evidence. The remaining experiment-level boundary is one
persisted source loader and state machine that owns all four folds without
accepting selected identifiers, actions, transitions, or statistics from an
external driver. It does **not** manufacture historical authority:
the synthetic canaries remain nonauthorizing, final reporting stays false,
and a real run still requires acquired payloads, partitions, features,
targets, source-qualified prequential blocks, and replayed fold authorities.
Intraday path tensors are also not yet materialized by the V1 tensor artifact.
Those data-first boundaries remain prerequisites for an H100 launch or any
profitability claim.

## Evidence boundary

Documentation and local tests are design evidence only. A reportable result
requires the exact Massive entitlement, source, delayed replay, PIT identity,
dual-universe, economic, target, tensor, package, lifecycle, training, compiler,
evaluation, and cleanup receipts.
