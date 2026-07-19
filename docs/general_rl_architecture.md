# General deep-reinforcement-learning architecture

This document describes the code that exists now. It also names the integration gaps that remain. The intended
audience is contributors adding environments, policies, algorithms, data adapters, or evaluation workflows.

QuantTrade has a **domain-neutral RL substrate** under `rl_quant.rl`. Trading is one application of those
contracts, implemented by the historical portfolio environment under `rl_quant.envs`. The older Phase-1
context/policy experiment remains available as a direct differentiable portfolio baseline; it has not yet been
converted into an on-policy PPO workflow.

## Current architecture

```text
domain data adapter
    -> ObservationBatch + ActionSpec
    -> Algorithm.act(...)
    -> requested ActionBatch
    -> VectorEnvironment.step(...)
    -> executed action + RewardComponents + next observation
    -> trajectory/replay representation
    -> Algorithm.update(...)
    -> domain evaluator and persisted research artifacts
```

The typed contracts through `Algorithm.update(...)` exist. QuantTrade provides separate on-policy and replay
storage, reusable collection coordinators, recurrent PPO, and offline IQL. What does not yet exist is a Phase-1
market-encoder adapter/CLI that assembles those pieces into a full RL experiment, or an artifact-driven evaluator
for that path.

## Domain-neutral contracts (`rl_quant.rl`)

### Specifications and batches

| Contract | Current responsibility |
|---|---|
| `TensorSpec` | Validates event shape, dtype, finiteness, and optional numeric bounds after caller-declared leading dimensions. |
| `ActionSpec` | Adds action kind, optional simplex semantics, CASH index, and action-mask key to a `TensorSpec`. |
| `ObservationBatch` | Carries named tensors with a common batch/device, an optional feasible-action mask, and episode-start flags. Missingness must use explicit masks/channels; floating observations fail on non-finite values. |
| `ActionBatch` | Carries the requested action, log probability, entropy, recurrent state, and algorithm-specific extras. |
| `RewardComponents` | Stores additive return-unit reward terms. Gross return is signed and added; execution, impact, risk, constraint, and liquidation costs are nonnegative and subtracted. |
| `TransitionBatch` | Records the old observation, requested and executed actions, reward ledger, next observation, terminal flags, discount, and diagnostics. |

Requested and executed actions are intentionally separate. An environment may project an action to its feasible
set; policy diagnostics and evaluation can then measure projection distance instead of silently treating the
request as the fill.

`terminated` and `truncated` are also separate:

- A true terminal has discount zero and cannot bootstrap.
- A rollout/time-limit truncation may bootstrap from its real continuation, but recurrent advantage recursion
  stops at the reset boundary.

### Environment and algorithm interfaces

`VectorEnvironment` is a synchronous, torch-native protocol with `batch_size`, `action_spec`, `reset()`, and
`step()`. It is deliberately independent of markets.

`Actor`, `Critic`, and `ActionValueCritic` are small structural protocols. `Algorithm` defines the lifecycle
shared by on-policy, off-policy, offline, and direct methods:

- `act(observation, deterministic=False, recurrent_state=None)`;
- `update(batch)` returning named diagnostics;
- `state_dict()` / `load_state_dict()` for algorithm-owned state;
- train/eval mode switching.

This checkpoint boundary is not the whole experiment boundary. Environment continuation, rollout state, and
global/action-sampling RNG remain orchestration responsibilities unless a concrete implementation explicitly owns
and documents them.

The generic `update` payload is intentional. PPO consumes recurrent on-policy sequences while IQL consumes a
validated `ReplayBatch`; future algorithms can add another validated batch without changing the environment.

### On-policy trajectories

`OnPolicyTrajectoryBuffer` stores time-major vectorized transitions, behavior log probabilities, value and
next-value estimates, and recurrent state. It currently supports:

- generalized advantage estimation with correct terminal/truncation bootstrapping;
- optional advantage normalization;
- fixed-length recurrent sequences;
- burn-in that updates hidden state but is excluded from losses;
- padding and episode-boundary masks;
- decomposed reward components and both requested/executed actions.

This buffer is intentionally on-policy. Replay/offline data use the separate contracts below, so their sampling
semantics cannot be confused with recurrent PPO trajectories.

### Replay collection and offline training

`ReplayBatch` preserves named current/next observations, requested and optional executed actions, scalar reward,
discount, termination versus truncation, current/next action masks, optional behavior log probability, decomposed
reward components, current/next episode-start flags, and extras. `TransitionReplayBuffer` is a schema-locked
circular store: the first addition defines its tensor schema, later additions must match, and sampling retains that
meaning. Full value checks occur at ingestion/checkpoint restore; trusted sampled slices retain structural checks
without repeating GPU-synchronizing finite scans. Prioritized and n-step replay are not implemented.

`ReplayRolloutCollector` gathers behavior-policy transitions through the same `VectorEnvironment` ledger used by
evaluation. It keeps requested and executed actions distinct, carries observation/recurrent/episode state across
collection windows, and reports projection, reward, and episode metrics. `OfflineTrainer` supplies seeded,
checkpointable replay sampling and tracks its update position for any replay-based `Algorithm`. This makes the
sample sequence reproducible; stochasticity inside the algorithm and nondeterministic backend kernels remain
separate responsibilities. Both coordinators use the base synchronous environment contract and fail on
partial-vector completion because that contract has no subset reset.

Replay extras can carry a globally unique int64 `decision_id`. `ReplayBatch.decision_keys()` and
`align_replay_batches(...)` then validate uniqueness and reorder by exact identity; they fail closed when identity
is absent, duplicated, or mismatched. This is opt-in infrastructure, not proof that every existing artifact is
aligned.

## Recurrent PPO (`rl_quant.rl.ppo`)

`RecurrentPPO` is the first algorithm implemented against the new contracts. It includes clipped policy and
value objectives, entropy regularization, gradient clipping, optional target-KL stopping, recurrent minibatches,
checkpointable optimizer/configuration state, and diagnostics including KL, clip fraction, entropy, gradient
norm, and explained variance. The checkpoint also carries the algorithm-owned recurrent-minibatch RNG position;
experiment orchestration must still persist rollout/action-sampling RNG and environment continuation together.

The supplied `RecurrentActorCritic` is a compact GRU reference model. It supports:

- masked categorical actions via `MaskedCategorical`;
- unconstrained continuous vectors via `DiagonalNormal`;
- masked simplex allocations via `MaskedDirichlet` and `action_kind="dirichlet"`;
- single-step acting and padded recurrent sequence updates.

`MaskedDirichlet` samples and scores the active simplex directly: masked dimensions stay exactly zero, active
dimensions are positive, and the one-active-action case is handled as a degenerate simplex. This connects the
reference recurrent PPO model to `VectorPortfolioEnv` without computing a likelihood before an unrelated
softmax/projection. The diagonal Normal is still not valid for simplex allocation, and hybrid actions are
representable by `ActionSpec` but have no hybrid PPO distribution.

`OnPolicyRolloutCoordinator` owns the reusable collection boundary. It stores recurrent state from immediately
before each observation, values the real next observation before reset, zeros bootstrap only at true termination,
allows truncation bootstrap, carries mid-episode continuation between rollout windows, computes GAE, and returns
the trajectory plus episode/throughput metrics. It is tested from toy-environment collection through a real PPO
update; Dirichlet PPO is separately tested against the portfolio action contract. It is a library coordinator,
not an experiment CLI or complete checkpoint/artifact manager.

## Replay-based IQL (`rl_quant.rl.iql`)

`ImplicitQLearning` is the implemented offline reference algorithm. `VectorIQLActorCritic` supplies twin action
critics, an expectile value network, and either a diagonal-Normal or masked-Dirichlet actor. The update follows
these semantics:

1. When replay contains `executed_actions`, the default `action_source="executed_if_available"` trains both
   critics and the actor likelihood on the action whose environment economics produced the reward. Requested
   actions remain available for projection diagnostics; callers may explicitly choose requested-action data.
2. Both critics regress to the Bellman target, while value and actor advantages use the target critics'
   elementwise minimum. Their per-row standard deviation is exposed as critic disagreement and can be subtracted
   from that minimum through `critic_uncertainty_penalty`.
3. Advantage-weighted behavior cloning never queries a counterfactual action inside the Bellman target. Sparse
   logged simplex weights are additively smoothed only on active dimensions and only for the Dirichlet
   log-likelihood; materially negative or off-sum logged weights fail before any optimizer step. The original
   logged/executed action remains the Q input. The L1 smoothing amount is reported.
4. Optional `TransitionTransform`s generate additional Bellman-target candidates. IQL takes the elementwise
   lower envelope across the original and transformed batches and reports their target spread. A private row
   identity sentinel prevents a transform from reordering decisions before the elementwise minimum, and target
   transforms that affect only the current observation fail closed.

Model, target model, all three optimizers, configuration, transform fingerprints, and update count are included
in IQL checkpoints. Replay-sampling state belongs to `OfflineTrainer`, while global stochastic action-sampling RNG
belongs to experiment orchestration. This is a tested algorithm primitive, not evidence that an offline market
dataset is complete, unbiased, or suitable for live deployment.

### Regime mixtures and uncertainty

`RegimeRouter` maps caller-supplied features to expert probabilities.
`RegimeMixtureIQLActorCritic` combines same-support Normal or Dirichlet experts with shared twin critics/value;
its router starts uniformly and can use utilization-balance and entropy regularization. The marginal mixture
log-likelihood is exact. Categorical mixture entropy is exact; continuous/simplex mixture entropy is explicitly
an upper bound. Deterministic categorical acting uses exact marginal MAP; continuous/simplex acting selects the
highest-router-probability expert's deterministic representative;
it does not average separated specialists into a potentially low-density compromise. For `MaskedDirichlet`, that
representative is the always-defined mean, because a mathematical interior mode does not exist for every
concentration vector.

IQL actions expose the twin-critic minimum and disagreement in `ActionBatch.extras`. `UncertaintyAbstention` can
replace an action above a caller-chosen threshold with a safe fallback for deterministic evaluation, shadow, or
deployment use. It clears log probability and entropy because the substituted action did not come from the
original stochastic distribution; it must not be reused as an on-policy sample. The helper provides a mechanism,
not uncertainty calibration or a live adaptation policy.

The domain-neutral layer also provides affine-observation and adverse-reward transforms. Market-specific
`TrendReturnFeatureReversal`, `LiquidityCostStress`, and named lower-envelope suites live under `rl_quant.envs`.
They transform explicitly named point-in-time fields and cost components, do not infer semantics from names, and
do not mutate source replay. These deterministic stresses do not substitute for an empirical impact model or
held-out evaluation.

## Portfolio application (`rl_quant.envs`)

`HistoricalMarketData` holds:

- point-in-time features with shape `[batch, horizon + 1, ...]`;
- one chronological simple-return vector per transition with shape `[batch, horizon, asset]`;
- an availability mask with shape `[batch, horizon + 1, asset]`;
- optional globally unique int64 decision IDs with shape `[batch, horizon]`.

`TensorPortfolioObservationAdapter` combines those market fields with current weights, equity, time index, the
availability/action mask, and episode-start state. The environment then enriches the result with peak equity,
drawdown, recent turnover, gross exposure, CASH weight, configured constraint limits/enabled flags, risk-halt
state, and valid-action fraction without changing the adapter signature. A learned encoder can replace the
adapter without changing the environment or algorithm contracts.

`VectorPortfolioEnv` is the canonical implementation for the new RL path. It currently:

1. accepts a requested long-only simplex allocation including CASH;
2. applies availability, per-risky-asset weight, and gross-exposure constraints;
3. caps discretionary one-way turnover, while hard feasibility changes may exceed the cap and report the forced
   excess;
4. treats optional maximum drawdown as a post-return breach threshold that latches a CASH halt, not a bound that
   can prevent an intra-step overshoot;
5. reports forced turnover and request-to-execution projection distance;
6. sends only the final feasible target to a separate `TargetWeightExecutionModel`, which is authoritative for
   costs but cannot substitute another allocation;
7. realizes exactly one chronological asset-return vector;
8. marks holdings through that return rather than rebalancing for free;
9. latches a drawdown breach, immediately moves to CASH with cost, and masks subsequent policy actions to CASH;
10. decomposes reward into explicit return-unit components;
11. liquidates to CASH, with cost, only at a true data terminal. Because liquidation occurs after the period return,
   its cost fraction is scaled to beginning-of-transition return units so the additive ledger matches sequential
   equity accounting exactly.

The default `ImmediateTargetWeightExecution` charges configured spread/fee and an optional linear-impact
sensitivity, and declares `models_market_fills=False`. This historical research simulator has no partial fills,
queue position, market-by-order data, venue latency, volume/capacity calibration, empirical nonlinear impact,
borrow model, built-in CASH yield, or live adaptation. Its output must not be described as executable trading
evidence.

Every transition includes runtime environment and decision positions for debugging. When `HistoricalMarketData`
receives optional decision IDs, the stable `decision_id` also flows through transition info into replay; exact
alignment APIs require it and reject absence or duplicates. Runtime batch position is deliberately not treated as
a stable cross-run identity.

## Causality, provenance, and normalization

The RL interfaces prevent shape and terminal-semantics drift, but they cannot make a dataset point-in-time by
themselves. The market adapter is responsible for ensuring every observation was available by the decision
timestamp.

The active raw-window pipeline now carries explicit per-field covariate validity and optional event-time universe
membership. `inspect_dataset_provenance()` fails reportability when the universe selection date is missing,
invalid, later than the sample start, or paired with a declared coverage start that differs from the earliest
actual date inside the bars files. A `point_in_time` or `rolling` universe requires
`universe_membership.parquet`; the table must be non-empty, cover every declared non-CASH action with a positive
event, and contain no undeclared symbols. Its events apply only after both their effective date and availability timestamp.
Reserved-name collisions use the explicit `universe.json.source_symbol_aliases` action-to-source mapping; raw
symbols are never deduplicated or silently assigned to synthetic CASH.
A prior-selected static universe can pass the mechanical check but retains a survivorship/delisting warning.

Stage-1 bar and covariate normalization is fixed-stat and explicit:

- `ContextEncoder.calibrate_normalization(...)` streams masked training samples into persistent per-field
  moments;
- masked and non-finite values are excluded;
- `forward()` never updates or derives statistics from its submitted session;
- train and eval use identical statistics;
- unfitted fields use the identity transform;
- normalization buffers are part of the encoder checkpoint.

The Phase-1 driver calibrates on a deterministic training-only sample before Stage-1 optimization and broadcasts
the result across distributed ranks. Validation/test data must never enter this calibration.

The reusable `generate_walk_forward_folds(...)` library API is implemented and tested. It accepts one
caller-supplied `label_horizon` and enforces `purge_size >= label_horizon`; the caller must calculate that value as
the execution/settlement delay plus the maximum target, selection, or reporting horizon. The API cannot discover
those dependencies from a dataset. It supports explicit embargo geometry
and expanding or bounded rolling training windows. Fold identities are bound to a required caller-supplied
`decision_axis_id`; validation checks its presence and binds it into the digest, but cannot prove that the string
really identifies an immutable/content-addressed dataset snapshot. The splitter is not yet wired into a launcher
or artifact evaluator: the Phase-1 command continues to use its existing chronological train/validation/test split.
Do not claim walk-forward model selection for a run until that end-to-end integration is complete.

## Relationship to legacy Phase 1

The `../training/train_phase1.py` workflow remains a useful market-specific baseline:

- Stage 1 self-supervises a causal context encoder, freezes it, and caches detached context.
- Stage 2 directly differentiates a gated portfolio allocation objective.
- The daily-raw path now uses one-day realized wealth change as its training and reporting reward; longer-horizon
  labels are auxiliary targets, and causal input-only prefixes provide out-of-sample memory warm-up.
- Legacy intraday/daily rollouts share pure execution accounting helpers for drift, forced availability changes,
  turnover, and terminal liquidation.

It does **not** instantiate `VectorEnvironment`, either rollout coordinator, either buffer, `RecurrentPPO`, or
`ImplicitQLearning`. Its gate is a continuous interpolation between the prior and target allocation, not a
sampled trade event, and its optimizer is not actor-critic RL. Treat it as a direct portfolio optimizer in
comparisons until a concrete adapter implements the RL observation/environment boundary.

The migration should preserve this baseline so new algorithms can be compared under the same data, execution,
cost, and evaluation contracts. See [the migration ledger](architecture_migration_plan.md).

## Evaluation and reportability boundary

The new RL core returns the ingredients needed for evaluation, but it does not yet persist a complete research
artifact. Optional int64 decision IDs and fail-closed replay alignment solve an important identity problem only
when callers supply them. The legacy Phase-1 results now also persist decision identifiers and join seed returns
by the exact identifier set, while older list-only checkpoints remain uncertified. Neither path is yet the full
RL artifact evaluator. A production-quality runner still needs to record, at minimum:

- date/decision identity and provenance hashes;
- requested and executed actions, masks, projection distance, and constraint violations;
- every reward component, equity, holdings, turnover, and terminal treatment;
- actor/critic losses, value calibration, KL, clip fraction, entropy, and seed;
- aligned baseline returns and cost/latency/impact stresses;
- train/validation/test boundaries and all checkpoint-selection decisions.

Passing unit tests is not a reportable result. In particular, the existing TOP50/TOP2000 datasets were built
from a universe ranked after the backtest began and must remain development-only until rebuilt with lagged or
rolling point-in-time membership. Current news scores are also research-only because their model-availability
metadata is anachronistic; the Phase-1 driver disables news by default.

The legacy verdict's alignment certification means complete, identity-matched scored-label coverage; it is not a
dataset-provenance certificate. Provenance is evaluated separately and now forces `positive=false` for
`--allow-unreportable` runs even if their `statistically_positive` diagnostics pass.

Statistical promotion should use aligned per-date returns across designs/seeds, purged and embargoed walk-forward
selection, paired baseline comparisons, block-bootstrap uncertainty, multiple-testing correction, and
cost/capacity stresses. Walk-forward launcher/evaluator integration is still in progress. No new RL run should use
the test split for seed or checkpoint selection.

## Tests that define the current contracts

- `tests/test_rl_core.py` — specs, batches, rewards, terminal semantics, GAE, recurrent burn-in.
- `tests/test_ppo.py` — categorical/Normal/Dirichlet distributions, PPO updates/checkpoints, portfolio integration.
- `tests/test_replay.py` — schema-locked replay, executed actions, terminal semantics, and exact identity alignment.
- `tests/test_offline.py` — behavior collection, continuation, replay sampling, and offline update coordination.
- `tests/test_iql.py` — twin-critic IQL, executed-action learning, sparse-simplex likelihood smoothing, uncertainty,
  conservative transforms, mixtures, and checkpoints.
- `tests/test_mixture.py` — router/marginal-distribution semantics and same-support expert validation.
- `tests/test_rollout.py` — recurrent continuation, correct bootstrap, GAE, and collect-to-PPO integration.
- `tests/test_market_robust_transforms.py` — named market stresses and lower-envelope suites.
- `tests/test_portfolio_env.py` — chronological reward, target-weight execution authority, risk state/halt, optional
  identity, constraints, costs, drift, and liquidation.
- `tests/test_context_normalization.py` — train-mode causality, fixed stats, masks, and serialization.
- `tests/test_dataset_provenance.py` — future-universe rejection and event-time membership.
- `tests/test_walk_forward.py` — purge/embargo geometry, expanding/rolling folds, axis-bound identity, and validation.
- `tests/test_partition_split.py` — immutable physical partition-split values.
- `tests/test_phase1_reporting.py` — identity-aligned seed aggregation, requested-replication completeness, and
  legacy fail-closed certification.
- `tests/test_daily_runtime_accounting.py` — compact daily storage and legacy accounting parity.
