# Deep-RL migration ledger

This ledger tracks the move from the market-specific Phase-1 direct optimizer to a general deep-RL framework.
Its implementation-status table is a 2026-07 snapshot; the documentation was
updated on 2026-08-04. Pending items are not promises that the implementation
already exists. The stable architecture contracts are documented in
[general_rl_architecture.md](general_rl_architecture.md).

The scientifically negative S0–S7 evaluation motivates a proposed next
generation rather than a wider search over the old grid. The strategy mandate
is recorded in [ADR-0006](adr/0006-daily-decision-soft-30-session-holding.md):
one portfolio decision per trading session, continuously carried positions,
and a soft target holding duration of roughly 30 sessions. The
[Hold-30 redesign RFC](daily_hold30_policy_rfc.md) and
[H0–H3 pre-lockbox experiment](prelockbox_hold30_h0_h3_experiment.md) are the
only active proposed implementation and experiment protocol. The earlier
A0–A5 draft was superseded before execution because it tested a materially
different holding mechanism. This ledger continues to describe implemented
behavior until the RFC code and blocking tests land.

## Non-negotiable boundary

> For the general RL path, only the environment/execution layer may mutate domain state, execute/project an
> action, or compute reward.

Algorithms consume observations and transitions. Data adapters establish causal observations. Evaluators replay
the same environment semantics and persist artifacts. A trainer must not carry a private copy of portfolio P&L,
turnover, drift, or liquidation logic.

The legacy Phase-1 trainer predates this boundary and still owns its differentiable rollout. It now shares pure
accounting primitives with the environment, but it is not yet evidence that the migration is complete.

## Current status

| Workstream | Status | Evidence / boundary |
|---|---|---|
| Domain-neutral specs and typed batches | Implemented | `rl_quant.rl.specs`, `rl_quant.rl.types`; fail-closed unit tests. |
| Algorithm/environment protocols | Implemented | `rl_quant.rl.algorithm`, `rl_quant.rl.environment`. |
| Recurrent on-policy trajectories and GAE | Implemented | Correct termination/truncation bootstrapping, burn-in, padding, and reward-component storage. |
| Recurrent PPO reference | Implemented | Masked categorical, diagonal-Normal, and masked-Dirichlet policies; recurrent minibatches, checkpointed minibatch RNG, and portfolio integration tests. |
| On-policy rollout coordinator | Implemented as a library primitive | Preserves pre-observation recurrent state, terminal/truncation bootstrap, mid-episode continuation, GAE, and episode metrics; no production CLI. |
| Typed replay and offline coordinators | Implemented | Schema-locked requested/executed transition replay including episode boundaries, behavior collection, checkpointed deterministic replay sampling, trusted sample fast path, and optional exact int64 decision alignment. |
| Twin-critic IQL | Implemented | Executed-action economics by default, expectile value, validated sparse-simplex likelihood smoothing, critic disagreement/penalty, row-identity-preserving lower-envelope targets, and checkpoints for IQL-owned model/optimizer/configuration state. Replay and global action RNG remain external. |
| Regime mixture and uncertainty helpers | Implemented | Learned same-support expert router, exact categorical marginal MAP, coherent highest-router continuous/simplex action, balance/entropy terms, critic-disagreement output, and deterministic fallback abstention. Calibration/live adaptation are not supplied. |
| Historical portfolio environment | Implemented | Long-only simplex projection, separate target-weight execution-cost authority, risk-state observations, maximum-drawdown CASH halt, chronological reward, drift, and sequentially scaled terminal liquidation. |
| Market robust transforms | Implemented | Explicit trend/return reversal and liquidity/cost stresses compose into named lower-envelope IQL scenarios. |
| Dataset universe provenance | Implemented mechanically | Future-selected universes fail; rolling/PIT membership is supported. Existing ranked datasets still need rebuilding. |
| Causal Stage-1 normalization | Implemented and integrated | Fixed training-only moments; forward is immutable and train/eval identical. |
| Legacy daily reward/accounting repairs | Implemented | One-step daily-raw reward, auxiliary H-day target, burn-in score mask, drift and liquidation helpers. |
| Hold-30 state/action/accounting contract | **Pending** | ADR-0006 is agreed, but canonical state carry, origin-indexed credit replay, fill-time age/cohort state, per-stock hazards, sleeves, and duration evidence are not implemented. |
| Walk-forward splitter | Implemented as a library primitive | Tested purge/embargo geometry, expanding/rolling windows, and fold identities bound to caller-supplied horizon/axis identity. The API cannot infer effective lookahead or verify dataset-snapshot authenticity; launcher/evaluator integration remains in progress. |
| General RL experiment CLI | **Pending** | Implemented collectors, algorithms, and environment still need configuration, checkpoint/RNG/provenance bundling, and command orchestration. |
| Phase-1 encoder-to-RL observation adapter | **Pending** | Frozen/raw market encoders are not connected to `ObservationBatch` for PPO. |
| Legacy trainer/evaluator routed through environment | **Pending** | Direct differentiable rollouts remain separate, although accounting primitives are shared. |
| Artifact-driven RL evaluation/reportability | **Pending** | Identity hooks exist, but the RL core does not yet write the complete decision log, manifest, baselines, stresses, and selection history. |
| Other algorithms | **Pending** | SAC, modern DQN/QR-DQN, CQL, prioritized replay, and n-step replay are not implemented. |

## Migration sequence

### 1. Establish trustworthy data — in progress

- Rebuild TOP50/TOP2000 from a universe selected no later than the training start, or provide an event-time
  `universe_membership.parquet` with delistings and availability timestamps.
- Keep `--require-reportable` and `--no-news` as defaults.
- Persist universe method/date/hash, membership mode, raw-source hashes, and known coverage gaps.
- Run-resume and shared Stage-1 context identities now bind provenance-manifest content, canonical raw-source
  signatures, ordered actions, and driver/package/distributed-helper source content in addition to configuration.
- Reserve a new future lockbox because the existing test period has already informed model design.

Exit condition: the provenance gate passes without override, and delisting/missing-return behavior is tested.

### 2. Complete the portfolio action adapters — simplex primitive implemented; Hold-30 pending

Delivered:

- `MaskedDirichlet` samples and scores the active simplex directly, with masked coordinates exactly zero;
- `RecurrentActorCritic(action_kind="dirichlet")` carries action masks into both acting and recurrent PPO update;
- tests cover masked recurrent PPO updates and acceptance of the sampled action unchanged by the portfolio
  action contract;
- projection distance remains available from the environment.

The simplex adapter remains valid as a generic PPO/reference compatibility
primitive. H0 instead uses the experiment's deterministic
target-softmax/scalar adapter under direct optimization. Neither is the target
Hold-30 action. Remaining work is to add a typed entry-score,
per-stock-hazard, and risky-exposure intent adapter with environment-owned age
state, then monitor projection distance and action causes in the artifact
evaluator. Sparse top-k allocation is not silently introduced; if added, its
probability semantics must be explicit.

### 3. Add one reusable rollout coordinator — library primitive implemented

Delivered:

- `OnPolicyRolloutCoordinator` owns reset/step collection, pre-observation recurrent state, next-value estimation,
  GAE, synchronous episode resets, continuation across fixed rollout windows, and low-cardinality metrics;
- true terminals zero bootstrap, truncations may bootstrap from the real next observation;
- tests cover toy-environment collection through a real recurrent PPO update; portfolio distribution compatibility
  is covered separately.

Remaining integration work: the CLI must checkpoint model, optimizer, RNG, normalization, rollout position, and
provenance identity together. The coordinator deliberately rejects partial-vector completion until an environment
defines subset-reset semantics.

### 4. Adapt Phase-1 observations without coupling the RL core — pending

- Wrap causal raw/context encoders in a `PortfolioObservationAdapter` or portfolio actor; do not import market
  models into `rl_quant.rl`.
- Calibrate fixed normalization on training data only and include its buffers in checkpoints.
- Use causal input-only prefixes for recurrent validation/test warm-up while excluding prefixes from scores.
- Keep longer-horizon forecasting and SSL targets auxiliary to the environment's one-step reward.

Exit condition: H0–H3 consume the same point-in-time observations, scored date
range, causal age summaries, availability masks, and censor masks under the
direct optimizer. A later registered PPO comparison must reuse that frozen
observation and Hold-30 state/action contract.

### 5. Make environment accounting universal — pending

- Compare the legacy direct rollout with `VectorPortfolioEnv` on deterministic action traces.
- Resolve parity for availability, missing/delisting returns, turnover convention, costs, holdings drift, and
  terminal liquidation.
- Route sequential evaluation through the environment first.
- Retain the differentiable legacy rollout only as an explicitly named baseline if training cannot use the
  non-differentiable environment.

Exit condition: evaluator rewards come from the environment; portfolio,
age/cohort, pending-intent, and model state cross intra-sweep boundaries without
an economic reset; turnover is cause-typed; continuing and separately
liquidated wealth reconcile; and any direct-baseline differences are declared
and regression-tested rather than accidental.

### 6. Persist evaluation artifacts before scaling — pending

- Use optional `HistoricalMarketData.decision_ids` so transitions/replay have globally unique int64 identities;
  exact replay alignment must fail closed when IDs are missing, duplicated, or mismatched.
- Write a dated decision log with observations/provenance IDs, masks, raw/
  constructed/filled action stages, pending intents, reward components,
  holdings, age/cohort or sleeve state, cause-typed turnover, equity,
  constraints, censoring, continuation, and terminal flags.
- Write a run manifest containing split boundaries, seeds, selection rules, configs, code/data identities,
  normalization state identity, and runtime/GPU-hour measurements.
- Evaluate cash, equal weight, aligned buy-and-hold, random-same-turnover, and simple linear/momentum baselines.
- Run paired multi-seed comparisons and 1x/2x/4x cost and capacity stresses.

The legacy Phase-1 result payload now persists per-decision identifiers and joins seeds by identity. That repair
does not supply the requested/executed action ledger or make the legacy artifact a complete RL evaluation record.
Its provenance gate now keeps `positive=false` for `--allow-unreportable` runs even when the separate
`statistically_positive` diagnostics pass.

Exit condition: reportability is derived from persisted artifacts, not a console summary or in-memory metric.

### 7. Integrate purged walk-forward selection — library primitive implemented

Delivered:

- `generate_walk_forward_folds(...)` has explicit decision-count geometry, accepts a caller-supplied effective
  `label_horizon`, and enforces `purge_size >= label_horizon`;
- configurable embargo and expanding or bounded rolling training windows;
- immutable stable fold identities over decision positions/dates and a required caller-supplied decision-axis ID;
- fail-closed validation and adversarial tests.

The caller must calculate `label_horizon` from execution/settlement delay plus the maximum target, selection, or
reporting horizon; the splitter cannot discover those dependencies. It binds `decision_axis_id` into each fold
digest but cannot prove that the supplied string is an immutable/content-addressed dataset snapshot.

Remaining integration work:

- Wire immutable fold identities and boundaries into the future RL run manifest.
- Fit normalization and select checkpoints/hyperparameters inside each permitted training fold only.
- Keep the final lockbox outside fold construction and selection.

Exit condition: the audited splitter is exercised through the launcher and evaluator, and every reported result
can identify the exact folds used. Until then, the Phase-1 launcher retains its single chronological split.

### 8. Extend replay-based algorithms behind the common contracts — partly implemented

Delivered:

- schema-locked `ReplayBatch` / `TransitionReplayBuffer` with requested and executed actions;
- `ReplayRolloutCollector` and `OfflineTrainer` with checkpointable seeded replay sampling (algorithm/backend
  stochasticity remains separate);
- twin-critic `ImplicitQLearning`, masked-Dirichlet behavior policy, regime-mixture actor, uncertainty diagnostics,
  and lower-envelope robust transition targets.

Remaining:

- categorical QR-DQN/Double-DQN for truly discrete actions;
- SAC for a validated continuous allocation distribution;
- constrained/Lagrangian or distributional risk objectives;
- n-step and prioritized replay;
- CQL or other alternatives only where their logged-action assumptions are documented.

Each algorithm must use the same environment, observation cutoff, execution semantics, compute budget, baselines,
and evaluation periods. IQL's presence in the library does not by itself establish that an existing market log is
causally complete or that counterfactual execution is identifiable.

## Scaling gate

Do not promote a TOP2000/H100 run merely because it completes. Scale only after:

1. the dataset provenance gate passes;
2. the Hold-30 planted-signal, age/cohort, state-carry, and accounting tests pass;
3. H2 qualifies under the frozen PIT-300 pre-lockbox protocol;
4. performance telemetry shows the job is not dominated by host RAM, padded attention, or repeated encoding;
5. a complete run artifact can be audited and replayed; and
6. any direct-versus-PPO comparison is registered as a later orthogonal
   ablation using the already-qualified Hold-30 mechanism.

The current compact EOD context storage, vectorized raw-window joins, batched evaluation encoding, and streaming
cache changes reduce known bottlenecks. They do not substitute for the correctness and artifact gates above.

## Compatibility policy

- Keep the Phase-1 command available during migration and label its optimizer accurately.
- Do not make `rl_quant.rl` depend on markets, datasets, or Phase-1 models.
- Prefer adapters over parallel copies of trajectory, reward, or execution logic.
- A result-moving semantic change requires a new manifest/config identity and a fresh comparison; cached artifacts
  must never silently cross that boundary.
- Remove legacy paths only after a reportable replacement exists and parity differences are understood.
