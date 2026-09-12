# Primary direction: raw seconds → Transformer → direct-allocation PPO

This is the user-approved replacement **research architecture**, introduced
2026-09-11. It is not a relabeling of native adaptive V5. The existing V5,
forecast-control, engineered daily/minute, normalized ContextEncoder,
five-minute AlphaHierarchical, and Hold-30 artifacts remain legacy evidence.
Their receipts and results must not be rewritten or accepted by this schema.

## Non-singleton learning contract

Raw-second learning requires at least two valid samples per PPO update. A
single sample becomes zero after rollout-wide advantage centering, so its
parameter changes can reflect only entropy/value objectives, not the clipped
reward-driven policy objective. The raw-second trainer rejects such an update
before changing parameters, optimizer state or RNG. Single-transition collection
remains available for non-learning diagnostics; it cannot authorize an update.
The shared generic PPO implementation and its other experiment contracts are
unchanged. Sequence length one and minibatches of one sequence remain valid
when normalization uses multiple learning samples from the complete rollout.

Both raw-second runners reject `rollout_steps < 2` and training catalogs with
fewer than two scored transitions. They freeze the complete rollout schedule
before collection. A would-be singleton tail is absorbed into the preceding
block **before** collecting it under one behavior policy: nominal size two over
five transitions produces `(2, 3)`, and nominal 63 over 64 produces `(64,)`.
The nominal bound may therefore grow by exactly one; resource planning must
allow this. No transition is dropped, duplicated or merged after an update.
The actual sizes and learning-sample counts are persisted and checked at report
replay. Learning changes use `raw-second-multitransition-ppo-v1`, chronological
report v3 and engineering summary v2. Old resume checkpoints without this
learning contract are rejected; old reports retain their original identities.
Portable frozen-policy inference retains its existing interface and provenance.

`test_raw_second_learning_v1.py` isolates the PPO policy objective with entropy
and value-loss coefficients zero. Actual sampled actions, ledger rewards and
GAE must produce finite nonzero gradients and parameter updates in the actor,
input projection and every attention tier over seeds 17/29/43. This proves
reward-objective gradient flow, **not** improvement in held-out profitability.
Even a multi-sample batch may have equal advantages; sample count alone does
not guarantee a nonzero policy gradient. A controlled economic-learning test,
representative one-H100 profiling and actual-data evaluation remain separate.

## Implemented boundary

`rl-quant.massive-raw-second-window-v1` admits Massive REST unadjusted second
aggregates only. `SecondQuery` constructs `/v2/aggs/ticker/{ticker}/range/1/second/{from}/{to}`
with `adjusted=false&sort=asc&limit=50000`. The offline capture handoff checks
the exact pagination chain, response status, ticker, adjustment flag, result
counts, duplicate fields/seconds, source hashes and chronological bounds.
It commits complete provider response bytes before its capture manifest.
Provider VWAP/count/other response fields stay archived; the model receives
only OHLCV, in that order, cast through JSON numeric parsing to FP32.

The capture handoff itself is **not an acquisition client, entitlement proof,
historical-vintage authority, or security-identity authority**. The separate
`raw_second_capture_v1` module now connects the existing header-authenticated,
no-redirect HTTPS transport to this handoff. Its explicit plan is bounded to
24 nonoverlapping QT200 queries, at most one hour per query, eight pages each
and 128 MB of response bytes. The plan must be published before its caller
opens a credential. This module neither reads a credential file nor introduces
any remote credential transport. No real request is authorized by its presence.

Provider pagination may advance the path's start timestamp within the original
query. Ticker, one-second interval, end time, origin, ordering and adjustment
semantics remain fixed; blank, duplicated or credential-bearing URL parameters
are rejected. These rules follow the [provider's aggregate pagination example](https://massive.com/docs/rest/stocks/aggregates/custom-bars)
and [header authentication contract](https://massive.com/docs/rest/quickstart).

`capture_seconds` preserves raw HTTP pages and receipts. `materialize_second_sources`
replays that complete capture before publishing the byte-identical raw-second
source objects; invalid OHLCV cannot pass this later boundary. Receipt times
are rounded up from nanoseconds to milliseconds, never made available early.
`verify_second_sources` reproduces that mapping without network access or writes.
Capture completeness and valid raw objects do not confer identity, corporate-
action, historical-availability or training qualification. Existing
daily/minute aggregates and trades cannot be expanded or relabeled as these
REST seconds. No real second-bar corpus has been qualified by this change.

`SecondPartitionSet` resolves bounded immutable captures for a single instrument
without manufacturing a combined provider response. Observations query their
own raw context; the catalog's independently indexed source sets cover execution
and ledger marks. Additional intervening captures can be bound explicitly through
`execution_sources`. Every read verifies original bytes. Overlap must agree in
both OHLCV and observed-versus-empty status; conflicting vintages fail closed.
Identical duplicate observations retain all source hashes and their earliest
actual receipt. An uncovered interval remains unknown, never zero liquidity.
Regular-session execution requires complete coverage of the eligible session
intervals between decisions, including marks after order expiry; calendar-closed
overnights need no capture. This does not relax coverage inside an open session.

The new catalog identity is `rl-quant.raw-second-catalog-v2`. The corrected ledger
and chronological report have distinct v2 identities, and old price-only resume
states are rejected. Previously published reports remain historical artifacts;
they must not be silently replayed or relabeled under changed execution semantics.

The allowed pre-encoder operations are parsing, validation, clock alignment,
masking and numeric casting. Configuration rejects returns, features,
scalers, resampling, adjusted values, forward fill, covariates and news.
The model's first learned market operation is `Linear(5, d_model)` in FP32.
LayerNorm follows that projection, not the raw input. There is no calibration
routine or raw normalizer state. Nonfinite casts/projections fail rather than
restoring a scaler. This is an explicit numerical risk to test and profile.

Observed, known-empty, unknown and padding/unavailable intervals remain
distinct. Unknown required coverage fails. A completed empty response is
known-empty, not fabricated OHLCV; a trainable missing token is used. Masked
payloads cannot affect the model. Every observed second in the bounded context
reaches the input projection and local causal attention before compression.

Local attention → learned attention readout → causal block attention → learned
temporal readout → cross-stock attention feeds a masked-Dirichlet actor and
critic. CASH is action index zero, then the frozen issue-ID order. No signal
forecasts, holding ages, duration bonuses or duration constraints enter the
policy. Cash and actual share quantities form a separate account branch after
the market encoder. Execution configuration declares observation/execution
sessions and either next-decision or session-close order expiry. Regular-only
operation requires a bound calendar, including early closes. Order expiry does
not liquidate held positions; there is no hidden persistent order queue.

Raw windows are reloaded from immutable, hash-bound references on **every**
PPO forward. The existing `RecurrentPPO` and on-policy trajectory/GAE code is
reused. Buffers contain raw-reference indices, catalog digests, account state,
behavior probabilities and actual transitions—not learned representations.
All encoder, actor and critic parameters join one optimizer. Checkpoints bind
the catalog, model/configuration, optimizer, shuffle/RNG and carried ledger.
Exact training resume retains those strict bindings. Separately exported
`raw-second-frozen-policy-v1` artifacts contain tensor weights, portable model/
OHLCV/instrument contracts and original training provenance, **no optimizer**.
`load_frozen_raw_second_policy` may bind later catalogs and other cost scenarios
with identical input semantics and issue order. It freezes gradients/training
mode and checks parameter identity before actions. The old `inference_only`
resume shortcut is rejected; it is not a way to update an evaluation policy.
No KV/recurrent/frozen embedding caches are accepted. Initial correctness is
bounded recomputation, not a scalable cross-day memory implementation.

## Ledger and chronology

One `RawSecondPortfolioEnv` is used for training and deterministic evaluation.
It reuses the existing decimal Book, split/dividend/receivable and terminal
accounting primitives; raw prices are never adjusted by that ledger. Requested
weights are capped by asset/gross limits, converted into orders at
decision-known marks, and executed against strictly **later** second bars.
Whole newly purchased shares, a declared volume-participation cap, proportional
cash-constrained buys and 10/20/40-bp costs are modeled. These are second-OPEN
execution proxies, **not observed bid/ask fills or empirical capacity**.
Terminal liquidation is a costed mark adjustment, not observed exit liquidity.

The book carries across sessions. Corporate events apply in the ledger only.
Ledger marks retain the raw source second, availability time and current share
basis. A pre-split close read after a split is translated into the current book
basis; it cannot overwrite a rebased cache with an old-basis price. Accounting
and decision-known marks have separate state, persisted for exact resume. A
newer accounting price does not make a delayed observation available to sizing.
Reward is exactly `log(equity_after / equity_before)`; costs are already in
equity and are not subtracted again. Cash/share, fill, fee, notional and terminal
diagnostics remain outside model market inputs. Finalized prices and event
applicability still need an explicit real-data research contract.

Availability has no silent default. Choose either:

- `historical-finalized-assumed-delay` with an explicit nonnegative delay;
- `developer-delayed-assumption` with at least 900,000 milliseconds;
- `captured-receipt-time`, using actual capture timestamps.

Historical delay is an assumption, not reconstructed data vintages. The
first learned layer cannot see a bar ending or becoming available after its
decision. Accounting marks are distinct from model observations. Publication
latency, security identity, listings, missing coverage and corporate-action
applicability must be resolved before claiming real-data readiness.

## Entry points and acceptance

`run_raw_second_engineering_episode` connects committed second references,
the actual trainable model, direct actions, carried cash/share execution,
chronological GAE/PPO updates and checkpoint/diagnostic publication. It
requires one CUDA GPU and a deterministic FP32 runtime. Its output explicitly
has `real_data_training_ready=false`, `native_v5_authorized=false` and no
positive profitability authorization. It is not a four-fold report or a
study-level selection/access gate.

`run_raw_second_experiment` adds one chronological train/validation/test
development experiment. Each PPO update yields a resume checkpoint and portable
frozen policy. Only validation net return selects a candidate, with earliest
update as deterministic tie-breaker. `selection.json` commits before test
captures open. Test evaluation includes CASH, **one-shot-entry** equal-weight buy-and-hold
subject to the same asset/gross caps, and the original same-seed untrained
policy. Unfilled initial benchmark orders expire; there are no subsequent catch-up
buys. These are constraint-matched, not necessarily exposure-matched comparisons.
The report includes first-entry share completion valued at decision marks,
average/maximum risky exposure and cash allocation. Exposure is explicitly an
unweighted decision-interval-end sample before terminal liquidation, not a
continuous-time average; these diagnostics never enter market inputs.
Cost scenarios rerun the same frozen policy on the same raw inputs;
actions may differ with cost-dependent account state. This is not a frozen-
target cost estimand and does not assert monotone returns.

The persisted report includes exact fill/fee/holdings/receivable records,
terminal mark liquidation separately from fills, net and relative returns,
fill-derived turnover and last-scored-daily-equity risk statistics. A negative
test return still yields a complete report. `verify_raw_second_experiment`
reconstructs validation selection and test economics without updates or writes;
it does not rerun training or confer native-V5 qualification. The caller pins
the report hash. Exact numerical replay requires the frozen evaluation stack.

Both runners now require `SecondEconomicInputs`, binding a normalized event
census, interval/issue coverage, session calendar, and original evidence hashes.
Missing coverage is not an event-free declaration. Splits/dividends flow into
every ledger; receivables are valued at entitlement but cash is paid only when
due. The document's integrity checks do not independently establish real
provider mapping or historical applicability. Synthetic censuses exercise the
workflow; actual-data identity/event review and acquisition remain separate
gates. Unsupported event populations are rejected.

The focused suite is `tests/test_massive_raw_second_rl_v1.py`. It uses complete
synthetic provider-shaped second responses and real learning/accounting,
including raw-value hooks, failed pagination, forbidden inputs, missingness,
unavailable-value perturbations, all attention-tier gradients, same-weight
recomputation, ledger replay, cash/share/fee reconciliation and fresh-process
checkpoint resume/CPU frozen inference. GPU tests fail without an assigned
GPU; they never silently use local CPU or qualify via skips. CPU inference
checks run only inside the allocated LSF job, not on the controller.

CPU GitHub jobs explicitly deselect `lsf_gpu`; their success is not GPU
acceptance. Release acceptance additionally requires a linked immutable LSF
result with exact source inventory, hardware/runtime, all required nodes and
zero failures/errors/skips. Publishing a required external GitHub check is an
operator integration, not established by the CPU workflow alone.

On an assigned LSF GPU run these in **separate processes**, in order:

```bash
python -B -m pytest -q -m lsf_gpu \
  tests/test_raw_second_capture_v1.py \
  tests/test_massive_raw_second_rl_v1.py \
  tests/test_raw_second_experiment_v1.py \
  tests/test_raw_second_partitions_v1.py \
  tests/test_raw_second_learning_v1.py
python -B -m pytest -q -m lsf_gpu tests/test_raw_second_fresh_process_v1.py
```

The fresh-process orchestrator never initializes CUDA. Its training, resume
and CPU-inference children exit sequentially; this supports LSF
`exclusive_process` allocations without competing with a live parent CUDA
context. Collect both test inventories, not merely the first command's result.

Regression scheduler: **LSF GPUs**. Eventual real training: **one Kubernetes
H100**, approved pinned ml2, existing narrowly scoped claim, no new PVC.
The older two-rank 36-trial proposal and controller-only GPU tests do not
qualify this model. Exact source/package/runtime/test receipts are required.

## Still required before real training at scale

1. Pass the exact new LSF regression package; inspect gradient/stability and
   fresh-process evidence, not just successful submission.
2. Pass the bounded REST transport/handoff tests, bind the approved controller
   acquisition owner and current entitlement receipt, then qualify an actual
   second-bar pilot, preserving pagination and raw bytes. Do not reacquire the
   trade archive or expose protected outcomes.
3. Bind the real issue identities, corporate-action coverage, second/session
   calendar and declared availability/execution assumptions. The numerical
   interface deliberately cannot manufacture those authorities.
4. Profile the selected context/universe on one H100. Defaults are 128 hidden,
   four heads, two local/two temporal layers and 300 raw seconds per local
   block; context is at most 23,400 clock seconds initially. Asset/block
   chunking and activation checkpointing are learned-model memory controls,
   not resampling. Cross-session gaps may need separate bounded contexts.
5. Register chronological selection/evaluation for this **new** direct-policy
   experiment before outer access. Do not reuse V5 controller checkpoints,
   normalizers or qualification flags. Extend observation/account state for
   pending-order or receivable-dependent policies before enabling such inputs.

An engineering episode's net return is diagnostic, not out-of-sample alpha.
No mandatory or rewarded holding duration is introduced.
