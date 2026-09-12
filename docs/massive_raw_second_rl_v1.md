# Primary direction: raw seconds → Transformer → direct-allocation PPO

This is the user-approved replacement **research architecture**, introduced
2026-09-11. It is not a relabeling of native adaptive V5. The existing V5,
forecast-control, engineered daily/minute, normalized ContextEncoder,
five-minute AlphaHierarchical, and Hold-30 artifacts remain legacy evidence.
Their receipts and results must not be rewritten or accepted by this schema.

## Implemented boundary

`rl-quant.massive-raw-second-window-v1` admits Massive REST unadjusted second
aggregates only. `SecondQuery` constructs `/v2/aggs/ticker/{ticker}/range/1/second/{from}/{to}`
with `adjusted=false&sort=asc&limit=50000`. The offline capture handoff checks
the exact pagination chain, response status, ticker, adjustment flag, result
counts, duplicate fields/seconds, source hashes and chronological bounds.
It commits complete provider response bytes before its capture manifest.
Provider VWAP/count/other response fields stay archived; the model receives
only OHLCV, in that order, cast through JSON numeric parsing to FP32.

The capture handoff is **not an acquisition client, entitlement proof,
historical-vintage authority, or security-identity authority**. Existing
daily/minute aggregates and trades cannot be expanded or relabeled as these
REST seconds. No real second-bar corpus has been qualified by this change.

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
the market encoder. Outstanding orders expire at the next decision boundary;
there is no hidden persistent order queue in this initial execution proxy.

Raw windows are reloaded from immutable, hash-bound references on **every**
PPO forward. The existing `RecurrentPPO` and on-policy trajectory/GAE code is
reused. Buffers contain raw-reference indices, catalog digests, account state,
behavior probabilities and actual transitions—not learned representations.
All encoder, actor and critic parameters join one optimizer. Checkpoints bind
the catalog, model/configuration, optimizer, shuffle/RNG and carried ledger.
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

The focused suite is `tests/test_massive_raw_second_rl_v1.py`. It uses complete
synthetic provider-shaped second responses and real learning/accounting,
including raw-value hooks, failed pagination, forbidden inputs, missingness,
unavailable-value perturbations, all attention-tier gradients, same-weight
recomputation, ledger replay, cash/share/fee reconciliation and fresh-process
checkpoint resume/CPU frozen inference. GPU tests fail without an assigned
GPU; they never silently use local CPU or qualify via skips. CPU inference
checks run only inside the allocated LSF job, not on the controller.

Regression scheduler: **LSF GPUs**. Eventual real training: **one Kubernetes
H100**, approved pinned ml2, existing narrowly scoped claim, no new PVC.
The older two-rank 36-trial proposal and controller-only GPU tests do not
qualify this model. Exact source/package/runtime/test receipts are required.

## Still required before real training at scale

1. Pass the exact new LSF regression package; inspect gradient/stability and
   fresh-process evidence, not just successful submission.
2. Implement the reviewed controller-side REST acquisition integration and
   qualify a bounded actual second-bar pilot, preserving pagination and raw
   bytes. Do not reacquire the trade archive or expose protected outcomes.
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
