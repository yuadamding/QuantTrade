# Wiring the execution engine into reward — shipped PR-3 contract + PR-4 plan (no number-moving code)

Companion to the 2026-06-17 target memo and [architecture_migration_plan.md](architecture_migration_plan.md).
This records the **shipped** PR-3 shadow-mode contract (§3) and the remaining PR-4 promotion plan (§4); the
document itself is non-executable and moves no numbers.

## 0. Target & the honesty line

Make **both** the learned Q-function and the reward depend on the **current held position → candidate action
transition**, priced through decision-time-safe execution (latency, spread, cost, impact, terminal handling,
real fill logs).

Hard honesty line (unchanged): *real executable trading requires quote-side fills, latency P&L, impact, AND
real fill-price logs; anything short is labelled causal-research / backtest only.* A config flag is not
sufficient evidence — reportability is judged from config AND the decision logs. The minute_to_hour data is
return-based: `HourFromMinuteDataSplit` carries `action_returns` + `action_names` (no per-row target-weight
tensor and no NBBO on disk; `action_target_weights` exists only on the unrelated `SecondContextDataSplit`).
So the work below is explicitly a **cost-model** experiment, NOT a real-executable claim.

## 1. Current state

The protocol-first layers exist; `envs/` is the authoritative owner of state/transition/reward. **PR-3 has
shipped:** `execution.py`'s weight-bps cost model is wired into `envs/minute_to_hour` in SHADOW mode (default
off) — it computes the execution reward/cost alongside the legacy reward and logs them; training still uses
the legacy reward, so nothing it does has moved a reported number. The engine is not yet the reward authority
of any trainer (that is PR-4).

Already landed on `main` (all default-preserving, gate-green):
- PR-3 shadow-mode execution reward (this doc's §3) — `execution_env_reward_shadow`, default off.
- The engine: fail-closed valuation/fills, `valuation_complete`/`execution_complete`/`impact_applied`,
  `SwitchFillPolicy.{INDEPENDENT_LEGS (default), ATOMIC_SWITCH}`, and the return-based **bps** cost model
  `WeightExecutionCostConfig(fee_bps, impact_kind, linear_impact_bps_per_weight)`.
- PR-D **D0–D3b**: dynamic position-state (`entry_index`/`unrealized_pnl`/`mae`/`mfe`), schema, network
  support, env/replay/TD wiring — behind `use_dynamic_transition_features` (default off, byte-identical),
  with the recent correctness fixes (fail-closed `dynamic_state`, dynamic env-state checkpoint, dynamic-aware
  eval, clean transition RNG, shared `advance_position_excursion`).
- The reportability layer (tiered base vs strict real-executable + semantic checks) and
  `evaluation/statistical.py` (PSR / expected-max-Sharpe / DSR).
- Governance infra: the flag registry (`protocol/flags.py`), the architecture import-boundary test, and the
  protocol anti-leakage validator wired into the real second-context builder.

## 2. Governing principle — opt-in, default-preserving (the law)

Every result-moving step is gated behind one flag whose default reproduces the old run **byte-for-byte**
(regression test asserts it). Turning the flag **on** is the only thing that moves a number, and that is a
separate, reviewed, A/B-gated act. Each flag carries promotion metadata in `protocol/flags.py`.

### 2.1 Promotion gate (every reward-changing flip passes it)

```
safe unwired capability → shadow-mode eval → latest-period A/B → strict reportability audit
                        → opt-in experiment → default flip ONLY on evidence  (rollback = one config change)
```

Merge checklist for a reward-changing flip: (1) default path byte-identical (regression test); (2) exactly
one explicit flag; (3) manifest records the flag value + `result_moving_flag: true`; (4) a latest-period A/B,
**test block untouched** (recency-focus protocol); (5) A/B reports Δ return/turnover/exposure/cash-share/
cost-drag/drawdown (+ PSR/DSR); (6) explicit reportability label + reasons; (7) one-flag rollback.

## 3. PR-3 — shadow-mode execution reward in the minute_to_hour env (SHIPPED)

`execution.py`'s weight-bps cost model is wired into `envs/minute_to_hour` behind
`MinuteToHourEnvConfig.execution_env_reward_shadow` (default off). When on, the env computes the
execution-engine reward/cost ALONGSIDE the legacy reward and logs them; training still uses the legacy
`rewards`, so the run is byte-identical to shadow-off (regression-tested at the env-step and train-trace level).

**Pricing: static single-slot weight-bps, sourced from action metadata (NOT a dataset field).**
`HourFromMinuteDataSplit` has no `action_target_weights`; the env is single-slot/leg-based. The per-action
weight is the STATIC `action_weight_tensor(build_action_metadata(action_names))` (cash zeroed), so
`previous_action` alone determines the prior held weight — **no `execution_shadow_holdings` state is needed**
(it would be required only if weights became time-varying per row). The shadow cost prices the transition's
two legs (sell prior weight + buy executed weight) via the vectorized `weight_transition_cost_bps`, pinned to
the dataclass engine by an equivalence test. Cost-model A/B (turnover-weighted vs leg-count); explicitly
**not** real-executable (no NBBO / quote-side fills / latency P&L).

**Open semantics question (resolve before PR-4):** the shadow weight is `action_metadata.max_weight`
(= 1/leverage for a leveraged ETF). This is correct only if `action_returns` are METADATA-WEIGHTED portfolio
returns. If they are FULL-CAPITAL single-slot returns (100% capital in the asset, leverage intrinsic), the
turnover weight should be 1.0 for every non-cash action and the current model UNDERCHARGES leveraged turnover.
The minute_to_hour builder takes `action_returns` from the upstream payload (applies no leverage), so this must
be confirmed against the gold builder before training on the execution reward. The artifact records
`execution_shadow_weight_source` so the assumption is auditable; for a non-leveraged universe (all max_weight
== 1) there is no discrepancy.

**Contract (env `step`, default off, byte-identical):**
- Legacy `rewards` (used for training) is computed first and never mutated.
- When on, the env additionally computes (no RNG, no reordering) via the SHARED `transition_trade_cost_bps`
  breakdown (`leg_cost_bps`, `switch_penalty_bps`, `cash_idle_bps`):
  - `execution_cost_bps_shadow` = `weight_transition_cost_bps` of the transition's legs (bps = 1e4 × the
    engine's return-unit `realized_execution_cost`);
  - `execution_env_reward_shadow = raw * reward_scale - (execution_cost_bps_shadow + switch_penalty_bps + cash_idle_bps) * reward_scale / 1e4`
    — it swaps ONLY the leg/execution cost and KEEPS the behavioural switch-penalty regularizer + cash-idle,
    so `reward_delta` isolates the cost-MODEL change (PR-4 would not silently drop the regularizer);
  - `reward_delta_shadow = execution_env_reward_shadow - rewards`;
    `cost_delta_shadow = execution_cost_bps_shadow - leg_cost_bps` (vs the LEG cost only — switch penalty +
    cash-idle are held constant across both rewards);
  - emitted into the step dict (replay stores only its declared fields → never reaches training).
- The train loop aggregates the per-step deltas and stamps the artifact: `execution_env_reward_shadow`,
  `execution_shadow_cost_model="static_single_slot_weight_bps"`, `execution_shadow_real_executable=False`,
  and the mean reward/cost deltas (reward units + scale-normalised bps).

**Why it's bounded.** Training reads only `rewards` (legacy); the shadow quantities are computed from existing
env state + the engine and only logged. Verification: (i) a regression test that `loss_trace`/`reward_trace`/
`best_val_*` are byte-identical with the shadow flag off vs on (shadow doesn't perturb training); (ii) the
default-off path is the current code path unchanged; (iii) a unit test that `execution_cost_bps_shadow`
matches a hand-computed weight-bps cost on a known transition.

**Manifest / governance:** register `execution_env_reward_shadow` and the eventual `use_execution_env_reward`
in `protocol/flags.py`; the shadow flag is label-changing (adds metrics), the training flip is result-moving.

## 4. PR-4 — the result-moving flip (`use_execution_env_reward`, needs sign-off + A/B)

A separate flag that makes the env **train on** `execution_env_reward_shadow` instead of the legacy reward.
This MOVES results, so it merges only through the §2.1 gate: default off + byte-identical; a latest-period A/B
(test block untouched) reporting the Δ metrics + PSR/DSR; explicit reportability label; one-flag rollback.
Never bundled with PR-3.

**Prerequisite (partly done; remaining drift):** the cash-idle drift is FIXED — `evaluate_minute_to_hour_policy`
now takes a `cash_idle_penalty_bps` argument and computes its ledger through the SHARED
`transition_trade_cost_bps` (leg/switch/cash-idle accounting matches the env). What remains is that the eval
still recomputes the REST of the rollout ledger (`net_return`/equity/action-selection/dynamic recurrence)
outside the env. Before PR-4, route eval scoring fully through the env (or one shared reward primitive used by
both `env.step` and the evaluator) so "only env/execution computes reward" holds end-to-end, not just for cost.
Also resolve the weight-semantics question (§3) and confirm action-selection/hysteresis + transition features
use the same cost basis as the execution reward (today they are leg-count, a behavioural prior).

## 5. Deferred / blocked (honesty line)

- **Real fill logs / quote-side execution (the old PR-A/B on second_context).** Blocked on a quote/NBBO
  dataset that is not on disk (grounded). Until then second_context stays `close_based_research_backtest`,
  base-reportable, not real-executable. PR-3's weight-bps shadow does NOT change this.
- **Stress grid (latency/cost/spread/impact sweep)** and **statistical-credibility expansion** (PBO/CSCV,
  walk-forward, reality-check) — additive reporting harnesses, valuable once results start moving.

## 6. What I will not do without explicit sign-off

Flip any flag from its default; make the env train on the execution reward (PR-4); or merge any step whose
default-off path is not proven byte-identical by a regression test. PR-3 (shadow) is itself sign-off-gated
because it is the first time the engine touches a trainer — even though it is byte-identical by default.
