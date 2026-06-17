# Wiring the execution engine into reward — design (DESIGN-ONLY, no number-moving code)

Companion to the 2026-06-17 target memo and [architecture_migration_plan.md](architecture_migration_plan.md).
This is the reviewable contract for the **next** step (PR-3, shadow mode); no code in this doc moves a number.

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

## 1. Current state (post-reorg)

The protocol-first layers exist; `envs/` is the authoritative owner of state/transition/reward, and
`execution.py` is the foundational engine — still **unwired** (no env/trainer computes reward through it, so
nothing it does has moved a reported number).

Already landed on `main` (all default-preserving, gate-green):
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

## 3. PR-3 — shadow-mode execution reward in the minute_to_hour env (THE NEXT STEP)

**Goal:** wire `execution.py` into the `envs/minute_to_hour` env for the first time, but only to compute the
execution-engine reward/cost **alongside** the legacy reward and **log** them — training still uses the legacy
reward. Default byte-identical; this is the safe first contact between the engine and a trainer.

**Pricing (decision made): weight-bps cost model — sourced from action metadata, NOT a dataset field.**
`HourFromMinuteDataSplit` has **no** `action_target_weights`; the env is single-slot/leg-based. So the
per-action weight VECTOR is derived from action metadata (`features/action_risk.action_weight_tensor`/
`action_leverage_tensor` on `build_action_metadata(data.action_names)`) — note these are **static per-symbol
caps** (a function of leverage), not time-varying targets, and PR-3 must state that limitation. The shadow
cost is `execution.py`'s `WeightExecutionCostConfig` (`fee_bps + linear_impact_bps_per_weight`) applied to the
transition **prior-held-weight vector → executed-weight vector** (indexed by `previous_actions`/`actions`,
not a dataset tensor). The env must therefore track a `execution_shadow_holdings` weight vector (set on reset,
updated to the executed weight each step, saved/restored on resume when the flag is on — mirror the dynamic
env-state checkpoint fix). This is a **cost-model A/B**: leg-aware bps cost vs the env's current
`legs * one_way_cost_bps + is_switch * extra_switch_penalty_bps`. Explicitly **not** real-executable (no
quote-side fills, no latency P&L) — strict reportability still requires NBBO data that isn't on disk.

**Contract (env `step`, behind `execution_env_reward_shadow: bool = False`, default off):**
- Legacy path unchanged: `rewards` (the tensor used for training) is computed exactly as today; the trained
  model and every reported number are identical with the flag off OR on (shadow is a pure side-channel).
- When on, additionally compute (no RNG draws, no reordering, no mutation of the legacy reward):
  - `execution_cost_bps_shadow = 1e4 * outcome.realized_execution_cost` — the engine returns
    `realized_execution_cost` in **return units** (`traded · total_cost_bps / 1e4`), so convert to bps with ×1e4;
  - `execution_env_reward_shadow = raw_returns * reward_scale - (execution_cost_bps_shadow + cash_idle_penalty_bps) * reward_scale / 1e4`
    — it **must carry the same `cash_idle_penalty_bps` term as the legacy reward** so `reward_delta` isolates the
    trade-cost-model change, not the cash-idle policy (the legacy reward subtracts `cost_bps + cash_idle_penalty_bps`);
  - `reward_delta = execution_env_reward_shadow - rewards`; `cost_delta = execution_cost_bps_shadow - legacy_trade_cost_bps`;
  - `impact_applied_shadow` — the **transition-actual** flag from the outcome (now fixed: a positive impact was
    charged on a filled leg), not the config-level capability;
  - `execution_transition_issues` (the engine's fail-closed flags, if any).
- Emit these into the step dict (replay filters unknown keys → harmless), surface per-run aggregates
  (mean/quantiles of `reward_delta`, `cost_delta`) and stamp `execution_env_reward_shadow: true` in the
  run manifest.

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

**Prerequisite (known latent drift):** `evaluate_minute_to_hour_policy` currently RECOMPUTES the reward/cost
ledger inline (`legs`/`cost_bps`/`net_return`/equity) outside the env, and omits the env's
`cash_idle_penalty_bps` (it isn't even passed to the evaluator). With the default `cash_idle_penalty_bps == 0`
the two agree today, but they diverge for any nonzero-penalty run, and PR-4 would make this worse (eval must
score the *execution* reward the env trains on). Before PR-4, route eval scoring through the env (or a single
shared pure transition/reward primitive called by both `env.step` and the evaluator) so "only env/execution
computes reward" actually holds. This is the eval-through-env refactor — bounded but multi-call-site (8+
`evaluate_minute_to_hour_policy` call sites), tracked here, not rushed.

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
