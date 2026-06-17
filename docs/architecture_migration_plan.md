# QuantTrade architecture migration — protocol-first, env-first, artifact-reportable (PHASED)

Executes the 2026-06-17 architectural review **safely and incrementally**. A package-wide reorg of a
300+-test codebase must be phased and gate-verified, never big-bang — and the review agrees (Phase 1 = *no
behavior change*; the monolith splits + env-owns-reward are explicitly Phases 3–6). This doc is the tracked
plan; each phase keeps the full gate green and old import paths working via shims.

## Status (2026-06-17)

All layer packages now exist and are populated; the flat top-level holds only back-compat shims,
cross-cutting infra (`cli`/`config`/`core`/`paths`), and the not-yet-split workflow monoliths.

```
protocol/      constraints (the contract), partition, validators            DONE
data_sources/  polygon_second_aggs, polygon_stock_covariates, quote_utils    DONE
features/      stock_second_context, stock_covariates, news_llm, action_risk DONE
datasets/      intraday, strategy                                            DONE
models/        minute_to_hour, hourly, second_context, intraday, strategy   DONE (Phase 4)
evaluation/    statistical, confidence, research_protocol, decision_framework, second_context  DONE
reportability/ decision_log                                                  DONE
workflows/     presets, cli, config                                          DONE
envs/          strategy, intraday, hourly, minute_to_hour                    DONE (Phase 4)
training/      strategy, intraday, hourly, minute_to_hour                    DONE (Phase 4)
```

**Phase 4 complete (verified CLEAN, commit 610e190):** all 5 transformer/dqn monoliths decomposed into
datasets/models/envs/training/evaluation, each move byte-for-byte verbatim (adversarially verified: 125
top-level blocks, 0 altered), old import paths preserved via shims, gate green (303), ruff clean, no import
cycles. The flat top level now holds only back-compat shims + the foundation (core/execution/paths).
Remaining (not yet done): Phase 3 (env-owns-reward behind a flag -- result-moving, needs sign-off + A/B) and
the optional further-split of eval out of training/minute_to_hour into evaluation/.

Every relocation was behavior-preserving: git-mv + a re-export shim at the old path, verified gate-green
(303 tests) and import-cycle-free, with identity checks where it mattered (the contract; the extracted model).

## The one rule the architecture must enforce

> **Only the environment/execution layer may change portfolio state and compute trading reward.**

Datasets prepare causal observations + labels; models score valid candidate actions; trainers optimize via
the env/dataset interface; **env/execution** applies the action, computes fill/cost/reward, updates state;
evaluation runs the policy through the env and collects logs; reportability judges artifact completeness +
claim validity; the statistical layer judges credibility. If this rule holds, many model families plug in;
if it doesn't, every workflow grows private execution semantics and reportability becomes untrustworthy.

## Target layers

```
protocol/      decision-tensor schema, mask semantics, validators, manifests  (the contract)
data_sources/  vendor ingestion only (OHLCV second-aggs, NBBO, news)          [exists]
features/      point-in-time feature construction                             [exists]
datasets/      gold dataset builders/loaders (compact tensors, manifests)
execution/     pure transition + cost semantics (scalar-$ vs weight-bps)      [execution.py today]
envs/          AUTHORITATIVE owner of state/transition/reward/next-mask/logs
models/         neural networks only (consume typed tensors -> scores/Q)
training/      DQN/replay loops (consume env+dataset+model; no raw data/reward)
evaluation/    sequential eval, baselines, statistical credibility            [partial: evaluation/]
reportability/ mechanical + execution + claim gates over PERSISTED artifacts  [partial: reportability/]
workflows/     versioned WorkflowSpecs + presets + CLI
```

## Migration discipline (every phase)

- Default flags reproduce current numbers **byte-for-byte** (regression test); result-moving behavior is
  opt-in, manifest-recorded, A/B-gated, one-flag rollback (the established promotion gate, §2.1 of the wiring
  design). Old import paths stay working via shims until the transition completes.
- **Bounded / non-reward-changing** phases I can land directly; **result-moving** phases (env owns reward,
  dynamic model inputs) need explicit sign-off + a latest-period A/B (test block untouched).

## Phases

### Phase 1 — pure/shared layers + shims — DONE (this commit), no behavior change
- `reportability.py` → `reportability/` package (`reportability/decision_log.py` + `__init__` re-export). Old
  `from rl_quant.reportability import ...` unchanged.
- `statistical_credibility.py` → `evaluation/statistical.py` (+ `evaluation/__init__` re-export); old path kept
  as a shim. Establishes the **evaluation** and **reportability** layers.
- Verified: old + new import paths both resolve; full gate green (302 tests), ruff clean.

### Phase 2 — protocol layer + enforced validation (bounded, additive)
- `protocol/` package: a real `DecisionTensorPayload.load()/validate()` wrapping the decision-tensor schema +
  the model-input / label / forbidden-key split (today documented in `decision_tensor_protocol.md` and
  `features/stock_second_context.py`), so trainers consume a *validated* payload, not a raw dict. Re-export
  the mask/transition/constraint contract from `trading_constraints` (16 importers -> keep the module, add the
  namespace). Add architecture tests: model never consumes forbidden/label keys; `selected_action` valid under
  the decision mask; `context_available_until <= decision_ts`. Label-only / additive.

### Phase 3 — env layer owns reward (RESULT-MOVING -> flag + A/B)
- `envs/` with a `TradingEnv` protocol (`reset/observe/valid_action_mask/step/decision_log_row`). Wrap the
  existing minute_to_hour env first (byte-identical), then add an opt-in `use_execution_env_reward` flag that
  makes the env (via the `execution.py` engine) the reward authority — the step the leg-level engine was built
  for. Gated, default-off byte-identical, A/B before any default flip. (Builds directly on PR-D D0–D3b.)

### Phase 4 — split the monoliths (bounded, behavior-preserving)
- Decompose `minute_to_hour_transformer.py` -> `datasets/hour_from_subhour.py`, `models/minute_to_hour.py`,
  `envs/hour_allocator_env.py`, `training/minute_to_hour_dqn.py`, `evaluation/hour_allocator.py`; same for
  `second_context_transformer.py`. Move the transition-feature builders into a shared module
  (static [A,A,F] vs dynamic per-env), keeping `execution.py`'s scalar-$ vs weight-bps split. Each move keeps a
  shim at the old import path; gate green per move. No logic change.

### Phase 5 — artifact-driven reportability (bounded)
- Reportability evaluates **persisted** `decision_log.jsonl` + `run_manifest.json`, not in-memory `metrics`,
  so a run can't be made (un)reportable by toggling whether logs are returned. Emit
  `mechanical_reportability.json` / `execution_reportability.json` / `statistical_credibility.json` +
  `reportability_issues.jsonl`.

### Phase 6 — statistical-credibility expansion + stress grid (bounded)
- Extend `evaluation/` (PSR/DSR already landed) with block-bootstrap CIs, walk-forward, PBO/CSCV, reality-check
  + the latency/cost/spread/impact stress grid and the baseline panel as a promotion gate.

## What I will NOT do without sign-off
Flip any flag from default, make the env the reward authority (Phase 3), or merge a phase whose gate-off path
isn't proven byte-identical. Phases 3 (result-moving) and the larger Phase-4 splits proceed one reviewable,
gate-green step at a time on your go-ahead — not as a single big-bang reorg.
