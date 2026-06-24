# QuantTrade

QuantTrade (`rl_quant`) is a compact, point‑in‑time‑correct research framework for learning trading policies
from raw market data. Its current centerpiece is a **decoupled two‑stage learning framework**: a self‑supervised
**context encoder** that learns market state from the raw 1‑second bars, and a **decision policy** that allocates
capital on top of the frozen context. Around it sit the kept data/evaluation/reportability infrastructure.

> **Status (2026‑06).** The earlier per‑second / subhour / second‑to‑hour transformer stack and its
> precomputed‑feature datasets were removed. The framework below is raw‑input only: **no precomputed/engineered
> features are stored or consumed** — the models organize and learn from raw seconds + covariates + news at
> train time. (The expensive LLM news scoring is the one precomputed artifact that is kept, as a raw input.)

---

## Design principles

1. **Point‑in‑time causality.** A bar, covariate, news item, constraint, or cost may be a model input only if it
   is available **at or before** the decision timestamp. Forward returns are used only as the training *label*.
2. **No precomputed features.** The dataset stores raw 1‑second OHLCV bars, raw point‑in‑time covariate records,
   and raw per‑article LLM news scores. All normalization / aggregation / representation is done **inside the
   models at train time** — nothing hand‑engineered is persisted.
3. **Context ⟂ policy split.** "What is the market doing" (context) is learned separately from "what to do about
   it" (policy). The context encoder is trained self‑supervised, then **frozen**; the policy trains on the
   frozen context and can never backprop into it.
4. **Reportability.** A result is trustworthy only with realistic masks/latency/costs, matched train/val/test
   schemas, cost‑paid baselines, and a statistical battery (see *Safety & reportability*).

---

## The two‑stage learning framework

### Stage 1 — context learning (`rl_quant.models.context_encoder`)

A **two‑tier causal transformer** that reads the **raw 1‑second bars directly** (one token per second, raw
OHLCV; in‑model `BatchNorm` + linear embedding only — no pooling, no scale‑free features):

- **Tier 1 (local):** causal attention *within* fixed `block_seconds` blocks of raw seconds; each block's
  most‑recent‑valid token is a **learned** summary (the model compresses raw seconds — nothing hand‑pooled).
- **Tier 2 (global):** causal attention *over* the block summaries across the session; the most‑recent‑valid
  block is the per‑stock session context. This makes the **full ~6.5 h RTH session** tractable at
  O(S·block) + O(n_blocks²) instead of flat O(S²); context rolls from the 09:30 open with no look‑ahead.
- The encoder **also learns from each stock's as‑of covariates** (fundamentals / market‑cap / news‑volume),
  fused with the temporal context; the cross‑sectional mean over **all involved stocks** → a **pure market
  context** (no policy state ever enters).
- **Self‑supervised pretext:** predict the next interval's equal‑weight market return + realized vol. Then the
  encoder is **frozen** and used to encode every decision into cached embeddings.

### Stage 2 — policy learning (`rl_quant.models.decision_policy`)

A **permutation‑equivariant set‑transformer** over the action set `{CASH, stock₁ … stock_N}`, trained on the
frozen context (it holds no encoder reference → its reward gradient cannot reach the context):

- Each action becomes a token `[ broadcast market ctx | per‑stock ctx | in‑model‑normalized covariates |
  in‑model‑aggregated raw news | previous weight ]` + a learned CASH token. Cross‑sectional attention values
  each action relative to the others; unavailable actions are masked.
- Raw per‑article **news scores are aggregated in‑model** (a learned masked sum), and **covariates are
  normalized in‑model** — so the policy also uses **no precomputed features**.
- A softmax (with **temperature**) over `{CASH, stocks}` yields **allocation weights**. CASH = abstain.
- **Objective — differentiable portfolio:** roll each window forward carrying the previous weights and maximize
  realized net return − turnover cost, with a downside‑variance penalty and an optional entropy bonus. Shared
  weights ⇒ the same head scales from tens to ~2000 actions.

### Training (`rl_quant.training`)

- `context_pretrain.py` — Stage‑1 SSL trainer: **streams** raw‑second micro‑batches from CPU‑resident windows to
  the GPU + **gradient accumulation** (effective batch ≫ peak VRAM); then `freeze_encoder` + `encode_windows`
  (cached embeddings). `decision_policy.py` — Stage‑2 differentiable‑portfolio trainer + `evaluate_policy` +
  cost‑paid baselines. `_optim.py` — step‑driven warmup+cosine/constant LR (resume‑exact, no scheduler state).
- **Per‑stage training strategy** (LR / warmup / weight decay / grad clip; bf16 AMP + TF32; policy
  cost / risk / entropy / temperature) is parameterized by `designs.py` — a series of `Phase1Design` presets
  spanning both transformers' architecture *and* the training setup, keyed on the `max_seconds` lookback
  (including a `full_session` design over the whole RTH session). `tiny` is the CPU smoke / CI design.

The actual **experiment driver lives outside this package**, in `../training/` (relative paths only): a thin
`train_phase1.py` (multi‑seed, resumable, time split, verdict) and `sweep_phase1.py` (multi‑GPU sweep). See
*Running*. The split is CI‑enforced (`tests/test_phase1_framework.py`).

---

## Package layout (`src/rl_quant`)

| subpackage / module | role |
|---|---|
| `models/` | `context_encoder` (Stage 1, two‑tier causal), `decision_policy` (Stage 2, set‑transformer) |
| `datasets/` | `raw_window` — organizes a raw time window (bars/covariates/news) into train‑time tensors; no features stored |
| `training/` | two‑stage trainers, LR/optim helpers, the `Phase1Design` series |
| `evaluation/` | the statistical battery — `statistical.py` (block‑bootstrap CI, White Reality Check, Hansen SPA, deflated Sharpe), ranking, run registry, research protocol |
| `features/` | `news_llm` (qwen3 news scoring), `stock_covariates`, `action_risk` — the **kept** feature producers |
| `data_sources/` | polygon second aggregates + stock‑covariate readers, quote utils |
| `protocol/`, `reportability/` | the reportability contract: action‑return basis, constraints, validators, decision log, baselines |
| `execution/` | fills / legs / cost model |
| core modules | `core`, `config`, `paths`, `decision_framework`, `trading_constraints`, `research_protocol`, `statistical_credibility`, `confidence`, `partition_protocol` |

Layering is enforced as a runtime DAG by `tests/test_import_boundaries.py`.

---

## Data

Forward‑only layers (training code never parses raw vendor files when a validated builder exists):

- **Bronze** — raw/minimally‑normalized vendor files (Polygon 1‑second aggregate Parquet
  `…/SYMBOL/YYYY/MM/YYYY-MM-DD.parquet` with a `manifest.csv` source‑of‑truth; Yahoo OHLCV; raw quotes).
- **Silver** — cleaned point‑in‑time tables (stock covariates; news‑LLM article scores).
- **Raw decision windows** — what the framework consumes: `partitions/<S_to_E>/{bars.parquet,
  covariates.parquet, news.jsonl}` + `universe.json` (e.g. the TOP50 dataset at `../TOP50`). The organizer
  builds, **at train time**, the raw per‑second bars, as‑of covariates, raw news, and forward‑return labels —
  nothing precomputed. The decision grid is 5 hourly RTH decisions/day, DST‑correct via `zoneinfo`.

**LLM news caveat:** the qwen3 news scores carry an anachronistic availability sentinel — fine as a model
*input*, but **not point‑in‑time clean for a reportable backtest**. Bars + covariates + forward‑return labels
**are** point‑in‑time clean. Large generated assets stay under `data/`/`derived/`, never in `src/`.

---

## Running

The package is the library; the runnable Phase‑1 experiment driver is in `../training/` (DATA read only from a
raw dataset root such as `../TOP50`; all paths relative to the script). Set `PYTHON` to the `quanttrade`
interpreter.

```bash
# correctness smoke (CPU ok, ~30 s): design=tiny
PYTHON=…/quanttrade/bin/python python ../training/train_phase1.py --smoke

# one design on one GPU (resumable, multi-seed)
PYTHON=… python ../training/train_phase1.py --design wide --device cuda:0 --seeds 3

# multi-GPU sweep over the design series (one (design,seed) job per GPU; resumable)
PYTHON=… python ../training/sweep_phase1.py --designs sweep --devices 0,2,3 --cpu-workers-per-gpu 8 --seeds 3
```

The repo‑root `run.sh` is a convenience launcher for the sweep. The **verdict** reports pooled OOS
mean/decision + a 95% block‑bootstrap CI, cost‑paid CASH / buy‑&‑hold baselines, and White Reality Check /
Hansen SPA / deflated‑Sharpe vs the CASH floor. A trustworthy positive = beats cash **and** CI excludes 0
**and** low WRC/SPA **and** deflated‑Sharpe credible.

---

## Safety & reportability

Research code. A result is **not** reportable unless: every input is available ≤ the decision timestamp;
rewards realize inside the evaluated split; train/val/test schemas match; fit windows end before val/test use;
trading constraints apply in both training and eval; costs are leg/action‑aware; terminal positions are
liquidated‑with‑cost or reported open; registered baselines + cost/frequency stress are included; source
completeness/limitations are declared; invalid action returns are stored as `NaN`, never silently zero.

No secrets in the repo (`.env.example` documents variable names only). Do not commit API/broker/S3 credentials,
large raw datasets, checkpoints, or run directories.

---

## Environment

Use the `quanttrade` conda env (Python 3.11):

```bash
cd QuantTrade
conda run -n quanttrade python -m pip install -e ".[dev,data]"   # + ".[llm]" for offline news scoring
```

Core deps are intentionally small: `torch>=2.6,<3`, `numpy>=2.2,<3`; optional `pandas`/`pyarrow` (data) and
`transformers`/`accelerate` (LLM); `pytest`/`ruff`/`mypy` (dev). The kept news‑LLM stack scores articles offline
(primary `Qwen/Qwen3‑1.7B`) and writes frozen tables; **training never calls an LLM** — it consumes only frozen
scores. Verify CUDA/AMP: `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"`.

---

## Testing & quality

```bash
cd QuantTrade
PYTHONPATH=src conda run -n quanttrade python -m pytest tests/ -q
conda run -n quanttrade ruff check src tests
```

`tests/test_phase1_framework.py` locks the design as executable assertions: the context/policy split (no policy
gradient reaches the frozen encoder), multi‑block cross‑block causality, simplex allocation + constraint
masking, the LR/temperature/AMP/entropy strategy knobs, and design‑series validity. `test_import_boundaries.py`
locks the layering DAG; `test_scripts_are_wrappers.py` keeps `scripts/` thin.

---

## Caveats / correctness contract

- **No precomputed features** anywhere (context or policy) — enforced by design + tests.
- **News is input‑only**, not reportable‑clean (anachronistic LLM availability). Bars/covariates/labels are clean.
- **VRAM is GPU‑measured but verify on your hardware.** Raw‑second sequences are heavy: the SSL batch is
  `ssl_batch × n_stocks` (the encoder runs every stock per decision), so `ssl_batch` is small and decoupled from
  the statistical batch via gradient accumulation. The `full_session` design needs a big‑RAM box (~134 GB of raw
  bars across windows) or lazy per‑window loading.
- Decision times are **DST‑correct** (`zoneinfo`); changing the build logic bumps the driver's cache version.

---

## Glossary

- **Decision grid** — the 5 hourly RTH decision timestamps per trading day (DST‑aware).
- **Context** — the frozen market‑state representation (Stage 1) the policy consumes.
- **Differentiable portfolio** — the Stage‑2 objective: maximize realized net return − turnover cost (with
  downside/entropy terms) over softmax allocation weights.
- **Reportable** — a result satisfying every point in *Safety & reportability*.
- **CASH** — action 0; the abstention/risk‑free floor (return identically 0).
