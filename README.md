# rl_quant

Compact reinforcement-learning trading framework for the research in this
workspace. The package code lives in `src/rl_quant`; executable workflows live
in `scripts`. Scripts are designed to be run from the repository root. Use
repo-relative `data/` and `derived/` paths unless explicit paths are passed.

This is research backtesting code under simplified data and execution
assumptions. It is not live-trading-ready; broker credentials and live order
placement must stay outside this repository and require explicit opt-in guards.

QuantTrade is organized around a point-in-time research protocol. A valid
experiment must declare the data manifest, feature fit windows, decision
schedule, action universe, trading constraints, cost/execution model,
validation protocol, baselines, and hyperparameter search space. A result is
not reportable unless features are available at or before decision time, reward
realization stays inside the split, schemas match exactly, trading constraints
are applied during training and evaluation, costs are leg-aware, cost/frequency
stress evidence is present, and registered baselines are included.

Use the `ml1` conda environment:

```bash
cd /path/to/QuantTrade
python -m pip install -e ".[dev]"
conda run -n ml1 python scripts/check_torch_cuda.py --device auto --matrix-size 1024 --repeats 1 --amp
```

## Package Layout

- `core.py`: shared torch runtime, CUDA/AMP helpers, replay buffer, metrics, and temporal Q-network blocks.
- `intraday_data.py`, `intraday_dqn.py`: QQQ NBBO feature loading and intraday DQN trading.
- `strategy_data.py`, `strategy_dqn.py`: daily strategy-allocation dataset loading and DQN allocator.
- `hourly_transformer.py`: causal-transformer DQN allocator for hourly or minute market context.
- `minute_to_hour_transformer.py`: hierarchical minute-to-hour causal transformer for hourly allocation decisions using causal minute context.
- `research_protocol.py`: dataset/model manifests, fit-window validation, benchmark registry, stress evidence, and experiment registry helpers.
- `bar_transformer.py`: interval-neutral aliases for the transformer allocator.
- `quote_utils.py`: raw quote parsing, NBBO construction, session utilities, and bucket formatting.

## Policies

Intraday DQN:

- State: rolling windows of QQQ NBBO bucket features.
- Actions: discrete `short`, `flat`, `long` exposure.
- Reward: realized mid-price PnL over a configurable step horizon after optional latency and costs.
- Frequency: raw quote data converted to fixed buckets; default is 1-second buckets.

Daily strategy allocator:

- State: daily market/strategy features, optionally augmented with ecological attention context.
- Actions: complete strategy curves such as `BH_QQQ`, cross-sectional momentum, dual momentum, and filtered variants.
- Reward: next daily close-to-close return of the selected strategy minus optional strategy-switch cost.
- Frequency: daily close-to-close rows, focused on 2026.

Bar causal-transformer allocator:

- State: rolling bar windows containing top-stock cross-section features, time features, and ETF context.
- Actions: `CASH` plus selected liquid ETF actions.
- Reward: next aligned simple bar return for the selected ETF action, net of leg-aware order costs.
- Model: masked transformer encoder with previous-action embedding; attention is causal across the lookback window.
- Frequency: Yahoo Finance `1h` or `1m` exchange-session bars.
- Constraints: optional action masks, minimum hold, cooldown, daily/episode switch caps, daily/episode order-leg caps, leg-aware Q-value hysteresis, and ETF-to-ETF two-leg transaction costs.

Minute-to-hour causal-context allocator:

- Decision frequency: hourly boundaries only.
- State: `H` historical hour tokens, each encoded from up to `M` causal minute bars ending no later than the decision timestamp.
- Actions: `CASH` plus selected liquid ETF actions.
- Reward: selected ETF simple return from the decision close to the next hourly decision close.
- Model: minute encoder learns intrahour path context; hour encoder learns multi-hour regime context.
- Constraints: action masks, minimum hold, daily/episode switch caps, daily/episode order-leg caps, cooldown, leg-aware Q-value hysteresis, and ETF-to-ETF two-leg transaction costs.

Correctness contract:

- Trading rewards are simple returns because environments compound equity with `equity *= 1 + return`.
- Log returns may appear as state features only, for example `bar_log_return`.
- Bar datasets must store both `timestamps` and `next_timestamps`; split builders reject rewards realized after a split end.
- Bar action rewards are computed from the current decision close to the next decision close, so filtered intermediate bars do not distort returns.
- Minute-to-hour datasets must satisfy `max(context_minute_ts) <= decision_ts < next_ts`.
- Evaluation walks `valid_start_indices` exactly and resets previous-action state when valid windows are not contiguous.
- Strategy and bar loaders validate feature/action names and order across train, validation, and test splits.
- Numeric strategy inputs are parsed strictly; missing action returns should be fixed upstream, not coerced to zero.

## Data Formats

Daily allocator dataset:

- `state_features.csv`: `Date` plus numeric state features available through that date.
- `action_returns.csv`: `Date` plus one numeric return column per strategy action.
- `action_manifest.csv`: action index, strategy name, benchmark flag, backtest fields, and variation-risk fields.

Bar transformer dataset:

- `hourly_transformer_dataset.pt` or `minute_transformer_dataset.pt`: trusted local torch payload with `timestamps`, `next_timestamps`, `feature_names`, `action_names`, `features`, `action_returns`, `bar_interval`, and `periods_per_year`.
- `state_features.csv`: human-readable copy of bar state features.
- `action_returns.csv`: human-readable copy of bar action returns.

Minute-to-hour dataset:

- `hour_from_minute_dataset.pt`: trusted local torch payload with `decision_timestamps`, `next_timestamps`, `minute_features`, `minute_mask`, `hour_features`, `action_returns`, feature names, and action names.
- Default grid: hourly decisions/rewards (`60` minutes) built from `1m` source bars.
- `minute_features`: tensor shaped `[decisions, hours_lookback, minutes_per_hour, minute_feature_count]`.
- `minute_mask`: boolean tensor where `True` marks causal valid minute bars.
- `action_returns`: close-to-close ETF returns from each hourly decision timestamp to the next hourly decision timestamp.

Research protocol artifacts:

- `dataset_manifest.json`: source/vendor, symbols, schema hashes, timestamp hashes, universe timing, known limitations, and feature fit-window metadata.
- Model manifests: algorithm, encoder, training dataset, validation protocol, search-space hash, selected metric, baselines, and cost/frequency stress results.
- `FitWindow`: every learned feature or normalizer should prove `fit_end < feature_asof`.

Intraday NBBO dataset:

- Raw quote files are expected under `QQQ_2025` unless `--raw-dir` is passed.
- Extracted bucket files live under `derived/nbbo_features` and contain OHLC mid, spread, depth, imbalance, microprice, and quote-quality fields.

## Main Commands

Check CUDA:

```bash
conda run -n ml1 python scripts/check_torch_cuda.py --device auto --matrix-size 2048 --repeats 2 --amp
```

Build daily market ecology and merge it into RL state:

```bash
conda run -n ml1 python scripts/learn_market_ecological_attention.py
conda run -n ml1 python scripts/merge_market_ecology_with_rl_states.py
```

Build the daily strategy RL dataset:

```bash
conda run -n ml1 python scripts/build_2026_rl_strategy_dataset.py
```

Train the daily strategy allocator:

```bash
conda run -n ml1 python scripts/train_strategy_allocator.py --device auto --amp
```

Build the hourly transformer dataset:

```bash
conda run -n ml1 python scripts/build_hourly_transformer_dataset.py \
  --output-dir derived/rl_hourly/top_volume_2026
```

Train the hourly causal-transformer DQN:

```bash
conda run -n ml1 python scripts/train_hourly_causal_transformer_rl.py \
  --dataset derived/rl_hourly/top_volume_2026/hourly_transformer_dataset.pt \
  --device auto --amp --target-vram-gb 9.5
```

The direct bar trainer is risk-aware by default: leveraged actions are
risk-scaled, same-group exposure is prospectively capped at 50% after the
minimum observation window, and leveraged exposure is capped at 30 bars per day
and 15 consecutive bars unless overridden. Its summary includes canonical
`cost_stress`, `RandomSameTurnover`, `RandomSameActionDistribution`,
risk-scaled baseline labels, action metadata hashes, and action-risk config
hashes.

Build the latest minute-level transformer dataset:

```bash
conda run -n ml1 python scripts/build_minute_transformer_dataset.py
```

The minute wrapper drops overnight reward gaps and requires same-session
lookback windows, so valid `1m` transformer states do not silently span an
exchange close. It uses the top 16 ETF actions by default for trainability; pass
`--action-count 500` to build a full top-500 ETF action dataset from the same
downloaded files.

In the local nested workspace used for the current experiments, defaults resolve
to `/home/yding1995/quant/data`. In a standalone root checkout, those defaults
resolve under the checkout's `data/` directory.

Train the minute-level causal-transformer DQN:

```bash
conda run -n ml1 python scripts/train_minute_causal_transformer_rl.py \
  --device auto --amp --target-vram-gb 9.5
```

The minute-level wrapper applies conservative direct-minute turnover defaults:
15-bar minimum hold, 5-bar cooldown, 4 switches per day, 8 switches per
episode, 8 order legs per day, 2 bps one-way leg cost, 1 bps extra switch
penalty, and 5 bps Q-value switch margin. Its `summary.json` includes
0/1/2/5/10 bps test cost stress under the same action-mask constraints.

Preferred RL path for minute data: build hourly decisions from minute-level context.
This preserves minute microstructure/path information while making decisions on
hourly boundaries, which is a more stable target than direct one-minute trading.
The default decision/reward grid is fixed at one hour while all context bars
remain `1m` source data.

```bash
conda run -n ml1 python scripts/build_hourly_from_minute_context_dataset.py
```

Train the hierarchical minute-to-hour causal transformer. The minute encoder
learns intrahour dynamics from causal 1-minute bars; the hour encoder learns
multi-hour regime dynamics for the RL action decision.

```bash
conda run -n ml1 python scripts/train_hourly_from_minute_context_rl.py \
  --device auto --amp --target-vram-gb 9.5 \
  --max-switches-per-episode 3 --max-order-legs-per-episode 6
```

For rolling 2026 training, fine-tune each new period from the last period's
minute-to-hour model instead of relearning the context transformer from scratch:

```bash
conda run -n ml1 python scripts/train_hourly_from_minute_context_rl.py \
  --device auto --amp --target-vram-gb 9.5 \
  --warm-start-model data/rl_hour_from_minute_runs/previous_period/model.pt
```

The warm-start checkpoint is loaded only after its minute feature names, hour
feature names, action names, and constraint feature schema match the current
dataset and code.

Validate research manifests:

```bash
conda run -n ml1 python scripts/validate_research_protocol.py \
  --dataset-manifest data/rl_hour_from_minute/top_volume_1m_recent/dataset_manifest.json
```

Train the QQQ intraday DQN:

```bash
conda run -n ml1 python scripts/train_dqn_agent.py \
  --train-dates 2025-01-02,2025-01-03 \
  --val-dates 2025-01-06 \
  --test-dates 2025-01-07 \
  --auto-extract --device auto --amp
```

## Data Bootstrap

Universe and OHLCV helpers are included so the RL data can be rebuilt:

```bash
conda run -n ml1 python scripts/fetch_top_us_market_cap_universe.py
conda run -n ml1 python scripts/fetch_top_volume_universes.py
conda run -n ml1 python scripts/download_daily_ohlcv.py --help
conda run -n ml1 python scripts/download_hourly_ohlcv.py --help
conda run -n ml1 python scripts/download_intraday_ohlcv.py --help
```

Yahoo Finance currently limits true `1m` data to a short recent window. Use the
intraday downloader with explicit dates and manifests when rebuilding minute
bars.

## Review Notes

- Scripts use `PACKAGE_ROOT` for imports and script-to-script calls.
- Scripts use `PROJECT_ROOT` for data paths; a root checkout resolves it to the
  checkout directory.
- New RL work should import from `rl_quant`.
- The market-ecology feature workflow is research-only unless it is run in a
  prior-only rolling or expanding walk-forward mode. Do not fit ecological
  context parameters on the full test period for backtest claims.
