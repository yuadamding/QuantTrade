# rl_quant

Compact reinforcement-learning trading framework for the research in this
workspace. The package code lives in `src/rl_quant`; executable workflows live
in `scripts`. Scripts are designed to be run from this directory or from the
parent project, while reading and writing datasets under `/home/yding1995/quant/derived`.

Use the `ml1` conda environment:

```bash
cd /home/yding1995/quant
conda run -n ml1 python rl_quant/scripts/check_torch_cuda.py --device auto --matrix-size 1024 --repeats 1 --amp
```

## Package Layout

- `core.py`: shared torch runtime, CUDA/AMP helpers, replay buffer, metrics, and temporal Q-network blocks.
- `intraday_data.py`, `intraday_dqn.py`: QQQ NBBO feature loading and intraday DQN trading.
- `strategy_data.py`, `strategy_dqn.py`: daily strategy-allocation dataset loading and DQN allocator.
- `hourly_transformer.py`: causal-transformer DQN allocator for hourly or minute market context.
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
- Reward: next aligned bar return for the selected ETF action, net of switch cost.
- Model: masked transformer encoder with previous-action embedding; attention is causal across the lookback window.
- Frequency: Yahoo Finance `1h` or `1m` exchange-session bars.

## Data Formats

Daily allocator dataset:

- `state_features.csv`: `Date` plus numeric state features available through that date.
- `action_returns.csv`: `Date` plus one numeric return column per strategy action.
- `action_manifest.csv`: action index, strategy name, benchmark flag, backtest fields, and variation-risk fields.

Bar transformer dataset:

- `hourly_transformer_dataset.pt` or `minute_transformer_dataset.pt`: torch payload with `timestamps`, `feature_names`, `action_names`, `features`, `action_returns`, `bar_interval`, and `periods_per_year`.
- `state_features.csv`: human-readable copy of bar state features.
- `action_returns.csv`: human-readable copy of bar action returns.

Intraday NBBO dataset:

- Raw quote files are expected under `/home/yding1995/quant/QQQ_2025`.
- Extracted bucket files live under `derived/nbbo_features` and contain OHLC mid, spread, depth, imbalance, microprice, and quote-quality fields.

## Main Commands

Check CUDA:

```bash
conda run -n ml1 python rl_quant/scripts/check_torch_cuda.py --device auto --matrix-size 2048 --repeats 2 --amp
```

Build daily market ecology and merge it into RL state:

```bash
conda run -n ml1 python rl_quant/scripts/learn_market_ecological_attention.py
conda run -n ml1 python rl_quant/scripts/merge_market_ecology_with_rl_states.py
```

Build the daily strategy RL dataset:

```bash
conda run -n ml1 python rl_quant/scripts/build_2026_rl_strategy_dataset.py
```

Train the daily strategy allocator:

```bash
conda run -n ml1 python rl_quant/scripts/train_strategy_allocator.py --device auto --amp
```

Build the hourly transformer dataset:

```bash
conda run -n ml1 python rl_quant/scripts/build_hourly_transformer_dataset.py \
  --output-dir /home/yding1995/quant/derived/rl_hourly/top_volume_2026
```

Train the hourly causal-transformer DQN:

```bash
conda run -n ml1 python rl_quant/scripts/train_hourly_causal_transformer_rl.py \
  --dataset /home/yding1995/quant/derived/rl_hourly/top_volume_2026/hourly_transformer_dataset.pt \
  --device auto --amp --target-vram-gb 9.5
```

Build the latest minute-level transformer dataset:

```bash
conda run -n ml1 python rl_quant/scripts/build_minute_transformer_dataset.py
```

The minute wrapper drops overnight reward gaps and requires same-session
lookback windows, so valid `1m` transformer states do not silently span an
exchange close. It uses the top 16 ETF actions by default for trainability; pass
`--action-count 500` to build a full top-500 ETF action dataset from the same
downloaded files.

Train the minute-level causal-transformer DQN:

```bash
conda run -n ml1 python rl_quant/scripts/train_minute_causal_transformer_rl.py \
  --device auto --amp --target-vram-gb 9.5
```

Train the QQQ intraday DQN:

```bash
conda run -n ml1 python rl_quant/scripts/train_dqn_agent.py \
  --train-dates 2025-01-02,2025-01-03 \
  --val-dates 2025-01-06 \
  --test-dates 2025-01-07 \
  --auto-extract --device auto --amp
```

## Data Bootstrap

Universe and OHLCV helpers are included so the RL data can be rebuilt:

```bash
conda run -n ml1 python rl_quant/scripts/fetch_top_us_market_cap_universe.py
conda run -n ml1 python rl_quant/scripts/fetch_top_volume_universes.py
conda run -n ml1 python rl_quant/scripts/download_daily_ohlcv.py --help
conda run -n ml1 python rl_quant/scripts/download_hourly_ohlcv.py --help
conda run -n ml1 python rl_quant/scripts/download_intraday_ohlcv.py --help
```

Yahoo Finance currently limits true `1m` data to a short recent window. Use the
intraday downloader with explicit dates and manifests when rebuilding minute
bars.

## Review Notes

- Scripts use `PACKAGE_ROOT` for imports and script-to-script calls.
- Scripts use `PROJECT_ROOT` for data, so outputs remain in the existing
  `/home/yding1995/quant/derived` tree.
- Existing legacy `src/quant_system` files are left untouched for compatibility;
  new RL work should import from `rl_quant`.
