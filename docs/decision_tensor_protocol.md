# Decision Tensor Protocol v1

This protocol defines the compact model-input format for QuantTrade decision
models. It is designed for second-level, minute-level, and future higher
resolution data while keeping the model interface stable.

The core idea is simple:

```text
raw market data -> causal compressed context tokens -> decision tensors
```

Raw 1-second bars, quotes, fundamentals, and news remain in source-specific
bronze/silver layers. Training consumes a compact gold dataset with one row per
decision timestamp.

## Goals

- Compact enough for GPU training.
- Causal by construction.
- Stable across model families.
- Easy to extend with new context groups.
- Explicit about masks, fills, costs, and reportability.
- Easy to audit with manifests and feature registries.

## Artifact Layout

A protocol-compliant dataset is a directory:

```text
dataset_root/
  arrays.zarr/
  timestamps.parquet
  dataset.pt
  dataset_manifest.json
  feature_manifest.json
  normalization_manifest.json
  data_quality_report.json
  action_metadata.json
  split_manifest.json
  schema.json
```

`dataset.pt` is the fast local training cache. It is a trusted local torch
payload and must not be treated as an untrusted interchange or archival format.
For large datasets, the canonical long-term store should be Zarr/Arrow/Parquet
plus JSON manifests; `dataset.pt` can be regenerated from those artifacts.

## Core Tensor Contract

Required tensors:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `decision_timestamps_ms` | `[N]` | `int64` | Decision timestamp in UTC milliseconds. |
| `next_timestamps_ms` | `[N]` | `int64` | Reward horizon end timestamp in UTC milliseconds. |
| `market_context` | `[N, L, Fm]` | `float16` or `float32` | Causal compressed market context tokens. |
| `market_context_mask` | `[N, L]` | `bool` | True where a context token is valid. |
| `market_context_available_timestamps_ms` | `[N, L]` | `int64` | When each context token became available. |
| `action_features` | `[N, A, Fa]` | `float16` or `float32` | Per-action features known at decision time. |
| `action_features_available_timestamps_ms` | `[N, A]` | `int64` | When each action feature row became available; `-1` means unavailable. |
| `action_valid_mask` | `[N, A]` | `bool` | True where the action can be selected. |
| `action_mask_reason_code` | `[N, A]` | `uint16` or `int32` | Numeric reason code for invalid actions. |
| `action_returns` | `[N, A]` | `float32` | Realized simple return for valid actions; invalid values must be `NaN`. |
| `action_cost_bps` | `[N, A]` | `float16` or `float32` | Estimated action execution cost in basis points. |
| `action_cost_available_timestamps_ms` | `[N, A]` | `int64` | When each cost estimate became available; `-1` means unavailable. |
| `action_target_weights` | `[N, A]` | `float16` or `float32` | Target portfolio weight for each action. |
| `action_quality_score` | `[N, A]` | `float16` or `float32` | Per-action data quality in `[0, 1]`. |
| `entry_execution_timestamps_ms` | `[N, A]` | `int64` | Simulated entry fill timestamp. |
| `exit_execution_timestamps_ms` | `[N, A]` | `int64` | Simulated exit/reward fill timestamp. |
| `portfolio_state` | `[N, Fp]` | `float16` or `float32` | State of the current portfolio before decision. |
| `portfolio_state_available_timestamps_ms` | `[N]` | `int64` | When portfolio state became available. |
| `constraint_state` | `[N, Fc]` | `float16` or `float32` | Risk, data-quality, and session constraints. |
| `constraint_state_available_timestamps_ms` | `[N]` | `int64` | When constraint state became available. |
| `decision_quality_score` | `[N]` | `float16` or `float32` | Row-level data quality in `[0, 1]`. |
| `force_cash_mask` | `[N]` | `bool` | True when data quality or risk requires cash-only action. |
| `valid_start_indices` | `[K]` | `int64` | Rows that may start valid sequential evaluation windows. |
| `segment_ids` | `[N]` | `int64` | Segment/session id used to reset sequential state. |
| `session_ids` | `[N]` | `string/list` | Human-readable session id for each row. |

Required metadata fields inside `dataset.pt`:

```text
schema_version
protocol_version
decision_tensor_protocol_version
dataset_schema_version
decision_timestamps
next_timestamps
action_names
execution_model
feature_names
feature_names_by_tensor
dataset_manifest
data_quality_report
action_metadata
split_manifest
model_input_keys
label_keys
forbidden_model_input_keys
config
payload_hash
```

Recommended values:

```text
protocol_version: decision_tensor_v1
decision_tensor_protocol_version: 1.0.0
schema_version: stock_second_context_decision_v3
dataset_schema_version: second_context_gold_v1
feature_schema_hash: stable hash of feature_names_by_tensor
action_schema_hash: stable hash of action_names
action_metadata_hash: stable hash of action_metadata
constraint_schema_hash: stable hash of constraint_state names
portfolio_state_schema_hash: stable hash of portfolio_state names
execution_schema_hash: stable hash of execution_model_detail
```

## Default Compact Shape

For current top-500 stock second-bar context:

```text
N  = decision rows
L  = 12 context blocks
Fm = 48 or fewer market features
A  = tradable actions, including CASH
Fa = 24 or fewer action features
Fp = 6 or fewer portfolio features
Fc = 8 or fewer constraint features
```

Recommended default:

```text
decision_interval: 15m or 30m
source_bar_interval: 1s
context_seconds: 3600
block_seconds: 300
lookback_blocks: 12
market_context: [N, 12, Fm]
```

The transformer should not consume raw `[N, 500, 3600, OHLCV]` tensors by
default. That form is too sparse, too expensive, and too tied to one universe.
The bronze layer may store raw bars; the gold model input should store compressed
tokens.

## Required Causality Rules

Every row must satisfy:

```text
market_context_available_timestamps_ms[n, l] <= decision_timestamps_ms[n]
action_features_available_timestamps_ms[n, a] <= decision_timestamps_ms[n] or == -1
action_cost_available_timestamps_ms[n, a] <= decision_timestamps_ms[n] or == -1
portfolio_state_available_timestamps_ms[n] <= decision_timestamps_ms[n]
constraint_state_available_timestamps_ms[n] <= decision_timestamps_ms[n]
next_timestamps_ms[n] > decision_timestamps_ms[n]
entry_execution_timestamps_ms[n, a] >= decision_timestamps_ms[n] + execution_latency_ms
exit_execution_timestamps_ms[n, a] >= next_timestamps_ms[n] + execution_latency_ms
```

For second aggregates:

```text
bar_latency_ms >= 1000
execution_latency_ms >= 1000
```

`CASH` is always action index `0` and must satisfy:

```text
action_names[0] == "CASH"
action_valid_mask[:, 0] == True
action_returns[:, 0] == 0
action_cost_bps[:, 0] == 0
action_target_weights[:, 0] == 0
```

Invalid actions must satisfy:

```text
action_valid_mask[n, a] == False => isNaN(action_returns[n, a])
```

## Feature Groups

The core protocol has four model-facing feature groups:

```text
market_context
action_features
portfolio_state
constraint_state
```

Future context groups are added through an optional `context_groups` dictionary:

```text
context_groups = {
  "sector_context.v1": {
    "tensor": [N, L, S, Fs],
    "mask": [N, L, S],
    "available_timestamps_ms": [N, L, S],
    "feature_names": [...]
  },
  "quote_context.v1": {
    "tensor": [N, Lq, A, Fq],
    "mask": [N, Lq, A],
    "available_timestamps_ms": [N, Lq, A],
    "feature_names": [...]
  }
}
```

Rules for extensions:

- New groups must be optional.
- New groups must include a mask and availability timestamps.
- New groups must declare feature names and a schema version.
- Existing required keys must not change shape or meaning within a protocol
  version.
- A model may ignore unknown groups and still train on the core tensors.

## Model Inputs And Labels

Every dataset manifest must separate inputs from labels:

```json
{
  "model_input_keys": [
    "market_context",
    "market_context_mask",
    "action_features",
    "action_valid_mask",
    "action_cost_bps",
    "action_target_weights",
    "portfolio_state",
    "constraint_state",
    "decision_quality_score",
    "force_cash_mask"
  ],
  "label_keys": [
    "action_returns",
    "next_timestamps",
    "entry_execution_timestamps_ms",
    "exit_execution_timestamps_ms"
  ],
  "forbidden_model_input_keys": [
    "action_returns",
    "next_timestamps",
    "exit_execution_timestamps_ms"
  ]
}
```

Loaders must reject datasets where `model_input_keys` overlap
`forbidden_model_input_keys`.

## Recommended Feature Content

### Market Context

Each context token summarizes a block of raw data, for example one 5-minute
block built from 1-second bars:

```text
active_symbol_count
active_fraction
equal_weight_return
dollar_volume_weighted_return
median_return
return_std
up_fraction
down_fraction
top_decile_return
bottom_decile_return
top_minus_bottom_return
abs_return_dollar_volume_weighted
dollar_volume_concentration
transaction_concentration
log_total_dollar_volume
log_total_volume
log_total_transactions
mean_range_bps
range_bps_std
mean_active_seconds
missing_symbol_fraction
large_move_fraction
quality_score
session flags
time since open / time to close
```

### Action Features

Action features should describe tradability and recent action state:

```text
action_index_scaled
is_cash
is_etf
is_stock
is_inverse
is_leveraged
leverage
valid_price_flag
feature_staleness_seconds
log_last_dollar_volume
estimated_cost_bps
recent_return_1m
recent_return_5m
recent_return_15m
recent_return_60m
recent_volatility
spread_bps
group_id_scaled
```

### Portfolio State

Minimum:

```text
cash_weight
gross_exposure
previous_action_index_scaled
```

Recommended extensions:

```text
bars_held
switches_today
order_legs_today
drawdown_from_session_high
previous_action_group_scaled
```

### Constraint State

Minimum:

```text
data_quality_score
valid_action_fraction
minutes_to_close_scaled
```

Recommended extensions:

```text
max_switches_remaining_scaled
max_order_legs_remaining_scaled
inverse_budget_remaining_scaled
leveraged_budget_remaining_scaled
force_cash_flag
session_warmup_flag
```

## Manifests

`dataset_manifest.json` must include:

```text
protocol_version
schema_version
dataset_id
created_at_utc
source_bar_interval
decision_interval
context_seconds
block_seconds
lookback_blocks
bar_latency_ms
execution_latency_ms
execution_model
feature_schema_hash
action_schema_hash
action_metadata_hash
constraint_schema_hash
portfolio_state_schema_hash
execution_schema_hash
split_manifest_hash
model_input_keys
label_keys
forbidden_model_input_keys
source_access
source_download_complete
universe_asof
universe_selection_date
reportable
reportability_errors
payload_hash
```

`action_metadata.json` must include one entry per action:

```json
{
  "actions": [
    {
      "action_index": 0,
      "action_name": "CASH",
      "symbol_id": "CASH",
      "asset_type": "cash",
      "group": "cash",
      "underlying": null,
      "leverage_factor": 0.0,
      "inverse": false,
      "max_weight": 0.0,
      "risk_bucket": "cash"
    }
  ],
  "action_metadata_hash": "..."
}
```

`feature_manifest.json` must include:

```text
feature_set_id
feature_groups
feature_names
feature_dtypes
feature_fit_windows
normalization_method
input_dataset_manifest_hash
```

`normalization_manifest.json` must include:

```text
fit_start
fit_end
market_feature_mean
market_feature_std
action_feature_mean
action_feature_std
clip_range
```

For reportable research:

```text
normalization_manifest.fit_end < first validation decision timestamp
universe_selection_date <= train_start_date
source_download_complete == true
reportability_errors == []
```

## Split Contract

Splits are defined in metadata, not inferred from file position:

```text
train_start
train_end
val_start
val_end
test_start
test_end
```

Every selected row must satisfy:

```text
split_start <= decision_timestamp <= split_end
next_timestamp <= split_end
```

This prevents train rows from using validation or test rewards.

The required `split_manifest.json` stores:

```json
{
  "train": {"start": "...", "end": "...", "rows": 0, "reward_end_max": "..."},
  "validation": {"start": "...", "end": "...", "rows": 0, "reward_end_max": "..."},
  "test": {"start": "...", "end": "...", "rows": 0, "reward_end_max": "..."},
  "rule": "decision_ts in split and next_ts <= split_end",
  "embargo": null
}
```

Sequential evaluators must reset previous action when `segment_id` changes or
when a valid row is not contiguous with the previous valid row.

## Execution And Terminal Position

Execution assumptions are first-class metadata:

```json
{
  "execution_model": {
    "name": "optimistic_close_plus_estimated_cost_bps",
    "entry_rule": "first action close at or after decision_ts + execution_latency_ms",
    "exit_rule": "first action close at or after next_ts + execution_latency_ms",
    "cost_rule": "action_cost_bps, charged on switches in sequential evaluation",
    "liquidate_at_end": false,
    "allow_extended_hours_exit": false
  },
  "terminal_position_policy": "report_open_or_liquidate_by_evaluator",
  "terminal_liquidation_cost_included": false
}
```

Reportable evaluations must either liquidate final non-cash positions and charge
the exit cost, or report `final_position_open=true`.

## Decision Logs

Every sequential evaluation should produce `decision_logs.jsonl` with:

```text
decision_ts
context_available_until
entry_execution_ts
reward_end_ts
exit_execution_ts
previous_action
selected_action
target_weight
order_legs
traded_notional
q_values
q_edge_vs_cash
q_edge_vs_current
action_mask
mask_reasons
data_quality_score
readiness_score
entry_price
exit_price
gross_return
cost_bps
net_return
equity_after
```

This log is not optional for reportable sequential backtests.

## Dtype Policy

Use compact dtypes where they are safe:

```text
float16: normalized model inputs
float32: returns, labels, costs when used in accounting
int64: timestamps
uint16/int32: reason codes and ids
bool: masks
```

Never store timestamps as float. Never store invalid action returns as zero.

## Forward Compatibility

Protocol v1 readers must:

- Require the core tensor contract.
- Ignore unknown optional `context_groups`.
- Reject unknown required keys only if they conflict with existing meanings.
- Validate all point-in-time timestamps.
- Validate `feature_names` length against the last tensor dimension.
- Validate `action_names` length against the action dimension.

Protocol v2 may add required fields only if it changes `protocol_version`.

## Minimal Validator Checklist

A dataset is valid if:

```text
required keys exist
tensor ranks and shapes match
CASH contract passes
invalid-return contract passes
context availability is causal
entry/exit execution latency is respected
costs are finite and nonnegative for valid actions
feature names match tensor widths
action names match action dimension
manifest hashes are present
source download/reportability flags are explicit
```

## Current Implementation And Next Migrations

The current `stock_second_context_decision_v3` payload implements the v1
contract for the second-context action-scorer path:

```text
protocol_version="decision_tensor_v1"
decision_tensor_protocol_version="1.0.0"
dataset_schema_version="second_context_gold_v1"
schema hashes for features/actions/action metadata/constraints/portfolio/execution
action availability timestamps and action mask reason codes
row/action quality scores and force-cash masks
valid_start_indices, segment_ids, and session_ids
action_metadata.json and split_manifest.json sidecars
model_input_keys, label_keys, and forbidden_model_input_keys leakage guards
```

Remaining migrations should be additive:

```text
write normalization_manifest.json when learned normalizers are persisted
store normalized model inputs as float16 when practical
keep action_returns and accounting fields as float32
promote Zarr/Arrow/Parquet arrays to the archival canonical store for large datasets
move optional future tensors into context_groups
```

This keeps the existing transformer path working while making room for richer
future data without a schema rewrite.
