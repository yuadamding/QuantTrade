"""Train-time data organizers for the learning framework (raw inputs -> consumable tensors)."""
from __future__ import annotations

from rl_quant.datasets.daily import (
    build_daily_episodes,
    build_daily_raw_episodes,
    cross_day_returns,
    horizon_close_returns,
    to_daily_raw_records,
)
from rl_quant.datasets.provenance import (
    DatasetProvenance,
    declared_universe_actions,
    inspect_dataset_provenance,
    point_in_time_membership,
    source_symbol_to_action_index,
)
from rl_quant.datasets.splits import day_sequence, flatten_days, split_days, time_split
from rl_quant.datasets.walk_forward import (
    DecisionBlock,
    DecisionPosition,
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardFoldIdentity,
    generate_walk_forward_folds,
)
from rl_quant.datasets.raw_window import (
    BAR_FEATS,
    BAR_FIELDS,
    COV_FIELDS,
    MAX_NEWS,
    NEWS_RAW_DIM,
    RAW_WINDOW_CACHE_VERSION,
    RawWindowConfig,
    build_window,
    list_windows,
    load_universe,
    news_is_reportable,
    raw_window_cache_key,
    raw_window_dependency_paths,
    raw_window_source_signature,
)

__all__ = [
    "BAR_FEATS",
    "BAR_FIELDS",
    "COV_FIELDS",
    "DatasetProvenance",
    "DecisionBlock",
    "DecisionPosition",
    "MAX_NEWS",
    "NEWS_RAW_DIM",
    "RAW_WINDOW_CACHE_VERSION",
    "RawWindowConfig",
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardFoldIdentity",
    "build_daily_episodes",
    "build_daily_raw_episodes",
    "build_window",
    "cross_day_returns",
    "declared_universe_actions",
    "horizon_close_returns",
    "inspect_dataset_provenance",
    "to_daily_raw_records",
    "day_sequence",
    "flatten_days",
    "generate_walk_forward_folds",
    "list_windows",
    "load_universe",
    "news_is_reportable",
    "point_in_time_membership",
    "raw_window_cache_key",
    "raw_window_dependency_paths",
    "raw_window_source_signature",
    "split_days",
    "source_symbol_to_action_index",
    "time_split",
]
