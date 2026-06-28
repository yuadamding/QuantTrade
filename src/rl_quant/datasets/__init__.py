"""Train-time data organizers for the learning framework (raw inputs -> consumable tensors)."""
from __future__ import annotations

from rl_quant.datasets.daily import (
    build_daily_episodes,
    build_daily_raw_episodes,
    cross_day_returns,
    horizon_close_returns,
    to_daily_raw_records,
)
from rl_quant.datasets.splits import day_sequence, flatten_days, split_days, time_split
from rl_quant.datasets.raw_window import (
    BAR_FEATS,
    BAR_FIELDS,
    COV_FIELDS,
    MAX_NEWS,
    NEWS_RAW_DIM,
    RawWindowConfig,
    build_window,
    list_windows,
    load_universe,
    news_is_reportable,
)

__all__ = [
    "BAR_FEATS",
    "BAR_FIELDS",
    "COV_FIELDS",
    "MAX_NEWS",
    "NEWS_RAW_DIM",
    "RawWindowConfig",
    "build_daily_episodes",
    "build_daily_raw_episodes",
    "build_window",
    "cross_day_returns",
    "horizon_close_returns",
    "to_daily_raw_records",
    "day_sequence",
    "flatten_days",
    "list_windows",
    "load_universe",
    "news_is_reportable",
    "split_days",
    "time_split",
]
