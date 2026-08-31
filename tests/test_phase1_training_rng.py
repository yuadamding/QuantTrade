"""Common-random-stream regressions for the external Phase-1 driver."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from rl_quant.models import DailyCrossSectionConfig, DailyCrossSectionPolicy
from rl_quant.training.designs import DESIGNS


DRIVER_PATH = Path(__file__).resolve().parents[2] / "training" / "train_phase1.py"
if not DRIVER_PATH.is_file():
    pytest.skip(
        "external Phase-1 training driver is not packaged in a clean checkout",
        allow_module_level=True,
    )
SPEC = importlib.util.spec_from_file_location("phase1_training_rng_driver", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def _daily_policy(name: str) -> DailyCrossSectionPolicy:
    design = DESIGNS[name]
    config = DailyCrossSectionConfig(
        context_dim=design.d_model,
        bar_feature_dim=driver.BAR_FEATS,
        raw_policy_dim=design.raw_policy_dim,
        raw_policy_layers=design.raw_policy_layers,
        raw_policy_heads=design.raw_policy_heads,
        raw_block_seconds=design.block_seconds // design.bar_seconds,
        session_seconds=design.session_seconds // design.bar_seconds,
        news_raw_dim=driver.NEWS_RAW_DIM,
        max_news=driver.MAX_NEWS,
        token_dim=design.policy_token_dim,
        temporal_layers=design.policy_layers,
        temporal_heads=design.policy_heads,
        daily_lookback=design.daily_lookback,
        max_days=design.episode_len + 2,
        alloc_layers=design.policy_layers,
        alloc_heads=design.policy_heads,
        feedforward_dim=design.policy_token_dim * 2,
        dropout=design.dropout,
        temperature=design.temperature,
        max_stock_weight=design.max_stock_weight,
        gate_init_bias=design.gate_init_bias,
        grad_checkpoint=design.grad_checkpoint,
        raw_norm=design.raw_norm,
        raw_recent_days=design.raw_recent_days,
        raw_stock_chunk=design.raw_stock_chunk,
    )
    return DailyCrossSectionPolicy(config)


def _stage2_draw_after_construction(name: str) -> list[int]:
    seed = 17
    driver._seed_stage2_policy_initialization(seed)
    _daily_policy(name)
    driver._seed_stage2_training_rng(seed)
    return torch.randperm(29)[:8].tolist()


def test_stage2_sampling_is_independent_of_real_top2000_policy_size() -> None:
    expected = _stage2_draw_after_construction("daily_raw_top2000")

    assert _stage2_draw_after_construction("top2000_h100_policy_small") == expected
    assert _stage2_draw_after_construction("top2000_h100_policy_large") == expected


def _stage1_draw_after_construction(width: int) -> list[int]:
    seed = 17
    torch.manual_seed(seed)
    torch.nn.Sequential(
        torch.nn.Linear(width, width * 2),
        torch.nn.GELU(),
        torch.nn.Linear(width * 2, width),
    )
    driver._seed_stage1_training_rng(seed)
    return torch.randperm(840)[:36].tolist()


def test_stage1_sampling_is_independent_of_initialization_rng_consumption() -> None:
    assert _stage1_draw_after_construction(8) == _stage1_draw_after_construction(128)
