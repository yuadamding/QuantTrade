"""Exact contracts for the exploratory TOP50/TOP2000 H100 screening matrices."""
from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from rl_quant.training import (
    DESIGNS,
    SWEEP,
    TOP2000_H100_CORE_SWEEP,
    TOP2000_H100_WIDE_SWEEP,
    TOP50_H100_CORE_SWEEP,
    TOP50_H100_WIDE_SWEEP,
)


CORE = (
    "top50_h100_42_1s",
    "top50_h100_42_60s",
    "top50_h100_126",
    "top50_h100_252",
)
WIDE = CORE + (
    "top50_h100_252_small",
    "top50_h100_252_large",
    "top50_h100_252_15s",
    "top50_h100_252_300s",
    "top50_h100_252_block1m",
    "top50_h100_252_block15m",
    "top50_h100_252_h5",
    "top50_h100_252_h63",
    "top50_h100_252_risk0",
    "top50_h100_252_risk25",
    "top50_h100_252_cost2bp",
    "top50_h100_252_cost10bp",
    "top50_h100_252_temp1",
    "top50_h100_252_cap20",
    "top50_h100_252_uncapped",
    "top50_h100_252_raw21",
)
TOP2000_CORE = (
    "daily_raw_top2000",
    "top2000_h100_policy_small",
    "top2000_h100_policy_large",
    "top2000_h100_lr1e4",
    "top2000_h100_lr6e4",
    "top2000_h100_bptt5",
    "top2000_h100_actions26",
    "top2000_h100_actions5",
    "top2000_h100_actions52",
    "top2000_h100_budget0",
)
TOP2000_WIDE = TOP2000_CORE + (
    "top2000_h100_cap005",
    "top2000_h100_raw21",
    "top2000_h100_raw84",
    "top2000_h100_cap02",
    "top2000_h100_uncapped",
    "top2000_h100_risk0",
    "top2000_h100_risk25",
    "top2000_h100_cost5bp",
    "top2000_h100_cost20bp",
    "top2000_h100_temp1",
    "top2000_h100_entropy1e5",
    "top2000_h100_wd10",
    "top2000_h100_wd60",
    "top2000_h100_bar300",
    "top2000_h100_block15m",
    "top2000_h100_h5",
    "top2000_h100_h63",
)


def test_top50_screen_membership_and_order_are_exact() -> None:
    assert tuple(TOP50_H100_CORE_SWEEP) == CORE
    assert tuple(TOP50_H100_WIDE_SWEEP) == WIDE
    assert len(WIDE) == len(set(WIDE)) == 20
    assert set(WIDE).issubset(DESIGNS)


def test_core_screen_holds_capacity_economics_and_exposure_constant() -> None:
    expected_geometry = {
        "top50_h100_42_1s": (1, 42, 5, 42, 0, 2, 4, 8, 3000),
        "top50_h100_42_60s": (60, 42, 5, 42, 0, 8, 1, 8, 3000),
        "top50_h100_126": (60, 126, 10, 126, 42, 8, 1, 8, 1500),
        "top50_h100_252": (60, 252, 15, 252, 42, 8, 1, 8, 1000),
    }
    base = DESIGNS["top50_h100_252"]
    capacity = (
        base.d_model,
        base.enc_layers,
        base.enc_heads,
        base.raw_policy_dim,
        base.raw_policy_layers,
        base.raw_policy_heads,
        base.policy_token_dim,
        base.policy_layers,
        base.policy_heads,
    )
    economics = (
        base.label_horizon_days,
        base.risk_lambda,
        base.cost,
        base.temperature,
        base.max_stock_weight,
    )

    for name in CORE:
        design = DESIGNS[name]
        assert (
            design.bar_seconds,
            design.episode_len,
            design.episode_stride,
            design.daily_lookback,
            design.raw_recent_days,
            design.ssl_batch_size,
            design.ssl_accum,
            design.batch_days,
            design.policy_steps,
        ) == expected_geometry[name]
        assert design.ssl_batch_size * design.ssl_accum == 8
        assert design.batch_days * design.episode_stride * design.policy_steps == 120_000
        assert (
            design.d_model,
            design.enc_layers,
            design.enc_heads,
            design.raw_policy_dim,
            design.raw_policy_layers,
            design.raw_policy_heads,
            design.policy_token_dim,
            design.policy_layers,
            design.policy_heads,
        ) == capacity
        assert (
            design.label_horizon_days,
            design.risk_lambda,
            design.cost,
            design.temperature,
            design.max_stock_weight,
        ) == economics
        assert design.horizon_mode == "daily_raw"
        assert design.min_gpus == 1
        assert design.bptt_window == 42


def test_wide_ablations_change_only_the_declared_fields() -> None:
    expected_overrides = {
        "top50_h100_42_1s": {
            "bar_seconds": 1,
            "episode_len": 42,
            "episode_stride": 5,
            "daily_lookback": 42,
            "raw_recent_days": 0,
            "ssl_batch_size": 2,
            "ssl_accum": 4,
            "policy_steps": 3000,
        },
        "top50_h100_42_60s": {
            "episode_len": 42,
            "episode_stride": 5,
            "daily_lookback": 42,
            "raw_recent_days": 0,
            "policy_steps": 3000,
        },
        "top50_h100_126": {
            "episode_len": 126,
            "episode_stride": 10,
            "daily_lookback": 126,
            "policy_steps": 1500,
        },
        "top50_h100_252": {},
        "top50_h100_252_small": {
            "d_model": 256,
            "enc_layers": 4,
            "raw_policy_dim": 64,
            "raw_policy_layers": 1,
            "raw_policy_heads": 4,
            "policy_token_dim": 128,
            "policy_layers": 2,
        },
        "top50_h100_252_large": {
            "d_model": 512,
            "enc_layers": 8,
            "raw_policy_dim": 192,
            "raw_policy_layers": 3,
            "policy_token_dim": 384,
            "policy_layers": 4,
        },
        "top50_h100_252_15s": {"bar_seconds": 15},
        "top50_h100_252_300s": {"bar_seconds": 300},
        "top50_h100_252_block1m": {"block_seconds": 60},
        "top50_h100_252_block15m": {"block_seconds": 900},
        "top50_h100_252_h5": {"label_horizon_days": 5},
        "top50_h100_252_h63": {"label_horizon_days": 63},
        "top50_h100_252_risk0": {"risk_lambda": 0.0},
        "top50_h100_252_risk25": {"risk_lambda": 0.25},
        "top50_h100_252_cost2bp": {"cost": 2e-4},
        "top50_h100_252_cost10bp": {"cost": 1e-3},
        "top50_h100_252_temp1": {"temperature": 1.0},
        "top50_h100_252_cap20": {"max_stock_weight": 0.20},
        "top50_h100_252_uncapped": {"max_stock_weight": 1.0},
        "top50_h100_252_raw21": {"raw_recent_days": 21},
    }
    base = asdict(DESIGNS["top50_h100_252"])

    for name in WIDE:
        candidate = asdict(DESIGNS[name])
        actual = {
            field: value
            for field, value in candidate.items()
            if field not in {"name", "note"} and value != base[field]
        }
        assert actual == expected_overrides[name]
        assert candidate["min_gpus"] == 1


def test_top2000_membership_order_and_base_geometry_are_exact() -> None:
    assert tuple(TOP2000_H100_CORE_SWEEP) == TOP2000_CORE
    assert tuple(TOP2000_H100_WIDE_SWEEP) == TOP2000_WIDE
    assert tuple(SWEEP) == TOP2000_WIDE
    assert len(TOP2000_WIDE) == len(set(TOP2000_WIDE)) == 27

    design = DESIGNS["daily_raw_top2000"]

    assert (design.session_seconds, design.block_seconds, design.bar_seconds) == (23_400, 300, 60)
    assert (design.episode_len, design.episode_stride, design.daily_lookback) == (252, 21, 252)
    assert (design.raw_recent_days, design.bptt_window) == (42, 21)
    assert (design.d_model, design.enc_layers, design.enc_heads) == (512, 8, 8)
    assert (design.policy_token_dim, design.policy_layers, design.policy_heads) == (96, 2, 6)
    assert (design.ssl_batch_size, design.ssl_accum, design.ssl_steps) == (12, 3, 1000)
    assert (design.batch_days, design.policy_steps) == (8, 716)
    assert design.batch_days * design.episode_stride * design.policy_steps == 120_288
    assert (design.max_actions_per_day, design.budget_lambda) == (12.0, 1e-3)
    assert (design.cost, design.max_stock_weight) == (1e-3, 0.01)
    assert design.context_storage_dtype == "bfloat16"
    assert (design.enc_stock_chunk, design.raw_stock_chunk) == (640, 1024)
    assert design.min_gpus == 4


def test_daily_raw_rejects_delays_without_a_pending_order_queue() -> None:
    with pytest.raises(ValueError, match=r"daily_raw supports exec_delay=1 only"):
        replace(DESIGNS["daily_raw_top2000"], name="unsupported_delay", exec_delay=2)


def test_top2000_wide_ablations_change_only_declared_fields() -> None:
    expected_overrides = {
        "daily_raw_top2000": {},
        "top2000_h100_policy_small": {
            "raw_policy_dim": 64,
            "raw_policy_layers": 1,
            "raw_policy_heads": 4,
            "policy_token_dim": 64,
            "policy_layers": 1,
            "policy_heads": 4,
        },
        "top2000_h100_policy_large": {
            "raw_policy_dim": 192,
            "raw_policy_layers": 3,
            "policy_token_dim": 192,
            "policy_layers": 3,
            "policy_heads": 8,
        },
        "top2000_h100_lr1e4": {"pol_lr": 1e-4},
        "top2000_h100_lr6e4": {"pol_lr": 6e-4},
        "top2000_h100_bptt5": {"bptt_window": 5},
        "top2000_h100_actions26": {"max_actions_per_day": 26.0},
        "top2000_h100_actions5": {"max_actions_per_day": 5.0},
        "top2000_h100_actions52": {"max_actions_per_day": 52.0},
        "top2000_h100_cap005": {"max_stock_weight": 0.005},
        "top2000_h100_budget0": {"budget_lambda": 0.0},
        "top2000_h100_raw21": {"raw_recent_days": 21},
        "top2000_h100_raw84": {"raw_recent_days": 84},
        "top2000_h100_cap02": {"max_stock_weight": 0.02},
        "top2000_h100_uncapped": {"max_stock_weight": 1.0},
        "top2000_h100_risk0": {"risk_lambda": 0.0},
        "top2000_h100_risk25": {"risk_lambda": 0.25},
        "top2000_h100_cost5bp": {"cost": 5e-4},
        "top2000_h100_cost20bp": {"cost": 2e-3},
        "top2000_h100_temp1": {"temperature": 1.0},
        "top2000_h100_entropy1e5": {"entropy_coef": 1e-5},
        "top2000_h100_wd10": {"pol_weight_decay": 0.10},
        "top2000_h100_wd60": {"pol_weight_decay": 0.60},
        "top2000_h100_bar300": {"bar_seconds": 300},
        "top2000_h100_block15m": {"block_seconds": 900},
        "top2000_h100_h5": {"label_horizon_days": 5},
        "top2000_h100_h63": {"label_horizon_days": 63},
    }
    base = asdict(DESIGNS["daily_raw_top2000"])

    for name in TOP2000_WIDE:
        candidate = asdict(DESIGNS[name])
        actual = {
            field: value
            for field, value in candidate.items()
            if field not in {"name", "note"} and value != base[field]
        }
        assert actual == expected_overrides[name]
        assert candidate["horizon_mode"] == "daily_raw"
        assert candidate["episode_len"] == candidate["daily_lookback"] == 252
        assert candidate["ssl_batch_size"] % 4 == candidate["batch_days"] % 4 == 0
        assert (candidate["enc_stock_chunk"], candidate["raw_stock_chunk"]) == (640, 1024)
        assert candidate["min_gpus"] == 4
