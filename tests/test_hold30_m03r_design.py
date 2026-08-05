"""Registration checks for the versioned M03R two-speed design."""

from rl_quant.training.designs import (
    DESIGNS,
    HOLD30_BASE_DESIGN,
    HOLD30_M03R_BASE_DESIGN,
    TOP2000_H100_CORE_SWEEP,
    TOP2000_H100_WIDE_SWEEP,
)


def test_m03r_design_is_new_and_keeps_temporal_units_distinct() -> None:
    assert HOLD30_M03R_BASE_DESIGN == "daily_raw_pit300_hold30_m03r"
    assert HOLD30_M03R_BASE_DESIGN != HOLD30_BASE_DESIGN

    design = DESIGNS[HOLD30_M03R_BASE_DESIGN]
    assert design.horizon_mode == "daily_raw"
    assert design.episode_len == 252
    assert design.daily_lookback == 252
    assert design.raw_recent_days == 42
    assert design.scored_tail_days == 63
    assert design.bptt_window == 63
    assert design.label_horizon_days == 30
    assert design.auxiliary_horizons == (5, 21, 30, 63)
    assert design.target_holding_days == 30
    assert design.exec_delay == 1
    assert design.terminal_liquidate is False
    assert design.min_gpus == 2


def test_m03r_registration_does_not_change_legacy_or_default_sweeps() -> None:
    legacy = DESIGNS[HOLD30_BASE_DESIGN]
    assert legacy.daily_lookback == 63
    assert legacy.auxiliary_horizons == ()

    assert HOLD30_M03R_BASE_DESIGN not in TOP2000_H100_CORE_SWEEP
    assert HOLD30_M03R_BASE_DESIGN not in TOP2000_H100_WIDE_SWEEP
