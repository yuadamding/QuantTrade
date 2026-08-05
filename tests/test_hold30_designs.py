"""Registration, geometry, and censoring contracts for Hold-30 v2."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.datasets.daily import build_daily_raw_episodes
from rl_quant.training.designs import (
    DESIGNS,
    HOLD30_BASE_DESIGN,
    TOP2000_H100_CORE_SWEEP,
    TOP2000_H100_WIDE_SWEEP,
)
from rl_quant.training.hold30_experiment import (
    HOLD30_BAR_FEATURE_DIM,
    HOLD30_COVARIATE_DIM,
    HOLD30_MECH8_BY_ID,
    HOLD30_MECH8_IDS,
    HOLD30_MECH8_SETTINGS,
    build_hold30_context_config,
    build_hold30_policy_config,
    hold30_parameter_counts,
    resolve_hold30_setting,
)

EXPECTED_SETTING_IDS = (
    "hold30-m00-legacy-gate",
    "hold30-m01-slow-gate",
    "hold30-m02-age-hazard",
    "hold30-m03-sleeve30",
    "hold30-a04-no-age-input",
    "hold30-a05-no-early-penalty",
    "hold30-a06-no-turn-penalty",
    "hold30-a07-no-exp-timing",
)


def _record(index: int) -> dict[str, object]:
    close = torch.tensor([1.0, 100.0 + index], dtype=torch.float32)
    return {
        "date": f"d{index:03d}",
        "day_close": close,
        "market": torch.tensor([float(index)], dtype=torch.float32),
        "per_stock": torch.tensor([[0.0], [float(index)]], dtype=torch.float32),
        "bars": torch.zeros(2, 2, 5),
        "bar_mask": torch.ones(2, 2, dtype=torch.bool),
        "news_raw": torch.zeros(2, 1, 1),
        "news_mask": torch.zeros(2, 1, dtype=torch.bool),
        "avail": torch.ones(2, dtype=torch.bool),
    }


def test_hold30_base_design_freezes_the_monthly_holding_geometry() -> None:
    assert HOLD30_BASE_DESIGN == "daily_raw_pit300_hold30"
    design = DESIGNS[HOLD30_BASE_DESIGN]

    assert design.horizon_mode == "daily_raw"
    assert design.episode_len == 252
    assert design.scored_tail_days == 63
    assert design.bptt_window == 63
    assert design.label_horizon_days == 30
    assert design.auxiliary_horizons == ()
    assert design.target_holding_days == 30
    assert design.target_discretionary_turnover == pytest.approx(1.0 / 30.0)
    assert design.raw_recent_days == 42
    assert design.daily_lookback == 63
    assert design.exec_delay == 1
    assert design.gate_init_bias == pytest.approx(-3.3844844191)
    assert design.gate_entropy_coef == 0.0
    assert design.terminal_liquidate is False
    assert design.min_gpus == 2
    assert design.d_model == 128
    assert design.enc_layers == 2
    assert design.enc_heads == 4
    assert design.policy_steps == 128
    assert design.pol_lr == pytest.approx(1e-4)
    assert design.pol_weight_decay == pytest.approx(1e-4)
    assert design.grad_clip == pytest.approx(0.5)
    assert design.cost == pytest.approx(0.002)
    assert design.dropout == 0.0
    assert design.schedule == "constant"


def test_hold30_registration_does_not_mutate_legacy_sweeps() -> None:
    assert HOLD30_BASE_DESIGN not in TOP2000_H100_CORE_SWEEP
    assert HOLD30_BASE_DESIGN not in TOP2000_H100_WIDE_SWEEP

    legacy = DESIGNS["daily_raw_top2000"]
    assert legacy.scored_tail_days is None
    assert legacy.target_holding_days is None
    assert legacy.target_discretionary_turnover is None
    assert legacy.auxiliary_horizons == ()
    assert legacy.terminal_liquidate is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"scored_tail_days": 30}, "must cover target_holding_days plus exec_delay"),
        ({"bptt_window": 29}, "bptt_window must be >= target_holding_days"),
        ({"label_horizon_days": 21}, "label_horizon_days must equal target_holding_days"),
        ({"auxiliary_horizons": (5, 30, 21, 63)}, "strictly increasing and unique"),
        ({"target_discretionary_turnover": 0.0}, "must lie in"),
        ({"terminal_liquidate": True}, "require terminal_liquidate=False"),
    ],
)
def test_hold30_design_rejects_incoherent_geometry(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(DESIGNS[HOLD30_BASE_DESIGN], name="invalid_hold30", **changes)


def test_hold30_setting_ids_and_promotion_metadata_are_frozen() -> None:
    assert HOLD30_MECH8_IDS == EXPECTED_SETTING_IDS
    assert tuple(HOLD30_MECH8_BY_ID) == EXPECTED_SETTING_IDS
    assert tuple(setting.setting_index for setting in HOLD30_MECH8_SETTINGS) == tuple(range(8))

    eligible = [setting for setting in HOLD30_MECH8_SETTINGS if setting.promotion_eligible]
    assert [setting.setting_id for setting in eligible] == ["hold30-m02-age-hazard"]
    assert eligible[0].mechanism == "H2"
    assert eligible[0].ablation_of is None

    for setting_id in EXPECTED_SETTING_IDS[4:]:
        setting = resolve_hold30_setting(setting_id)
        assert setting.mechanism == "H2"
        assert setting.ablation_of == "hold30-m02-age-hazard"
        assert not setting.promotion_eligible

    with pytest.raises(ValueError, match="unknown Hold-30 setting"):
        resolve_hold30_setting("H2")


def test_hold30_ablation_switches_change_one_declared_component() -> None:
    canonical = HOLD30_MECH8_BY_ID["hold30-m02-age-hazard"]
    switch_fields = (
        "use_position_age",
        "use_early_exit_penalty",
        "use_turnover_penalty",
        "use_exposure_timing",
    )
    expected_disabled = {
        "hold30-a04-no-age-input": "use_position_age",
        "hold30-a05-no-early-penalty": "use_early_exit_penalty",
        "hold30-a06-no-turn-penalty": "use_turnover_penalty",
        "hold30-a07-no-exp-timing": "use_exposure_timing",
    }

    for setting_id, disabled_field in expected_disabled.items():
        ablation = HOLD30_MECH8_BY_ID[setting_id]
        changed = {
            field
            for field in switch_fields
            if getattr(ablation, field) != getattr(canonical, field)
        }
        assert changed == {disabled_field}
        assert getattr(ablation, disabled_field) is False


def test_package_owned_policy_config_is_compact_and_frozen() -> None:
    from rl_quant.models.daily_policy import DailyCrossSectionPolicy

    config = build_hold30_policy_config("hold30-m02-age-hazard")
    assert config.raw_policy_dim == 64
    assert config.raw_policy_layers == 2
    assert config.raw_policy_heads == 4
    assert config.token_dim == 128
    assert config.temporal_layers == 2
    assert config.temporal_heads == 4
    assert config.feedforward_dim == 256
    assert config.daily_lookback == 63
    policy = DailyCrossSectionPolicy(config)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    assert 500_000 <= parameter_count <= 5_000_000


@pytest.mark.parametrize("setting_id", EXPECTED_SETTING_IDS)
def test_complete_hold30_model_respects_frozen_capacity_caps(setting_id: str) -> None:
    context = build_hold30_context_config()
    assert context.bar_feature_dim == HOLD30_BAR_FEATURE_DIM == 5
    assert context.covariate_dim == HOLD30_COVARIATE_DIM == 0
    assert context.d_model == 128
    assert context.n_layers == 2
    assert context.n_heads == 4
    assert context.feedforward_dim == 256
    assert context.dropout == 0.0
    assert context.max_seconds == 390
    assert context.block_seconds == 5

    counts = hold30_parameter_counts(setting_id)
    assert 0 < counts.context_encoder < 2_000_000
    assert 500_000 <= counts.actor_path <= 5_000_000
    assert counts.total_unique == counts.context_encoder + counts.actor_path
    assert counts.total_unique <= 7_000_000


def test_daily_builder_materializes_ordered_multi_horizon_labels() -> None:
    episodes = build_daily_raw_episodes(
        [_record(index) for index in range(100)],
        episode_len=80,
        horizon=30,
        auxiliary_horizons=(5, 21, 30, 63),
    )
    episode = episodes[0]

    assert episode["auxiliary_horizons"] == (5, 21, 30, 63)
    assert episode["aux_ret_multi"].shape == (80, 4, 2)
    assert episode["aux_ret_valid_multi"].shape == (80, 4, 2)
    torch.testing.assert_close(episode["aux_ret_multi"][:, 2], episode["aux_ret"])
    assert torch.equal(episode["aux_ret_valid_multi"][:, 2], episode["aux_ret_valid"])


def test_daily_builder_marks_complete_and_right_censored_entry_lifecycles() -> None:
    episode = build_daily_raw_episodes(
        [_record(index) for index in range(100)],
        episode_len=80,
        stride=80,
        horizon=30,
        score_tail=63,
        entry_credit_horizon_days=30,
    )[0]

    # The 63-row score tail is local rows 17..79.  With a one-session fill
    # delay, rows 17..48 have a terminal state after 30 holding transitions;
    # rows 49..79 are explicitly right-censored.
    assert torch.where(episode["score_mask"])[0].tolist() == list(range(17, 80))
    assert torch.where(episode["entry_credit_mask"])[0].tolist() == list(range(17, 49))
    assert torch.where(episode["entry_censored_mask"])[0].tolist() == list(range(49, 80))
    assert not bool((episode["entry_credit_mask"] & episode["entry_censored_mask"]).any())
    assert torch.equal(
        episode["entry_credit_mask"] | episode["entry_censored_mask"],
        episode["score_mask"],
    )


def test_daily_builder_validates_multi_horizon_and_credit_configuration() -> None:
    records = [_record(index) for index in range(70)]

    with pytest.raises(ValueError, match="must include horizon"):
        build_daily_raw_episodes(records, 60, horizon=30, auxiliary_horizons=(5, 21, 63))
    with pytest.raises(ValueError, match="strictly increasing and unique"):
        build_daily_raw_episodes(records, 60, horizon=30, auxiliary_horizons=(5, 30, 21))
    with pytest.raises(ValueError, match="positive integer or None"):
        build_daily_raw_episodes(records, 60, horizon=30, entry_credit_horizon_days=0)
