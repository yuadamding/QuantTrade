from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.alpha.massive_universe_adapter import checked_pit_universe_rule
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    assert_no_adaptive_hold_semantics,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS,
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
    MASSIVE_FINALIZED_VALIDATION_V0_SETTINGS,
    MassiveFinalizedValidationProtocolError,
    build_massive_finalized_validation_v0_protocol,
)


def test_finalized_validation_protocol_freezes_the_minimum_experiment() -> None:
    protocol = build_massive_finalized_validation_v0_protocol()

    assert protocol == MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL
    assert protocol.receipt_sha256 == MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256
    assert protocol.protocol_id == "massive-finalized-alpha-validation-v0"
    assert protocol.dataset_id == "MassiveFinalizedPIT500ValidationV0"
    assert protocol.production_equivalence is False
    assert protocol.historical_delayed_stream_replay_required is False
    assert protocol.universe_rule.target_size == 500
    assert protocol.context_universe_rule_receipt_sha256 == (
        protocol.action_universe_rule_receipt_sha256
    )
    assert tuple(
        row.horizon_id for row in MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS
    ) == (
        "H01",
        "H05",
        "H21",
        "H63",
    )
    assert all(row.loss_weight == 1.0 for row in protocol.horizons)
    assert protocol.horizons_equal_status is True
    assert tuple(
        row.setting_id for row in MASSIVE_FINALIZED_VALIDATION_V0_SETTINGS
    ) == (
        "MV00",
        "MV01",
        "MV02",
        "MV03",
        "MV04",
    )
    assert protocol.primary_contrast == ("MV04", "MV02")


def test_finalized_validation_source_contract_cannot_authorize_later_stages() -> None:
    for field in (
        "predictive_training_authorized",
        "diagnostic_portfolio_evaluation_authorized",
        "economic_optimization_authorized",
        "historical_lockbox_access_authorized",
        "prospective_access_authorized",
        "reinforcement_learning_authorized",
    ):
        with pytest.raises(
            MassiveFinalizedValidationProtocolError,
            match="cannot authorize",
        ):
            replace(
                MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
                **{field: True},
            ).validate()


def test_finalized_validation_has_no_duration_or_primary_horizon_semantics() -> None:
    protocol = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL
    assert_no_adaptive_hold_semantics(protocol.payload())

    with pytest.raises(
        MassiveFinalizedValidationProtocolError,
        match="primary forecast horizon",
    ):
        replace(protocol, horizons_equal_status=False).validate()
    with pytest.raises(
        MassiveFinalizedValidationProtocolError,
        match="cannot authorize",
    ):
        replace(protocol, position_age_input_authorized=True).validate()


def test_finalized_validation_reuses_the_generic_pit500_contract() -> None:
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    materializer_rule = checked_pit_universe_rule(rule)

    rule.validate()
    assert materializer_rule.unsigned() == rule.unsigned()
    assert rule.ranking_metric == "trailing-mean-dollar-volume"
    assert rule.ranking_lookback_sessions == 63
    assert rule.ranking_lag_sessions == 1
    assert rule.minimum_observed_sessions == 50
    assert rule.minimum_close_price == 3.0
    assert rule.minimum_average_dollar_volume == 5_000_000.0
    assert rule.rebalance_frequency == "monthly"
    assert rule.uses_future_survival is False
