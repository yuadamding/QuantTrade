from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS,
    MassiveAdaptiveAlphaProtocolError,
    build_massive_adaptive_universe_rule,
    build_massive_adaptive_alpha_v1_protocol,
)


def test_adaptive_protocol_has_frozen_dual_universes_and_term_structure() -> None:
    protocol = build_massive_adaptive_alpha_v1_protocol()

    assert protocol.receipt_sha256 == MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    assert protocol == MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL
    assert protocol.context_universe_rule.target_size == 1_500
    assert protocol.action_universe_rule.target_size == 500
    assert tuple(row.bucket_id for row in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS) == (
        "B01",
        "B02_05",
        "B06_10",
        "B11_21",
        "B22_42",
        "B43_63",
        "B64_126",
    )
    assert tuple(row.end_offset_sessions for row in protocol.return_buckets) == (
        1,
        5,
        10,
        21,
        42,
        63,
        126,
    )


def test_only_ad11_is_initially_promotion_eligible() -> None:
    assert len(MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS) == 12
    assert [row.setting_id for row in MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS] == [
        f"AD{index:02d}" for index in range(12)
    ]
    assert [row.setting_id for row in MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS if row.promotion_eligible] == [
        "AD11"
    ]


def test_source_protocol_cannot_authorize_downstream_stages() -> None:
    for field in (
        "economic_optimization_authorized",
        "reinforcement_learning_authorized",
        "historical_lockbox_access_authorized",
        "prospective_access_authorized",
    ):
        with pytest.raises(MassiveAdaptiveAlphaProtocolError, match="cannot authorize"):
            replace(MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL, **{field: True}).validate()


def test_protocol_rejects_duration_or_universe_drift() -> None:
    with pytest.raises(MassiveAdaptiveAlphaProtocolError, match="duration priors"):
        replace(
            MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
            position_age_input_authorized=True,
        ).validate()
    with pytest.raises(MassiveAdaptiveAlphaProtocolError, match="dual PIT"):
        replace(
            MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
            action_universe_rule=build_massive_adaptive_universe_rule(
                rule_id="massive-pit500-monthly-dollar-volume-v1",
                target_size=501,
                minimum_close_price=3.0,
                minimum_average_dollar_volume=5_000_000.0,
            ),
        ).validate()
