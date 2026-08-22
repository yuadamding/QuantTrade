from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.alpha.massive_universe_adapter import checked_pit_universe_rule
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MassiveAdaptiveAlphaProtocolError,
)


def test_protocol_universe_rules_pass_the_materializer_validator_exactly() -> None:
    for rule in (
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.context_universe_rule,
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.action_universe_rule,
    ):
        materializer = checked_pit_universe_rule(rule)

        assert materializer.unsigned() == rule.unsigned()
        assert materializer.receipt_sha256 == rule.receipt_sha256


def test_materializer_rejects_a_protocol_rule_with_wire_schema_drift() -> None:
    rule = MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.action_universe_rule

    with pytest.raises(MassiveAdaptiveAlphaProtocolError, match="schema"):
        checked_pit_universe_rule(
            replace(rule, schema="rl-quant.incompatible-universe-rule-v1")
        )
