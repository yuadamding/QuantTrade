"""Checked bridge from the protocol-layer universe view to PIT materialization."""

from __future__ import annotations

from rl_quant.alpha.pit_universe import PITUniverseRuleSpec
from rl_quant.protocol.massive_adaptive_alpha_v1 import MassiveAdaptiveUniverseRule


def checked_pit_universe_rule(
    rule: MassiveAdaptiveUniverseRule,
) -> PITUniverseRuleSpec:
    """Require both validators to accept the exact same wire representation."""

    rule.validate()
    materializer_rule = PITUniverseRuleSpec(
        rule_id=rule.rule_id,
        target_size=rule.target_size,
        ranking_metric=rule.ranking_metric,
        ranking_lookback_sessions=rule.ranking_lookback_sessions,
        ranking_lag_sessions=rule.ranking_lag_sessions,
        minimum_observed_sessions=rule.minimum_observed_sessions,
        minimum_close_price=rule.minimum_close_price,
        minimum_average_dollar_volume=rule.minimum_average_dollar_volume,
        eligible_security_types=rule.eligible_security_types,
        rebalance_frequency=rule.rebalance_frequency,
        tie_breaker=rule.tie_breaker,
        uses_future_survival=rule.uses_future_survival,
        receipt_sha256=rule.receipt_sha256,
        schema=rule.schema,
    )
    materializer_rule.validate()
    if materializer_rule.unsigned() != rule.unsigned():
        raise ValueError("protocol and PIT universe wire representations differ")
    return materializer_rule


__all__ = ["checked_pit_universe_rule"]
