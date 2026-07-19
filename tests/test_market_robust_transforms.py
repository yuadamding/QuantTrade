from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.envs import (
    LiquidityCostStress,
    LowerEnvelopeScenario,
    LowerEnvelopeTransformSuite,
    SequentialTransitionTransform,
    TrendReturnFeatureReversal,
    market_lower_envelope_suite,
)
from rl_quant.rl.iql import ImplicitQLearning, VectorIQLActorCritic
from rl_quant.rl.replay import ReplayBatch


def _batch() -> ReplayBatch:
    gross = torch.tensor([0.04, 0.03])
    execution = torch.tensor([0.01, 0.002])
    impact = torch.tensor([0.003, 0.004])
    rewards = gross - execution - impact
    current = {
        "trend_20d": torch.tensor([[0.4], [-0.2]]),
        "return_1d": torch.tensor([[0.1], [-0.3]]),
        "dollar_volume": torch.tensor([[10.0], [20.0]]),
        "spread_proxy": torch.tensor([[0.02], [0.03]]),
        "regime_id": torch.tensor([[1], [2]]),
    }
    return ReplayBatch(
        observations=current,
        actions=torch.tensor([[0.25], [-0.5]]),
        rewards=rewards,
        next_observations={
            name: value + 1 if value.is_floating_point() else value.clone()
            for name, value in current.items()
        },
        discounts=torch.full((2,), 0.99),
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
        executed_actions=torch.tensor([[0.2], [-0.4]]),
        reward_components={
            "gross_return": gross,
            "execution_cost": execution,
            "impact_cost": impact,
            "risk_penalty": torch.zeros(2),
        },
        extras={"row_id": torch.tensor([7, 8])},
    )


def test_trend_return_reversal_is_explicit_and_does_not_mutate_source() -> None:
    batch = _batch()
    current_before = {name: value.clone() for name, value in batch.observations.items()}
    next_before = {name: value.clone() for name, value in batch.next_observations.items()}
    transform = TrendReturnFeatureReversal(("trend_20d", "return_1d"))

    transformed = transform(batch)

    for name in ("trend_20d", "return_1d"):
        torch.testing.assert_close(transformed.observations[name], -current_before[name])
        torch.testing.assert_close(transformed.next_observations[name], -next_before[name])
        assert transformed.observations[name].data_ptr() != batch.observations[name].data_ptr()
    torch.testing.assert_close(transformed.observations["dollar_volume"], current_before["dollar_volume"])
    assert transformed.rewards is batch.rewards
    for name, value in current_before.items():
        torch.testing.assert_close(batch.observations[name], value)
    for name, value in next_before.items():
        torch.testing.assert_close(batch.next_observations[name], value)

    with pytest.raises(ValueError, match="missing"):
        TrendReturnFeatureReversal(("unlisted_return",))(batch)
    with pytest.raises(ValueError, match="floating point"):
        TrendReturnFeatureReversal(("regime_id",))(batch)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"feature_names": ()}, "non-empty tuple"),
        ({"feature_names": ("trend", "trend")}, "duplicate"),
        (
            {"feature_names": ("trend",), "transform_current": False, "transform_next": False},
            "at least one",
        ),
    ],
)
def test_trend_return_reversal_rejects_ambiguous_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrendReturnFeatureReversal(**kwargs)  # type: ignore[arg-type]


def test_liquidity_cost_stress_updates_observations_components_and_total_reward() -> None:
    batch = _batch()
    reward_before = batch.rewards.clone()
    execution_before = batch.reward_components["execution_cost"].clone()
    impact_before = batch.reward_components["impact_cost"].clone()
    volume_before = batch.observations["dollar_volume"].clone()
    spread_before = batch.observations["spread_proxy"].clone()
    transform = LiquidityCostStress(
        liquidity_feature_factors=(("dollar_volume", 0.5), ("spread_proxy", 2.0)),
        cost_component_factors=(("execution_cost", 2.0), ("impact_cost", 3.0)),
    )

    transformed = transform(batch)

    torch.testing.assert_close(transformed.observations["dollar_volume"], volume_before * 0.5)
    torch.testing.assert_close(transformed.observations["spread_proxy"], spread_before * 2.0)
    torch.testing.assert_close(transformed.reward_components["execution_cost"], execution_before * 2.0)
    torch.testing.assert_close(transformed.reward_components["impact_cost"], impact_before * 3.0)
    added_cost = execution_before + impact_before * 2.0
    torch.testing.assert_close(transformed.rewards, reward_before - added_cost)
    assert transformed.reward_components["gross_return"] is batch.reward_components["gross_return"]
    torch.testing.assert_close(batch.rewards, reward_before)
    torch.testing.assert_close(batch.reward_components["execution_cost"], execution_before)
    torch.testing.assert_close(batch.reward_components["impact_cost"], impact_before)
    torch.testing.assert_close(batch.observations["dollar_volume"], volume_before)
    torch.testing.assert_close(batch.observations["spread_proxy"], spread_before)


def test_liquidity_cost_stress_fails_closed_on_unknown_or_invalid_ledger_fields() -> None:
    batch = _batch()
    feature_missing = LiquidityCostStress(
        liquidity_feature_factors=(("unknown_liquidity", 0.5),),
        cost_component_factors=(("execution_cost", 2.0),),
    )
    with pytest.raises(ValueError, match="observation fields are missing"):
        feature_missing(batch)

    with pytest.raises(ValueError, match="only execution_cost"):
        LiquidityCostStress(
            liquidity_feature_factors=(("dollar_volume", 0.5),),
            cost_component_factors=(("unknown_cost", 2.0),),
        )

    missing_ledger_entry = replace(
        batch,
        reward_components={
            name: value for name, value in batch.reward_components.items() if name != "execution_cost"
        },
    )
    with pytest.raises(ValueError, match="Cost reward components are missing"):
        LiquidityCostStress(
            liquidity_feature_factors=(("dollar_volume", 0.5),),
            cost_component_factors=(("execution_cost", 2.0),),
        )(missing_ledger_entry)

    bad_component = dict(batch.reward_components)
    bad_component["execution_cost"] = -bad_component["execution_cost"]
    with pytest.raises(ValueError, match="must be nonnegative"):
        LiquidityCostStress(
            liquidity_feature_factors=(("dollar_volume", 0.5),),
            cost_component_factors=(("execution_cost", 2.0),),
        )(replace(batch, reward_components=bad_component))


@pytest.mark.parametrize(
    "feature_factors, component_factors, message",
    [
        ((), (("execution_cost", 2.0),), "non-empty"),
        ((("volume", 0.5),), (), "non-empty"),
        ((("volume", -0.1),), (("execution_cost", 2.0),), ">= 0.0"),
        ((("volume", 0.5),), (("execution_cost", 0.9),), ">= 1.0"),
        ((("volume", 0.5),), (("risk_penalty", 2.0),), "only execution_cost"),
        ((("volume", 1.0),), (("execution_cost", 1.0),), "no-op"),
    ],
)
def test_liquidity_cost_stress_rejects_ambiguous_configuration(
    feature_factors: tuple[tuple[str, float], ...],
    component_factors: tuple[tuple[str, float], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LiquidityCostStress(feature_factors, component_factors)


def test_market_lower_envelope_suite_composes_scenarios_for_iql() -> None:
    batch = _batch()
    reversal = TrendReturnFeatureReversal(("trend_20d", "return_1d"))
    cost_stress = LiquidityCostStress(
        liquidity_feature_factors=(("dollar_volume", 0.5),),
        cost_component_factors=(("execution_cost", 2.0),),
    )
    suite = market_lower_envelope_suite(reversal, cost_stress)
    assert suite.names == (
        "trend_return_reversal",
        "liquidity_cost_stress",
        "joint_reversal_liquidity_cost",
    )
    outputs = suite.apply(batch)
    joint = outputs["joint_reversal_liquidity_cost"]
    torch.testing.assert_close(joint.next_observations["trend_20d"], -batch.next_observations["trend_20d"])
    torch.testing.assert_close(joint.observations["dollar_volume"], batch.observations["dollar_volume"] * 0.5)
    torch.testing.assert_close(
        joint.rewards,
        batch.rewards - batch.reward_components["execution_cost"],
    )

    model = VectorIQLActorCritic(
        observation_key="trend_20d",
        observation_dim=1,
        action_dim=1,
        hidden_dims=(8,),
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    algorithm = ImplicitQLearning(model, transforms=suite.transforms)
    target, spread = algorithm._conservative_target(batch)
    expected_penalty = batch.reward_components["execution_cost"]
    torch.testing.assert_close(target, batch.rewards - expected_penalty)
    torch.testing.assert_close(spread, expected_penalty)


def test_lower_envelope_composition_and_scenario_names_fail_closed() -> None:
    reversal = TrendReturnFeatureReversal(("trend_20d",))
    stress = LiquidityCostStress(
        liquidity_feature_factors=(("dollar_volume", 0.5),),
        cost_component_factors=(("execution_cost", 2.0),),
    )
    sequential = SequentialTransitionTransform((reversal, stress))
    suite = LowerEnvelopeTransformSuite((LowerEnvelopeScenario("joint", sequential),))
    assert tuple(suite.apply(_batch())) == ("joint",)

    with pytest.raises(ValueError, match="unique"):
        LowerEnvelopeTransformSuite(
            (
                LowerEnvelopeScenario("duplicate", reversal),
                LowerEnvelopeScenario("duplicate", stress),
            )
        )
    with pytest.raises(ValueError, match="non-empty"):
        LowerEnvelopeScenario("", reversal)
