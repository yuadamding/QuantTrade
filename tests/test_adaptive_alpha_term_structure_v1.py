from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

import rl_quant.models.adaptive_alpha_term_structure_v1 as model_module
from rl_quant.features.massive_profitability_origin_features_v2 import (
    BARS_MIN_V2_FIELDS,
    TAPE_MIN_V2_FIELDS,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1,
    MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
    MassiveAdaptiveAlphaModelSpecV1,
    MassiveAdaptiveAlphaModelV1Error,
    MassiveAdaptiveAlphaTermStructureModelV1,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    assert_adaptive_import_firewall,
    assert_no_adaptive_hold_semantics,
)


def _spec() -> MassiveAdaptiveAlphaModelSpecV1:
    return replace(
        MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
        token_dimension=16,
        fast_window_sessions=3,
        maximum_context_sessions=6,
        maximum_intraday_intervals=4,
        market_latent_count=4,
        attention_heads=4,
        dropout_probability=0.0,
    )


def _inputs(
    *,
    batch: int = 2,
    sessions: int = 6,
    assets: int = 5,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(112)
    membership = torch.ones(batch, sessions, assets, dtype=torch.bool)
    membership[0, :2, -1] = False
    membership[0, -1, -1] = False
    bars_valid = membership.unsqueeze(-1).expand(
        -1, -1, -1, len(BARS_MIN_V2_FIELDS)
    ).clone()
    tape_valid = membership.unsqueeze(-1).expand(
        -1, -1, -1, len(TAPE_MIN_V2_FIELDS)
    ).clone()
    bars = torch.randn(
        batch,
        sessions,
        assets,
        len(BARS_MIN_V2_FIELDS),
        generator=generator,
    )
    tape = torch.randn(
        batch,
        sessions,
        assets,
        len(TAPE_MIN_V2_FIELDS),
        generator=generator,
    )
    bars = torch.where(bars_valid, bars, torch.zeros_like(bars))
    tape = torch.where(tape_valid, tape, torch.zeros_like(tape))
    staleness = torch.full((batch, sessions, assets, 2), 0.25)
    staleness = torch.where(
        membership.unsqueeze(-1), staleness, torch.zeros_like(staleness)
    )
    intraday_valid = membership.unsqueeze(-1).expand(-1, -1, -1, 4).clone()
    intraday = torch.randn(
        batch, sessions, assets, 4, 5, generator=generator
    )
    intraday = torch.where(
        intraday_valid.unsqueeze(-1), intraday, torch.zeros_like(intraday)
    )
    action_mask = membership.clone()
    action_mask[:, :, 1::2] = False
    return {
        "bars_values": bars,
        "bars_valid": bars_valid,
        "tape_values": tape,
        "tape_valid": tape_valid,
        "source_staleness": staleness,
        "context_membership": membership,
        "action_mask": action_mask,
        "intraday_values": intraday,
        "intraday_valid": intraday_valid,
    }


def _model() -> MassiveAdaptiveAlphaTermStructureModelV1:
    torch.manual_seed(19)
    return MassiveAdaptiveAlphaTermStructureModelV1(_spec()).eval()


def test_model_emits_two_seven_bucket_distributions_and_backpropagates() -> None:
    model = _model().train()
    inputs = _inputs()

    output = model(**inputs)
    bucket_count = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
    assert output.residual_distribution.mean.shape == (2, 5, bucket_count)
    assert output.raw_distribution.mean.shape == (2, 5, bucket_count)
    assert output.factor_return_mean.shape == (2, bucket_count)
    assert output.executable_score.shape == (2, 5)
    assert output.bucket_router_weights.shape == (2, 5, bucket_count)
    assert output.router_weights.shape == (
        2,
        5,
        bucket_count,
        len(MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1),
    )
    assert output.market_context.shape == (2, 4, 16)
    assert bool(
        (
            output.residual_distribution.downside_quantile
            <= output.residual_distribution.median
        ).all()
    )
    assert bool(
        (
            output.residual_distribution.median
            <= output.residual_distribution.upside_quantile
        ).all()
    )
    assert bool((output.residual_distribution.scale > 0.0).all())
    torch.testing.assert_close(
        output.bucket_router_weights[output.valid].sum(dim=-1),
        torch.ones_like(output.executable_score[output.valid]),
    )

    loss = (
        output.residual_distribution.mean.sum()
        + output.raw_distribution.mean.sum()
        + output.factor_return_mean.sum()
        + output.executable_score.sum()
    )
    loss.backward()
    assert model.bars_projection[0].weight.grad is not None
    assert model.tape_projection[0].weight.grad is not None
    assert model.router.weight.grad is not None
    assert model.term_router.weight.grad is not None


def test_full_sequence_is_strictly_causal_across_sessions() -> None:
    model = _model()
    original = _inputs(batch=1)
    changed = {name: value.clone() for name, value in original.items()}
    for value_name, valid_name in (
        ("bars_values", "bars_valid"),
        ("tape_values", "tape_valid"),
    ):
        changed[value_name][:, 4:] = torch.where(
            changed[valid_name][:, 4:],
            changed[value_name][:, 4:] * -31.0 + 7.0,
            changed[value_name][:, 4:],
        )
    changed["intraday_values"][:, 4:] = torch.where(
        changed["intraday_valid"][:, 4:].unsqueeze(-1),
        changed["intraday_values"][:, 4:] * -31.0 + 7.0,
        changed["intraday_values"][:, 4:],
    )
    changed["source_staleness"][:, 4:] += (
        changed["context_membership"][:, 4:].unsqueeze(-1) * 20.0
    )

    first = model.forward_sequence(**original)
    second = model.forward_sequence(**changed)

    torch.testing.assert_close(
        first.residual_distribution.mean[:, :4],
        second.residual_distribution.mean[:, :4],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        first.router_weights[:, :4],
        second.router_weights[:, :4],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        first.executable_score[:, :4],
        second.executable_score[:, :4],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert not torch.allclose(
        first.residual_distribution.mean[:, -1],
        second.residual_distribution.mean[:, -1],
    )


def test_model_is_permutation_equivariant_on_the_security_axis() -> None:
    model = _model()
    inputs = _inputs()
    permutation = torch.tensor((4, 2, 0, 3, 1))
    inverse = torch.argsort(permutation)
    permuted = {
        name: (
            value.index_select(2, permutation)
            if value.ndim >= 3
            else value.index_select(1, permutation)
        )
        for name, value in inputs.items()
    }

    original = model(**inputs)
    changed = model(**permuted)

    torch.testing.assert_close(
        original.residual_distribution.mean,
        changed.residual_distribution.mean.index_select(1, inverse),
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    torch.testing.assert_close(
        original.router_weights,
        changed.router_weights.index_select(1, inverse),
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    torch.testing.assert_close(
        original.executable_score,
        changed.executable_score.index_select(1, inverse),
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    torch.testing.assert_close(
        original.market_context,
        changed.market_context,
        atol=3.0e-6,
        rtol=3.0e-6,
    )


def test_missing_sources_receive_zero_router_weight() -> None:
    model = _model()
    inputs = _inputs()
    inputs["tape_values"].zero_()
    inputs["tape_valid"].zero_()
    inputs.pop("intraday_values")
    inputs.pop("intraday_valid")

    output = model(**inputs)
    valid_weights = output.router_weights[output.valid]
    tape_fast = MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1.index("tape-fast")
    tape_slow = MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1.index("tape-slow")
    intraday = MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1.index("intraday-path")

    assert torch.count_nonzero(valid_weights[..., tape_fast]) == 0
    assert torch.count_nonzero(valid_weights[..., tape_slow]) == 0
    assert torch.count_nonzero(valid_weights[..., intraday]) == 0
    torch.testing.assert_close(
        valid_weights.sum(dim=-1), torch.ones_like(valid_weights[..., 0])
    )


def test_real_source_staleness_changes_forecasts_and_is_differentiable() -> None:
    model = _model()
    inputs = _inputs(batch=1)
    current = model(**inputs)
    stale_inputs = dict(inputs)
    stale_inputs["source_staleness"] = inputs["source_staleness"].clone()
    stale_inputs["source_staleness"][..., 1] += (
        inputs["context_membership"] * 50.0
    )
    stale = model(**stale_inputs)

    assert not torch.allclose(
        current.residual_distribution.mean,
        stale.residual_distribution.mean,
    )
    gradient_inputs = dict(inputs)
    gradient_inputs["source_staleness"] = (
        inputs["source_staleness"].clone().requires_grad_()
    )
    model(**gradient_inputs).residual_distribution.mean.sum().backward()
    gradient = gradient_inputs["source_staleness"].grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[inputs["context_membership"]]) > 0


def test_action_mask_zeroes_nonexecutable_outputs_without_hiding_context() -> None:
    model = _model()
    inputs = _inputs(batch=1)
    output = model(**inputs)
    excluded = ~inputs["action_mask"][:, -1]

    assert torch.count_nonzero(output.residual_distribution.mean[excluded]) == 0
    assert torch.count_nonzero(output.stock_context[excluded]) == 0
    assert torch.count_nonzero(output.router_weights[excluded]) == 0
    assert torch.count_nonzero(output.bucket_router_weights[excluded]) == 0
    assert torch.count_nonzero(output.executable_score[excluded]) == 0
    assert bool((output.residual_distribution.scale[excluded] == 1.0).all())
    assert torch.count_nonzero(output.market_context) > 0


@pytest.mark.parametrize("corruption", ("payload", "staleness", "action"))
def test_malformed_or_noncausal_support_fails_closed(corruption: str) -> None:
    model = _model()
    inputs = _inputs(batch=1)
    if corruption == "payload":
        inputs["bars_valid"][0, 0, 0, 0] = False
    elif corruption == "staleness":
        inputs["source_staleness"][0, 0, 0, 0] = -1.0
    else:
        inputs["context_membership"][0, -1, 1] = False
        inputs["bars_valid"][0, -1, 1] = False
        inputs["tape_valid"][0, -1, 1] = False
        inputs["intraday_valid"][0, -1, 1] = False
        inputs["bars_values"][0, -1, 1] = 0.0
        inputs["tape_values"][0, -1, 1] = 0.0
        inputs["intraday_values"][0, -1, 1] = 0.0
        inputs["source_staleness"][0, -1, 1] = 0.0
        inputs["action_mask"][0, -1, 1] = True

    with pytest.raises(MassiveAdaptiveAlphaModelV1Error, match="malformed"):
        model(**inputs)


def test_model_spec_is_engineering_only_and_has_no_duration_semantics() -> None:
    spec = _spec()
    spec.validate()
    assert len(spec.receipt_sha256) == 64
    assert_no_adaptive_hold_semantics(spec)
    assert_adaptive_import_firewall((Path(model_module.__file__),))
    assert not spec.economic_training_authorized
    assert not spec.outer_evaluation_authorized
    assert not spec.profitability_reporting_authorized
    assert not spec.lockbox_access_authorized
    assert not spec.reinforcement_learning_authorized
    forbidden_fragments = ("age", "duration", "persistence", "scheduled_exit")
    assert all(
        not any(fragment in field.name for fragment in forbidden_fragments)
        for field in fields(MassiveAdaptiveAlphaModelSpecV1)
    )
