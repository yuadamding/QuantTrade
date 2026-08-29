from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

import rl_quant.training.adaptive_alpha_supervised_v1 as objective_module
from rl_quant.alpha.targets import OriginExposurePanel
from rl_quant.features.massive_adaptive_alpha_targets_v1 import (
    MassiveAdaptiveEconomicPathV1,
    build_massive_adaptive_alpha_targets_v1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
    MassiveAdaptiveAlphaTermStructureModelV1,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    assert_adaptive_import_firewall,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.adaptive_alpha_supervised_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_OBJECTIVE_CONFIG_V1,
    MassiveAdaptiveAlphaObjectiveConfigV1,
    MassiveAdaptiveAlphaSupervisedV1Error,
    MassiveAdaptiveAlphaTrainingBatchV1,
    massive_adaptive_alpha_supervised_loss_v1,
)
from rl_quant.workflows.adaptive_alpha_training_inputs_v1 import (
    build_massive_adaptive_alpha_training_batch_from_targets_v1,
)


_DIGEST = "a" * 64
_TARGET_ASSETS = ("SEC-A", "SEC-B", "SEC-C", "SEC-D")


def _target_artifact(session_index: int):
    decision_at_ms = 100_000 + 200_000 * session_index
    fill_at_ms = decision_at_ms + 1_000
    economic = tuple(fill_at_ms + 1_000 * offset for offset in range(127))
    paths = []
    for asset_index, security_id in enumerate(_TARGET_ASSETS):
        values = tuple(
            (100.0 + 5.0 * asset_index)
            * (1.0 + (0.0005 + 0.0001 * asset_index) * offset)
            for offset in range(127)
        )
        body = {
            "schema": "rl-quant.massive-adaptive-economic-path-v1",
            "security_id": security_id,
            "decision_at_ms": decision_at_ms,
            "fill_at_ms": fill_at_ms,
            "economic_at_ms": economic,
            "available_at_ms": tuple(value + 50 for value in economic),
            "values": values,
            "valid": (True,) * 127,
            "terminal": (False,) * 127,
            "mark_kinds": ("market",) * 127,
            "mark_receipts": tuple(
                semantic_sha256((session_index, security_id, offset))
                for offset in range(127)
            ),
            "unresolved_terminal_fallback_session_offset": None,
            "conservative_total_loss_fallback": False,
            "source_economic_path_receipt_sha256": semantic_sha256(
                (session_index, security_id, "source-path")
            ),
        }
        paths.append(
            MassiveAdaptiveEconomicPathV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    exposure_panel = OriginExposurePanel(
        origin_at_ms=decision_at_ms,
        available_at_ms=decision_at_ms,
        asset_ids=_TARGET_ASSETS,
        exposure_names=("intercept", "style"),
        exposures=((1.0, -1.5), (1.0, -0.5), (1.0, 0.5), (1.0, 1.5)),
        regression_weights=(1.0,) * 4,
        qualified_asset_mask=(True,) * 4,
        source_receipt_sha256=semantic_sha256((session_index, "exposures")),
    )
    return build_massive_adaptive_alpha_targets_v1(
        decision_session_date=f"2024-01-{session_index + 2:02d}",
        built_at_ms=economic[-1] + 50,
        paths=tuple(paths),
        exposure_panel=exposure_panel,
        origin_receipt_sha256=semantic_sha256((session_index, "origin")),
        economic_accounting_receipt_sha256=semantic_sha256(
            (session_index, "accounting")
        ),
        fill_source_receipt_sha256=semantic_sha256((session_index, "fills")),
        terminal_authority_receipt_sha256=semantic_sha256(
            (session_index, "terminal")
        ),
        economic_coverage_receipt_sha256=semantic_sha256(
            (session_index, "coverage")
        ),
    )


def _model_and_output() -> tuple[
    MassiveAdaptiveAlphaTermStructureModelV1,
    object,
]:
    spec = replace(
        MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
        token_dimension=16,
        fast_window_sessions=2,
        maximum_context_sessions=4,
        maximum_intraday_intervals=4,
        market_latent_count=4,
        attention_heads=4,
        dropout_probability=0.0,
    )
    torch.manual_seed(81)
    model = MassiveAdaptiveAlphaTermStructureModelV1(spec)
    batch, sessions, assets = 1, 4, 4
    generator = torch.Generator().manual_seed(82)
    bars = torch.randn(batch, sessions, assets, 19, generator=generator)
    tape = torch.randn(batch, sessions, assets, 15, generator=generator)
    bars_valid = torch.ones_like(bars, dtype=torch.bool)
    tape_valid = torch.ones_like(tape, dtype=torch.bool)
    membership = torch.ones(batch, sessions, assets, dtype=torch.bool)
    output = model.forward_sequence(
        bars_values=bars,
        bars_valid=bars_valid,
        tape_values=tape,
        tape_valid=tape_valid,
        source_staleness=torch.full((batch, sessions, assets, 2), 0.25),
        context_membership=membership,
        action_mask=membership,
    )
    return model, output


def _batch() -> tuple[
    MassiveAdaptiveAlphaTermStructureModelV1,
    MassiveAdaptiveAlphaTrainingBatchV1,
]:
    model, output = _model_and_output()
    bucket_count = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
    cross_section = torch.tensor((-0.02, -0.01, 0.01, 0.02)).view(1, 1, 4, 1)
    bucket_shift = torch.linspace(0.0, 0.006, bucket_count).view(1, 1, 1, -1)
    raw = (cross_section + bucket_shift).expand(1, 4, 4, -1).clone()
    factor_component = torch.full_like(raw, 0.002)
    residual = raw - factor_component
    valid = torch.ones_like(raw, dtype=torch.bool)
    benchmark = torch.full((1, 4, 4), 0.25)
    benchmark_return = (benchmark * raw[..., 0]).sum(dim=2)
    result = MassiveAdaptiveAlphaTrainingBatchV1(
        output=output,  # type: ignore[arg-type]
        raw_return_target=raw,
        factor_component_target=factor_component,
        residual_return_target=residual,
        target_valid=valid,
        factor_return_target=torch.full((1, 4, bucket_count), 0.002),
        factor_valid=torch.ones((1, 4, bucket_count), dtype=torch.bool),
        action_mask=torch.ones((1, 4, 4), dtype=torch.bool),
        benchmark_weights=benchmark,
        benchmark_net_returns=benchmark_return,
        initial_pretrade_weights=benchmark[:, 0].clone(),
        portfolio_utility_valid=torch.ones((1, 4), dtype=torch.bool),
        origin_indices=torch.tensor(((10, 11, 12, 13),), dtype=torch.long),
        split_start_inclusive=0,
        split_stop_exclusive=200,
        split_role="training",
        source_bundle_receipt_sha256=_DIGEST,
        target_bundle_receipt_sha256="b" * 64,
        factor_operator_receipt_sha256="c" * 64,
        split_plan_receipt_sha256="d" * 64,
    )
    result.validate()
    return model, result


def test_joint_distribution_factor_and_profit_objective_backpropagates() -> None:
    model, batch = _batch()
    raw_target = batch.raw_return_target.clone().requires_grad_()
    batch = replace(batch, raw_return_target=raw_target)

    loss = massive_adaptive_alpha_supervised_loss_v1(batch)

    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.residual.total)
    assert torch.isfinite(loss.raw.total)
    assert torch.isfinite(loss.factor)
    assert torch.isfinite(loss.soft_active_log_utility)
    assert loss.soft_one_way_turnover >= 0.0
    loss.total.backward()
    assert model.residual_head.projection.weight.grad is not None
    assert model.raw_head.projection.weight.grad is not None
    assert model.factor_head.weight.grad is not None
    assert model.term_router.weight.grad is not None
    assert raw_target.grad is None


def test_profit_surrogate_rewards_scores_aligned_with_realized_returns() -> None:
    _model, batch = _batch()
    realized = batch.raw_return_target[..., 0]
    aligned = batch.output._replace(executable_score=realized.clone())
    reversed_score = batch.output._replace(executable_score=-realized.clone())

    aligned_loss = massive_adaptive_alpha_supervised_loss_v1(
        replace(batch, output=aligned)
    )
    reversed_loss = massive_adaptive_alpha_supervised_loss_v1(
        replace(batch, output=reversed_score)
    )

    assert aligned_loss.soft_active_log_utility > reversed_loss.soft_active_log_utility
    assert aligned_loss.total < reversed_loss.total


def test_real_cost_ladder_reduces_soft_net_utility_monotonically() -> None:
    _model, batch = _batch()
    values = []
    costs = []
    for cost in (0.0, 0.002, 0.004):
        config = replace(
            MASSIVE_ADAPTIVE_ALPHA_OBJECTIVE_CONFIG_V1,
            one_way_cost_return=cost,
        )
        result = massive_adaptive_alpha_supervised_loss_v1(batch, config)
        values.append(float(result.soft_active_log_utility))
        costs.append(float(result.soft_execution_cost))

    assert values[0] > values[1] > values[2]
    assert costs[0] < costs[1] < costs[2]


def test_raw_factor_and_residual_targets_must_reconcile() -> None:
    _model, batch = _batch()
    changed = batch.residual_return_target.clone()
    changed[0, 0, 0, 0] += 0.01

    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="do not reconcile",
    ):
        replace(batch, residual_return_target=changed).validate()


def test_source_target_artifacts_define_tensor_payload_and_receipts() -> None:
    _model, output = _model_and_output()
    artifacts = tuple(_target_artifact(index) for index in range(4))
    benchmark = torch.full((1, 4, 4), 0.25)
    raw_one = torch.tensor(
        tuple(
            tuple(row.raw_bucket_returns[0] for row in artifact.rows)
            for artifact in artifacts
        )
    ).unsqueeze(0)
    batch = build_massive_adaptive_alpha_training_batch_from_targets_v1(
        output=output,  # type: ignore[arg-type]
        target_artifacts=(artifacts,),
        action_security_ids=_TARGET_ASSETS,
        benchmark_weights=benchmark,
        benchmark_net_returns=(benchmark * raw_one).sum(dim=2),
        initial_pretrade_weights=benchmark[:, 0].clone(),
        origin_indices=torch.tensor(((10, 11, 12, 13),), dtype=torch.long),
        split_start_inclusive=0,
        split_stop_exclusive=200,
        split_role="training",
        source_bundle_receipt_sha256="a" * 64,
        split_plan_receipt_sha256="b" * 64,
    )

    expected_raw = torch.tensor(
        tuple(
            tuple(row.raw_bucket_returns for row in artifact.rows)
            for artifact in artifacts
        )
    ).unsqueeze(0)
    assert torch.allclose(batch.raw_return_target, expected_raw)
    assert torch.allclose(
        batch.raw_return_target,
        batch.factor_component_target + batch.residual_return_target,
    )
    assert batch.target_bundle_receipt_sha256 == semantic_sha256(
        (tuple(artifact.semantic_receipt_sha256 for artifact in artifacts),)
    )
    assert batch.factor_operator_receipt_sha256 == semantic_sha256(
        (tuple(artifact.residual_operator.receipt_sha256 for artifact in artifacts),)
    )
    assert bool(batch.portfolio_utility_valid.all())


def test_target_adapter_rejects_duplicate_origin_artifacts() -> None:
    _model, output = _model_and_output()
    artifact = _target_artifact(0)
    benchmark = torch.full((1, 4, 4), 0.25)
    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="chronology or security axis",
    ):
        build_massive_adaptive_alpha_training_batch_from_targets_v1(
            output=output,  # type: ignore[arg-type]
            target_artifacts=((artifact, artifact, artifact, artifact),),
            action_security_ids=_TARGET_ASSETS,
            benchmark_weights=benchmark,
            benchmark_net_returns=torch.zeros(1, 4),
            initial_pretrade_weights=benchmark[:, 0].clone(),
            origin_indices=torch.tensor(((10, 11, 12, 13),), dtype=torch.long),
            split_start_inclusive=0,
            split_stop_exclusive=200,
            split_role="training",
            source_bundle_receipt_sha256="a" * 64,
            split_plan_receipt_sha256="b" * 64,
        )


def test_targets_cannot_cross_the_frozen_split() -> None:
    _model, batch = _batch()

    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="crosses its frozen split",
    ):
        replace(batch, split_stop_exclusive=100).validate()


def test_soft_portfolio_cannot_drop_a_missing_outcome_ex_post() -> None:
    _model, batch = _batch()
    valid = batch.target_valid.clone()
    valid[0, 1, 0, 0] = False
    raw = batch.raw_return_target.clone()
    factor = batch.factor_component_target.clone()
    residual = batch.residual_return_target.clone()
    raw[0, 1, 0, 0] = 0.0
    factor[0, 1, 0, 0] = 0.0
    residual[0, 1, 0, 0] = 0.0

    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="incomplete one-session outcomes",
    ):
        replace(
            batch,
            raw_return_target=raw,
            factor_component_target=factor,
            residual_return_target=residual,
            target_valid=valid,
        ).validate()


def test_invalid_target_payload_and_action_support_fail_closed() -> None:
    _model, batch = _batch()
    valid = batch.target_valid.clone()
    valid[0, 0, 0, 1] = False
    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="noncanonical missing payload",
    ):
        replace(batch, target_valid=valid).validate()

    changed_action = batch.action_mask.clone()
    changed_action[0, 0, 0] = False
    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="action or target support",
    ):
        replace(batch, action_mask=changed_action).validate()


def test_executable_score_and_bucket_router_cannot_be_substituted() -> None:
    _model, batch = _batch()
    changed_score = batch.output.executable_score.clone()
    changed_score[0, 0, 0] = torch.nan
    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="executable-score",
    ):
        replace(
            batch,
            output=batch.output._replace(executable_score=changed_score),
        ).validate()

    changed_router = batch.output.bucket_router_weights.clone()
    changed_router[0, 0, 0] = 0.0
    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="executable-score",
    ):
        replace(
            batch,
            output=batch.output._replace(bucket_router_weights=changed_router),
        ).validate()


def test_outer_and_lockbox_inputs_are_rejected() -> None:
    _model, batch = _batch()
    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="development splits only",
    ):
        replace(batch, outer_test_accessed=True).validate()
    with pytest.raises(
        MassiveAdaptiveAlphaSupervisedV1Error,
        match="development splits only",
    ):
        replace(batch, lockbox_accessed=True).validate()


def test_objective_has_no_duration_semantics_or_forbidden_imports() -> None:
    config = MASSIVE_ADAPTIVE_ALPHA_OBJECTIVE_CONFIG_V1
    config.validate()
    assert len(config.receipt_sha256) == 64
    assert_no_adaptive_hold_semantics(config)
    assert_adaptive_import_firewall((Path(objective_module.__file__),))
    assert not config.economic_training_authorized
    assert not config.outer_evaluation_authorized
    assert not config.profitability_reporting_authorized
    assert not config.lockbox_access_authorized
    assert not config.reinforcement_learning_authorized
    forbidden_fragments = ("age", "duration", "persistence", "scheduled_exit")
    assert all(
        not any(fragment in field.name for fragment in forbidden_fragments)
        for field in fields(MassiveAdaptiveAlphaObjectiveConfigV1)
    )
