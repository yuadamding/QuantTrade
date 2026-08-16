from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_SETTINGS,
    M03R_V16_SURVIVAL_WEIGHTS,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    qualify_m03r_v9_origin_risk_exposures,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v15_residual_operator import (
    apply_m03r_v15_residual_operator,
)
from rl_quant.training.top2000_m03r_v16_checkpoint import (
    M03RV16LoadedEpochCheckpoint,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
    render_m03r_v16_training_update_plan,
)
from rl_quant.training.top2000_m03r_v16_pretraining_optimizer import (
    build_m03r_v16_optimizer,
)
from rl_quant.training.top2000_m03r_v16_pretraining_runtime import (
    M03RV16PretrainingRuntimeError,
    build_m03r_v16_batch_from_origin_states,
    m03r_v16_selection_weights,
)
from rl_quant.training.top2000_m03r_v16_pretraining_step import (
    train_m03r_v16_score_batch_update,
)
from rl_quant.training.top2000_m03r_v16_validation_runtime import (
    M03RV16ValidationError,
    evaluate_m03r_v16_inner_validation_batch,
)


def _surfaces() -> tuple[Hold30Sequence, object]:
    states = 345
    assets = 14
    initial = torch.zeros((1, assets), dtype=torch.float32)
    initial[0, 0] = 0.87
    initial[0, 1:] = 0.01
    availability = torch.ones((states, 1, assets), dtype=torch.bool)
    availability[252:283, 0, 12] = False
    availability[251, 0, 11] = False
    returns = torch.zeros((states - 1, 1, assets), dtype=torch.float32)
    returns[:, 0, 1:] = torch.linspace(-0.001, 0.001, assets - 1)
    sequence = Hold30Sequence(
        decision_state=torch.zeros((states, 1, assets, 1)),
        asset_returns=returns,
        decision_available=availability,
        fill_membership=availability.clone(),
        fill_availability=availability.clone(),
        benchmark_weights=initial.unsqueeze(0).expand(states, -1, -1).clone(),
        risk_asset_caps=torch.ones((states, 1, assets)),
        risk_gross_max=torch.ones((states, 1)),
        benchmark_net_returns=torch.zeros((states - 1, 1)),
        initial_ledger=CohortLedger.from_weights(
            initial, cash_index=0, initial_age=0, track_initial_units=True
        ),
        cost_rate=0.002,
        axis_id="v16-runtime-test",
    )
    exposure_states = 1001
    loadings = torch.zeros((exposure_states, assets, 6), dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, assets - 1, dtype=torch.float64)
    loadings[:, 1:, 0] = 1.0
    loadings[:, 1:7, 1] = 1.0
    loadings[:, 7:, 2] = 1.0
    loadings[:, 1:, 3] = x
    loadings[:, 1:, 4] = x.square()
    loadings[:, 1:, 5] = x.pow(3)
    weights = torch.ones((exposure_states, assets), dtype=torch.float64)
    weights[:, 0] = 0.0
    decision_time = torch.arange(exposure_states, dtype=torch.int64) * 86_400_000
    exposure = qualify_m03r_v9_origin_risk_exposures(
        state_start_index=0,
        cash_index=0,
        projector_exposure_names=(
            "sector-a",
            "sector-b",
            "active-beta",
            "style-return",
            "style-volatility",
        ),
        projector_exposure_families=(
            "sector",
            "sector",
            "active-beta",
            "style-risk",
            "style-risk",
        ),
        asset_axis_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
        exposure_loadings=loadings,
        regression_weights=weights,
        decision_timestamp_ms=decision_time,
        exposure_available_timestamp_ms=decision_time[:, None, None]
        .expand(exposure_states, assets, 3)
        .clone(),
    )
    return sequence, exposure


def test_v16_target_weights_preserve_declared_economic_units() -> None:
    h21 = m03r_v16_selection_weights(M03R_V16_SETTINGS[0])
    h30 = m03r_v16_selection_weights(M03R_V16_SETTINGS[1])
    survival = m03r_v16_selection_weights(M03R_V16_SETTINGS[2])
    assert torch.equal(h21, torch.ones(21, dtype=torch.float64))
    assert torch.equal(h30, torch.ones(30, dtype=torch.float64))
    assert tuple(float(value) for value in survival) == pytest.approx(
        M03R_V16_SURVIVAL_WEIGHTS
    )
    assert float(survival.sum()) > 20.0


def test_v16_survival_weighting_is_not_a_mandatory_expiry() -> None:
    survival = m03r_v16_selection_weights(M03R_V16_SETTINGS[2])
    assert bool((survival > 0.0).all())
    assert float(survival[-1]) > 0.0
    assert float(survival[0]) > float(survival[-1])


def test_v16_qualification_authority_recomputes_checkpoint_score_projection() -> None:
    import rl_quant.training.top2000_m03r_v16_qualification_runtime as runtime

    sequence, exposure = _surfaces()
    policy = Top2000M03RV16PredictivePolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    origins = torch.arange(251, 314, dtype=torch.long)
    batch = build_m03r_v16_batch_from_origin_states(
        policy,
        M03R_V16_SETTINGS[0],
        torch.randn((63, 1, sequence.num_assets, 16)),
        sequence,
        origins,
        sequence_global_state_start=284,
        split="qualification",
        split_start_inclusive=535,
        split_stop_exclusive=629,
        fold_index=0,
        source_array_sha256="c" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,  # type: ignore[arg-type]
    )
    checkpoint = M03RV16LoadedEpochCheckpoint(
        setting_index=0,
        setting_id=M03R_V16_SETTINGS[0].setting_id,
        fold_index=0,
        epoch_index=7,
        completed_score_updates=24,
        model_state_sha256=model_state_sha256(policy),
        score_component_state_sha256="1" * 64,
        checkpoint_file_sha256="2" * 64,
        panel_schedule_sha256="3" * 64,
        selection_target_operator_root_sha256="4" * 64,
        action_operator_root_sha256="5" * 64,
        source_array_sha256="c" * 64,
        asset_axis_sha256="a" * 64,
        head_identity=policy.v16_head_identity(),
    )

    class _DeviceOperator:
        def __init__(self, operator: Any) -> None:
            self.operator = operator

        def apply(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            residual = apply_m03r_v15_residual_operator(value, self.operator).residual
            return residual, residual.new_zeros(())

    class _Slab:
        def device_origin(self, origin: int, _device: torch.device) -> object:
            row = int(origin - int(batch.origin_indices[0]))
            return SimpleNamespace(
                action_operator=_DeviceOperator(batch.action_operators[row])
            )

    authority = runtime._issue_score_authority(
        checkpoint,
        batch,
        torch.randn((63, 1, sequence.num_assets, 16)),
        cast(Any, _Slab()),
    )
    authority.validate()
    assert authority.batch is batch
    tampered_objective = replace(
        batch.objective,
        executable_selection_score_z=(
            batch.objective.executable_selection_score_z + 0.01
        ),
    )
    with pytest.raises(ValueError, match="not reproduced"):
        runtime._issue_score_authority(
            checkpoint,
            replace(batch, objective=tampered_objective),
            torch.randn((63, 1, sequence.num_assets, 16)),
            cast(Any, _Slab()),
        )


def test_v16_public_qualification_api_does_not_accept_a_prebuilt_batch() -> None:
    import inspect

    from rl_quant.training.top2000_m03r_v16_qualification_runtime import (
        run_m03r_v16_fold_qualification,
    )

    assert "batch" not in inspect.signature(run_m03r_v16_fold_qualification).parameters


def test_v16_fold_qualification_rebuilds_states_scores_and_fill_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v16_qualification_runtime as runtime

    class _Validated(SimpleNamespace):
        def validate(self) -> None:
            return None

        def validate_unmodified(self) -> None:
            return None

        def require_fast_identity(self) -> None:
            return None

    geometry = render_m03r_v16_fold_geometries(1001)[0]
    sequence, _exposure = _surfaces()
    policy = Top2000M03RV16PredictivePolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    checkpoint = M03RV16LoadedEpochCheckpoint(
        setting_index=0,
        setting_id=M03R_V16_SETTINGS[0].setting_id,
        fold_index=0,
        epoch_index=7,
        completed_score_updates=24,
        model_state_sha256=model_state_sha256(policy),
        score_component_state_sha256="1" * 64,
        checkpoint_file_sha256="2" * 64,
        panel_schedule_sha256="3" * 64,
        selection_target_operator_root_sha256="4" * 64,
        action_operator_root_sha256="5" * 64,
        source_array_sha256="6" * 64,
        asset_axis_sha256="a" * 64,
        head_identity=policy.v16_head_identity(),
    )
    cache = _Validated(cache_sha256="b" * 64, action_hash="a" * 64)
    exposures = _Validated(receipt_sha256="c" * 64)
    risk_source = _Validated(
        cache_sha256="b" * 64,
        action_hash="a" * 64,
        receipt_sha256="d" * 64,
        exposures=exposures,
    )
    steps = 63 + 29
    risk_state = _Validated(
        origin_state_indices=tuple(range(535, 535 + steps)),
        asset_axis_sha256="a" * 64,
        source_exposure_receipt_sha256="c" * 64,
    )
    slab = _Validated(
        receipt=SimpleNamespace(
            cache_sha256="b" * 64,
            asset_axis_sha256="a" * 64,
            risk_source_receipt_sha256="d" * 64,
            exposure_receipt_sha256="c" * 64,
            common_target_operator_root_sha256="4" * 64,
            action_operator_root_sha256="5" * 64,
        )
    )
    built = SimpleNamespace(
        sequence=object(),
        identity=SimpleNamespace(receipt_sha256="e" * 64),
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        runtime,
        "build_top2000_hold30_development_sequence_from_loaded_cache",
        lambda *args, **kwargs: built,
    )
    monkeypatch.setattr(
        runtime,
        "move_and_bind_m03r_v16_sequence",
        lambda *args, **kwargs: sequence,
    )
    monkeypatch.setattr(runtime, "top2000_m03r_v7_decision_inputs", lambda _: object())

    class _Provider:
        def __init__(self, _inputs: object) -> None:
            return None

        def replay_origin_states(
            self,
            _source_policy: object,
            _sequence: object,
            local_origins: torch.Tensor,
        ) -> torch.Tensor:
            captured["local_origins"] = tuple(int(value) for value in local_origins)
            return torch.ones((63, 1, sequence.num_assets, 16))

    monkeypatch.setattr(runtime, "Top2000M03RV7DecisionStateProvider", _Provider)
    batch = _Validated(
        fold_index=0,
        objective=SimpleNamespace(
            setting=M03R_V16_SETTINGS[0],
            executable_selection_score_z=torch.ones((63, sequence.num_assets)),
            selection_valid=torch.ones((63, sequence.num_assets), dtype=torch.bool),
        ),
        action_valid=torch.ones((63, sequence.num_assets), dtype=torch.bool),
        origin_indices=torch.arange(535, 598, dtype=torch.long),
        asset_axis_sha256="a" * 64,
        receipt_sha256="7" * 64,
    )

    def _build_batch(*args: object, **kwargs: Any) -> object:
        captured["batch_kwargs"] = kwargs
        return batch

    monkeypatch.setattr(
        runtime, "build_m03r_v16_batch_from_origin_states", _build_batch
    )
    authority = _Validated(batch=batch)
    monkeypatch.setattr(runtime, "_issue_score_authority", lambda *args: authority)
    trace = _Validated(
        fold_index=0,
        setting_index=0,
        checkpoint_file_sha256="2" * 64,
        checkpoint_model_state_sha256=checkpoint.model_state_sha256,
        qualification_batch_receipt_sha256="7" * 64,
    )

    def _cohort(*args: object, **kwargs: Any) -> object:
        captured["cohort_kwargs"] = kwargs
        return trace

    monkeypatch.setattr(runtime, "run_m03r_v16_horizon_matched_cohort_sleeve", _cohort)
    result = runtime.run_m03r_v16_fold_qualification(
        cache,
        geometry,
        risk_source,
        risk_state,
        cast(Any, slab),
        policy,
        checkpoint,
        device=torch.device("cpu"),
    )
    result.validate()
    assert captured["local_origins"] == tuple(range(251, 314))
    cohort_kwargs = cast(dict[str, Any], captured["cohort_kwargs"])
    assert cohort_kwargs["post_fill_asset_returns"].shape == (
        steps,
        sequence.num_assets,
    )
    assert cohort_kwargs["benchmark_weights"].shape == (
        steps,
        sequence.num_assets,
    )


def test_v16_settings_use_identical_common30_masks_and_operators() -> None:
    sequence, exposure = _surfaces()
    state = torch.randn((1, 1, sequence.num_assets, 16))
    batches = []
    for setting_index in range(3):
        torch.manual_seed(17)
        policy = Top2000M03RV16PredictivePolicy(
            setting_index,
            token_dim=16,
            raw_stock_chunk=8,
            activation_checkpointing=False,
        )
        batches.append(
            build_m03r_v16_batch_from_origin_states(
                policy,
                M03R_V16_SETTINGS[setting_index],
                state,
                sequence,
                torch.tensor([251]),
                sequence_global_state_start=0,
                split="training",
                split_start_inclusive=0,
                split_stop_exclusive=345,
                fold_index=0,
                source_array_sha256="c" * 64,
                asset_axis_sha256="a" * 64,
                origin_risk_exposures=exposure,  # type: ignore[arg-type]
            )
        )
    assert all(
        batch.objective.executable_selection_score_z.shape == (1, 14)
        for batch in batches
    )
    assert (
        len({batch.selection_target_operators[0].receipt_sha256 for batch in batches})
        == 1
    )
    assert len({batch.action_operators[0].receipt_sha256 for batch in batches}) == 1
    assert all(
        torch.equal(
            batches[0].objective.selection_valid, batch.objective.selection_valid
        )
        for batch in batches[1:]
    )
    assert bool(batches[0].action_operators[0].qualified_asset_mask[12])
    assert not bool(batches[0].selection_target_operators[0].qualified_asset_mask[12])
    assert bool(batches[0].action_valid[0, 12])
    assert not bool(batches[0].objective.selection_valid[0, 12])
    assert not bool(
        (batches[0].objective.selection_valid & ~batches[0].action_valid).any()
    )
    assert not bool(batches[0].action_operators[0].qualified_asset_mask[11])
    assert batches[0].objective.executable_selection_score_z.grad_fn is not None
    assert max(batches[0].action_returned_dtype_exposure_errors) <= 1.0e-5


def test_v16_score_step_mutates_encoder_and_selection_head_only() -> None:
    sequence, exposure = _surfaces()
    geometries = render_m03r_v16_fold_geometries(1001)
    schedule = M03RV16PanelSchedule(
        protocol_common_data_sha256="d" * 64,
        cache_sha256="e" * 64,
        asset_axis_sha256="a" * 64,
        fold_geometry_sha256=tuple(value.receipt_sha256 for value in geometries),
    )
    geometry = geometries[0]
    plans = tuple(
        render_m03r_v16_training_update_plan(
            schedule, geometry, setting_index=0, completed_update=cursor
        )
        for cursor in range(geometry.training_block_count)
    )
    plan = next(value for value in plans if value.episode_start == 0)
    policy = Top2000M03RV16PredictivePolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    optimizer, partition = build_m03r_v16_optimizer(policy, "score")
    batch = build_m03r_v16_batch_from_origin_states(
        policy,
        M03R_V16_SETTINGS[0],
        torch.randn((len(plan.global_origins), 1, sequence.num_assets, 16)),
        sequence,
        torch.tensor(plan.global_origins),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=geometry.training_origin_start_inclusive,
        split_stop_exclusive=geometry.inner_validation_origin_start_inclusive,
        fold_index=0,
        source_array_sha256="f" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,  # type: ignore[arg-type]
    )
    receipt = train_m03r_v16_score_batch_update(
        policy,
        batch,
        optimizer,
        partition,
        plan,
        completed_updates=plan.completed_update,
        distributed_rank=0,
        distributed_world_size=1,
    )
    receipt.validate()
    assert receipt.encoder_version_root_before != receipt.encoder_version_root_after
    assert (
        receipt.selection_head_version_root_before
        != receipt.selection_head_version_root_after
    )
    assert receipt.timing_optimizer_updates == 0
    assert receipt.uncertainty_calibration_updates == 0


def test_v16_inner_validation_scores_the_executable_selection_tensor() -> None:
    sequence, exposure = _surfaces()
    geometry = render_m03r_v16_fold_geometries(1001)[0]
    policy = Top2000M03RV16PredictivePolicy(
        1,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    local_origins = torch.arange(251, 314, dtype=torch.long)
    batch = build_m03r_v16_batch_from_origin_states(
        policy,
        M03R_V16_SETTINGS[1],
        torch.randn((63, 1, sequence.num_assets, 16)),
        sequence,
        local_origins,
        sequence_global_state_start=160,
        split="inner_validation",
        split_start_inclusive=geometry.inner_validation_origin_start_inclusive,
        split_stop_exclusive=geometry.training_target_stop_exclusive,
        fold_index=0,
        source_array_sha256="f" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,  # type: ignore[arg-type]
    )
    receipt = evaluate_m03r_v16_inner_validation_batch(
        batch,
        geometry,
        epoch_index=0,
        completed_score_updates=geometry.training_block_count,
        model_state_sha256=batch.policy_state_binding_sha256,
        epoch_checkpoint_file_sha256="2" * 64,
    )
    receipt.validate()
    assert receipt.origin_count == 63
    assert receipt.setting_index == 1
    assert receipt.batch_receipt_sha256 == batch.receipt_sha256
    with pytest.raises(M03RV16ValidationError, match="batch or epoch"):
        evaluate_m03r_v16_inner_validation_batch(
            batch,
            geometry,
            epoch_index=0,
            completed_score_updates=geometry.training_block_count,
            model_state_sha256="9" * 64,
            epoch_checkpoint_file_sha256="2" * 64,
        )


def test_v16_validation_batch_rejects_training_state_binding() -> None:
    sequence, exposure = _surfaces()
    policy = Top2000M03RV16PredictivePolicy(
        0, token_dim=16, raw_stock_chunk=8, activation_checkpointing=False
    )
    batch = build_m03r_v16_batch_from_origin_states(
        policy,
        M03R_V16_SETTINGS[0],
        torch.randn((1, 1, sequence.num_assets, 16)),
        sequence,
        torch.tensor([251]),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=345,
        fold_index=0,
        source_array_sha256="c" * 64,
        asset_axis_sha256="a" * 64,
        origin_risk_exposures=exposure,  # type: ignore[arg-type]
    )
    with pytest.raises(M03RV16PretrainingRuntimeError, match="built batch drifted"):
        replace(
            batch,
            split="inner_validation",
            policy_state_binding_kind="parameter-version-root",
        ).validate()
