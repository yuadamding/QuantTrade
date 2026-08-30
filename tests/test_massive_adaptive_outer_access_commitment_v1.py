from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v1 import (
    MassiveAdaptiveOuterAccessCommitmentV1Error,
    authorize_massive_adaptive_outer_access_commitment_v1,
    materialize_massive_adaptive_outer_access_commitment_v1,
    parse_massive_adaptive_outer_access_commitment_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_forecast_archive_v1 import (
    MassiveAdaptiveRLOuterForecastArchiveV1Error,
    materialize_massive_adaptive_rl_outer_forecast_archive_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    build_massive_adaptive_rl_fixed_control_registry_v1,
    registered_massive_adaptive_rl_constant_actions_v1,
)


def _d(value: object) -> str:
    return semantic_sha256(value)


def _fixture():
    fold = 0
    dates = ("2024-07-01", "2024-07-02")
    outer_plan_receipt = _d("outer-inference")
    selected_supervised = _d("selected-supervised")
    supervised_state = _d("supervised-state")
    outer_plan = SimpleNamespace(
        validate=lambda: None,
        fold_index=fold,
        rows=tuple(SimpleNamespace(decision_session_date=date) for date in dates),
        semantic_receipt_sha256=outer_plan_receipt,
        selected_checkpoint_receipt_sha256=selected_supervised,
        outer_inference_authorized=True,
    )
    calibration = SimpleNamespace(
        validate=lambda: None,
        fold_index=fold,
        checkpoint_receipt_sha256=selected_supervised,
        model_state_receipt_sha256=supervised_state,
        semantic_receipt_sha256=_d("calibration"),
        development_calibration_authorized=True,
    )
    policy_selection = SimpleNamespace(
        fold_index=fold,
        semantic_receipt_sha256=_d("policy-selection"),
        selected_checkpoint_receipt_sha256=_d("ppo-checkpoint"),
    )
    policy_selection_authority = SimpleNamespace(
        validate=lambda: None,
        runtime_selection=policy_selection,
        runtime_selection_replayed=True,
        semantic_receipt_sha256=_d("policy-selection-authority"),
        outer_evaluation_authorized=True,
    )
    frozen_policy = SimpleNamespace(
        validate=lambda: None,
        fold_index=fold,
        policy_selection_authority_receipt_sha256=(
            policy_selection_authority.semantic_receipt_sha256
        ),
        policy_selection_receipt_sha256=policy_selection.semantic_receipt_sha256,
        selected_rl_checkpoint_receipt_sha256=(
            policy_selection.selected_checkpoint_receipt_sha256
        ),
        semantic_receipt_sha256=_d("frozen-policy"),
        frozen_model_state_receipt_sha256=_d("frozen-policy-state"),
        observation_specification_sha256=_d("observation-spec"),
        action_specification_sha256=_d("action-spec"),
        reward_specification_sha256=_d("reward-spec"),
        development_outer_policy_authorized=True,
    )
    registry = build_massive_adaptive_rl_fixed_control_registry_v1()
    actions = registered_massive_adaptive_rl_constant_actions_v1()
    candidates = tuple(
        SimpleNamespace(
            control_id=control_id,
            action_receipt_sha256=action.semantic_receipt_sha256,
            semantic_receipt_sha256=_d(("candidate", control_id)),
        )
        for control_id, action in actions
    )
    candidate_inventory = _d(tuple(row.semantic_receipt_sha256 for row in candidates))
    chronology = SimpleNamespace(
        validate=lambda: None,
        fold_index=fold,
        outer_inference_plan_receipt_sha256=outer_plan_receipt,
        outer_origin_dates=dates,
        outer_origin_inventory_sha256=_d(dates),
        rl_fit_origin_inventory_sha256=_d(("2023-01-03",)),
        semantic_receipt_sha256=_d("chronology"),
        outer_evaluation_authorized=True,
    )
    fixed_selection = SimpleNamespace(
        fold_index=fold,
        training_origin_inventory_sha256=chronology.rl_fit_origin_inventory_sha256,
        selected_control_id=actions[-1][0],
        selected_action_receipt_sha256=actions[-1][1].semantic_receipt_sha256,
        candidate_inventory_sha256=candidate_inventory,
        semantic_receipt_sha256=_d("fixed-selection"),
    )
    fixed_selection_authority = SimpleNamespace(
        validate=lambda: None,
        runtime_candidates=candidates,
        runtime_selection=fixed_selection,
        runtime_selection_replayed=True,
        semantic_receipt_sha256=_d("fixed-selection-authority"),
        development_control_selection_authorized=True,
    )
    fit_run = SimpleNamespace(
        fixed_control_registry_receipt_sha256=registry.semantic_receipt_sha256,
        chronology_authority_receipt_sha256=chronology.semantic_receipt_sha256,
        training_origin_inventory_sha256=chronology.rl_fit_origin_inventory_sha256,
        candidate_inventory_sha256=candidate_inventory,
        candidates=candidates,
    )
    fit_authority = SimpleNamespace(
        validate=lambda: None,
        runtime_fit_run=fit_run,
        runtime_fit_replayed=True,
        semantic_receipt_sha256=_d("fixed-fit-authority"),
        development_control_fit_authorized=True,
    )
    compiler = SimpleNamespace(validate=lambda: None, receipt_sha256=_d("compiler"))
    return (
        outer_plan,
        calibration,
        policy_selection_authority,
        frozen_policy,
        registry,
        fit_authority,
        fixed_selection_authority,
        chronology,
        compiler,
    )


def test_outer_access_commitment_is_create_only_and_replay_required(tmp_path) -> None:
    values = _fixture()
    commitment = materialize_massive_adaptive_outer_access_commitment_v1(
        root=tmp_path,
        artifact_id="fold0-outer-access",
        outer_inference_plan=values[0],
        calibration=values[1],
        policy_selection_authority=values[2],
        frozen_policy=values[3],
        fixed_control_registry=values[4],
        fixed_control_fit_authority=values[5],
        fixed_control_selection_authority=values[6],
        chronology_authority=values[7],
        compiler_config=values[8],
        committed_at_ms=10,
    )
    assert commitment.runtime_commitment_replayed
    assert commitment.outer_forecast_access_authorized

    generic = parse_massive_adaptive_outer_access_commitment_v1(
        root=tmp_path,
        loaded_source=commitment.loaded_source,
    )
    assert not generic.runtime_commitment_replayed
    assert not generic.outer_forecast_access_authorized
    with pytest.raises(
        MassiveAdaptiveRLOuterForecastArchiveV1Error,
        match="commitment is absent",
    ):
        materialize_massive_adaptive_rl_outer_forecast_archive_v1(
            root=tmp_path,
            artifact_id="blocked-outer-forecast",
            commitment=generic,
            checkpoint_selection=None,  # type: ignore[arg-type]
            selected_checkpoint=None,  # type: ignore[arg-type]
            training_window_plan=None,  # type: ignore[arg-type]
            outer_tensor=None,  # type: ignore[arg-type]
            outer_decision_roots=(),
            outer_plan=values[0],
            model_spec=None,  # type: ignore[arg-type]
            committed_at_ms=11,
        )

    reopened = authorize_massive_adaptive_outer_access_commitment_v1(
        root=tmp_path,
        commitment=generic,
        outer_inference_plan=values[0],
        calibration=values[1],
        policy_selection_authority=values[2],
        frozen_policy=values[3],
        fixed_control_registry=values[4],
        fixed_control_fit_authority=values[5],
        fixed_control_selection_authority=values[6],
        chronology_authority=values[7],
        compiler_config=values[8],
    )
    assert reopened.semantic_receipt_sha256 == commitment.semantic_receipt_sha256

    wrong_outer = SimpleNamespace(**vars(values[0]))
    wrong_outer.semantic_receipt_sha256 = _d("different-outer-plan")
    with pytest.raises(
        MassiveAdaptiveOuterAccessCommitmentV1Error,
        match="components differ",
    ):
        authorize_massive_adaptive_outer_access_commitment_v1(
            root=tmp_path,
            commitment=generic,
            outer_inference_plan=wrong_outer,
            calibration=values[1],
            policy_selection_authority=values[2],
            frozen_policy=values[3],
            fixed_control_registry=values[4],
            fixed_control_fit_authority=values[5],
            fixed_control_selection_authority=values[6],
            chronology_authority=values[7],
            compiler_config=values[8],
        )
