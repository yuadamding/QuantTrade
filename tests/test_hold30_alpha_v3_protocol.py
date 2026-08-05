"""Frozen identities and manifest schema for Hold-30 alpha V3."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, timedelta

import pytest

from rl_quant.models.daily_policy import (
    HOLD30_ALPHA_MODEL_SETTING_IDS,
    HOLD30_MODEL_SETTING_IDS,
    resolve_hold30_model_switches,
)
from rl_quant.protocol.hold30 import HOLD30_MECH8_IDS as V2_SETTING_IDS
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_C1_BENCHMARK_ID,
    HOLD30_ALPHA_C1_USAGE,
    HOLD30_ALPHA_HORIZONS,
    HOLD30_ALPHA_MECH8_BY_ID,
    HOLD30_ALPHA_MECH8_IDS,
    HOLD30_ALPHA_PROTOCOL_GENERATION,
    HOLD30_ALPHA_V3_CANONICAL_ID,
    HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
    HOLD30_ALPHA_V3_DESIGN,
    HOLD30_ALPHA_V3_SUPERSEDED_GENERATION,
    HOLD30_ALPHA_VALIDATION_COSTS_BPS,
    Hold30AlphaV3ProtocolError,
    hold30_alpha_v3_design_payload,
    resolve_hold30_alpha_setting,
    validate_hold30_alpha_v3_artifact_identity,
)
from rl_quant.protocol.hold30_alpha_v3_freeze import (
    Hold30AlphaV3DataContract,
    Hold30AlphaV3FreezeBindings,
    Hold30AlphaV3FreezeError,
    hold30_alpha_v3_trial_inventory,
    render_hold30_alpha_v3_manifest,
)
from rl_quant.protocol.hold30_freeze import (
    HOLD30_MIN_AXIS_POSITIONS,
    render_hold30_folds,
    sha256_payload,
)
from rl_quant.training.hold30_alpha import Hold30AlphaObjectiveConfig
from rl_quant.training.hold30_alpha_plan import (
    Hold30AlphaTrainingPlan,
    unresolved_hold30_alpha_training_plan,
)

EXPECTED_IDS = (
    "hold30a-m00-legacy-absolute",
    "hold30a-m01-persistent-absolute",
    "hold30a-m02-active-te",
    "hold30a-m03-alpha-core",
    "hold30a-a04-no-uncertainty",
    "hold30a-a05-no-te-floor",
    "hold30a-a06-sharpe-overlay",
    "hold30a-a07-direct-sharpe",
)


def _axis() -> tuple[str, ...]:
    start = date(2018, 1, 1)
    return tuple(
        (start + timedelta(days=index)).isoformat()
        for index in range(HOLD30_MIN_AXIS_POSITIONS)
    )


def _data_contract() -> Hold30AlphaV3DataContract:
    digest = "7" * 64
    return Hold30AlphaV3DataContract(
        protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
        source_axis_id=digest,
        training_benchmark_id=HOLD30_ALPHA_C1_BENCHMARK_ID,
        c1_usage=HOLD30_ALPHA_C1_USAGE,
        c1_trace_sha256=digest,
        panel_schema="rl-quant.hold30-alpha-evaluator-data-v3",
        provenance_receipt_id=digest,
        panel_id=digest,
        binding_receipt_id=digest,
        label_schema="rl-quant.hold30-alpha-residual-labels-v3",
        label_rule=(
            "decision-t-fills-t-plus-1;returns-t-plus-1-through-t-plus-H;"
            "forced-exit-then-explicit-cash;subtract-identical-window-C1-net-log;"
            "right-censor-at-split"
        ),
        labels_id=digest,
        horizons=(5, 21, 30, 63),
        risk_free_usage=(
            "portfolio-accounting",
            "a06-a07-total-sharpe-objective",
            "checkpoint-ranking",
            "evaluation",
        ),
        market_usage=(
            "beta-objective",
            "checkpoint-eligibility",
            "evaluation",
        ),
        factor_usage=("evaluation-only",),
        policy_feature_access=False,
        actor_access=False,
        auxiliary_only=True,
    )


def _bindings(
    axis: tuple[str, ...],
    data_contract: Hold30AlphaV3DataContract,
    training_plan: Hold30AlphaTrainingPlan,
) -> Hold30AlphaV3FreezeBindings:
    digest = "1" * 64
    folds = [asdict(fold) for fold in render_hold30_folds(axis)]
    return Hold30AlphaV3FreezeBindings(
        repository_url="ssh://example/QuantTrade.git",
        git_commit="2" * 40,
        git_tree="3" * 40,
        clean_worktree=True,
        dirty_patch_sha256=None,
        source_archive_sha256=digest,
        dependency_lock_sha256=digest,
        container_image_digest="sha256:" + digest,
        v3_rfc_sha256=digest,
        v3_adr_sha256=digest,
        v3_design_sha256=sha256_payload(
            hold30_alpha_v3_design_payload(
                checkpoint_contract=training_plan.checkpoint_contract
            )
        ),
        v3_data_contract_sha256=sha256_payload(data_contract.manifest_payload()),
        v3_checkpoint_contract_sha256=sha256_payload(
            asdict(training_plan.checkpoint_contract)
        ),
        superseded_v2_specification_sha256=digest,
        data_snapshot_sha256=digest,
        decision_axis_sha256=sha256_payload(axis),
        split_arrays_sha256=sha256_payload(folds),
        component_qualification_sha256=digest,
        software_qualification_sha256=digest,
        data_qualification_sha256=digest,
        capacity_qualification_sha256=digest,
        training_plan_sha256=training_plan.receipt_id,
        evaluation_plan_sha256=digest,
        artifact_inventory_sha256=digest,
        recovery_policy_sha256=digest,
        worker_template_sha256=digest,
        admitted_job_template_sha256=digest,
        namespace="yn-gpu-workload",
        service_account="hold30-alpha-v3-runner",
    )


def _synthetic_numerically_resolved_configs() -> tuple[Hold30AlphaObjectiveConfig, ...]:
    common = {
        "lambda_te_ceiling": 1.0,
        "lambda_turnover": 1.0,
        "lambda_early_exit": 1.0,
    }
    alpha = {
        **common,
        "lambda_beta": 1.0,
        "lambda_auxiliary_alpha": 1.0,
        "active_log_scale_bounds": (-1.0, 1.0),
        "auxiliary_horizon_weights": (0.15, 0.20, 0.50, 0.15),
        "auxiliary_horizon_scales": (1.0, 1.0, 1.0, 1.0),
    }
    uncertain = {
        **alpha,
        "downside_penalty_kappa": 1.0,
        "lambda_uncertainty": 1.0,
        "uncertainty_log_scale_bounds": (-4.0, 2.0),
    }
    return (
        Hold30AlphaObjectiveConfig(
            setting_id=EXPECTED_IDS[2],
            lambda_te_floor=1.0,
            **common,
        ),
        Hold30AlphaObjectiveConfig(
            setting_id=EXPECTED_IDS[3],
            lambda_te_floor=1.0,
            **uncertain,
        ),
        Hold30AlphaObjectiveConfig(
            setting_id=EXPECTED_IDS[4],
            lambda_te_floor=1.0,
            **alpha,
        ),
        Hold30AlphaObjectiveConfig(setting_id=EXPECTED_IDS[5], **uncertain),
        Hold30AlphaObjectiveConfig(
            setting_id=EXPECTED_IDS[6],
            lambda_te_floor=1.0,
            lambda_total_excess_mean=1.0,
            lambda_total_sharpe_overlay=1.0,
            total_sharpe_epsilon=1e-6,
            lambda_volatility_ratio=1.0,
            target_volatility_ratio=1.0,
            lambda_drawdown=1.0,
            drawdown_limit=1.0,
            a06_total_risk_step=1.0,
            alpha_core_parameter_selector="alpha-core-only",
            overlay_parameter_selector="a06-overlay-only",
            stop_gradient_core_to_overlay=True,
            stop_gradient_overlay_to_core=True,
            separate_optimizer_spec_receipt_sha256="9" * 64,
            **uncertain,
        ),
        Hold30AlphaObjectiveConfig(
            setting_id=EXPECTED_IDS[7],
            lambda_te_floor=1.0,
            lambda_direct_sharpe=1.0,
            direct_sharpe_epsilon=1e-6,
            **uncertain,
        ),
    )


def test_v3_design_freezes_alpha_risk_cost_and_checkpoint_fields() -> None:
    assert HOLD30_ALPHA_PROTOCOL_GENERATION == "prelockbox-hold30-alpha-mech8-v3"
    assert HOLD30_ALPHA_V3_SUPERSEDED_GENERATION == "prelockbox-hold30-mech8-v2"
    assert HOLD30_ALPHA_V3_DESIGN.alpha_horizons == HOLD30_ALPHA_HORIZONS == (
        5,
        21,
        30,
        63,
    )
    assert HOLD30_ALPHA_V3_DESIGN.primary_alpha_horizon == 30
    assert HOLD30_ALPHA_V3_DESIGN.training_benchmark_id == HOLD30_ALPHA_C1_BENCHMARK_ID
    assert HOLD30_ALPHA_V3_DESIGN.training_cost_bps == 20
    assert HOLD30_ALPHA_VALIDATION_COSTS_BPS == (10, 20, 40)
    assert (
        HOLD30_ALPHA_V3_DESIGN.te_min_annual,
        HOLD30_ALPHA_V3_DESIGN.te_target_annual,
        HOLD30_ALPHA_V3_DESIGN.te_max_annual,
    ) == (0.02, 0.04, 0.06)
    assert (
        HOLD30_ALPHA_V3_DESIGN.beta_target,
        HOLD30_ALPHA_V3_DESIGN.beta_tolerance,
    ) == (1.0, 0.1)
    assert HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT.validation_every_updates == 8
    assert HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT.shared_six_fold_five_seed_update
    assert HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT.required_active_costs_bps == (20, 40)
    assert HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT.eligibility_order[:5] == (
        "complete-coverage",
        "active-return-available-at-20bp-and-40bp",
        "annual-tracking-error-in-[0.02,0.06]",
        "market-beta-in-[0.9,1.1]",
        "median-discretionary-sold-age-in-[20,40]",
    )
    assert HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT.ranking_order[:5] == (
        "median-20bp-active-return-across-folds-and-seeds-desc",
        "20bp-active-information-ratio-across-folds-and-seeds-desc",
        "20bp-total-sharpe-across-folds-and-seeds-desc",
        "20bp-maximum-drawdown-across-folds-and-seeds-asc",
        "20bp-turnover-and-cost-across-folds-and-seeds-asc",
    )
    assert not HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT.result_moving_thresholds_complete


def test_v3_exact_eight_settings_and_sole_promotion_candidate() -> None:
    assert HOLD30_ALPHA_MECH8_IDS == EXPECTED_IDS
    assert tuple(HOLD30_ALPHA_MECH8_BY_ID) == EXPECTED_IDS
    eligible = [row.setting_id for row in HOLD30_ALPHA_MECH8_BY_ID.values() if row.promotion_eligible]
    assert eligible == [HOLD30_ALPHA_V3_CANONICAL_ID]

    assert HOLD30_ALPHA_MECH8_BY_ID[EXPECTED_IDS[0]].objective_mode == (
        "absolute-net-log-return"
    )
    assert HOLD30_ALPHA_MECH8_BY_ID[EXPECTED_IDS[1]].age_aware
    assert HOLD30_ALPHA_MECH8_BY_ID[EXPECTED_IDS[2]].te_band_mode == "min-target-max"
    canonical = HOLD30_ALPHA_MECH8_BY_ID[HOLD30_ALPHA_V3_CANONICAL_ID]
    assert not any(
        HOLD30_ALPHA_MECH8_BY_ID[setting_id].supervised_residual_alpha_heads
        for setting_id in EXPECTED_IDS[:3]
    )
    assert all(
        HOLD30_ALPHA_MECH8_BY_ID[setting_id].supervised_residual_alpha_heads
        for setting_id in EXPECTED_IDS[3:]
    )
    assert canonical.uncertainty_downside_heads
    assert canonical.beta_band == pytest.approx((0.9, 1.1))

    no_head = HOLD30_ALPHA_MECH8_BY_ID[EXPECTED_IDS[4]]
    assert no_head.supervised_residual_alpha_heads
    assert not no_head.uncertainty_downside_heads
    no_floor = HOLD30_ALPHA_MECH8_BY_ID[EXPECTED_IDS[5]]
    assert no_floor.te_floor_annual is None
    assert no_floor.te_target_annual == pytest.approx(0.04)
    assert no_floor.te_ceiling_annual == pytest.approx(0.06)
    assert HOLD30_ALPHA_MECH8_BY_ID[EXPECTED_IDS[6]].sharpe_mode == (
        "separate-total-risk-overlay"
    )
    assert HOLD30_ALPHA_MECH8_BY_ID[EXPECTED_IDS[7]].sharpe_mode == (
        "direct-two-pass-gradient"
    )


def test_v2_and_v3_model_setting_inventories_remain_disjoint() -> None:
    assert HOLD30_MODEL_SETTING_IDS == V2_SETTING_IDS
    assert HOLD30_ALPHA_MODEL_SETTING_IDS == EXPECTED_IDS
    assert not resolve_hold30_model_switches(EXPECTED_IDS[2]).use_alpha_head
    assert all(
        resolve_hold30_model_switches(setting_id).use_alpha_head
        for setting_id in EXPECTED_IDS[3:]
    )


def test_v3_artifact_identity_rejects_every_v2_generation_and_setting_id() -> None:
    for setting_id in V2_SETTING_IDS:
        with pytest.raises(Hold30AlphaV3ProtocolError, match="V2 setting ID"):
            resolve_hold30_alpha_setting(setting_id)
        with pytest.raises(Hold30AlphaV3ProtocolError, match="superseded before launch"):
            validate_hold30_alpha_v3_artifact_identity(
                protocol_generation=HOLD30_ALPHA_V3_SUPERSEDED_GENERATION,
                setting_id=setting_id,
            )


def test_v3_inventory_is_exactly_eight_by_six_by_five() -> None:
    inventory = hold30_alpha_v3_trial_inventory()
    assert len(inventory) == 240
    assert {row["protocol_generation"] for row in inventory} == {
        HOLD30_ALPHA_PROTOCOL_GENERATION
    }
    assert tuple(dict.fromkeys(row["setting_id"] for row in inventory)) == EXPECTED_IDS
    assert {row["fold_index"] for row in inventory} == set(range(6))
    assert {row["seed"] for row in inventory} == {17, 29, 43, 71, 101}


def test_v3_manifest_binds_design_data_and_disjoint_identity() -> None:
    axis = _axis()
    data_contract = _data_contract()
    training_plan = unresolved_hold30_alpha_training_plan()
    manifest = render_hold30_alpha_v3_manifest(
        axis,
        _bindings(axis, data_contract, training_plan),
        data_contract,
        training_plan,
        protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
        setting_ids=HOLD30_ALPHA_MECH8_IDS,
        approval_state="software_qualified",
    )
    assert manifest["schema_version"] == 3
    assert manifest["protocol_generation"] == HOLD30_ALPHA_PROTOCOL_GENERATION
    assert manifest["superseded_before_launch"] is True
    assert manifest["v2_artifacts_reusable_as_implementation_history_only"] is True
    assert manifest["trial_inventory_count"] == 240
    assert manifest["render_grants_launch_authority"] is False
    assert manifest["compute"]["world_size_per_trial"] == 2
    assert manifest["compute"]["local_paths_per_rank"] == 1
    assert manifest["compute"]["rank_sharding"] == "distinct-global-paths"
    assert manifest["compute"]["effective_paths_per_trial"] == 2
    assert manifest["data_contract"] == data_contract.manifest_payload()
    assert manifest["training_plan"]["resolved_for_executable"] is False
    assert manifest["checkpoint_contract_source"] == "typed-training-plan"
    assert (
        manifest["design"]["design"]["checkpoint"]
        == manifest["training_plan"]["checkpoint_contract"]
        == manifest["checkpoint_contract"]
    )
    assert [
        row["setting_id"] for row in manifest["training_plan"]["objective_configs"]
    ] == list(EXPECTED_IDS[2:])
    a06_config = manifest["training_plan"]["objective_configs"][4]
    assert a06_config["setting_id"] == "hold30a-a06-sharpe-overlay"
    assert a06_config["alpha_core_parameter_selector"] is None
    assert a06_config["overlay_parameter_selector"] is None
    assert a06_config["stop_gradient_core_to_overlay"] is None
    assert a06_config["stop_gradient_overlay_to_core"] is None
    assert a06_config["separate_optimizer_spec_receipt_sha256"] is None
    external = manifest["data_contract"]["external_return_data"]
    assert external["risk_free_usage"] == [
        "portfolio-accounting",
        "a06-a07-total-sharpe-objective",
        "checkpoint-ranking",
        "evaluation",
    ]
    assert external["market_usage"] == [
        "beta-objective",
        "checkpoint-eligibility",
        "evaluation",
    ]
    assert external["factor_usage"] == ["evaluation-only"]
    assert external["policy_feature_access"] is False


def test_v3_manifest_uses_one_plan_checkpoint_contract_without_stale_thresholds() -> None:
    checkpoint = replace(
        HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
        projection_distance_max=0.125,
        forced_turnover_fraction_max=0.25,
    )
    base_plan = unresolved_hold30_alpha_training_plan()
    training_plan = replace(base_plan, checkpoint_contract=checkpoint)
    axis = _axis()
    data_contract = _data_contract()
    bindings = _bindings(axis, data_contract, training_plan)

    manifest = render_hold30_alpha_v3_manifest(
        axis,
        bindings,
        data_contract,
        training_plan,
        protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
        setting_ids=HOLD30_ALPHA_MECH8_IDS,
        approval_state="software_qualified",
    )
    authoritative = manifest["checkpoint_contract"]
    assert authoritative["projection_distance_max"] == pytest.approx(0.125)
    assert authoritative["forced_turnover_fraction_max"] == pytest.approx(0.25)
    assert manifest["design"]["design"]["checkpoint"] == authoritative
    assert manifest["training_plan"]["checkpoint_contract"] == authoritative

    stale_design_digest = sha256_payload(hold30_alpha_v3_design_payload())
    with pytest.raises(Hold30AlphaV3FreezeError, match="design digest"):
        render_hold30_alpha_v3_manifest(
            axis,
            replace(bindings, v3_design_sha256=stale_design_digest),
            data_contract,
            training_plan,
            protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
            setting_ids=HOLD30_ALPHA_MECH8_IDS,
            approval_state="software_qualified",
        )


def test_v3_manifest_rejects_v2_ids_and_stale_data_contract() -> None:
    axis = _axis()
    data_contract = _data_contract()
    training_plan = unresolved_hold30_alpha_training_plan()
    bindings = _bindings(axis, data_contract, training_plan)
    with pytest.raises(Hold30AlphaV3FreezeError, match="V2 setting IDs"):
        render_hold30_alpha_v3_manifest(
            axis,
            bindings,
            data_contract,
            training_plan,
            protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
            setting_ids=V2_SETTING_IDS,
        )
    with pytest.raises(Hold30AlphaV3FreezeError, match="exact C1 benchmark"):
        replace(data_contract, training_benchmark_id="not-C1")
    changed = replace(data_contract, labels_id="8" * 64)
    with pytest.raises(Hold30AlphaV3FreezeError, match="data-contract digest"):
        render_hold30_alpha_v3_manifest(
            axis,
            bindings,
            changed,
            training_plan,
            protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
            setting_ids=HOLD30_ALPHA_MECH8_IDS,
        )
    untyped = data_contract.manifest_payload()
    with pytest.raises(Hold30AlphaV3FreezeError, match="typed Hold30AlphaV3DataContract"):
        render_hold30_alpha_v3_manifest(
            axis,
            bindings,
            untyped,  # type: ignore[arg-type]
            training_plan,
            protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
            setting_ids=HOLD30_ALPHA_MECH8_IDS,
        )
    with pytest.raises(Hold30AlphaV3FreezeError, match="typed Hold30AlphaTrainingPlan"):
        render_hold30_alpha_v3_manifest(
            axis,
            bindings,
            data_contract,
            training_plan.manifest_payload(),  # type: ignore[arg-type]
            protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
            setting_ids=HOLD30_ALPHA_MECH8_IDS,
        )
    with pytest.raises(Hold30AlphaV3FreezeError, match="design digest"):
        render_hold30_alpha_v3_manifest(
            axis,
            replace(bindings, v3_design_sha256="0" * 64),
            data_contract,
            training_plan,
            protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
            setting_ids=HOLD30_ALPHA_MECH8_IDS,
        )
    with pytest.raises(Hold30AlphaV3FreezeError, match="typed training-plan digest"):
        render_hold30_alpha_v3_manifest(
            axis,
            replace(bindings, training_plan_sha256="0" * 64),
            data_contract,
            training_plan,
            protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
            setting_ids=HOLD30_ALPHA_MECH8_IDS,
        )


def test_v3_executable_manifest_is_blocked_by_unfrozen_result_moving_thresholds() -> None:
    axis = _axis()
    data_contract = _data_contract()
    training_plan = unresolved_hold30_alpha_training_plan()
    bindings = replace(
        _bindings(axis, data_contract, training_plan),
        executable_approval_sha256="4" * 64,
    )
    with pytest.raises(Hold30AlphaV3FreezeError, match="typed training plan is unresolved"):
        render_hold30_alpha_v3_manifest(
            axis,
            bindings,
            data_contract,
            training_plan,
            protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
            setting_ids=HOLD30_ALPHA_MECH8_IDS,
            approval_state="executable",
        )


def test_typed_plan_and_manifest_accept_resolved_a06_optimizer_contract() -> None:
    checkpoint = replace(
        HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
        projection_distance_max=0.1,
        forced_turnover_fraction_max=0.1,
    )
    plan = Hold30AlphaTrainingPlan(
        objective_configs=_synthetic_numerically_resolved_configs(),
        checkpoint_contract=checkpoint,
        scientific_decision_receipt_sha256="8" * 64,
    )
    assert plan.resolved_for_executable
    plan.require_resolved()

    axis = _axis()
    data_contract = _data_contract()
    bindings = replace(
        _bindings(axis, data_contract, plan),
        executable_approval_sha256="4" * 64,
    )
    manifest = render_hold30_alpha_v3_manifest(
        axis,
        bindings,
        data_contract,
        plan,
        protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
        setting_ids=HOLD30_ALPHA_MECH8_IDS,
        approval_state="executable",
    )
    assert manifest["approval_state"] == "executable"
    assert manifest["render_grants_launch_authority"] is False
    assert manifest["training_plan"]["resolved_for_executable"] is True
