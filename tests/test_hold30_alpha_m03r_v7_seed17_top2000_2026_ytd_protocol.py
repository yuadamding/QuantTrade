"""Blocking tests for the contaminated seed-17 2026-YTD evaluation plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_CANONICAL_SETTING_ID,
    M03R_V7_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_COHORT_BOOTSTRAP_SEED_SHA256,
    M03R_SEED17_TOP2000_2026_YTD_CONTRASTS,
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT,
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256,
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID,
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION,
    M03R_SEED17_TOP2000_2026_YTD_REFERENCE_SETTING_ID,
    M03R_SEED17_TOP2000_2026_YTD_REQUIRED_METRIC_IDS,
    M03R_SEED17_TOP2000_2026_YTD_SETTING_BINDINGS,
    M03R_SEED17_TOP2000_2026_YTD_UNAVAILABLE_OPTIONAL_METRIC_IDS,
    M03RSeed17Top20002026YTDContaminationContract,
    M03RSeed17Top20002026YTDContrast,
    M03RSeed17Top20002026YTDCostContract,
    M03RSeed17Top20002026YTDEvaluationContract,
    M03RSeed17Top20002026YTDEvaluationProtocolError,
    M03RSeed17Top20002026YTDEvaluationWindow,
    M03RSeed17Top20002026YTDFactorContract,
    M03RSeed17Top20002026YTDJointBootstrapPlan,
    M03RSeed17Top20002026YTDReversalEpisodeContract,
    m03r_seed17_top2000_2026_ytd_evaluation_payload,
    resolve_m03r_seed17_top2000_2026_ytd_setting,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_DESIGN_ID,
    M03R_SEED17_TOP2000_PROTOCOL_GENERATION,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_RUNTIME_SETTING_BY_ID,
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID,
)


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_evaluation_identity_is_new_and_binds_unchanged_seed17_training() -> None:
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
    assert "2026-ytd-evaluation" in contract.protocol_generation
    assert contract.protocol_generation != M03R_SEED17_TOP2000_PROTOCOL_GENERATION
    assert contract.protocol_generation != M03R_V7_PROTOCOL_GENERATION
    assert contract.design_id == (
        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID
    )
    assert contract.design_id != M03R_SEED17_TOP2000_DESIGN_ID
    assert contract.source_training_protocol_generation == (
        M03R_SEED17_TOP2000_PROTOCOL_GENERATION
    )
    assert contract.source_training_protocol_sha256 == (
        M03R_SEED17_TOP2000_PROTOCOL_SHA256
    )
    assert contract.source_training_design_id == M03R_SEED17_TOP2000_DESIGN_ID


def test_exact_twelve_setting_map_is_nonreportable_and_nonpromotable() -> None:
    rows = M03R_SEED17_TOP2000_2026_YTD_SETTING_BINDINGS
    assert len(rows) == 12
    assert tuple(row.setting_index for row in rows) == tuple(range(12))
    assert tuple(row.seed17_setting_id for row in rows) == (
        M03R_SEED17_TOP2000_SETTING_IDS
    )
    for row in rows:
        assert row.runtime_setting_id == (
            M03R_SEED17_TOP2000_RUNTIME_SETTING_BY_ID[row.seed17_setting_id]
        )
        assert row.reviewed_v7_setting_id == (
            M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID[row.runtime_setting_id]
        )
        assert row.future_selected_universe
        assert not row.scientific_reporting_eligible
        assert not row.promotion_eligible
        assert resolve_m03r_seed17_top2000_2026_ytd_setting(
            row.seed17_setting_id
        ) == row

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="unknown seed-17",
    ):
        resolve_m03r_seed17_top2000_2026_ytd_setting(
            M03R_V7_CANONICAL_SETTING_ID
        )


def test_fold05_seed17_rank00_is_headline_and_other_folds_never_combine() -> None:
    rule = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.checkpoint_rule
    assert rule.seed == 17
    assert rule.checkpoint_writer_rank == 0
    assert rule.headline_training_fold_index == 5
    assert rule.sensitivity_training_fold_indices == (0, 1, 2, 3, 4)
    assert rule.checkpoint_hashes_frozen_before_2026_outcome_access
    assert rule.fold_05_is_only_headline
    assert rule.sensitivities_are_separate_paths
    assert not rule.fold_outputs_averaged
    assert not rule.fold_outputs_pooled
    assert not rule.fold_outputs_ensembled
    assert not rule.fold_outputs_ranked_or_selected_on_2026
    assert not rule.five_seed_ensemble
    assert not rule.retraining_on_2026_authorized
    assert not rule.recalibration_on_2026_authorized
    assert not rule.policy_update_after_2026_access_authorized

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="sole frozen headline",
    ):
        replace(rule, fold_outputs_ensembled=True)
    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="rank 00",
    ):
        replace(rule, checkpoint_writer_rank=1)


def test_scored_window_and_cost_paths_are_frozen() -> None:
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
    window = contract.window
    assert window.first_scored_date == "2026-01-02"
    assert window.last_scored_date == "2026-06-23"
    assert window.date_bounds_are_inclusive
    assert window.pre_start_causal_context_allowed
    assert window.pre_start_context_is_unscored
    assert window.post_end_outcomes_forbidden
    assert not window.calendar_year_complete
    assert window.year_to_date_only

    costs = contract.costs
    assert costs.one_way_costs_basis_points == (10, 20, 40)
    assert costs.primary_one_way_cost_basis_points == 20
    assert costs.primary_execution_mode == (
        "authoritative-20bp-closed-loop-chronological"
    )
    assert costs.sensitivity_execution_mode == (
        "reprice-frozen-20bp-executed-turnover"
    )
    assert costs.sensitivity_costs_basis_points == (10, 40)
    assert costs.primary_closed_loop_path_reused_for_cost_sensitivities

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="inclusive 2026-01-02",
    ):
        M03RSeed17Top20002026YTDEvaluationWindow(
            last_scored_date="2026-06-22"
        )
    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="20-bp closed-loop",
    ):
        M03RSeed17Top20002026YTDCostContract(
            primary_closed_loop_path_reused_for_cost_sensitivities=False
        )


def test_future_selected_contamination_can_never_be_upgraded() -> None:
    provenance = (
        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.contamination
    )
    assert provenance.dataset_built_at_utc == "2026-06-23"
    assert provenance.universe_selection_date == "2026-06-12"
    assert provenance.membership_mode == "static"
    assert provenance.future_selected_universe
    assert provenance.universe_selection_overlaps_scored_window
    assert not provenance.point_in_time_universe
    assert not provenance.delisting_history_complete
    assert provenance.retrospective_only
    assert provenance.development_only
    assert not provenance.scientific_reporting_eligible
    assert not provenance.promotion_eligible
    assert not provenance.lockbox_eligible
    assert not provenance.confirmatory_evidence_eligible
    assert not provenance.investment_performance_claim_eligible
    assert "future-selected-universe" in provenance.required_result_labels
    assert "promotion-evidence" in provenance.prohibited_claims

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="future-selected TOP2000 contamination",
    ):
        replace(provenance, promotion_eligible=True)
    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="future-selected TOP2000 contamination",
    ):
        M03RSeed17Top20002026YTDContaminationContract(
            universe_selection_date="2025-12-31"
        )


def test_official_daily_ff5_momentum_factor_family_fails_closed() -> None:
    factors = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.factors
    assert factors.source_library == "official-Kenneth-French-Data-Library"
    assert factors.five_factor_dataset_id == (
        "F-F_Research_Data_5_Factors_2x3_daily"
    )
    assert factors.momentum_dataset_id == "F-F_Momentum_Factor_daily"
    assert factors.five_factor_download_url == (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    )
    assert factors.momentum_download_url == (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Momentum_Factor_daily_CSV.zip"
    )
    assert factors.five_factor_source_columns == (
        "Mkt-RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
        "RF",
    )
    assert factors.regression_factor_columns == (
        "Mkt-RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
        "Mom",
    )
    assert factors.date_alignment == "exact-date-inner-join"
    assert factors.coverage_requirement == "every-scored-exchange-date"
    assert not factors.factor_join_may_shorten_scored_window
    assert factors.official_source_transport == (
        "package-owned-https-default-tls-v1"
    )
    assert factors.official_source_retrieval_receipt_required
    assert not factors.caller_staged_archives_are_official_evidence
    assert factors.source_containers_may_include_unused_post_end_rows
    assert factors.extraction_rule == "exact-frozen-score-dates-only"
    assert not factors.post_end_source_rows_may_enter_evaluator_arrays
    assert factors.unit_conversion == "divide-by-100"
    assert factors.missing_value_policy == "no-imputation"
    assert factors.portfolio_dependent_variable == "policy-return-minus-RF"
    assert factors.benchmark_dependent_variable == "C1-return-minus-RF"
    assert factors.active_dependent_variable == "policy-return-minus-C1-return"
    assert factors.source_receipt_required
    assert factors.coverage_receipt_required
    assert factors.exact_array_hash_receipt_required
    assert factors.incomplete_evidence_behavior == (
        "multifactor-outputs-unavailable-score-window-unchanged"
    )
    assert not factors.incomplete_evidence_may_be_replaced_by_zero_or_imputed_values

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="exact-date, non-imputed",
    ):
        M03RSeed17Top20002026YTDFactorContract(
            missing_value_policy="forward-fill"
        )
    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="exact-date, non-imputed",
    ):
        M03RSeed17Top20002026YTDFactorContract(
            caller_staged_archives_are_official_evidence=True
        )


def test_joint_bootstrap_uses_one_paired_date_schedule_per_checkpoint_fold() -> None:
    plan = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.bootstrap
    assert plan.method == "joint-date-index-circular-moving-block"
    assert plan.primary_block_length_trading_sessions == 21
    assert plan.sensitivity_block_lengths_trading_sessions == (10, 30)
    assert plan.replicate_count == 10_000
    assert plan.confidence_level == pytest.approx(0.95)
    assert plan.familywise_error_rate == pytest.approx(0.05)
    assert plan.same_date_indices_for_all_settings
    assert plan.same_date_indices_for_c1_and_factor_rows
    assert plan.same_date_indices_for_all_cost_rungs
    assert plan.same_date_indices_for_all_contrasts
    assert plan.raw_one_sided_p_value_method == (
        "null-centered-paired-bootstrap-upper-tail"
    )
    assert plan.dispersion_standard_deviation_degrees_of_freedom == 1
    assert plan.cohort_rmst_method == (
        "kaplan-meier-complete-score-origin-return-neutral-trajectories"
    )
    assert plan.cohort_rmst_resampling == (
        "joint-complete-origin-trajectory-circular-block-by-entry-date"
    )
    assert plan.cohort_rmst_block_length_origin_sessions == 21
    assert plan.cohort_rmst_bootstrap_seed_sha256 == (
        M03R_SEED17_TOP2000_2026_YTD_COHORT_BOOTSTRAP_SEED_SHA256
    )
    assert plan.same_origin_block_draws_for_all_settings
    assert plan.cohort_trajectory_receipt_required
    assert plan.date_by_age_snapshot_survival_is_descriptive_only
    assert plan.one_joint_family_per_checkpoint_fold
    assert plan.headline_and_sensitivity_families_separate
    assert not plan.checkpoint_fold_paths_pooled
    assert not plan.seed_17_treated_as_independent_market_replication

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="paired joint moving-block",
    ):
        M03RSeed17Top20002026YTDJointBootstrapPlan(
            same_date_indices_for_all_settings=False
        )

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="paired joint moving-block",
    ):
        M03RSeed17Top20002026YTDJointBootstrapPlan(
            dispersion_standard_deviation_degrees_of_freedom=0
        )


def test_all_eleven_reference_ablation_contrasts_are_present_including_a09() -> (
    None
):
    contrasts = M03R_SEED17_TOP2000_2026_YTD_CONTRASTS
    reference = M03R_SEED17_TOP2000_2026_YTD_REFERENCE_SETTING_ID
    assert len(contrasts) == 11
    assert tuple(row.contrast_index for row in contrasts) == tuple(range(11))
    covered = {
        row.subtrahend_setting_id
        if row.minuend_setting_id == reference
        else row.minuend_setting_id
        for row in contrasts
    }
    assert covered == set(M03R_SEED17_TOP2000_SETTING_IDS[1:])
    assert sum("a09-no-long-context" in row.contrast_id for row in contrasts) == 1
    p10 = contrasts[1]
    assert p10.minuend_setting_id == M03R_SEED17_TOP2000_SETTING_IDS[2]
    assert p10.subtrahend_setting_id == reference

    without_a09 = contrasts[:4] + contrasts[5:]
    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="eleven ordered causal contrasts",
    ):
        replace(M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT, contrasts=without_a09)

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="pair the reference",
    ):
        M03RSeed17Top20002026YTDContrast(
            contrast_index=0,
            contrast_id="invalid",
            minuend_setting_id=M03R_SEED17_TOP2000_SETTING_IDS[1],
            subtrahend_setting_id=M03R_SEED17_TOP2000_SETTING_IDS[2],
            causal_field="persistence_coefficient_basis_points",
        )


def test_payload_hash_and_receipt_gates_are_deterministic_and_fail_closed() -> None:
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
    assert contract.required_metric_ids == (
        M03R_SEED17_TOP2000_2026_YTD_REQUIRED_METRIC_IDS
    )
    assert contract.unavailable_optional_metric_ids == (
        M03R_SEED17_TOP2000_2026_YTD_UNAVAILABLE_OPTIONAL_METRIC_IDS
    )
    assert "reversal-episode-performance" not in contract.required_metric_ids
    assert contract.unavailable_optional_metric_ids == (
        "reversal-episode-performance",
    )
    assert contract.reversal_episodes == (
        M03RSeed17Top20002026YTDReversalEpisodeContract()
    )
    assert contract.reversal_episodes.availability == "unavailable-v1"
    assert contract.reversal_episodes.frozen_typed_preoutcome_artifact_required
    assert not contract.reversal_episodes.frozen_typed_preoutcome_artifact_implemented
    assert not contract.reversal_episodes.caller_authored_mask_accepted
    assert not contract.reversal_episodes.caller_authored_receipt_hash_accepted
    assert len(contract.required_metric_ids) == len(set(contract.required_metric_ids))
    assert contract.receipt_sha256 == _sha256(asdict(contract))
    assert contract.receipt_sha256 == (
        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
    )
    assert m03r_seed17_top2000_2026_ytd_evaluation_payload() == (
        m03r_seed17_top2000_2026_ytd_evaluation_payload()
    )
    assert contract.requires_complete_12_setting_72_cell_training_receipt
    assert contract.requires_exact_checkpoint_hashes
    assert contract.requires_exact_2026_data_manifest_hash
    assert contract.requires_exact_evaluator_source_hash
    assert not contract.training_mutation_authorized
    assert not contract.evaluation_execution_authorized_by_this_contract_alone
    assert not contract.promotion_authorized
    assert not contract.scientific_reporting_authorized

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="receipt-gated",
    ):
        M03RSeed17Top20002026YTDEvaluationContract(
            promotion_authorized=True
        )

    with pytest.raises(
        M03RSeed17Top20002026YTDEvaluationProtocolError,
        match="frozen typed pre-outcome",
    ):
        M03RSeed17Top20002026YTDReversalEpisodeContract(
            caller_authored_mask_accepted=True
        )


def test_public_evaluation_protocol_constant_matches_contract() -> None:
    assert M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION == (
        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.protocol_generation
    )
