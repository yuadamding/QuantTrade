"""Frozen retrospective 2026-YTD contract for the seed-17 TOP2000 panel.

This is a development-only evaluation identity layered on the completed
seed-17 training generation.  It neither changes that immutable training
identity nor upgrades its future-selected TOP2000 evidence.  In particular,
the evaluation is not a point-in-time lockbox, a five-seed ensemble, a
reportable performance study, or promotion evidence.

Fold 5 supplies the single predeclared headline checkpoint for every setting.
Folds 0 through 4 are evaluated as five separate checkpoint sensitivities;
fold outputs are never averaged, pooled, selected, or ensembled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_DESIGN_ID,
    M03R_SEED17_TOP2000_PROTOCOL_GENERATION,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_RUNTIME_SETTING_BY_ID,
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID,
    M03R_TOP2000_DEV_SETTINGS_BY_ID,
)

M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v7-seed17-2026-ytd-evaluation-v1"
)
M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID = (
    "daily_ohlcv_aggregated_top2000_dev_hold30_m03r_v7_seed17_2026_ytd_eval_v1"
)
M03R_SEED17_TOP2000_2026_YTD_EVALUATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-ytd-evaluation-contract-v1"
)
M03R_SEED17_TOP2000_2026_YTD_FIRST_SCORED_DATE = "2026-01-02"
M03R_SEED17_TOP2000_2026_YTD_LAST_SCORED_DATE = "2026-06-23"
M03R_SEED17_TOP2000_UNIVERSE_SELECTION_DATE = "2026-06-12"
M03R_SEED17_TOP2000_DATASET_BUILT_AT_UTC = "2026-06-23"
M03R_SEED17_TOP2000_2026_YTD_PRIMARY_COST_BASIS_POINTS = 20
M03R_SEED17_TOP2000_2026_YTD_COSTS_BASIS_POINTS = (10, 20, 40)
M03R_SEED17_TOP2000_2026_YTD_HEADLINE_FOLD_INDEX = 5
M03R_SEED17_TOP2000_2026_YTD_SENSITIVITY_FOLD_INDICES = (0, 1, 2, 3, 4)
M03R_SEED17_TOP2000_2026_YTD_REFERENCE_SETTING_ID = (
    M03R_SEED17_TOP2000_SETTING_IDS[0]
)

_BOOTSTRAP_SEED_DOMAIN = (
    b"rl-quant.top2000-dev.m03r-v7-seed17-2026-ytd-joint-bootstrap-v1"
)
M03R_SEED17_TOP2000_2026_YTD_BOOTSTRAP_SEED_SHA256 = hashlib.sha256(
    _BOOTSTRAP_SEED_DOMAIN
).hexdigest()
M03R_SEED17_TOP2000_2026_YTD_COHORT_BOOTSTRAP_SEED_SHA256 = hashlib.sha256(
    b"rl-quant.top2000-dev.m03r-v7-seed17-2026-cohort-origin-bootstrap-v1"
).hexdigest()

M03R_SEED17_TOP2000_2026_YTD_REQUIRED_METRIC_IDS = (
    "net-portfolio-return-by-cost",
    "net-active-return-versus-c1-by-cost",
    "20bp-net-active-return-bootstrap-lcb95",
    "information-ratio",
    "portfolio-sharpe",
    "c1-sharpe",
    "portfolio-minus-c1-sharpe-and-lcb95",
    "active-market-beta-and-equivalence-upper-bound",
    "portfolio-multifactor-regression",
    "benchmark-multifactor-regression",
    "active-multifactor-regression",
    "active-multifactor-alpha-and-lcb95",
    "annualized-tracking-error",
    "portfolio-and-active-maximum-drawdown",
    "discretionary-and-forced-turnover-by-cause",
    "transaction-cost-by-rung",
    "rmst60-and-censoring-aware-uncertainty",
    "notional-survival-s10-s20-s30",
    "discretionary-exit-notional-by-age",
    "forced-exit-notional-by-age-and-cause",
    "hold-continuous-exit-action-frequency",
    "continuous-hazard-quantiles-and-saturation",
    "requested-to-executed-projection-distance",
    "startup-availability-risk-repair-terminal-accounting",
)
M03R_SEED17_TOP2000_2026_YTD_UNAVAILABLE_OPTIONAL_METRIC_IDS = (
    "reversal-episode-performance",
)


class M03RSeed17Top20002026YTDEvaluationProtocolError(ValueError):
    """The retrospective evaluation identity or a frozen invariant drifted."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M03RSeed17Top20002026YTDEvaluationProtocolError(
            "2026-YTD evaluation payload is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDEvaluationWindow:
    """Inclusive scored dates; earlier causal context is never scored."""

    first_scored_date: str = M03R_SEED17_TOP2000_2026_YTD_FIRST_SCORED_DATE
    last_scored_date: str = M03R_SEED17_TOP2000_2026_YTD_LAST_SCORED_DATE
    date_bounds_are_inclusive: bool = True
    pre_start_causal_context_allowed: bool = True
    pre_start_context_is_unscored: bool = True
    post_end_outcomes_forbidden: bool = True
    same_chronological_date_grid_for_every_setting: bool = True
    calendar_year_complete: bool = False
    year_to_date_only: bool = True

    def __post_init__(self) -> None:
        try:
            first = date.fromisoformat(self.first_scored_date)
            last = date.fromisoformat(self.last_scored_date)
        except ValueError as exc:
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD scored dates must be ISO calendar dates"
            ) from exc
        if (
            first != date(2026, 1, 2)
            or last != date(2026, 6, 23)
            or first > last
            or not self.date_bounds_are_inclusive
            or not self.pre_start_causal_context_allowed
            or not self.pre_start_context_is_unscored
            or not self.post_end_outcomes_forbidden
            or not self.same_chronological_date_grid_for_every_setting
            or self.calendar_year_complete
            or not self.year_to_date_only
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "the retrospective scored window must remain inclusive 2026-01-02 "
                "through 2026-06-23 and must be labelled YTD"
            )


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDCheckpointRule:
    """One headline checkpoint plus five strictly separate sensitivities."""

    seed: int = 17
    checkpoint_writer_rank: int = 0
    headline_training_fold_index: int = (
        M03R_SEED17_TOP2000_2026_YTD_HEADLINE_FOLD_INDEX
    )
    sensitivity_training_fold_indices: tuple[int, ...] = (
        M03R_SEED17_TOP2000_2026_YTD_SENSITIVITY_FOLD_INDICES
    )
    source_checkpoint_selection_rule: str = (
        "frozen-final-optimizer-update-no-validation-selection-v1"
    )
    checkpoint_hashes_frozen_before_2026_outcome_access: bool = True
    fold_05_is_only_headline: bool = True
    sensitivities_are_separate_paths: bool = True
    fold_outputs_averaged: bool = False
    fold_outputs_pooled: bool = False
    fold_outputs_ensembled: bool = False
    fold_outputs_ranked_or_selected_on_2026: bool = False
    five_seed_ensemble: bool = False
    retraining_on_2026_authorized: bool = False
    recalibration_on_2026_authorized: bool = False
    policy_update_after_2026_access_authorized: bool = False

    def __post_init__(self) -> None:
        if (
            self.seed != 17
            or self.checkpoint_writer_rank != 0
            or self.headline_training_fold_index != 5
            or self.sensitivity_training_fold_indices != tuple(range(5))
            or self.source_checkpoint_selection_rule
            != "frozen-final-optimizer-update-no-validation-selection-v1"
            or not self.checkpoint_hashes_frozen_before_2026_outcome_access
            or not self.fold_05_is_only_headline
            or not self.sensitivities_are_separate_paths
            or self.fold_outputs_averaged
            or self.fold_outputs_pooled
            or self.fold_outputs_ensembled
            or self.fold_outputs_ranked_or_selected_on_2026
            or self.five_seed_ensemble
            or self.retraining_on_2026_authorized
            or self.recalibration_on_2026_authorized
            or self.policy_update_after_2026_access_authorized
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "fold 5 must remain the sole frozen headline checkpoint and folds "
                "0-4 must remain separate, non-ensembled sensitivities; checkpoint "
                "evidence must come from seed 17 rank 00"
            )


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDCostContract:
    """Frozen one-way evaluation-cost ladder."""

    one_way_costs_basis_points: tuple[int, ...] = (
        M03R_SEED17_TOP2000_2026_YTD_COSTS_BASIS_POINTS
    )
    primary_one_way_cost_basis_points: int = (
        M03R_SEED17_TOP2000_2026_YTD_PRIMARY_COST_BASIS_POINTS
    )
    primary_execution_mode: str = "authoritative-20bp-closed-loop-chronological"
    sensitivity_execution_mode: str = "reprice-frozen-20bp-executed-turnover"
    sensitivity_costs_basis_points: tuple[int, ...] = (10, 40)
    primary_closed_loop_path_reused_for_cost_sensitivities: bool = True
    cost_rungs_used_for_policy_or_checkpoint_selection: bool = False

    def __post_init__(self) -> None:
        if (
            self.one_way_costs_basis_points != (10, 20, 40)
            or self.primary_one_way_cost_basis_points != 20
            or self.primary_execution_mode
            != "authoritative-20bp-closed-loop-chronological"
            or self.sensitivity_execution_mode
            != "reprice-frozen-20bp-executed-turnover"
            or self.sensitivity_costs_basis_points != (10, 40)
            or not self.primary_closed_loop_path_reused_for_cost_sensitivities
            or self.cost_rungs_used_for_policy_or_checkpoint_selection
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD costs must remain 10/20/40 bp with an authoritative "
                "20-bp closed-loop path and repriced 10/40-bp sensitivities"
            )


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDFactorContract:
    """Frozen daily factor source and exact-date alignment semantics."""

    source_library: str = "official-Kenneth-French-Data-Library"
    five_factor_dataset_id: str = "F-F_Research_Data_5_Factors_2x3_daily"
    momentum_dataset_id: str = "F-F_Momentum_Factor_daily"
    five_factor_download_url: str = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    )
    momentum_download_url: str = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Momentum_Factor_daily_CSV.zip"
    )
    five_factor_source_columns: tuple[str, ...] = (
        "Mkt-RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
        "RF",
    )
    momentum_source_column: str = "Mom"
    regression_factor_columns: tuple[str, ...] = (
        "Mkt-RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
        "Mom",
    )
    risk_free_column: str = "RF"
    source_frequency: str = "daily"
    date_alignment: str = "exact-date-inner-join"
    coverage_requirement: str = "every-scored-exchange-date"
    factor_join_may_shorten_scored_window: bool = False
    official_source_transport: str = "package-owned-https-default-tls-v1"
    official_source_retrieval_receipt_required: bool = True
    caller_staged_archives_are_official_evidence: bool = False
    source_containers_may_include_unused_post_end_rows: bool = True
    extraction_rule: str = "exact-frozen-score-dates-only"
    post_end_source_rows_may_enter_evaluator_arrays: bool = False
    source_unit: str = "percent"
    evaluator_unit: str = "decimal-return"
    unit_conversion: str = "divide-by-100"
    missing_value_policy: str = "no-imputation"
    portfolio_dependent_variable: str = "policy-return-minus-RF"
    benchmark_dependent_variable: str = "C1-return-minus-RF"
    active_dependent_variable: str = "policy-return-minus-C1-return"
    source_receipt_required: bool = True
    coverage_receipt_required: bool = True
    exact_array_hash_receipt_required: bool = True
    incomplete_evidence_behavior: str = (
        "multifactor-outputs-unavailable-score-window-unchanged"
    )
    incomplete_evidence_may_be_replaced_by_zero_or_imputed_values: bool = False

    def __post_init__(self) -> None:
        if (
            self.source_library != "official-Kenneth-French-Data-Library"
            or self.five_factor_dataset_id
            != "F-F_Research_Data_5_Factors_2x3_daily"
            or self.momentum_dataset_id != "F-F_Momentum_Factor_daily"
            or self.five_factor_download_url
            != (
                "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
            )
            or self.momentum_download_url
            != (
                "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                "F-F_Momentum_Factor_daily_CSV.zip"
            )
            or self.five_factor_source_columns
            != ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF")
            or self.momentum_source_column != "Mom"
            or self.regression_factor_columns
            != ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
            or self.risk_free_column != "RF"
            or self.source_frequency != "daily"
            or self.date_alignment != "exact-date-inner-join"
            or self.coverage_requirement != "every-scored-exchange-date"
            or self.factor_join_may_shorten_scored_window
            or self.official_source_transport
            != "package-owned-https-default-tls-v1"
            or not self.official_source_retrieval_receipt_required
            or self.caller_staged_archives_are_official_evidence
            or not self.source_containers_may_include_unused_post_end_rows
            or self.extraction_rule != "exact-frozen-score-dates-only"
            or self.post_end_source_rows_may_enter_evaluator_arrays
            or self.source_unit != "percent"
            or self.evaluator_unit != "decimal-return"
            or self.unit_conversion != "divide-by-100"
            or self.missing_value_policy != "no-imputation"
            or self.portfolio_dependent_variable != "policy-return-minus-RF"
            or self.benchmark_dependent_variable != "C1-return-minus-RF"
            or self.active_dependent_variable != "policy-return-minus-C1-return"
            or not self.source_receipt_required
            or not self.coverage_receipt_required
            or not self.exact_array_hash_receipt_required
            or self.incomplete_evidence_behavior
            != "multifactor-outputs-unavailable-score-window-unchanged"
            or self.incomplete_evidence_may_be_replaced_by_zero_or_imputed_values
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "multifactor evidence must use exact-date, non-imputed official "
                "Kenneth French daily FF5+momentum data converted to decimals"
            )


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDReversalEpisodeContract:
    """V1 has no frozen pre-outcome episode artifact and cannot accept a mask."""

    metric_id: str = "reversal-episode-performance"
    availability: str = "unavailable-v1"
    frozen_typed_preoutcome_artifact_required: bool = True
    frozen_typed_preoutcome_artifact_implemented: bool = False
    caller_authored_mask_accepted: bool = False
    caller_authored_receipt_hash_accepted: bool = False
    result_status: str = "unavailable"
    unavailable_reason: str = (
        "frozen-typed-pre-outcome-reversal-episode-artifact-unavailable-v1"
    )

    def __post_init__(self) -> None:
        if (
            self.metric_id != "reversal-episode-performance"
            or self.availability != "unavailable-v1"
            or not self.frozen_typed_preoutcome_artifact_required
            or self.frozen_typed_preoutcome_artifact_implemented
            or self.caller_authored_mask_accepted
            or self.caller_authored_receipt_hash_accepted
            or self.result_status != "unavailable"
            or self.unavailable_reason
            != "frozen-typed-pre-outcome-reversal-episode-artifact-unavailable-v1"
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "v1 reversal episodes require a future frozen typed pre-outcome "
                "artifact and must reject caller-authored masks and hashes"
            )


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDContaminationContract:
    """Truthful provenance for the static, future-selected TOP2000 universe."""

    source_manifest_schema: str = "top2000_raw_time_partitioned_v1"
    dataset_built_at_utc: str = M03R_SEED17_TOP2000_DATASET_BUILT_AT_UTC
    universe_selection_date: str = M03R_SEED17_TOP2000_UNIVERSE_SELECTION_DATE
    universe_selection_method: str = (
        "one-day S3 dollar-volume rank intersected with common stocks active on "
        "selection date"
    )
    membership_mode: str = "static"
    future_selected_universe: bool = True
    universe_selection_overlaps_scored_window: bool = True
    point_in_time_universe: bool = False
    delisting_history_complete: bool = False
    retrospective_only: bool = True
    development_only: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    lockbox_eligible: bool = False
    confirmatory_evidence_eligible: bool = False
    investment_performance_claim_eligible: bool = False
    required_result_labels: tuple[str, ...] = (
        "development-only",
        "retrospective",
        "2026-YTD",
        "future-selected-universe",
        "nonreportable",
        "nonpromotable",
        "one-seed",
    )
    prohibited_claims: tuple[str, ...] = (
        "point-in-time-out-of-sample",
        "untouched-lockbox",
        "five-seed-robustness",
        "reportable-investment-performance",
        "promotion-evidence",
    )

    def __post_init__(self) -> None:
        try:
            built = date.fromisoformat(self.dataset_built_at_utc)
            selected = date.fromisoformat(self.universe_selection_date)
        except ValueError as exc:
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "TOP2000 provenance dates must be ISO calendar dates"
            ) from exc
        first = date.fromisoformat(M03R_SEED17_TOP2000_2026_YTD_FIRST_SCORED_DATE)
        last = date.fromisoformat(M03R_SEED17_TOP2000_2026_YTD_LAST_SCORED_DATE)
        if (
            self.source_manifest_schema != "top2000_raw_time_partitioned_v1"
            or built != last
            or not first <= selected <= last
            or self.membership_mode != "static"
            or not self.future_selected_universe
            or not self.universe_selection_overlaps_scored_window
            or self.point_in_time_universe
            or self.delisting_history_complete
            or not self.retrospective_only
            or not self.development_only
            or self.scientific_reporting_eligible
            or self.promotion_eligible
            or self.lockbox_eligible
            or self.confirmatory_evidence_eligible
            or self.investment_performance_claim_eligible
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "future-selected TOP2000 contamination must remain explicit and "
                "must block reporting, lockbox, and promotion claims"
            )
        required_labels = {
            "development-only",
            "retrospective",
            "2026-YTD",
            "future-selected-universe",
            "nonreportable",
            "nonpromotable",
            "one-seed",
        }
        if set(self.required_result_labels) != required_labels:
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD result labels are incomplete"
            )
        required_prohibitions = {
            "point-in-time-out-of-sample",
            "untouched-lockbox",
            "five-seed-robustness",
            "reportable-investment-performance",
            "promotion-evidence",
        }
        if set(self.prohibited_claims) != required_prohibitions:
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD prohibited claims are incomplete"
            )


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDJointBootstrapPlan:
    """Paired date-index bootstrap shared across the complete setting panel."""

    method: str = "joint-date-index-circular-moving-block"
    primary_block_length_trading_sessions: int = 21
    sensitivity_block_lengths_trading_sessions: tuple[int, ...] = (10, 30)
    replicate_count: int = 10_000
    bootstrap_seed_sha256: str = (
        M03R_SEED17_TOP2000_2026_YTD_BOOTSTRAP_SEED_SHA256
    )
    confidence_level: float = 0.95
    familywise_error_rate: float = 0.05
    same_date_indices_for_all_settings: bool = True
    same_date_indices_for_c1_and_factor_rows: bool = True
    same_date_indices_for_all_cost_rungs: bool = True
    same_date_indices_for_all_contrasts: bool = True
    primary_contrast_family: str = (
        "eleven-paired-20bp-net-active-return-causal-contrasts"
    )
    simultaneous_interval_method: str = (
        "joint-max-absolute-centered-contrast-fwer-0.05"
    )
    raw_one_sided_p_value_method: str = (
        "null-centered-paired-bootstrap-upper-tail"
    )
    dispersion_standard_deviation_degrees_of_freedom: int = 1
    cohort_rmst_method: str = (
        "kaplan-meier-complete-score-origin-return-neutral-trajectories"
    )
    cohort_rmst_resampling: str = (
        "joint-complete-origin-trajectory-circular-block-by-entry-date"
    )
    cohort_rmst_block_length_origin_sessions: int = 21
    cohort_rmst_bootstrap_seed_sha256: str = (
        M03R_SEED17_TOP2000_2026_YTD_COHORT_BOOTSTRAP_SEED_SHA256
    )
    same_origin_block_draws_for_all_settings: bool = True
    cohort_trajectory_receipt_required: bool = True
    date_by_age_snapshot_survival_is_descriptive_only: bool = True
    one_joint_family_per_checkpoint_fold: bool = True
    headline_and_sensitivity_families_separate: bool = True
    checkpoint_fold_paths_pooled: bool = False
    seed_17_treated_as_independent_market_replication: bool = False

    def __post_init__(self) -> None:
        if (
            self.method != "joint-date-index-circular-moving-block"
            or self.primary_block_length_trading_sessions != 21
            or self.sensitivity_block_lengths_trading_sessions != (10, 30)
            or isinstance(self.replicate_count, bool)
            or self.replicate_count != 10_000
            or self.bootstrap_seed_sha256
            != M03R_SEED17_TOP2000_2026_YTD_BOOTSTRAP_SEED_SHA256
            or self.confidence_level != 0.95
            or self.familywise_error_rate != 0.05
            or not self.same_date_indices_for_all_settings
            or not self.same_date_indices_for_c1_and_factor_rows
            or not self.same_date_indices_for_all_cost_rungs
            or not self.same_date_indices_for_all_contrasts
            or self.primary_contrast_family
            != "eleven-paired-20bp-net-active-return-causal-contrasts"
            or self.simultaneous_interval_method
            != "joint-max-absolute-centered-contrast-fwer-0.05"
            or self.raw_one_sided_p_value_method
            != "null-centered-paired-bootstrap-upper-tail"
            or self.dispersion_standard_deviation_degrees_of_freedom != 1
            or self.cohort_rmst_method
            != "kaplan-meier-complete-score-origin-return-neutral-trajectories"
            or self.cohort_rmst_resampling
            != "joint-complete-origin-trajectory-circular-block-by-entry-date"
            or self.cohort_rmst_block_length_origin_sessions != 21
            or self.cohort_rmst_bootstrap_seed_sha256
            != M03R_SEED17_TOP2000_2026_YTD_COHORT_BOOTSTRAP_SEED_SHA256
            or not self.same_origin_block_draws_for_all_settings
            or not self.cohort_trajectory_receipt_required
            or not self.date_by_age_snapshot_survival_is_descriptive_only
            or not self.one_joint_family_per_checkpoint_fold
            or not self.headline_and_sensitivity_families_separate
            or self.checkpoint_fold_paths_pooled
            or self.seed_17_treated_as_independent_market_replication
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD inference must use the frozen paired joint moving-block "
                "plan without pooling checkpoint folds or treating seed 17 as a "
                "market replication"
            )


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDSettingBinding:
    """One evaluated seed-17 row and its unchanged training identities."""

    setting_index: int
    seed17_setting_id: str
    runtime_setting_id: str
    reviewed_v7_setting_id: str
    promotion_eligible: bool = False
    scientific_reporting_eligible: bool = False
    future_selected_universe: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.setting_index < len(M03R_SEED17_TOP2000_SETTING_IDS):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "invalid 2026-YTD setting index"
            )
        expected_seed17 = M03R_SEED17_TOP2000_SETTING_IDS[self.setting_index]
        if self.seed17_setting_id != expected_seed17:
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD setting ID/index map drifted"
            )
        expected_runtime = M03R_SEED17_TOP2000_RUNTIME_SETTING_BY_ID[expected_seed17]
        expected_reviewed = M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID[
            expected_runtime
        ]
        if (
            self.runtime_setting_id != expected_runtime
            or self.reviewed_v7_setting_id != expected_reviewed
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD setting must preserve its seed-17, runtime, and reviewed "
                "v7 identity map"
            )
        if (
            self.promotion_eligible
            or self.scientific_reporting_eligible
            or not self.future_selected_universe
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "every 2026-YTD TOP2000 row is contaminated, nonreportable, and "
                "nonpromotable"
            )


def _build_setting_bindings() -> tuple[M03RSeed17Top20002026YTDSettingBinding, ...]:
    rows: list[M03RSeed17Top20002026YTDSettingBinding] = []
    for index, seed17_setting_id in enumerate(M03R_SEED17_TOP2000_SETTING_IDS):
        runtime_id = M03R_SEED17_TOP2000_RUNTIME_SETTING_BY_ID[seed17_setting_id]
        rows.append(
            M03RSeed17Top20002026YTDSettingBinding(
                setting_index=index,
                seed17_setting_id=seed17_setting_id,
                runtime_setting_id=runtime_id,
                reviewed_v7_setting_id=(
                    M03R_TOP2000_DEV_SETTING_TO_REVIEWED_V7_ID[runtime_id]
                ),
            )
        )
    return tuple(rows)


M03R_SEED17_TOP2000_2026_YTD_SETTING_BINDINGS = _build_setting_bindings()
M03R_SEED17_TOP2000_2026_YTD_SETTINGS_BY_ID = {
    row.seed17_setting_id: row
    for row in M03R_SEED17_TOP2000_2026_YTD_SETTING_BINDINGS
}


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDContrast:
    """One signed paired causal contrast; value is minuend minus subtrahend."""

    contrast_index: int
    contrast_id: str
    minuend_setting_id: str
    subtrahend_setting_id: str
    causal_field: str

    def __post_init__(self) -> None:
        valid = set(M03R_SEED17_TOP2000_SETTING_IDS)
        reference = M03R_SEED17_TOP2000_2026_YTD_REFERENCE_SETTING_ID
        if (
            self.contrast_index < 0
            or not self.contrast_id
            or self.minuend_setting_id not in valid
            or self.subtrahend_setting_id not in valid
            or self.minuend_setting_id == self.subtrahend_setting_id
            or (self.minuend_setting_id == reference)
            == (self.subtrahend_setting_id == reference)
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "every 2026-YTD contrast must pair the reference with one ablation"
            )
        ablation_id = (
            self.subtrahend_setting_id
            if self.minuend_setting_id == reference
            else self.minuend_setting_id
        )
        runtime_id = M03R_SEED17_TOP2000_RUNTIME_SETTING_BY_ID[ablation_id]
        expected_field = M03R_TOP2000_DEV_SETTINGS_BY_ID[
            runtime_id
        ].declared_causal_field
        if self.causal_field != expected_field:
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD contrast causal field drifted from the training row"
            )


def _contrast(
    index: int,
    contrast_id: str,
    ablation_index: int,
    *,
    ablation_minus_reference: bool = False,
) -> M03RSeed17Top20002026YTDContrast:
    reference = M03R_SEED17_TOP2000_2026_YTD_REFERENCE_SETTING_ID
    ablation = M03R_SEED17_TOP2000_SETTING_IDS[ablation_index]
    runtime_id = M03R_SEED17_TOP2000_RUNTIME_SETTING_BY_ID[ablation]
    causal_field = M03R_TOP2000_DEV_SETTINGS_BY_ID[runtime_id].declared_causal_field
    if causal_field is None:  # pragma: no cover - protected by the setting registry
        raise M03RSeed17Top20002026YTDEvaluationProtocolError(
            "an ablation contrast has no causal field"
        )
    return M03RSeed17Top20002026YTDContrast(
        contrast_index=index,
        contrast_id=contrast_id,
        minuend_setting_id=ablation if ablation_minus_reference else reference,
        subtrahend_setting_id=reference if ablation_minus_reference else ablation,
        causal_field=causal_field,
    )


M03R_SEED17_TOP2000_2026_YTD_CONTRASTS = (
    _contrast(0, "reference-minus-p00-soft-persistence", 1),
    _contrast(
        1,
        "p10-soft-persistence-minus-reference",
        2,
        ablation_minus_reference=True,
    ),
    _contrast(2, "reference-minus-a08-fixed-exit-hazard", 3),
    _contrast(3, "reference-minus-a11-no-exact-hold-atom", 4),
    _contrast(4, "reference-minus-a09-no-long-context", 5),
    _contrast(5, "reference-minus-m02-no-alpha-heads", 6),
    _contrast(6, "reference-minus-a04-no-downside-adjustment", 7),
    _contrast(7, "reference-minus-a12-fixed-active-risk", 8),
    _contrast(8, "reference-minus-a10-no-factor-neutral-projection", 9),
    _contrast(9, "reference-minus-a06-sharpe-overlay", 10),
    _contrast(10, "reference-minus-a07-direct-sharpe", 11),
)


@dataclass(frozen=True, slots=True)
class M03RSeed17Top20002026YTDEvaluationContract:
    """Complete immutable plan; artifact access remains separately receipt-gated."""

    schema: str = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_SCHEMA
    protocol_generation: str = (
        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION
    )
    design_id: str = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID
    source_training_protocol_generation: str = (
        M03R_SEED17_TOP2000_PROTOCOL_GENERATION
    )
    source_training_protocol_sha256: str = M03R_SEED17_TOP2000_PROTOCOL_SHA256
    source_training_design_id: str = M03R_SEED17_TOP2000_DESIGN_ID
    settings: tuple[M03RSeed17Top20002026YTDSettingBinding, ...] = (
        M03R_SEED17_TOP2000_2026_YTD_SETTING_BINDINGS
    )
    contrasts: tuple[M03RSeed17Top20002026YTDContrast, ...] = (
        M03R_SEED17_TOP2000_2026_YTD_CONTRASTS
    )
    required_metric_ids: tuple[str, ...] = (
        M03R_SEED17_TOP2000_2026_YTD_REQUIRED_METRIC_IDS
    )
    unavailable_optional_metric_ids: tuple[str, ...] = (
        M03R_SEED17_TOP2000_2026_YTD_UNAVAILABLE_OPTIONAL_METRIC_IDS
    )
    window: M03RSeed17Top20002026YTDEvaluationWindow = (
        M03RSeed17Top20002026YTDEvaluationWindow()
    )
    checkpoint_rule: M03RSeed17Top20002026YTDCheckpointRule = (
        M03RSeed17Top20002026YTDCheckpointRule()
    )
    costs: M03RSeed17Top20002026YTDCostContract = (
        M03RSeed17Top20002026YTDCostContract()
    )
    factors: M03RSeed17Top20002026YTDFactorContract = (
        M03RSeed17Top20002026YTDFactorContract()
    )
    reversal_episodes: M03RSeed17Top20002026YTDReversalEpisodeContract = (
        M03RSeed17Top20002026YTDReversalEpisodeContract()
    )
    bootstrap: M03RSeed17Top20002026YTDJointBootstrapPlan = (
        M03RSeed17Top20002026YTDJointBootstrapPlan()
    )
    contamination: M03RSeed17Top20002026YTDContaminationContract = (
        M03RSeed17Top20002026YTDContaminationContract()
    )
    requires_complete_12_setting_72_cell_training_receipt: bool = True
    requires_exact_checkpoint_hashes: bool = True
    requires_exact_2026_data_manifest_hash: bool = True
    requires_exact_evaluator_source_hash: bool = True
    training_mutation_authorized: bool = False
    evaluation_execution_authorized_by_this_contract_alone: bool = False
    promotion_authorized: bool = False
    scientific_reporting_authorized: bool = False

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_SEED17_TOP2000_2026_YTD_EVALUATION_SCHEMA
            or self.protocol_generation
            != M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION
            or self.design_id
            != M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID
            or self.source_training_protocol_generation
            != M03R_SEED17_TOP2000_PROTOCOL_GENERATION
            or self.source_training_protocol_sha256
            != M03R_SEED17_TOP2000_PROTOCOL_SHA256
            or self.source_training_design_id != M03R_SEED17_TOP2000_DESIGN_ID
            or self.settings != M03R_SEED17_TOP2000_2026_YTD_SETTING_BINDINGS
            or self.required_metric_ids
            != M03R_SEED17_TOP2000_2026_YTD_REQUIRED_METRIC_IDS
            or self.unavailable_optional_metric_ids
            != M03R_SEED17_TOP2000_2026_YTD_UNAVAILABLE_OPTIONAL_METRIC_IDS
            or self.reversal_episodes
            != M03RSeed17Top20002026YTDReversalEpisodeContract()
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD evaluation identity, setting map, or metric inventory drifted"
            )
        if (
            len(self.settings) != 12
            or len(set(self.required_metric_ids)) != len(self.required_metric_ids)
            or set(self.required_metric_ids)
            & set(self.unavailable_optional_metric_ids)
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD evaluation requires 12 settings and disjoint metric "
                "inventories"
            )
        if len(self.contrasts) != 11 or tuple(
            row.contrast_index for row in self.contrasts
        ) != tuple(range(11)):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "2026-YTD evaluation requires exactly eleven ordered causal contrasts"
            )
        reference = M03R_SEED17_TOP2000_2026_YTD_REFERENCE_SETTING_ID
        covered_ablation_ids = tuple(
            row.subtrahend_setting_id
            if row.minuend_setting_id == reference
            else row.minuend_setting_id
            for row in self.contrasts
        )
        if set(covered_ablation_ids) != set(M03R_SEED17_TOP2000_SETTING_IDS[1:]):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "every non-reference setting, including A09, must have one contrast"
            )
        if len(set(covered_ablation_ids)) != 11:
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "an ablation cannot appear in more than one primary contrast"
            )
        if (
            not self.requires_complete_12_setting_72_cell_training_receipt
            or not self.requires_exact_checkpoint_hashes
            or not self.requires_exact_2026_data_manifest_hash
            or not self.requires_exact_evaluator_source_hash
            or self.training_mutation_authorized
            or self.evaluation_execution_authorized_by_this_contract_alone
            or self.promotion_authorized
            or self.scientific_reporting_authorized
        ):
            raise M03RSeed17Top20002026YTDEvaluationProtocolError(
                "the plan must remain receipt-gated, non-mutating, nonreportable, "
                "and nonpromotable"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT = (
    M03RSeed17Top20002026YTDEvaluationContract()
)
M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256 = (
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.receipt_sha256
)


def resolve_m03r_seed17_top2000_2026_ytd_setting(
    setting_id: str,
) -> M03RSeed17Top20002026YTDSettingBinding:
    """Resolve only an exact seed-17 setting accepted by this evaluation."""

    try:
        return M03R_SEED17_TOP2000_2026_YTD_SETTINGS_BY_ID[setting_id]
    except KeyError as exc:
        raise M03RSeed17Top20002026YTDEvaluationProtocolError(
            f"unknown seed-17 2026-YTD evaluation setting {setting_id!r}"
        ) from exc


def m03r_seed17_top2000_2026_ytd_evaluation_payload() -> dict[str, Any]:
    """Return the deterministic, non-authorizing evaluation payload."""

    payload = asdict(M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT)
    payload["contract_sha256"] = (
        M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256
    )
    return payload


__all__ = [
    "M03R_SEED17_TOP2000_2026_YTD_BOOTSTRAP_SEED_SHA256",
    "M03R_SEED17_TOP2000_2026_YTD_COHORT_BOOTSTRAP_SEED_SHA256",
    "M03R_SEED17_TOP2000_2026_YTD_CONTRASTS",
    "M03R_SEED17_TOP2000_2026_YTD_COSTS_BASIS_POINTS",
    "M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT",
    "M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT_SHA256",
    "M03R_SEED17_TOP2000_2026_YTD_EVALUATION_DESIGN_ID",
    "M03R_SEED17_TOP2000_2026_YTD_EVALUATION_PROTOCOL_GENERATION",
    "M03R_SEED17_TOP2000_2026_YTD_FIRST_SCORED_DATE",
    "M03R_SEED17_TOP2000_2026_YTD_HEADLINE_FOLD_INDEX",
    "M03R_SEED17_TOP2000_2026_YTD_LAST_SCORED_DATE",
    "M03R_SEED17_TOP2000_2026_YTD_PRIMARY_COST_BASIS_POINTS",
    "M03R_SEED17_TOP2000_2026_YTD_REFERENCE_SETTING_ID",
    "M03R_SEED17_TOP2000_2026_YTD_REQUIRED_METRIC_IDS",
    "M03R_SEED17_TOP2000_2026_YTD_SENSITIVITY_FOLD_INDICES",
    "M03R_SEED17_TOP2000_2026_YTD_SETTING_BINDINGS",
    "M03R_SEED17_TOP2000_2026_YTD_UNAVAILABLE_OPTIONAL_METRIC_IDS",
    "M03RSeed17Top20002026YTDCheckpointRule",
    "M03RSeed17Top20002026YTDContaminationContract",
    "M03RSeed17Top20002026YTDContrast",
    "M03RSeed17Top20002026YTDCostContract",
    "M03RSeed17Top20002026YTDEvaluationContract",
    "M03RSeed17Top20002026YTDEvaluationProtocolError",
    "M03RSeed17Top20002026YTDEvaluationWindow",
    "M03RSeed17Top20002026YTDFactorContract",
    "M03RSeed17Top20002026YTDJointBootstrapPlan",
    "M03RSeed17Top20002026YTDReversalEpisodeContract",
    "M03RSeed17Top20002026YTDSettingBinding",
    "m03r_seed17_top2000_2026_ytd_evaluation_payload",
    "resolve_m03r_seed17_top2000_2026_ytd_setting",
]
