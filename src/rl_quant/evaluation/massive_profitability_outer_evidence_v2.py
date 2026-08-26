"""Source-bound, calendar-stitched outer profitability evidence for Massive P0."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from statistics import fmean, pstdev

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.alpha_panel import (
    FoldBootstrapConfig,
    fold_cluster_block_bootstrap_lcb,
)
from rl_quant.evaluation.massive_fixed_horizon_tranches_v1 import (
    MassiveProfitabilityResidualInputRowV1,
    MassiveProfitabilityResidualScoresV1,
    MassiveProfitabilitySelectedTranchesV1,
    build_massive_profitability_residual_scores_v1,
    select_massive_profitability_tranches_v1,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v1 import (
    MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1,
    MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
    MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
    MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1,
    MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_RECEIPTS_V2,
    MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
    MassiveProfitabilityEvaluationPlanV1,
    parse_massive_profitability_evaluation_plan_v1,
)
from rl_quant.evaluation.massive_profitability_predictions_v1 import (
    MassiveProfitabilityOuterPredictionsV1,
    MassiveProfitabilityPredictionRowV1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    MassiveProfitabilityFeatureAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_feature_accounting_v1 import (
    MassiveProfitabilityFeatureEconomicValueRowV1,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    BARS_MIN_V2_FIELDS,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityPhasePlanV1,
)
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    MassiveProfitabilityTargetAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_target_accounting_v1 import (
    MassiveProfitabilityTargetEconomicPathRowV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SCHEMA = (
    "rl-quant.massive-profitability-outer-evidence-v2"
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_DATASET = (
    "massive-profitability-outer-evidence-v2"
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256 = semantic_sha256(
    {
        "schema": "rl-quant.massive-profitability-fixed-horizon-scaling-v2",
        "horizons": MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
        "weights": MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
        "source": "pre-registered-constant-no-outcome-input",
    }
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "residual_rows": "package-derived-from-frozen-prediction-and-source",
        "eligibility": "causal-source-only-max-300-or-80pct-pit-membership",
        "horizon_weights": MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
        "ledger": "entry-fold-model-one-global-calendar-row",
        "capacity": "intended-and-executed-participation-2pct-adv",
        "bootstrap": "calendar-unique-four-cluster-nonwrapping-63-session",
        "outer_only": True,
        "profitability_reporting": False,
        "lockbox": False,
        "retuning": False,
        "rl": False,
    }
)


class MassiveProfitabilityOuterEvidenceV2Error(ValueError):
    """The source-bound stitched profitability evidence differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(f"{name} must be finite")
    return float(value)


def _percentile_95(values: Sequence[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "participation inventory is empty"
        )
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityCausalEligibilityDateV2:
    decision_session_date: str
    pit_member_count: int
    required_eligible_count: int
    eligible_security_ids: tuple[str, ...]
    eligible_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        required = max(300, math.ceil(0.8 * self.pit_member_count))
        if (
            not self.decision_session_date
            or isinstance(self.pit_member_count, bool)
            or not isinstance(self.pit_member_count, int)
            or self.pit_member_count <= 0
            or self.required_eligible_count != required
            or self.eligible_security_ids
            != tuple(sorted(set(self.eligible_security_ids)))
            or len(self.eligible_security_ids) < required
            or len(self.eligible_security_ids) > self.pit_member_count
            or self.eligible_inventory_sha256
            != semantic_sha256(self.eligible_security_ids)
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "causal eligibility threshold or inventory differs"
            )
        _digest("causal eligibility inventory", self.eligible_inventory_sha256)
        _digest("causal eligibility row", self.receipt_sha256)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityAuthorizedResidualScoresV2:
    setting_id: str
    fold_index: int
    residual_scores: MassiveProfitabilityResidualScoresV1
    eligibility_dates: tuple[MassiveProfitabilityCausalEligibilityDateV2, ...]
    evaluation_plan_semantic_receipt_sha256: str
    prediction_semantic_receipt_sha256: str
    feature_inventory_sha256: str
    feature_accounting_inventory_sha256: str
    daily_input_authority_semantic_receipt_sha256: str
    eligibility_inventory_sha256: str
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = "rl-quant.massive-profitability-authorized-residual-scores-v2"

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("semantic_receipt_sha256")
        return body

    def validate(self) -> None:
        dates = tuple(row.decision_session_date for row in self.eligibility_dates)
        self.residual_scores.validate()
        if (
            self.setting_id != self.residual_scores.setting_id
            or self.fold_index != self.residual_scores.fold_index
            or dates != tuple(sorted(set(dates)))
            or not dates
            or not self.residual_scores.outer_evaluation_authorized
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "authorized residual score identity differs"
            )
        for row in self.eligibility_dates:
            row.validate()
        if self.eligibility_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.eligibility_dates)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "authorized residual score inventory differs"
            )
        for digest in (
            self.evaluation_plan_semantic_receipt_sha256,
            self.prediction_semantic_receipt_sha256,
            self.feature_inventory_sha256,
            self.feature_accounting_inventory_sha256,
            self.daily_input_authority_semantic_receipt_sha256,
            self.eligibility_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("authorized residual scores", digest)


def _source_residual_rows(
    *,
    prediction: MassiveProfitabilityOuterPredictionsV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    feature_accounting: Sequence[MassiveProfitabilityFeatureAccountingAuthorityV2],
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
) -> tuple[
    tuple[MassiveProfitabilityResidualInputRowV1, ...],
    tuple[MassiveProfitabilityCausalEligibilityDateV2, ...],
]:
    """Derive rows and causal eligibility without reading any target authority."""

    prediction.validate()
    daily_input_authority.validate()
    if not daily_input_authority.daily_input_data_qualified:
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "authorized residualization requires qualified daily inputs"
        )
    feature_by_receipt = {}
    for feature in features:
        feature.validate()
        if (
            not feature.source_inputs_data_qualified
            or feature.daily_input_authority_semantic_receipt_sha256
            != daily_input_authority.semantic_receipt_sha256
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "feature source differs from the frozen daily authority"
            )
        feature_by_receipt[feature.semantic_receipt_sha256] = feature
    accounting_by_receipt = {}
    for authority in feature_accounting:
        authority.validate()
        if (
            not authority.economic_values_data_qualified
            or authority.daily_input_authority_semantic_receipt_sha256
            != daily_input_authority.semantic_receipt_sha256
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "feature accounting differs from the frozen daily authority"
            )
        accounting_by_receipt[authority.semantic_receipt_sha256] = authority

    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    reversal_index = BARS_MIN_V2_FIELDS.index("reversal_5")
    momentum_index = BARS_MIN_V2_FIELDS.index("trend_21_minus_5")
    prediction_by_date: defaultdict[str, list[MassiveProfitabilityPredictionRowV1]] = (
        defaultdict(list)
    )
    for row in prediction.rows:
        prediction_by_date[row.decision_session_date].append(row)
    output: list[MassiveProfitabilityResidualInputRowV1] = []
    eligibility: list[MassiveProfitabilityCausalEligibilityDateV2] = []
    for session_date in prediction.outer_test_session_dates:
        prediction_rows = prediction_by_date[session_date]
        feature_receipts = {
            row.feature_semantic_receipt_sha256 for row in prediction_rows
        }
        if len(feature_receipts) != 1:
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "one outer date must use one source feature artifact"
            )
        selected_feature = feature_by_receipt.get(next(iter(feature_receipts)))
        if (
            selected_feature is None
            or selected_feature.decision_session_date != session_date
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "prediction is detached from its source feature artifact"
            )
        accounting = accounting_by_receipt.get(
            selected_feature.feature_accounting_authority_semantic_receipt_sha256
        )
        if accounting is None:
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "feature artifact lacks source-derived accounting"
            )
        feature_rows = {row.security_id: row for row in selected_feature.rows}
        prediction_rows_by_security = {row.security_id: row for row in prediction_rows}
        accounting_rows: defaultdict[
            str, list[MassiveProfitabilityFeatureEconomicValueRowV1]
        ] = defaultdict(list)
        for economic_row in accounting.accounting.rows:
            accounting_rows[economic_row.security_id].append(economic_row)
        eligible_rows: list[MassiveProfitabilityResidualInputRowV1] = []
        eligible_ids: list[str] = []
        for security_id in sorted(feature_rows):
            feature_row = feature_rows[security_id]
            prediction_row = prediction_rows_by_security.get(security_id)
            economic_rows = tuple(
                sorted(
                    accounting_rows.get(security_id, []),
                    key=lambda row: row.source_session_offset,
                )
            )
            if len(economic_rows) != 64 or any(
                not row.valid or row.economic_value <= 0.0 for row in economic_rows
            ):
                continue
            daily_rows = tuple(
                daily_input_authority.row(
                    session_date=date_value,
                    security_id=security_id,
                )
                for date_value in selected_feature.input_session_dates[-63:]
            )
            if any(
                not row.bars_valid[close_index]
                or row.bars_values[close_index] <= 0.0
                or not row.bars_valid[dollar_index]
                or row.bars_values[dollar_index] <= 0.0
                for row in daily_rows
            ) or not (
                feature_row.bars_valid[reversal_index]
                and feature_row.bars_valid[momentum_index]
            ):
                continue
            if prediction_row is None:
                raise MassiveProfitabilityOuterEvidenceV2Error(
                    "prediction omits a causally eligible security"
                )
            economic_values = tuple(row.economic_value for row in economic_rows)
            trailing_adv = fmean(row.bars_values[dollar_index] for row in daily_rows)
            accounting_inventory = semantic_sha256(
                tuple(row.receipt_sha256 for row in economic_rows)
            )
            if (
                accounting_inventory
                != feature_row.feature_accounting_security_inventory_sha256
            ):
                raise MassiveProfitabilityOuterEvidenceV2Error(
                    "eligible feature accounting inventory differs"
                )
            log_returns = tuple(
                math.log(right / left) for left, right in pairwise(economic_values)
            )
            body = {
                "decision_session_date": session_date,
                "security_id": security_id,
                "raw_scores": prediction_row.mean,
                "exposures": (
                    1.0,
                    math.log(economic_values[-1]),
                    math.log(trailing_adv),
                    feature_row.bars_values[reversal_index],
                    feature_row.bars_values[momentum_index],
                    pstdev(log_returns),
                ),
                "trailing_63_session_adv": trailing_adv,
                "prediction_row_receipt_sha256": prediction_row.receipt_sha256,
                "feature_row_receipt_sha256": feature_row.receipt_sha256,
                "feature_accounting_row_inventory_sha256": accounting_inventory,
            }
            eligible_rows.append(
                MassiveProfitabilityResidualInputRowV1(
                    **body,  # type: ignore[arg-type]
                    receipt_sha256=semantic_sha256(body),
                )
            )
            eligible_ids.append(security_id)
        pit_count = len(selected_feature.rows)
        required = max(300, math.ceil(0.8 * pit_count))
        if len(eligible_ids) < required:
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "causal source-time support is below the frozen threshold"
            )
        eligibility_body = {
            "decision_session_date": session_date,
            "pit_member_count": pit_count,
            "required_eligible_count": required,
            "eligible_security_ids": tuple(eligible_ids),
            "eligible_inventory_sha256": semantic_sha256(tuple(eligible_ids)),
        }
        eligibility.append(
            MassiveProfitabilityCausalEligibilityDateV2(
                **eligibility_body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(eligibility_body),
            )
        )
        output.extend(eligible_rows)
    return tuple(output), tuple(eligibility)


def build_massive_profitability_residual_scores_authorized_v2(
    *,
    evaluation_plan: MassiveProfitabilityEvaluationPlanV1,
    evaluation_plan_root: str | Path,
    prediction: MassiveProfitabilityOuterPredictionsV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    feature_accounting: Sequence[MassiveProfitabilityFeatureAccountingAuthorityV2],
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
) -> MassiveProfitabilityAuthorizedResidualScoresV2:
    """Derive and authorize residual scores without accepting caller rows."""

    evaluation_plan.validate()
    prediction.validate()
    reloaded = parse_massive_profitability_evaluation_plan_v1(
        root=evaluation_plan_root,
        loaded_source=evaluation_plan.loaded_source,
    )
    registered = next(
        (
            row
            for row in reloaded.predictions
            if row.fold_index == prediction.fold_index
            and row.setting_id == prediction.setting_id
        ),
        None,
    )
    if (
        reloaded.semantic_receipt_sha256 != evaluation_plan.semantic_receipt_sha256
        or registered is None
        or registered.prediction_semantic_receipt_sha256
        != prediction.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "prediction is outside the committed evaluation plan"
        )
    rows, eligibility = _source_residual_rows(
        prediction=prediction,
        features=features,
        feature_accounting=feature_accounting,
        daily_input_authority=daily_input_authority,
    )
    research_scores = build_massive_profitability_residual_scores_v1(
        setting_id=prediction.setting_id,
        fold_index=prediction.fold_index,
        evaluation_plan_semantic_receipt_sha256=evaluation_plan.semantic_receipt_sha256,
        prediction_semantic_receipt_sha256=prediction.semantic_receipt_sha256,
        rows=rows,
    )
    authorized_body = research_scores.unsigned()
    authorized_body["outer_evaluation_authorized"] = True
    residual_scores = replace(
        research_scores,
        outer_evaluation_authorized=True,
        semantic_receipt_sha256=semantic_sha256(authorized_body),
    )
    residual_scores.validate()
    feature_inventory = semantic_sha256(
        tuple(sorted(row.semantic_receipt_sha256 for row in features))
    )
    accounting_inventory = semantic_sha256(
        tuple(sorted(row.semantic_receipt_sha256 for row in feature_accounting))
    )
    eligibility_inventory = semantic_sha256(
        tuple(row.receipt_sha256 for row in eligibility)
    )
    body = {
        "schema": "rl-quant.massive-profitability-authorized-residual-scores-v2",
        "setting_id": prediction.setting_id,
        "fold_index": prediction.fold_index,
        "residual_scores": asdict(residual_scores),
        "eligibility_dates": tuple(asdict(row) for row in eligibility),
        "evaluation_plan_semantic_receipt_sha256": evaluation_plan.semantic_receipt_sha256,
        "prediction_semantic_receipt_sha256": prediction.semantic_receipt_sha256,
        "feature_inventory_sha256": feature_inventory,
        "feature_accounting_inventory_sha256": accounting_inventory,
        "daily_input_authority_semantic_receipt_sha256": (
            daily_input_authority.semantic_receipt_sha256
        ),
        "eligibility_inventory_sha256": eligibility_inventory,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityAuthorizedResidualScoresV2(
        setting_id=prediction.setting_id,
        fold_index=prediction.fold_index,
        residual_scores=residual_scores,
        eligibility_dates=eligibility,
        evaluation_plan_semantic_receipt_sha256=evaluation_plan.semantic_receipt_sha256,
        prediction_semantic_receipt_sha256=prediction.semantic_receipt_sha256,
        feature_inventory_sha256=feature_inventory,
        feature_accounting_inventory_sha256=accounting_inventory,
        daily_input_authority_semantic_receipt_sha256=(
            daily_input_authority.semantic_receipt_sha256
        ),
        eligibility_inventory_sha256=eligibility_inventory,
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


def _target_path_map(
    values: Sequence[MassiveProfitabilityTargetAccountingAuthorityV2],
) -> tuple[
    dict[
        tuple[str, str],
        tuple[MassiveProfitabilityTargetEconomicPathRowV1, tuple[str, ...]],
    ],
    str,
]:
    result = {}
    receipts = []
    for authority in values:
        authority.validate()
        if (
            not authority.fill_sources_qualified
            or not authority.economic_values_data_qualified
            or not authority.terminal_accounting_complete
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched target paths are not source-qualified"
            )
        receipts.append(authority.semantic_receipt_sha256)
        for row in authority.rows:
            key = (authority.decision_session_date, row.security_id)
            if key in result:
                raise MassiveProfitabilityOuterEvidenceV2Error(
                    "stitched target accounting duplicates a path"
                )
            result[key] = (row, authority.session_dates)
    return result, semantic_sha256(tuple(sorted(receipts)))


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityStitchedDailyHorizonPnlRowV2:
    session_date: str
    horizon_sessions: int
    gross_return: float
    entry_costs: tuple[float, ...]
    exit_costs: tuple[float, ...]
    short_borrow_cost: float
    net_returns: tuple[float, ...]
    entry_fold_net_return_20bp: tuple[float, ...]
    active_position_count: int
    contribution_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        expected_net = tuple(
            self.gross_return
            - self.entry_costs[index]
            - self.exit_costs[index]
            - self.short_borrow_cost
            for index in range(3)
        )
        if (
            not self.session_date
            or self.horizon_sessions not in MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1
            or len(self.entry_costs) != 3
            or len(self.exit_costs) != 3
            or len(self.net_returns) != 3
            or len(self.entry_fold_net_return_20bp) != 4
            or isinstance(self.active_position_count, bool)
            or not isinstance(self.active_position_count, int)
            or self.active_position_count < 0
            or any(value < 0.0 for value in self.entry_costs + self.exit_costs)
            or self.short_borrow_cost < 0.0
            or any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)
                for left, right in zip(self.net_returns, expected_net, strict=True)
            )
            or not math.isclose(
                math.fsum(self.entry_fold_net_return_20bp),
                self.net_returns[1],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched daily horizon P&L differs"
            )
        for value in (
            self.gross_return,
            *self.entry_costs,
            *self.exit_costs,
            self.short_borrow_cost,
            *self.net_returns,
            *self.entry_fold_net_return_20bp,
        ):
            _finite("stitched daily horizon P&L", value)
        _digest("stitched contribution inventory", self.contribution_inventory_sha256)
        _digest("stitched daily horizon row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched daily horizon receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityStitchedPnlV2:
    setting_id: str
    rows: tuple[MassiveProfitabilityStitchedDailyHorizonPnlRowV2, ...]
    selected_tranche_receipts: tuple[str, ...]
    target_accounting_inventory_sha256: str
    cost_rates: tuple[float, ...]
    annual_short_borrow_rate: float
    row_inventory_sha256: str
    semantic_receipt_sha256: str
    calendar_unique: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = "rl-quant.massive-profitability-stitched-pnl-v2"

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("semantic_receipt_sha256")
        return body

    def validate(self) -> None:
        keys = tuple((row.session_date, row.horizon_sessions) for row in self.rows)
        if (
            self.setting_id not in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
            or not self.rows
            or keys != tuple(sorted(set(keys)))
            or len(self.selected_tranche_receipts) != 4
            or self.cost_rates != MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1
            or self.annual_short_borrow_rate != 0.01
            or not self.calendar_unique
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched P&L identity differs"
            )
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched P&L inventory differs"
            )
        for value in (
            *self.selected_tranche_receipts,
            self.target_accounting_inventory_sha256,
            self.row_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("stitched P&L", value)


def stitch_massive_profitability_fixed_tranches_v2(
    *,
    selected_by_fold: Sequence[MassiveProfitabilitySelectedTranchesV1],
    target_accounting: Sequence[MassiveProfitabilityTargetAccountingAuthorityV2],
) -> MassiveProfitabilityStitchedPnlV2:
    """Carry entry-fold tranches in one global calendar ledger until exit."""

    selected = tuple(sorted(selected_by_fold, key=lambda row: row.fold_index))
    for row in selected:
        row.validate()
    if (
        len(selected) != 4
        or tuple(row.fold_index for row in selected) != tuple(range(4))
        or len({row.setting_id for row in selected}) != 1
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "stitched ledger requires four folds for one setting"
        )
    paths, target_inventory = _target_path_map(target_accounting)
    if any(
        row.target_accounting_inventory_sha256 != target_inventory for row in selected
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "stitched ledger target inventory differs from guarded selection"
        )
    contributions: defaultdict[
        tuple[str, int],
        list[
            tuple[
                str,
                int,
                float,
                tuple[float, ...],
                tuple[float, ...],
                float,
            ]
        ],
    ] = defaultdict(list)
    for selected_fold in selected:
        for position in selected_fold.positions:
            path, session_dates = paths[
                (position.decision_session_date, position.security_id)
            ]
            entry_value = path.values[0]
            horizon = position.horizon_sessions
            for offset in range(horizon + 1):
                ratio = path.values[offset] / entry_value
                gross = (
                    0.0
                    if offset == 0
                    else position.signed_entry_weight
                    * (path.values[offset] - path.values[offset - 1])
                    / entry_value
                )
                entry_costs = tuple(
                    rate * abs(position.signed_entry_weight) if offset == 0 else 0.0
                    for rate in MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1
                )
                exit_costs = tuple(
                    rate * abs(position.signed_entry_weight * ratio)
                    if offset == horizon
                    else 0.0
                    for rate in MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1
                )
                borrow = (
                    0.01
                    / 252.0
                    * abs(
                        position.signed_entry_weight
                        * path.values[offset - 1]
                        / entry_value
                    )
                    if position.side == "short" and offset > 0
                    else 0.0
                )
                contributions[(session_dates[offset], horizon)].append(
                    (
                        position.receipt_sha256,
                        selected_fold.fold_index,
                        gross,
                        entry_costs,
                        exit_costs,
                        borrow,
                    )
                )
    rows = []
    for session_date in sorted({key[0] for key in contributions}):
        for horizon in MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1:
            values = contributions.get((session_date, horizon), [])
            gross = math.fsum(value[2] for value in values)
            entry_costs = tuple(
                math.fsum(value[3][index] for value in values) for index in range(3)
            )
            exit_costs = tuple(
                math.fsum(value[4][index] for value in values) for index in range(3)
            )
            borrow = math.fsum(value[5] for value in values)
            net_returns = tuple(
                gross - entry_costs[index] - exit_costs[index] - borrow
                for index in range(3)
            )
            by_entry_fold = tuple(
                math.fsum(
                    value[2] - value[3][1] - value[4][1] - value[5]
                    for value in values
                    if value[1] == fold_index
                )
                for fold_index in range(4)
            )
            body = {
                "session_date": session_date,
                "horizon_sessions": horizon,
                "gross_return": gross,
                "entry_costs": entry_costs,
                "exit_costs": exit_costs,
                "short_borrow_cost": borrow,
                "net_returns": net_returns,
                "entry_fold_net_return_20bp": by_entry_fold,
                "active_position_count": len(values),
                "contribution_inventory_sha256": semantic_sha256(
                    tuple(value[0] for value in values)
                ),
            }
            rows.append(
                MassiveProfitabilityStitchedDailyHorizonPnlRowV2(
                    **body,  # type: ignore[arg-type]
                    receipt_sha256=semantic_sha256(body),
                )
            )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "schema": "rl-quant.massive-profitability-stitched-pnl-v2",
        "setting_id": selected[0].setting_id,
        "rows": tuple(asdict(row) for row in rows),
        "selected_tranche_receipts": tuple(
            row.semantic_receipt_sha256 for row in selected
        ),
        "target_accounting_inventory_sha256": target_inventory,
        "cost_rates": MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
        "annual_short_borrow_rate": 0.01,
        "row_inventory_sha256": row_inventory,
        "calendar_unique": True,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityStitchedPnlV2(
        setting_id=selected[0].setting_id,
        rows=tuple(rows),
        selected_tranche_receipts=tuple(
            row.semantic_receipt_sha256 for row in selected
        ),
        target_accounting_inventory_sha256=target_inventory,
        cost_rates=MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
        annual_short_borrow_rate=0.01,
        row_inventory_sha256=row_inventory,
        semantic_receipt_sha256=semantic_sha256(body),
        calendar_unique=True,
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityStitchedCompositeDailyRowV2:
    session_date: str
    gross_return: float
    net_returns: tuple[float, ...]
    entry_fold_net_return_20bp: tuple[float, ...]
    horizon_row_receipts: tuple[str, ...]
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            not self.session_date
            or len(self.net_returns) != 3
            or len(self.entry_fold_net_return_20bp) != 4
            or len(self.horizon_row_receipts) != 4
            or not math.isclose(
                math.fsum(self.entry_fold_net_return_20bp),
                self.net_returns[1],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched composite daily row differs"
            )
        for value in (
            self.gross_return,
            *self.net_returns,
            *self.entry_fold_net_return_20bp,
        ):
            _finite("stitched composite return", value)
        for digest in (*self.horizon_row_receipts, self.receipt_sha256):
            _digest("stitched composite row", digest)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched composite receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityStitchedCompositeV2:
    setting_id: str
    rows: tuple[MassiveProfitabilityStitchedCompositeDailyRowV2, ...]
    stitched_pnl_semantic_receipt_sha256: str
    fixed_horizon_scaling_receipt_sha256: str
    horizon_weights: tuple[float, ...]
    row_inventory_sha256: str
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = "rl-quant.massive-profitability-stitched-composite-v2"

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("semantic_receipt_sha256")
        return body

    def validate(self) -> None:
        dates = tuple(row.session_date for row in self.rows)
        if (
            self.setting_id not in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
            or not self.rows
            or dates != tuple(sorted(set(dates)))
            or self.fixed_horizon_scaling_receipt_sha256
            != MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
            or self.horizon_weights != MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched composite identity differs"
            )
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched composite inventory differs"
            )
        for value in (
            self.stitched_pnl_semantic_receipt_sha256,
            self.fixed_horizon_scaling_receipt_sha256,
            self.row_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("stitched composite", value)


def build_massive_profitability_stitched_composite_v2(
    *, pnl: MassiveProfitabilityStitchedPnlV2
) -> MassiveProfitabilityStitchedCompositeV2:
    """Apply fixed pre-registered 25 percent weights to all four horizons."""

    pnl.validate()
    by_key = {(row.session_date, row.horizon_sessions): row for row in pnl.rows}
    rows = []
    for session_date in sorted({row.session_date for row in pnl.rows}):
        horizon_rows = tuple(
            by_key[(session_date, horizon)]
            for horizon in MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1
        )
        body = {
            "session_date": session_date,
            "gross_return": math.fsum(
                weight * row.gross_return
                for weight, row in zip(
                    MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
                    horizon_rows,
                    strict=True,
                )
            ),
            "net_returns": tuple(
                math.fsum(
                    MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2[index]
                    * horizon_rows[index].net_returns[cost_index]
                    for index in range(4)
                )
                for cost_index in range(3)
            ),
            "entry_fold_net_return_20bp": tuple(
                math.fsum(
                    MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2[index]
                    * horizon_rows[index].entry_fold_net_return_20bp[fold_index]
                    for index in range(4)
                )
                for fold_index in range(4)
            ),
            "horizon_row_receipts": tuple(row.receipt_sha256 for row in horizon_rows),
        }
        rows.append(
            MassiveProfitabilityStitchedCompositeDailyRowV2(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    panel_body = {
        "schema": "rl-quant.massive-profitability-stitched-composite-v2",
        "setting_id": pnl.setting_id,
        "rows": tuple(asdict(row) for row in rows),
        "stitched_pnl_semantic_receipt_sha256": pnl.semantic_receipt_sha256,
        "fixed_horizon_scaling_receipt_sha256": (
            MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
        ),
        "horizon_weights": MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
        "row_inventory_sha256": inventory,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityStitchedCompositeV2(
        setting_id=pnl.setting_id,
        rows=tuple(rows),
        stitched_pnl_semantic_receipt_sha256=pnl.semantic_receipt_sha256,
        fixed_horizon_scaling_receipt_sha256=(
            MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
        ),
        horizon_weights=MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
        row_inventory_sha256=inventory,
        semantic_receipt_sha256=semantic_sha256(panel_body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityStitchedCapacityRowV2:
    capital_usd: float
    intended_order_count: int
    clipped_order_count: int
    mean_intended_participation: float
    intended_participation_p95: float
    maximum_intended_participation: float
    mean_executed_participation: float
    executed_participation_p95: float
    maximum_executed_participation: float
    lost_intended_notional_fraction: float
    clipped_mean_daily_net_return_20bp: float
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        participation = (
            self.mean_intended_participation,
            self.intended_participation_p95,
            self.maximum_intended_participation,
            self.mean_executed_participation,
            self.executed_participation_p95,
            self.maximum_executed_participation,
        )
        if (
            self.capital_usd not in MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1
            or isinstance(self.intended_order_count, bool)
            or not isinstance(self.intended_order_count, int)
            or self.intended_order_count <= 0
            or isinstance(self.clipped_order_count, bool)
            or not isinstance(self.clipped_order_count, int)
            or not 0 <= self.clipped_order_count <= self.intended_order_count
            or any(
                _finite("capacity participation", value) < 0.0
                for value in participation
            )
            or self.maximum_executed_participation > 0.02 + 1e-15
            or not 0.0 <= self.lost_intended_notional_fraction <= 1.0
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched capacity row differs"
            )
        _finite("stitched clipped return", self.clipped_mean_daily_net_return_20bp)
        _digest("stitched capacity row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched capacity receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityStitchedCapacityPanelV2:
    setting_id: str
    rows: tuple[MassiveProfitabilityStitchedCapacityRowV2, ...]
    selected_tranche_receipts: tuple[str, ...]
    target_accounting_inventory_sha256: str
    fixed_horizon_scaling_receipt_sha256: str
    row_inventory_sha256: str
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = "rl-quant.massive-profitability-stitched-capacity-v2"

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("semantic_receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            self.setting_id not in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
            or tuple(row.capital_usd for row in self.rows)
            != MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1
            or len(self.selected_tranche_receipts) != 4
            or self.fixed_horizon_scaling_receipt_sha256
            != MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched capacity panel differs"
            )
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "stitched capacity inventory differs"
            )
        for digest in (
            *self.selected_tranche_receipts,
            self.target_accounting_inventory_sha256,
            self.fixed_horizon_scaling_receipt_sha256,
            self.row_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("stitched capacity panel", digest)


def evaluate_massive_profitability_stitched_capacity_v2(
    *,
    selected_by_fold: Sequence[MassiveProfitabilitySelectedTranchesV1],
    target_accounting: Sequence[MassiveProfitabilityTargetAccountingAuthorityV2],
) -> MassiveProfitabilityStitchedCapacityPanelV2:
    """Report intended and clipped participation on the stitched ledger."""

    selected = tuple(sorted(selected_by_fold, key=lambda row: row.fold_index))
    for row in selected:
        row.validate()
    if (
        len(selected) != 4
        or tuple(row.fold_index for row in selected) != tuple(range(4))
        or len({row.setting_id for row in selected}) != 1
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "stitched capacity requires four folds for one setting"
        )
    paths, target_inventory = _target_path_map(target_accounting)
    if any(
        row.target_accounting_inventory_sha256 != target_inventory for row in selected
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "stitched capacity target inventory differs"
        )
    positions = tuple(position for row in selected for position in row.positions)
    rows = []
    for capital in MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1:
        intended_participation = []
        executed_participation = []
        intended_total = 0.0
        executed_total = 0.0
        clipped = 0
        daily_horizon_net: defaultdict[tuple[str, int], float] = defaultdict(float)
        for position in positions:
            horizon_index = MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1.index(
                position.horizon_sessions
            )
            horizon_weight = MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2[
                horizon_index
            ]
            intended = horizon_weight * abs(position.signed_entry_weight) * capital
            limit = 0.02 * position.trailing_63_session_adv
            executed = min(intended, limit)
            scale = executed / intended
            intended_total += intended
            executed_total += executed
            intended_participation.append(intended / position.trailing_63_session_adv)
            executed_participation.append(executed / position.trailing_63_session_adv)
            clipped += intended > limit
            path, session_dates = paths[
                (position.decision_session_date, position.security_id)
            ]
            entry_value = path.values[0]
            for offset in range(position.horizon_sessions + 1):
                ratio = path.values[offset] / entry_value
                gross = (
                    0.0
                    if offset == 0
                    else scale
                    * position.signed_entry_weight
                    * (path.values[offset] - path.values[offset - 1])
                    / entry_value
                )
                cost = (
                    0.002 * scale * abs(position.signed_entry_weight)
                    if offset == 0
                    else 0.0
                )
                if offset == position.horizon_sessions:
                    cost += 0.002 * scale * abs(position.signed_entry_weight * ratio)
                borrow = (
                    0.01
                    / 252.0
                    * scale
                    * abs(
                        position.signed_entry_weight
                        * path.values[offset - 1]
                        / entry_value
                    )
                    if position.side == "short" and offset > 0
                    else 0.0
                )
                daily_horizon_net[
                    (session_dates[offset], position.horizon_sessions)
                ] += gross - cost - borrow
        dates = tuple(sorted({key[0] for key in daily_horizon_net}))
        daily_composite = tuple(
            math.fsum(
                MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2[index]
                * daily_horizon_net.get((session_date, horizon), 0.0)
                for index, horizon in enumerate(
                    MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1
                )
            )
            for session_date in dates
        )
        body = {
            "capital_usd": capital,
            "intended_order_count": len(positions),
            "clipped_order_count": clipped,
            "mean_intended_participation": fmean(intended_participation),
            "intended_participation_p95": _percentile_95(intended_participation),
            "maximum_intended_participation": max(intended_participation),
            "mean_executed_participation": fmean(executed_participation),
            "executed_participation_p95": _percentile_95(executed_participation),
            "maximum_executed_participation": max(executed_participation),
            "lost_intended_notional_fraction": 1.0 - executed_total / intended_total,
            "clipped_mean_daily_net_return_20bp": fmean(daily_composite),
        }
        rows.append(
            MassiveProfitabilityStitchedCapacityRowV2(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    panel_body = {
        "schema": "rl-quant.massive-profitability-stitched-capacity-v2",
        "setting_id": selected[0].setting_id,
        "rows": tuple(asdict(row) for row in rows),
        "selected_tranche_receipts": tuple(
            row.semantic_receipt_sha256 for row in selected
        ),
        "target_accounting_inventory_sha256": target_inventory,
        "fixed_horizon_scaling_receipt_sha256": (
            MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
        ),
        "row_inventory_sha256": inventory,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityStitchedCapacityPanelV2(
        setting_id=selected[0].setting_id,
        rows=tuple(rows),
        selected_tranche_receipts=tuple(
            row.semantic_receipt_sha256 for row in selected
        ),
        target_accounting_inventory_sha256=target_inventory,
        fixed_horizon_scaling_receipt_sha256=(
            MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
        ),
        row_inventory_sha256=inventory,
        semantic_receipt_sha256=semantic_sha256(panel_body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


def _maximum_drawdown(values: Sequence[float]) -> float:
    wealth = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1.0)
    return drawdown


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityOuterEvidenceV2:
    evaluation_plan_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    authorized_residual_inventory_sha256: str
    selected_tranche_inventory_sha256: str
    stitched_pnl_receipts: tuple[str, ...]
    stitched_composite_receipts: tuple[str, ...]
    stitched_capacity_receipts: tuple[str, ...]
    fixed_horizon_scaling_receipt_sha256: str
    mean_mv04_net_20bp: float
    mean_mv04_minus_mv02_net_20bp: float
    mean_mv04_minus_shuffle_net_20bp: float
    mv04_net_20bp_lcb95: float
    mv04_minus_mv02_net_20bp_lcb95: float
    mv04_minus_shuffle_net_20bp_lcb95: float
    mean_mv04_net_40bp: float
    mean_mv04_clipped_10m_net_20bp: float
    annualized_mv04_net_return_20bp: float
    annualized_mv04_net_volatility_20bp: float
    mv04_net_sharpe_20bp: float
    mv04_net_sortino_20bp: float
    mv04_maximum_drawdown_20bp: float
    mv04_hit_rate_20bp: float
    mv04_mean_one_way_turnover: float
    mv04_break_even_one_way_cost: float
    positive_mv04_entry_fold_count: int
    calendar_date_count: int
    outer_profitability_gate_passed: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    development_conclusion_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    evaluator_retuning_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "loaded_source"}
        }

    def validate(self) -> None:
        passed = (
            self.mv04_net_20bp_lcb95 > 0.0
            and self.mv04_minus_mv02_net_20bp_lcb95 > 0.0
            and self.mv04_minus_shuffle_net_20bp_lcb95 > 0.0
            and self.positive_mv04_entry_fold_count >= 3
            and self.mean_mv04_net_40bp >= 0.0
            and self.mean_mv04_clipped_10m_net_20bp > 0.0
        )
        if (
            self.schema != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SCHEMA
            or len(self.stitched_pnl_receipts) != 4
            or len(self.stitched_composite_receipts) != 4
            or len(self.stitched_capacity_receipts) != 4
            or self.fixed_horizon_scaling_receipt_sha256
            != MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
            or not 0 <= self.positive_mv04_entry_fold_count <= 4
            or self.calendar_date_count <= 0
            or self.outer_profitability_gate_passed != passed
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SOURCE_SHA256
            or not self.development_conclusion_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.evaluator_retuning_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "outer evidence V2 identity or authorization differs"
            )
        for value in (
            self.mean_mv04_net_20bp,
            self.mean_mv04_minus_mv02_net_20bp,
            self.mean_mv04_minus_shuffle_net_20bp,
            self.mv04_net_20bp_lcb95,
            self.mv04_minus_mv02_net_20bp_lcb95,
            self.mv04_minus_shuffle_net_20bp_lcb95,
            self.mean_mv04_net_40bp,
            self.mean_mv04_clipped_10m_net_20bp,
            self.annualized_mv04_net_return_20bp,
            self.annualized_mv04_net_volatility_20bp,
            self.mv04_net_sharpe_20bp,
            self.mv04_net_sortino_20bp,
            self.mv04_maximum_drawdown_20bp,
            self.mv04_hit_rate_20bp,
            self.mv04_mean_one_way_turnover,
            self.mv04_break_even_one_way_cost,
        ):
            _finite("outer evidence V2 statistic", value)
        if (
            self.annualized_mv04_net_volatility_20bp < 0.0
            or self.mv04_maximum_drawdown_20bp > 0.0
            or not 0.0 <= self.mv04_hit_rate_20bp <= 1.0
            or self.mv04_mean_one_way_turnover < 0.0
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "outer evidence V2 diagnostic range differs"
            )
        for digest in (
            self.evaluation_plan_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.authorized_residual_inventory_sha256,
            self.selected_tranche_inventory_sha256,
            *self.stitched_pnl_receipts,
            *self.stitched_composite_receipts,
            *self.stitched_capacity_receipts,
            self.fixed_horizon_scaling_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("outer evidence V2", digest)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "outer evidence V2 semantic receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.evaluation_plan_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "outer evidence V2 committed source differs"
            )


def _calendar_fold_values(
    *,
    rows: Sequence[MassiveProfitabilityStitchedCompositeDailyRowV2],
    phase_plan: MassiveProfitabilityPhasePlanV1,
    value_index: int,
) -> tuple[tuple[float, ...], ...]:
    starts = tuple(fold.outer_test_session_dates[0] for fold in phase_plan.outer_folds)
    grouped: list[list[float]] = [[], [], [], []]
    for row in rows:
        fold_index = max(
            index for index, start in enumerate(starts) if row.session_date >= start
        )
        grouped[fold_index].append(row.net_returns[value_index])
    if any(not values for values in grouped):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "stitched inference lacks one calendar fold cluster"
        )
    return tuple(tuple(values) for values in grouped)


def _statistics(
    *,
    evaluation_plan: MassiveProfitabilityEvaluationPlanV1,
    phase_plan: MassiveProfitabilityPhasePlanV1,
    composites: Sequence[MassiveProfitabilityStitchedCompositeV2],
    capacity: Sequence[MassiveProfitabilityStitchedCapacityPanelV2],
) -> dict[str, float | int | bool]:
    by_setting = {row.setting_id: row for row in composites}
    if set(by_setting) != set(MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "stitched inference lacks one setting"
        )
    dates = tuple(row.session_date for row in by_setting["MV04"].rows)
    if any(
        tuple(row.session_date for row in by_setting[setting].rows) != dates
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "stitched settings do not share one exact calendar"
        )
    primary = {
        setting: tuple(row.net_returns[1] for row in by_setting[setting].rows)
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    }
    delta_bars = tuple(
        left - right
        for left, right in zip(primary["MV04"], primary["MV02"], strict=True)
    )
    delta_shuffle = tuple(
        left - right
        for left, right in zip(primary["MV04"], primary["MV04-SHUFFLE"], strict=True)
    )
    mv04_folds = _calendar_fold_values(
        rows=by_setting["MV04"].rows,
        phase_plan=phase_plan,
        value_index=1,
    )
    delta_bars_folds = []
    delta_shuffle_folds = []
    starts = tuple(fold.outer_test_session_dates[0] for fold in phase_plan.outer_folds)
    for fold_index in range(4):
        indices = tuple(
            index
            for index, session_date in enumerate(dates)
            if max(i for i, start in enumerate(starts) if session_date >= start)
            == fold_index
        )
        delta_bars_folds.append(tuple(delta_bars[index] for index in indices))
        delta_shuffle_folds.append(tuple(delta_shuffle[index] for index in indices))
    bootstrap = FoldBootstrapConfig(
        replicates=evaluation_plan.bootstrap_replicates,
        block_sessions=evaluation_plan.bootstrap_block_sessions,
        seed=evaluation_plan.bootstrap_seed,
        lower_probability=0.025,
    )
    mv04 = primary["MV04"]
    daily_mean = fmean(mv04)
    daily_volatility = math.sqrt(
        math.fsum((value - daily_mean) ** 2 for value in mv04) / len(mv04)
    )
    downside_deviation = math.sqrt(
        math.fsum(min(value, 0.0) ** 2 for value in mv04) / len(mv04)
    )
    mv04_low = tuple(row.net_returns[0] for row in by_setting["MV04"].rows)
    mv04_high = tuple(row.net_returns[2] for row in by_setting["MV04"].rows)
    turnover = fmean(
        (low - high) / (0.004 - 0.001)
        for low, high in zip(mv04_low, mv04_high, strict=True)
    )
    entry_fold_returns = tuple(
        math.fsum(
            row.entry_fold_net_return_20bp[index] for row in by_setting["MV04"].rows
        )
        for index in range(4)
    )
    capacity_by_setting = {row.setting_id: row for row in capacity}
    mv04_10m = next(
        row.clipped_mean_daily_net_return_20bp
        for row in capacity_by_setting["MV04"].rows
        if row.capital_usd == 10_000_000.0
    )
    result: dict[str, float | int | bool] = {
        "mean_mv04_net_20bp": daily_mean,
        "mean_mv04_minus_mv02_net_20bp": fmean(delta_bars),
        "mean_mv04_minus_shuffle_net_20bp": fmean(delta_shuffle),
        "mv04_net_20bp_lcb95": fold_cluster_block_bootstrap_lcb(mv04_folds, bootstrap),
        "mv04_minus_mv02_net_20bp_lcb95": fold_cluster_block_bootstrap_lcb(
            tuple(delta_bars_folds), bootstrap
        ),
        "mv04_minus_shuffle_net_20bp_lcb95": fold_cluster_block_bootstrap_lcb(
            tuple(delta_shuffle_folds), bootstrap
        ),
        "mean_mv04_net_40bp": fmean(mv04_high),
        "mean_mv04_clipped_10m_net_20bp": mv04_10m,
        "annualized_mv04_net_return_20bp": 252.0 * daily_mean,
        "annualized_mv04_net_volatility_20bp": math.sqrt(252.0) * daily_volatility,
        "mv04_net_sharpe_20bp": 0.0
        if daily_volatility == 0.0
        else math.sqrt(252.0) * daily_mean / daily_volatility,
        "mv04_net_sortino_20bp": 0.0
        if downside_deviation == 0.0
        else math.sqrt(252.0) * daily_mean / downside_deviation,
        "mv04_maximum_drawdown_20bp": _maximum_drawdown(mv04),
        "mv04_hit_rate_20bp": sum(value > 0.0 for value in mv04) / len(mv04),
        "mv04_mean_one_way_turnover": turnover,
        "mv04_break_even_one_way_cost": 0.0
        if turnover <= 0.0
        else 0.002 + daily_mean / turnover,
        "positive_mv04_entry_fold_count": sum(
            value > 0.0 for value in entry_fold_returns
        ),
        "calendar_date_count": len(dates),
    }
    result["outer_profitability_gate_passed"] = (
        result["mv04_net_20bp_lcb95"] > 0.0
        and result["mv04_minus_mv02_net_20bp_lcb95"] > 0.0
        and result["mv04_minus_shuffle_net_20bp_lcb95"] > 0.0
        and result["positive_mv04_entry_fold_count"] >= 3
        and result["mean_mv04_net_40bp"] >= 0.0
        and result["mean_mv04_clipped_10m_net_20bp"] > 0.0
    )
    return result


def materialize_massive_profitability_outer_evidence_v2(
    *,
    root: str | Path,
    artifact_id: str,
    evaluation_plan: MassiveProfitabilityEvaluationPlanV1,
    phase_plan: MassiveProfitabilityPhasePlanV1,
    predictions: Sequence[MassiveProfitabilityOuterPredictionsV1],
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    feature_accounting: Sequence[MassiveProfitabilityFeatureAccountingAuthorityV2],
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    target_accounting: Sequence[MassiveProfitabilityTargetAccountingAuthorityV2],
    committed_at_ms: int,
) -> MassiveProfitabilityOuterEvidenceV2:
    """Derive every authorizing intermediate from frozen typed source roots."""

    evaluation_plan.validate()
    phase_plan.validate()
    daily_input_authority.validate()
    reloaded_plan = parse_massive_profitability_evaluation_plan_v1(
        root=root, loaded_source=evaluation_plan.loaded_source
    )
    if (
        reloaded_plan.semantic_receipt_sha256 != evaluation_plan.semantic_receipt_sha256
        or evaluation_plan.phase_plan_semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
        or evaluation_plan.horizon_risk_scaling_receipts
        != MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_RECEIPTS_V2
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "outer evidence V2 differs from the committed plan or fixed weights"
        )
    outer_dates = tuple(
        session_date
        for fold in phase_plan.outer_folds
        for session_date in fold.outer_test_session_dates
    )
    if (
        tuple(sorted(feature.decision_session_date for feature in features))
        != outer_dates
        or len(feature_accounting) != len(outer_dates)
        or {authority.semantic_receipt_sha256 for authority in feature_accounting}
        != {
            feature.feature_accounting_authority_semantic_receipt_sha256
            for feature in features
        }
        or tuple(
            sorted(authority.decision_session_date for authority in target_accounting)
        )
        != outer_dates
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "outer evidence inputs must contain exactly the frozen outer dates"
        )
    ordered_predictions = tuple(
        sorted(predictions, key=lambda row: (row.fold_index, row.setting_id))
    )
    expected = tuple(
        (fold_index, setting)
        for fold_index in range(4)
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    if (
        tuple((row.fold_index, row.setting_id) for row in ordered_predictions)
        != expected
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "outer evidence requires one exact prediction inventory"
        )
    authorized_residuals = tuple(
        build_massive_profitability_residual_scores_authorized_v2(
            evaluation_plan=evaluation_plan,
            evaluation_plan_root=root,
            prediction=prediction,
            features=features,
            feature_accounting=feature_accounting,
            daily_input_authority=daily_input_authority,
        )
        for prediction in ordered_predictions
    )
    for fold_index in range(4):
        eligibility = {
            row.eligibility_inventory_sha256
            for row in authorized_residuals
            if row.fold_index == fold_index
        }
        if len(eligibility) != 1:
            raise MassiveProfitabilityOuterEvidenceV2Error(
                "outer settings do not share one causal eligibility inventory"
            )
    selected = tuple(
        select_massive_profitability_tranches_v1(
            residual_scores=row.residual_scores,
            target_accounting=target_accounting,
        )
        for row in authorized_residuals
    )
    stitched = tuple(
        stitch_massive_profitability_fixed_tranches_v2(
            selected_by_fold=tuple(
                row for row in selected if row.setting_id == setting
            ),
            target_accounting=target_accounting,
        )
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    composites = tuple(
        build_massive_profitability_stitched_composite_v2(pnl=row) for row in stitched
    )
    capacity = tuple(
        evaluate_massive_profitability_stitched_capacity_v2(
            selected_by_fold=tuple(
                row for row in selected if row.setting_id == setting
            ),
            target_accounting=target_accounting,
        )
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    statistics = _statistics(
        evaluation_plan=evaluation_plan,
        phase_plan=phase_plan,
        composites=composites,
        capacity=capacity,
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SCHEMA,
        "evaluation_plan_semantic_receipt_sha256": (
            evaluation_plan.semantic_receipt_sha256
        ),
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "authorized_residual_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in authorized_residuals)
        ),
        "selected_tranche_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in selected)
        ),
        "stitched_pnl_receipts": tuple(row.semantic_receipt_sha256 for row in stitched),
        "stitched_composite_receipts": tuple(
            row.semantic_receipt_sha256 for row in composites
        ),
        "stitched_capacity_receipts": tuple(
            row.semantic_receipt_sha256 for row in capacity
        ),
        "fixed_horizon_scaling_receipt_sha256": (
            MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256
        ),
        **statistics,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SOURCE_SHA256
        ),
        "development_conclusion_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "evaluator_retuning_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    if (
        not artifact_id
        or artifact_id != artifact_id.strip()
        or any(not (value.isalnum() or value in "-_") for value in artifact_id)
    ):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "outer evidence V2 artifact ID is not path safe"
        )
    relative = f"massive-profitability-outer-evidence-v2/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=evaluation_plan.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    result = MassiveProfitabilityOuterEvidenceV2(
        **semantic,  # type: ignore[arg-type]
        loaded_source=loaded,
    )
    result.validate()
    return result


def parse_massive_profitability_outer_evidence_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityOuterEvidenceV2:
    """Reload a committed V2 evidence summary and regenerate its bytes."""

    payload = json.loads(
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    )
    for name in (
        "stitched_pnl_receipts",
        "stitched_composite_receipts",
        "stitched_capacity_receipts",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveProfitabilityOuterEvidenceV2(**payload, loaded_source=loaded_source)
    result.validate()
    if canonical_json_file_bytes(
        result.semantic_unsigned()
        | {"semantic_receipt_sha256": result.semantic_receipt_sha256}
    ) != read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source):
        raise MassiveProfitabilityOuterEvidenceV2Error(
            "outer evidence V2 canonical bytes differ"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_V2_RECEIPT_SHA256",
    "MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V2_SCHEMA",
    "MassiveProfitabilityAuthorizedResidualScoresV2",
    "MassiveProfitabilityCausalEligibilityDateV2",
    "MassiveProfitabilityOuterEvidenceV2",
    "MassiveProfitabilityOuterEvidenceV2Error",
    "MassiveProfitabilityStitchedCapacityPanelV2",
    "MassiveProfitabilityStitchedCapacityRowV2",
    "MassiveProfitabilityStitchedCompositeDailyRowV2",
    "MassiveProfitabilityStitchedCompositeV2",
    "MassiveProfitabilityStitchedDailyHorizonPnlRowV2",
    "MassiveProfitabilityStitchedPnlV2",
    "build_massive_profitability_residual_scores_authorized_v2",
    "build_massive_profitability_stitched_composite_v2",
    "evaluate_massive_profitability_stitched_capacity_v2",
    "materialize_massive_profitability_outer_evidence_v2",
    "parse_massive_profitability_outer_evidence_v2",
    "stitch_massive_profitability_fixed_tranches_v2",
]
