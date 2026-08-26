"""Deterministic residual-score selection and fixed-horizon Massive P0 tranches."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from statistics import fmean, pstdev

import torch

from rl_quant.evaluation.massive_profitability_evaluation_plan_v1 import (
    MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
    MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
    MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA,
    MassiveProfitabilityEvaluationPlanV1,
    parse_massive_profitability_evaluation_plan_v1,
)
from rl_quant.evaluation.massive_profitability_predictions_v1 import (
    MassiveProfitabilityOuterPredictionsV1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    MassiveProfitabilityFeatureAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    BARS_MIN_V2_FIELDS,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    MassiveProfitabilityTargetAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_target_accounting_v1 import (
    MassiveProfitabilityTargetEconomicPathRowV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_FIXED_HORIZON_TRANCHES_V1_SCHEMA = "rl-quant.massive-fixed-horizon-tranches-v1"
MASSIVE_FIXED_HORIZON_TRANCHES_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_FIXED_HORIZON_TRANCHES_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "evaluation_plan": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA,
        "score": "predicted-mean-residualized-before-selection",
        "selection": "top-bottom-floor-20pct-ties-permanent-security-id",
        "target_support": "never-read-before-selection",
        "path": "all-offsets-entry-through-scheduled-exit-valid",
        "terminal_short": "unresolved-fallback-before-exit-hard-failure",
        "weight": "signed-0.5-divided-by-horizon-and-tail-count",
        "cost": "no-cross-tranche-netting-entry-and-exit",
        "borrow": "100bp-annualized-on-prior-short-economic-notional",
        "capacity": "entry-only-2pct-trailing-63-session-adv-no-reallocation",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveFixedHorizonTranchesV1Error(ValueError):
    """One fixed-tranche profitability input is incomplete or ex-post selected."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveFixedHorizonTranchesV1Error(f"{name} must be a lowercase SHA-256")
    return value


def _finite(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MassiveFixedHorizonTranchesV1Error(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityResidualInputRowV1:
    decision_session_date: str
    security_id: str
    raw_scores: tuple[float, ...]
    exposures: tuple[float, ...]
    trailing_63_session_adv: float
    prediction_row_receipt_sha256: str
    feature_row_receipt_sha256: str
    feature_accounting_row_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            not self.decision_session_date
            or not self.security_id
            or len(self.raw_scores) != 4
            or len(self.exposures) != 6
            or self.exposures[0] != 1.0
            or _finite("trailing ADV", self.trailing_63_session_adv) <= 0.0
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "residual input identity or dimensions differ"
            )
        for value in self.raw_scores + self.exposures:
            _finite("residual input", value)
        for digest in (
            self.prediction_row_receipt_sha256,
            self.feature_row_receipt_sha256,
            self.feature_accounting_row_inventory_sha256,
            self.receipt_sha256,
        ):
            _digest("residual input", digest)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFixedHorizonTranchesV1Error("residual input receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityResidualScoreRowV1:
    decision_session_date: str
    security_id: str
    raw_scores: tuple[float, ...]
    residual_scores: tuple[float, ...]
    trailing_63_session_adv: float
    residual_input_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            not self.decision_session_date
            or not self.security_id
            or len(self.raw_scores) != 4
            or len(self.residual_scores) != 4
            or _finite("residual score ADV", self.trailing_63_session_adv) <= 0.0
        ):
            raise MassiveFixedHorizonTranchesV1Error("residual score identity differs")
        for value in self.raw_scores + self.residual_scores:
            _finite("residual score", value)
        _digest("residual input", self.residual_input_receipt_sha256)
        _digest("residual row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFixedHorizonTranchesV1Error(
                "residual score row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityResidualScoresV1:
    setting_id: str
    fold_index: int
    rows: tuple[MassiveProfitabilityResidualScoreRowV1, ...]
    evaluation_plan_semantic_receipt_sha256: str
    prediction_semantic_receipt_sha256: str
    input_inventory_sha256: str
    row_inventory_sha256: str
    ridge_lambda: float
    exposure_fields: tuple[str, ...]
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = "rl-quant.massive-profitability-residual-scores-v1"

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("semantic_receipt_sha256")
        return body

    def validate(self) -> None:
        keys = tuple((row.decision_session_date, row.security_id) for row in self.rows)
        if (
            self.setting_id not in {"MV00", "MV02", "MV04", "MV04-SHUFFLE"}
            or isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < 4
            or not self.rows
            or keys != tuple(sorted(set(keys)))
            or self.ridge_lambda != 1e-6
            or self.exposure_fields
            != (
                "intercept",
                "log_source_economic_value",
                "log_trailing_63_session_adv",
                "reversal_5",
                "momentum_21_minus_5",
                "economic_volatility_63",
            )
            or not isinstance(self.outer_evaluation_authorized, bool)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveFixedHorizonTranchesV1Error("residual score artifact differs")
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFixedHorizonTranchesV1Error("residual score inventory differs")
        for value in (
            self.evaluation_plan_semantic_receipt_sha256,
            self.prediction_semantic_receipt_sha256,
            self.input_inventory_sha256,
            self.row_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("residual score artifact", value)


def _standardize_exposures(values: torch.Tensor) -> torch.Tensor:
    result = values.clone()
    for index in range(1, result.shape[1]):
        column = result[:, index]
        scale = column.std(correction=0)
        result[:, index] = (
            column - column.mean()
            if float(scale) <= 1e-12
            else (column - column.mean()) / scale
        )
    return result


def build_massive_profitability_residual_scores_v1(
    *,
    setting_id: str,
    fold_index: int,
    evaluation_plan_semantic_receipt_sha256: str,
    prediction_semantic_receipt_sha256: str,
    rows: Sequence[MassiveProfitabilityResidualInputRowV1],
    evaluation_plan: MassiveProfitabilityEvaluationPlanV1 | None = None,
    evaluation_plan_root: str | Path | None = None,
    prediction: MassiveProfitabilityOuterPredictionsV1 | None = None,
) -> MassiveProfitabilityResidualScoresV1:
    """Residualize all settings through one target-free deterministic operator."""

    authorizing = False
    optional_authority = (evaluation_plan, evaluation_plan_root, prediction)
    if any(value is not None for value in optional_authority):
        if any(value is None for value in optional_authority):
            raise MassiveFixedHorizonTranchesV1Error(
                "authorizing residualization requires the complete frozen plan root"
            )
        assert evaluation_plan is not None
        assert evaluation_plan_root is not None
        assert prediction is not None
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
                if row.fold_index == fold_index and row.setting_id == setting_id
            ),
            None,
        )
        if (
            reloaded.semantic_receipt_sha256 != evaluation_plan_semantic_receipt_sha256
            or prediction.semantic_receipt_sha256 != prediction_semantic_receipt_sha256
            or registered is None
            or registered.prediction_semantic_receipt_sha256
            != prediction.semantic_receipt_sha256
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "residualization is detached from the frozen evaluation plan"
            )
        authorizing = True
    ordered = tuple(
        sorted(rows, key=lambda row: (row.decision_session_date, row.security_id))
    )
    for row in ordered:
        row.validate()
    if not ordered:
        raise MassiveFixedHorizonTranchesV1Error("residualization input is empty")
    grouped: defaultdict[str, list[MassiveProfitabilityResidualInputRowV1]] = (
        defaultdict(list)
    )
    for row in ordered:
        grouped[row.decision_session_date].append(row)
    output: list[MassiveProfitabilityResidualScoreRowV1] = []
    ridge = 1e-6
    for session_date in sorted(grouped):
        date_rows = grouped[session_date]
        if len(date_rows) < 5:
            raise MassiveFixedHorizonTranchesV1Error(
                "residualization requires at least five securities per date"
            )
        design = _standardize_exposures(
            torch.tensor([row.exposures for row in date_rows], dtype=torch.float64)
        )
        scores = torch.tensor(
            [row.raw_scores for row in date_rows], dtype=torch.float64
        )
        gram = design.T @ design + ridge * torch.eye(
            design.shape[1], dtype=torch.float64
        )
        coefficients = torch.linalg.solve(gram, design.T @ scores)
        residuals = scores - design @ coefficients
        for index, source in enumerate(date_rows):
            body = {
                "decision_session_date": session_date,
                "security_id": source.security_id,
                "raw_scores": source.raw_scores,
                "residual_scores": tuple(float(value) for value in residuals[index]),
                "trailing_63_session_adv": source.trailing_63_session_adv,
                "residual_input_receipt_sha256": source.receipt_sha256,
            }
            output.append(
                MassiveProfitabilityResidualScoreRowV1(
                    **body,  # type: ignore[arg-type]
                    receipt_sha256=semantic_sha256(body),
                )
            )
    input_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in ordered))
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in output))
    body = {
        "schema": "rl-quant.massive-profitability-residual-scores-v1",
        "setting_id": setting_id,
        "fold_index": fold_index,
        "rows": tuple(asdict(row) for row in output),
        "evaluation_plan_semantic_receipt_sha256": (
            evaluation_plan_semantic_receipt_sha256
        ),
        "prediction_semantic_receipt_sha256": prediction_semantic_receipt_sha256,
        "input_inventory_sha256": input_inventory,
        "row_inventory_sha256": row_inventory,
        "ridge_lambda": ridge,
        "exposure_fields": (
            "intercept",
            "log_source_economic_value",
            "log_trailing_63_session_adv",
            "reversal_5",
            "momentum_21_minus_5",
            "economic_volatility_63",
        ),
        "outer_evaluation_authorized": authorizing,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityResidualScoresV1(
        setting_id=setting_id,
        fold_index=fold_index,
        rows=tuple(output),
        evaluation_plan_semantic_receipt_sha256=(
            evaluation_plan_semantic_receipt_sha256
        ),
        prediction_semantic_receipt_sha256=prediction_semantic_receipt_sha256,
        input_inventory_sha256=input_inventory,
        row_inventory_sha256=row_inventory,
        ridge_lambda=ridge,
        exposure_fields=body["exposure_fields"],  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=authorizing,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


def derive_massive_profitability_residual_inputs_v1(
    *,
    prediction: MassiveProfitabilityOuterPredictionsV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    feature_accounting: Sequence[MassiveProfitabilityFeatureAccountingAuthorityV2],
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
) -> tuple[MassiveProfitabilityResidualInputRowV1, ...]:
    """Derive the target-free residual design from frozen source authorities."""

    prediction.validate()
    daily_input_authority.validate()
    if not daily_input_authority.daily_input_data_qualified:
        raise MassiveFixedHorizonTranchesV1Error(
            "residual exposures require source-qualified daily inputs"
        )
    feature_by_receipt: dict[str, MassiveProfitabilityOriginFeaturesV3] = {}
    for artifact in features:
        artifact.validate()
        if (
            not artifact.source_inputs_data_qualified
            or artifact.daily_input_authority_semantic_receipt_sha256
            != daily_input_authority.semantic_receipt_sha256
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "residual feature source differs from the daily authority"
            )
        feature_by_receipt[artifact.semantic_receipt_sha256] = artifact
    accounting_by_receipt: dict[
        str, MassiveProfitabilityFeatureAccountingAuthorityV2
    ] = {}
    for authority in feature_accounting:
        authority.validate()
        if (
            not authority.economic_values_data_qualified
            or authority.daily_input_authority_semantic_receipt_sha256
            != daily_input_authority.semantic_receipt_sha256
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "residual accounting is not source-qualified"
            )
        accounting_by_receipt[authority.semantic_receipt_sha256] = authority

    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    reversal_index = BARS_MIN_V2_FIELDS.index("reversal_5")
    momentum_index = BARS_MIN_V2_FIELDS.index("trend_21_minus_5")
    rows: list[MassiveProfitabilityResidualInputRowV1] = []
    for prediction_row in prediction.rows:
        prediction_row.validate()
        feature = feature_by_receipt.get(prediction_row.feature_semantic_receipt_sha256)
        if feature is None:
            raise MassiveFixedHorizonTranchesV1Error(
                "prediction is detached from its source feature artifact"
            )
        feature_row = next(
            (
                row
                for row in feature.rows
                if row.security_id == prediction_row.security_id
            ),
            None,
        )
        if feature_row is None:
            raise MassiveFixedHorizonTranchesV1Error(
                "prediction lacks its source feature row"
            )
        accounting = accounting_by_receipt.get(
            feature.feature_accounting_authority_semantic_receipt_sha256
        )
        if accounting is None:
            raise MassiveFixedHorizonTranchesV1Error(
                "feature artifact lacks its source accounting authority"
            )
        economic_rows = tuple(
            row
            for row in accounting.accounting.rows
            if row.security_id == prediction_row.security_id
        )
        if len(economic_rows) != 64 or any(
            not row.valid or row.economic_value <= 0.0 for row in economic_rows
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "residual exposures require one complete 64-session economic path"
            )
        daily_rows = tuple(
            daily_input_authority.row(
                session_date=session_date,
                security_id=prediction_row.security_id,
            )
            for session_date in feature.input_session_dates[-63:]
        )
        if any(
            not row.bars_valid[close_index]
            or row.bars_values[close_index] <= 0.0
            or not row.bars_valid[dollar_index]
            or row.bars_values[dollar_index] <= 0.0
            for row in daily_rows
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "residual exposures require complete price and ADV support"
            )
        if not (
            feature_row.bars_valid[reversal_index]
            and feature_row.bars_valid[momentum_index]
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "residual exposures require causal reversal and momentum"
            )
        economic_values = tuple(row.economic_value for row in economic_rows)
        daily_log_returns = tuple(
            math.log(right / left) for left, right in pairwise(economic_values)
        )
        trailing_adv = fmean(row.bars_values[dollar_index] for row in daily_rows)
        accounting_inventory = semantic_sha256(
            tuple(row.receipt_sha256 for row in economic_rows)
        )
        if (
            accounting_inventory
            != feature_row.feature_accounting_security_inventory_sha256
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "feature row accounting inventory differs"
            )
        body = {
            "decision_session_date": prediction_row.decision_session_date,
            "security_id": prediction_row.security_id,
            "raw_scores": prediction_row.mean,
            "exposures": (
                1.0,
                math.log(economic_values[-1]),
                math.log(trailing_adv),
                feature_row.bars_values[reversal_index],
                feature_row.bars_values[momentum_index],
                pstdev(daily_log_returns),
            ),
            "trailing_63_session_adv": trailing_adv,
            "prediction_row_receipt_sha256": prediction_row.receipt_sha256,
            "feature_row_receipt_sha256": feature_row.receipt_sha256,
            "feature_accounting_row_inventory_sha256": accounting_inventory,
        }
        rows.append(
            MassiveProfitabilityResidualInputRowV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    ordered = tuple(
        sorted(rows, key=lambda row: (row.decision_session_date, row.security_id))
    )
    if tuple(row.receipt_sha256 for row in ordered) != tuple(
        row.receipt_sha256 for row in rows
    ):
        raise MassiveFixedHorizonTranchesV1Error(
            "prediction rows are not in canonical decision-security order"
        )
    return ordered


@dataclass(frozen=True, slots=True)
class MassiveProfitabilitySelectedTranchePositionV1:
    decision_session_date: str
    security_id: str
    horizon_sessions: int
    side: str
    tail_rank: int
    signed_entry_weight: float
    residual_score: float
    trailing_63_session_adv: float
    residual_score_row_receipt_sha256: str
    target_path_receipt_sha256: str
    unresolved_terminal_fallback_session_offset: int | None
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            not self.decision_session_date
            or not self.security_id
            or self.horizon_sessions not in MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1
            or self.side not in {"long", "short"}
            or isinstance(self.tail_rank, bool)
            or not isinstance(self.tail_rank, int)
            or self.tail_rank < 1
            or (self.side == "long" and self.signed_entry_weight <= 0.0)
            or (self.side == "short" and self.signed_entry_weight >= 0.0)
            or _finite("selected ADV", self.trailing_63_session_adv) <= 0.0
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "selected tranche position differs"
            )
        _finite("selected weight", self.signed_entry_weight)
        _finite("selected score", self.residual_score)
        fallback = self.unresolved_terminal_fallback_session_offset
        if fallback is not None and (
            isinstance(fallback, bool)
            or not isinstance(fallback, int)
            or not 1 <= fallback <= 63
        ):
            raise MassiveFixedHorizonTranchesV1Error("selected fallback offset differs")
        for value in (
            self.residual_score_row_receipt_sha256,
            self.target_path_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("selected tranche", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFixedHorizonTranchesV1Error("selected tranche receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveProfitabilitySelectedTranchesV1:
    setting_id: str
    fold_index: int
    positions: tuple[MassiveProfitabilitySelectedTranchePositionV1, ...]
    residual_scores_semantic_receipt_sha256: str
    target_accounting_inventory_sha256: str
    position_inventory_sha256: str
    path_support_complete: bool
    direction_safe_terminal_support_complete: bool
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("semantic_receipt_sha256")
        return body

    def validate(self) -> None:
        keys = tuple(
            (row.decision_session_date, row.horizon_sessions, row.side, row.tail_rank)
            for row in self.positions
        )
        if (
            not self.positions
            or keys != tuple(sorted(set(keys)))
            or not self.path_support_complete
            or not self.direction_safe_terminal_support_complete
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "selected tranche artifact differs"
            )
        for row in self.positions:
            row.validate()
        if self.position_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.positions)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFixedHorizonTranchesV1Error(
                "selected tranche inventory differs"
            )
        for value in (
            self.residual_scores_semantic_receipt_sha256,
            self.target_accounting_inventory_sha256,
            self.position_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("selected tranche artifact", value)


def _path_map(
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
            raise MassiveFixedHorizonTranchesV1Error(
                "selected paths are not source-qualified"
            )
        receipts.append(authority.semantic_receipt_sha256)
        for row in authority.rows:
            key = (authority.decision_session_date, row.security_id)
            if key in result:
                raise MassiveFixedHorizonTranchesV1Error(
                    "target accounting duplicates a selected path"
                )
            result[key] = (row, authority.session_dates)
    return result, semantic_sha256(tuple(sorted(receipts)))


def select_massive_profitability_tranches_v1(
    *,
    residual_scores: MassiveProfitabilityResidualScoresV1,
    target_accounting: Sequence[MassiveProfitabilityTargetAccountingAuthorityV2],
) -> MassiveProfitabilitySelectedTranchesV1:
    """Rank without targets, then fail closed on every selected economic path."""

    residual_scores.validate()
    if not residual_scores.outer_evaluation_authorized:
        raise MassiveFixedHorizonTranchesV1Error(
            "target selection requires plan-authorized residual scores"
        )
    grouped: defaultdict[str, list[MassiveProfitabilityResidualScoreRowV1]] = (
        defaultdict(list)
    )
    for row in residual_scores.rows:
        grouped[row.decision_session_date].append(row)
    selected_keys: list[
        tuple[MassiveProfitabilityResidualScoreRowV1, int, str, int]
    ] = []
    for session_date in sorted(grouped):
        date_rows = grouped[session_date]
        count = math.floor(0.20 * len(date_rows))
        if count < 1 or 2 * count > len(date_rows):
            raise MassiveFixedHorizonTranchesV1Error(
                "tail selection has insufficient cross-sectional support"
            )
        for horizon_index, horizon in enumerate(
            MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1
        ):
            short_ordered = sorted(
                date_rows,
                key=lambda row: (row.residual_scores[horizon_index], row.security_id),
            )
            long_ordered = sorted(
                date_rows,
                key=lambda row: (-row.residual_scores[horizon_index], row.security_id),
            )
            for rank, row in enumerate(short_ordered[:count], start=1):
                selected_keys.append((row, horizon, "short", rank))
            for rank, row in enumerate(long_ordered[:count], start=1):
                selected_keys.append((row, horizon, "long", rank))

    # Target paths are opened only after the complete deterministic selection exists.
    paths, target_inventory = _path_map(target_accounting)
    positions: list[MassiveProfitabilitySelectedTranchePositionV1] = []
    for score, horizon, side, rank in selected_keys:
        selected = paths.get((score.decision_session_date, score.security_id))
        if selected is None:
            raise MassiveFixedHorizonTranchesV1Error(
                "selected position lacks its target-accounting path"
            )
        path, _ = selected
        path.validate()
        if not all(path.valid[: horizon + 1]) or path.values[0] <= 0.0:
            raise MassiveFixedHorizonTranchesV1Error(
                "selected position lacks a complete marked path or scheduled exit"
            )
        fallback = path.unresolved_terminal_fallback_session_offset
        if side == "short" and fallback is not None and fallback <= horizon:
            raise MassiveFixedHorizonTranchesV1Error(
                "unresolved terminal fallback cannot credit a selected short"
            )
        tail_count = math.floor(0.20 * len(grouped[score.decision_session_date]))
        weight = (0.5 if side == "long" else -0.5) / (horizon * tail_count)
        body = {
            "decision_session_date": score.decision_session_date,
            "security_id": score.security_id,
            "horizon_sessions": horizon,
            "side": side,
            "tail_rank": rank,
            "signed_entry_weight": weight,
            "residual_score": score.residual_scores[
                MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1.index(horizon)
            ],
            "trailing_63_session_adv": score.trailing_63_session_adv,
            "residual_score_row_receipt_sha256": score.receipt_sha256,
            "target_path_receipt_sha256": path.receipt_sha256,
            "unresolved_terminal_fallback_session_offset": fallback,
        }
        positions.append(
            MassiveProfitabilitySelectedTranchePositionV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    ordered_positions = tuple(
        sorted(
            positions,
            key=lambda row: (
                row.decision_session_date,
                row.horizon_sessions,
                row.side,
                row.tail_rank,
            ),
        )
    )
    position_inventory = semantic_sha256(
        tuple(row.receipt_sha256 for row in ordered_positions)
    )
    body = {
        "setting_id": residual_scores.setting_id,
        "fold_index": residual_scores.fold_index,
        "positions": tuple(asdict(row) for row in ordered_positions),
        "residual_scores_semantic_receipt_sha256": (
            residual_scores.semantic_receipt_sha256
        ),
        "target_accounting_inventory_sha256": target_inventory,
        "position_inventory_sha256": position_inventory,
        "path_support_complete": True,
        "direction_safe_terminal_support_complete": True,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilitySelectedTranchesV1(
        setting_id=residual_scores.setting_id,
        fold_index=residual_scores.fold_index,
        positions=ordered_positions,
        residual_scores_semantic_receipt_sha256=(
            residual_scores.semantic_receipt_sha256
        ),
        target_accounting_inventory_sha256=target_inventory,
        position_inventory_sha256=position_inventory,
        path_support_complete=True,
        direction_safe_terminal_support_complete=True,
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDailyTranchePnlRowV1:
    session_date: str
    horizon_sessions: int
    gross_return: float
    entry_costs: tuple[float, ...]
    exit_costs: tuple[float, ...]
    short_borrow_cost: float
    net_returns: tuple[float, ...]
    active_position_count: int
    contribution_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            not self.session_date
            or self.horizon_sessions not in MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1
            or len(self.entry_costs) != 3
            or len(self.exit_costs) != 3
            or len(self.net_returns) != 3
            or isinstance(self.active_position_count, bool)
            or not isinstance(self.active_position_count, int)
            or self.active_position_count < 0
        ):
            raise MassiveFixedHorizonTranchesV1Error("daily tranche P&L differs")
        for value in (
            self.gross_return,
            *self.entry_costs,
            *self.exit_costs,
            self.short_borrow_cost,
            *self.net_returns,
        ):
            _finite("daily tranche P&L", value)
        if any(value < 0.0 for value in self.entry_costs + self.exit_costs) or (
            self.short_borrow_cost < 0.0
        ):
            raise MassiveFixedHorizonTranchesV1Error("daily costs are negative")
        expected = tuple(
            self.gross_return
            - self.entry_costs[index]
            - self.exit_costs[index]
            - self.short_borrow_cost
            for index in range(3)
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)
            for left, right in zip(self.net_returns, expected, strict=True)
        ):
            raise MassiveFixedHorizonTranchesV1Error("daily net P&L differs")
        _digest("daily contribution inventory", self.contribution_inventory_sha256)
        _digest("daily tranche row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFixedHorizonTranchesV1Error("daily tranche receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFixedTranchePnlV1:
    setting_id: str
    fold_index: int
    rows: tuple[MassiveProfitabilityDailyTranchePnlRowV1, ...]
    selected_tranches_semantic_receipt_sha256: str
    target_accounting_inventory_sha256: str
    cost_rates: tuple[float, ...]
    annual_short_borrow_rate: float
    row_inventory_sha256: str
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_FIXED_HORIZON_TRANCHES_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("semantic_receipt_sha256")
        return body

    def validate(self) -> None:
        keys = tuple((row.session_date, row.horizon_sessions) for row in self.rows)
        if (
            self.schema != MASSIVE_FIXED_HORIZON_TRANCHES_V1_SCHEMA
            or not self.rows
            or keys != tuple(sorted(set(keys)))
            or self.cost_rates != MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1
            or self.annual_short_borrow_rate != 0.01
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveFixedHorizonTranchesV1Error(
                "fixed tranche P&L artifact differs"
            )
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFixedHorizonTranchesV1Error(
                "fixed tranche P&L inventory differs"
            )
        for value in (
            self.selected_tranches_semantic_receipt_sha256,
            self.target_accounting_inventory_sha256,
            self.row_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("fixed tranche P&L", value)


def evaluate_massive_profitability_fixed_tranches_v1(
    *,
    selected: MassiveProfitabilitySelectedTranchesV1,
    target_accounting: Sequence[MassiveProfitabilityTargetAccountingAuthorityV2],
) -> MassiveProfitabilityFixedTranchePnlV1:
    """Mark selected positions daily and charge conservative entry/exit costs."""

    selected.validate()
    paths, target_inventory = _path_map(target_accounting)
    if target_inventory != selected.target_accounting_inventory_sha256:
        raise MassiveFixedHorizonTranchesV1Error(
            "P&L target paths differ from guarded selection"
        )
    contributions: defaultdict[
        tuple[str, int],
        list[tuple[str, float, tuple[float, ...], tuple[float, ...], float]],
    ] = defaultdict(list)
    for position in selected.positions:
        path, session_dates = paths[
            (position.decision_session_date, position.security_id)
        ]
        horizon = position.horizon_sessions
        entry_value = path.values[0]
        for offset in range(horizon + 1):
            session_date = session_dates[offset]
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
                    * (path.values[max(0, offset - 1)] / entry_value)
                )
                if position.side == "short" and offset > 0
                else 0.0
            )
            contributions[(session_date, horizon)].append(
                (
                    position.receipt_sha256,
                    gross,
                    entry_costs,
                    exit_costs,
                    borrow,
                )
            )
    rows: list[MassiveProfitabilityDailyTranchePnlRowV1] = []
    all_dates = tuple(sorted({key[0] for key in contributions}))
    for session_date in all_dates:
        for horizon in MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1:
            values = contributions.get((session_date, horizon), [])
            gross = math.fsum(value[1] for value in values)
            entry_cost_vector = tuple(
                math.fsum(value[2][index] for value in values) for index in range(3)
            )
            exit_ = tuple(
                math.fsum(value[3][index] for value in values) for index in range(3)
            )
            borrow = math.fsum(value[4] for value in values)
            net = tuple(
                gross - entry_cost_vector[index] - exit_[index] - borrow
                for index in range(3)
            )
            body = {
                "session_date": session_date,
                "horizon_sessions": horizon,
                "gross_return": gross,
                "entry_costs": entry_cost_vector,
                "exit_costs": exit_,
                "short_borrow_cost": borrow,
                "net_returns": net,
                "active_position_count": len(values),
                "contribution_inventory_sha256": semantic_sha256(
                    tuple(value[0] for value in values)
                ),
            }
            rows.append(
                MassiveProfitabilityDailyTranchePnlRowV1(
                    **body,  # type: ignore[arg-type]
                    receipt_sha256=semantic_sha256(body),
                )
            )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "schema": MASSIVE_FIXED_HORIZON_TRANCHES_V1_SCHEMA,
        "setting_id": selected.setting_id,
        "fold_index": selected.fold_index,
        "rows": tuple(asdict(row) for row in rows),
        "selected_tranches_semantic_receipt_sha256": selected.semantic_receipt_sha256,
        "target_accounting_inventory_sha256": target_inventory,
        "cost_rates": MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
        "annual_short_borrow_rate": 0.01,
        "row_inventory_sha256": row_inventory,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityFixedTranchePnlV1(
        setting_id=selected.setting_id,
        fold_index=selected.fold_index,
        rows=tuple(rows),
        selected_tranches_semantic_receipt_sha256=selected.semantic_receipt_sha256,
        target_accounting_inventory_sha256=target_inventory,
        cost_rates=MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
        annual_short_borrow_rate=0.01,
        row_inventory_sha256=row_inventory,
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityCapacityRowV1:
    setting_id: str
    fold_index: int
    capital_usd: float
    intended_order_count: int
    clipped_order_count: int
    mean_participation: float
    participation_p95: float
    maximum_participation: float
    lost_intended_notional_fraction: float
    clipped_mean_daily_net_return_20bp: float
    selected_tranches_semantic_receipt_sha256: str
    target_accounting_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            self.setting_id not in {"MV00", "MV02", "MV04", "MV04-SHUFFLE"}
            or isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < 4
            or self.capital_usd not in {1_000_000.0, 10_000_000.0, 50_000_000.0}
            or isinstance(self.intended_order_count, bool)
            or not isinstance(self.intended_order_count, int)
            or self.intended_order_count <= 0
            or isinstance(self.clipped_order_count, bool)
            or not isinstance(self.clipped_order_count, int)
            or not 0 <= self.clipped_order_count <= self.intended_order_count
        ):
            raise MassiveFixedHorizonTranchesV1Error("capacity counts differ")
        for value in (
            self.mean_participation,
            self.participation_p95,
            self.maximum_participation,
            self.lost_intended_notional_fraction,
        ):
            if not 0.0 <= _finite("capacity value", value):
                raise MassiveFixedHorizonTranchesV1Error("capacity value is negative")
        if not 0.0 <= self.lost_intended_notional_fraction <= 1.0:
            raise MassiveFixedHorizonTranchesV1Error("lost notional differs")
        _finite(
            "clipped mean daily net return",
            self.clipped_mean_daily_net_return_20bp,
        )
        for digest in (
            self.selected_tranches_semantic_receipt_sha256,
            self.target_accounting_inventory_sha256,
            self.receipt_sha256,
        ):
            _digest("capacity row", digest)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFixedHorizonTranchesV1Error("capacity row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityCapacityPanelV1:
    setting_id: str
    fold_index: int
    rows: tuple[MassiveProfitabilityCapacityRowV1, ...]
    selected_tranches_semantic_receipt_sha256: str
    target_accounting_inventory_sha256: str
    row_inventory_sha256: str
    semantic_receipt_sha256: str
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = "rl-quant.massive-profitability-capacity-panel-v1"

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("semantic_receipt_sha256")
        return body

    def validate(self) -> None:
        keys = tuple(row.capital_usd for row in self.rows)
        if (
            self.setting_id not in {"MV00", "MV02", "MV04", "MV04-SHUFFLE"}
            or isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < 4
            or keys != (1_000_000.0, 10_000_000.0, 50_000_000.0)
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveFixedHorizonTranchesV1Error("capacity panel identity differs")
        for row in self.rows:
            row.validate()
            if (
                row.setting_id != self.setting_id
                or row.fold_index != self.fold_index
                or row.selected_tranches_semantic_receipt_sha256
                != self.selected_tranches_semantic_receipt_sha256
                or row.target_accounting_inventory_sha256
                != self.target_accounting_inventory_sha256
            ):
                raise MassiveFixedHorizonTranchesV1Error(
                    "capacity row is detached from its panel"
                )
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFixedHorizonTranchesV1Error("capacity panel inventory differs")
        for digest in (
            self.selected_tranches_semantic_receipt_sha256,
            self.target_accounting_inventory_sha256,
            self.row_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("capacity panel", digest)


def evaluate_massive_profitability_entry_capacity_v1(
    *,
    selected: MassiveProfitabilitySelectedTranchesV1,
    target_accounting: Sequence[MassiveProfitabilityTargetAccountingAuthorityV2],
) -> MassiveProfitabilityCapacityPanelV1:
    """Clip entry orders at two percent of trailing ADV without reallocation."""

    selected.validate()
    paths, target_inventory = _path_map(target_accounting)
    if target_inventory != selected.target_accounting_inventory_sha256:
        raise MassiveFixedHorizonTranchesV1Error(
            "capacity target paths differ from guarded selection"
        )
    rows = []
    for capital in (1_000_000.0, 10_000_000.0, 50_000_000.0):
        participation = []
        intended_total = 0.0
        executable_total = 0.0
        clipped = 0
        daily_net: defaultdict[str, float] = defaultdict(float)
        for position in selected.positions:
            intended = abs(position.signed_entry_weight) * capital
            maximum = 0.02 * position.trailing_63_session_adv
            executable = min(intended, maximum)
            intended_total += intended
            executable_total += executable
            participation.append(executable / position.trailing_63_session_adv)
            clipped += intended > maximum
            scale = executable / intended
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
                daily_net[session_dates[offset]] += gross - cost - borrow
        ordered = sorted(participation)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        body = {
            "setting_id": selected.setting_id,
            "fold_index": selected.fold_index,
            "capital_usd": capital,
            "intended_order_count": len(selected.positions),
            "clipped_order_count": clipped,
            "mean_participation": math.fsum(participation) / len(participation),
            "participation_p95": ordered[p95_index],
            "maximum_participation": ordered[-1],
            "lost_intended_notional_fraction": (
                1.0 - executable_total / intended_total
            ),
            "clipped_mean_daily_net_return_20bp": (
                math.fsum(daily_net.values()) / len(daily_net)
            ),
            "selected_tranches_semantic_receipt_sha256": (
                selected.semantic_receipt_sha256
            ),
            "target_accounting_inventory_sha256": target_inventory,
        }
        rows.append(
            MassiveProfitabilityCapacityRowV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "schema": "rl-quant.massive-profitability-capacity-panel-v1",
        "setting_id": selected.setting_id,
        "fold_index": selected.fold_index,
        "rows": tuple(asdict(row) for row in rows),
        "selected_tranches_semantic_receipt_sha256": (selected.semantic_receipt_sha256),
        "target_accounting_inventory_sha256": target_inventory,
        "row_inventory_sha256": row_inventory,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityCapacityPanelV1(
        setting_id=selected.setting_id,
        fold_index=selected.fold_index,
        rows=tuple(rows),
        selected_tranches_semantic_receipt_sha256=selected.semantic_receipt_sha256,
        target_accounting_inventory_sha256=target_inventory,
        row_inventory_sha256=row_inventory,
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_FIXED_HORIZON_TRANCHES_V1_SCHEMA",
    "MassiveFixedHorizonTranchesV1Error",
    "MassiveProfitabilityCapacityPanelV1",
    "MassiveProfitabilityCapacityRowV1",
    "MassiveProfitabilityDailyTranchePnlRowV1",
    "MassiveProfitabilityFixedTranchePnlV1",
    "MassiveProfitabilityResidualInputRowV1",
    "MassiveProfitabilityResidualScoreRowV1",
    "MassiveProfitabilityResidualScoresV1",
    "MassiveProfitabilitySelectedTranchePositionV1",
    "MassiveProfitabilitySelectedTranchesV1",
    "build_massive_profitability_residual_scores_v1",
    "derive_massive_profitability_residual_inputs_v1",
    "evaluate_massive_profitability_entry_capacity_v1",
    "evaluate_massive_profitability_fixed_tranches_v1",
    "select_massive_profitability_tranches_v1",
]
