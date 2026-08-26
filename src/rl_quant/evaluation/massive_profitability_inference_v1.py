"""Equal-risk composite and paired outer-fold inference for Massive P0."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

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
    MassiveProfitabilityCapacityPanelV1,
    MassiveProfitabilityFixedTranchePnlV1,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v1 import (
    MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
    MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1,
    MassiveProfitabilityEvaluationPlanV1,
    parse_massive_profitability_evaluation_plan_v1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_INFERENCE_V1_SCHEMA = (
    "rl-quant.massive-profitability-inference-v1"
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_SCHEMA = (
    "rl-quant.massive-profitability-outer-evidence-v1"
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_DATASET = (
    "massive-profitability-outer-evidence-v1"
)
MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_INFERENCE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_INFERENCE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "composite": "fit-only-inverse-volatility-equal-risk-four-horizon",
        "primary": "mean-daily-net-return-20bp",
        "contrasts": (("MV04", "MV02"), ("MV04", "MV04-SHUFFLE")),
        "bootstrap": "four-fold-cluster-nonwrapping-63-session-2000-replicate",
        "diagnostics": (
            "annualized-return-volatility-sharpe-sortino-drawdown-hit-rate",
            "implied-one-way-turnover-and-break-even-cost",
            "10m-clipped-net-return",
        ),
        "outer_only": True,
        "final_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveProfitabilityInferenceV1Error(ValueError):
    """Outer-fold profitability evidence differs from the frozen plan."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityInferenceV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MassiveProfitabilityInferenceV1Error(f"{name} must be finite")
    return float(value)


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
class MassiveProfitabilityHorizonRiskScalingV1:
    fold_index: int
    horizon_volatility: tuple[float, ...]
    horizon_weights: tuple[float, ...]
    fit_source_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < 4
            or len(self.horizon_volatility) != 4
            or len(self.horizon_weights) != 4
            or any(
                _finite("horizon volatility", value) <= 0.0
                for value in self.horizon_volatility
            )
            or any(
                _finite("horizon weight", value) <= 0.0
                for value in self.horizon_weights
            )
            or not math.isclose(
                math.fsum(self.horizon_weights), 1.0, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise MassiveProfitabilityInferenceV1Error("horizon risk scaling differs")
        _digest("fit risk source", self.fit_source_receipt_sha256)
        _digest("risk scaling", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityInferenceV1Error("risk scaling receipt differs")


def build_massive_profitability_horizon_risk_scaling_v1(
    *,
    fold_index: int,
    fit_daily_returns_by_horizon: Mapping[int, Sequence[float]],
    fit_source_receipt_sha256: str,
) -> MassiveProfitabilityHorizonRiskScalingV1:
    """Estimate inverse-volatility horizon weights only from frozen fit returns."""

    if tuple(sorted(fit_daily_returns_by_horizon)) != (
        MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1
    ):
        raise MassiveProfitabilityInferenceV1Error(
            "fit risk source lacks one exact horizon inventory"
        )
    volatility = []
    for horizon in MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1:
        values = tuple(
            _finite("fit sleeve return", value)
            for value in fit_daily_returns_by_horizon[horizon]
        )
        if len(values) < 2:
            raise MassiveProfitabilityInferenceV1Error(
                "fit sleeve volatility has insufficient history"
            )
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        if variance <= 0.0:
            raise MassiveProfitabilityInferenceV1Error(
                "fit sleeve volatility is nonpositive"
            )
        volatility.append(math.sqrt(variance))
    inverse = tuple(1.0 / value for value in volatility)
    total = math.fsum(inverse)
    weights = tuple(value / total for value in inverse)
    body = {
        "fold_index": fold_index,
        "horizon_volatility": tuple(volatility),
        "horizon_weights": weights,
        "fit_source_receipt_sha256": fit_source_receipt_sha256,
    }
    result = MassiveProfitabilityHorizonRiskScalingV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityCompositeDailyRowV1:
    session_date: str
    gross_return: float
    net_returns: tuple[float, ...]
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
            or len(self.horizon_row_receipts) != 4
        ):
            raise MassiveProfitabilityInferenceV1Error("composite daily row differs")
        for value in (self.gross_return, *self.net_returns):
            _finite("composite return", value)
        for digest in (*self.horizon_row_receipts, self.receipt_sha256):
            _digest("composite row", digest)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityInferenceV1Error("composite row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityCompositePnlV1:
    setting_id: str
    fold_index: int
    rows: tuple[MassiveProfitabilityCompositeDailyRowV1, ...]
    fixed_tranche_pnl_semantic_receipt_sha256: str
    risk_scaling_receipt_sha256: str
    row_inventory_sha256: str
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
        dates = tuple(row.session_date for row in self.rows)
        if (
            self.setting_id not in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
            or not self.rows
            or dates != tuple(sorted(set(dates)))
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityInferenceV1Error("composite P&L identity differs")
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ) or self.semantic_receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityInferenceV1Error(
                "composite P&L inventory differs"
            )
        for value in (
            self.fixed_tranche_pnl_semantic_receipt_sha256,
            self.risk_scaling_receipt_sha256,
            self.row_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("composite P&L", value)


def build_massive_profitability_composite_pnl_v1(
    *,
    pnl: MassiveProfitabilityFixedTranchePnlV1,
    risk_scaling: MassiveProfitabilityHorizonRiskScalingV1,
) -> MassiveProfitabilityCompositePnlV1:
    """Combine four horizon sleeves with fit-only frozen risk weights."""

    pnl.validate()
    risk_scaling.validate()
    if pnl.fold_index != risk_scaling.fold_index:
        raise MassiveProfitabilityInferenceV1Error(
            "composite risk scaling belongs to another fold"
        )
    by_key = {(row.session_date, row.horizon_sessions): row for row in pnl.rows}
    dates = tuple(sorted({row.session_date for row in pnl.rows}))
    rows = []
    for session_date in dates:
        horizon_rows = []
        for horizon in MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1:
            row = by_key.get((session_date, horizon))
            if row is None:
                raise MassiveProfitabilityInferenceV1Error(
                    "composite date lacks one horizon sleeve"
                )
            horizon_rows.append(row)
        gross = math.fsum(
            risk_scaling.horizon_weights[index] * row.gross_return
            for index, row in enumerate(horizon_rows)
        )
        net = tuple(
            math.fsum(
                risk_scaling.horizon_weights[horizon_index]
                * horizon_rows[horizon_index].net_returns[cost_index]
                for horizon_index in range(4)
            )
            for cost_index in range(3)
        )
        body = {
            "session_date": session_date,
            "gross_return": gross,
            "net_returns": net,
            "horizon_row_receipts": tuple(row.receipt_sha256 for row in horizon_rows),
        }
        rows.append(
            MassiveProfitabilityCompositeDailyRowV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "setting_id": pnl.setting_id,
        "fold_index": pnl.fold_index,
        "rows": tuple(asdict(row) for row in rows),
        "fixed_tranche_pnl_semantic_receipt_sha256": pnl.semantic_receipt_sha256,
        "risk_scaling_receipt_sha256": risk_scaling.receipt_sha256,
        "row_inventory_sha256": row_inventory,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityCompositePnlV1(
        setting_id=pnl.setting_id,
        fold_index=pnl.fold_index,
        rows=tuple(rows),
        fixed_tranche_pnl_semantic_receipt_sha256=pnl.semantic_receipt_sha256,
        risk_scaling_receipt_sha256=risk_scaling.receipt_sha256,
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
class MassiveProfitabilityOuterEvidenceV1:
    evaluation_plan_semantic_receipt_sha256: str
    composite_receipts: tuple[str, ...]
    capacity_inventory_sha256: str
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
    positive_mv04_fold_count: int
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
    schema: str = MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "loaded_source"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_INFERENCE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_INFERENCE_V1_SOURCE_SHA256
            or len(self.composite_receipts) != 16
            or not 0 <= self.positive_mv04_fold_count <= 4
            or self.outer_profitability_gate_passed
            != (
                self.mv04_net_20bp_lcb95 > 0.0
                and self.mv04_minus_mv02_net_20bp_lcb95 > 0.0
                and self.mv04_minus_shuffle_net_20bp_lcb95 > 0.0
                and self.positive_mv04_fold_count >= 3
                and self.mean_mv04_net_40bp >= 0.0
                and self.mean_mv04_clipped_10m_net_20bp > 0.0
            )
            or not self.development_conclusion_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.evaluator_retuning_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityInferenceV1Error(
                "outer profitability evidence differs"
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
            _finite("outer evidence statistic", value)
        if (
            self.annualized_mv04_net_volatility_20bp < 0.0
            or self.mv04_maximum_drawdown_20bp > 0.0
            or not 0.0 <= self.mv04_hit_rate_20bp <= 1.0
            or self.mv04_mean_one_way_turnover < 0.0
        ):
            raise MassiveProfitabilityInferenceV1Error(
                "outer profitability diagnostic range differs"
            )
        for digest in (
            self.evaluation_plan_semantic_receipt_sha256,
            *self.composite_receipts,
            self.capacity_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("outer evidence", digest)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityInferenceV1Error(
                "outer evidence semantic receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.evaluation_plan_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityInferenceV1Error("outer evidence source differs")


def materialize_massive_profitability_outer_evidence_v1(
    *,
    root: str | Path,
    artifact_id: str,
    evaluation_plan: MassiveProfitabilityEvaluationPlanV1,
    composites: Sequence[MassiveProfitabilityCompositePnlV1],
    capacity_panels: Sequence[MassiveProfitabilityCapacityPanelV1],
    committed_at_ms: int,
) -> MassiveProfitabilityOuterEvidenceV1:
    """Publish paired outer evidence while preserving all downstream prohibitions."""

    evaluation_plan.validate()
    reloaded_plan = parse_massive_profitability_evaluation_plan_v1(
        root=root, loaded_source=evaluation_plan.loaded_source
    )
    if reloaded_plan.semantic_receipt_sha256 != evaluation_plan.semantic_receipt_sha256:
        raise MassiveProfitabilityInferenceV1Error(
            "evaluation plan differs after committed-byte reload"
        )
    ordered = tuple(
        sorted(composites, key=lambda row: (row.fold_index, row.setting_id))
    )
    for row in ordered:
        row.validate()
        if (
            row.risk_scaling_receipt_sha256
            != evaluation_plan.horizon_risk_scaling_receipts[row.fold_index]
        ):
            raise MassiveProfitabilityInferenceV1Error(
                "outer composite uses a risk scale outside the frozen plan"
            )
    expected = tuple(
        (fold, setting)
        for fold in range(4)
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    if tuple((row.fold_index, row.setting_id) for row in ordered) != expected:
        raise MassiveProfitabilityInferenceV1Error(
            "outer evidence lacks one complete setting-fold inventory"
        )
    by_key = {(row.fold_index, row.setting_id): row for row in ordered}
    mv04_folds: list[tuple[float, ...]] = []
    delta_bars_folds: list[tuple[float, ...]] = []
    delta_shuffle_folds: list[tuple[float, ...]] = []
    mv04_40: list[float] = []
    positive_folds = 0
    for fold in range(4):
        models = {
            setting: by_key[(fold, setting)]
            for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
        }
        dates = tuple(row.session_date for row in models["MV04"].rows)
        if any(
            tuple(row.session_date for row in models[setting].rows) != dates
            for setting in models
        ):
            raise MassiveProfitabilityInferenceV1Error(
                "paired outer settings do not share exact dates"
            )
        values = {
            setting: tuple(row.net_returns[1] for row in models[setting].rows)
            for setting in models
        }
        mv04_folds.append(values["MV04"])
        delta_bars_folds.append(
            tuple(
                left - right
                for left, right in zip(values["MV04"], values["MV02"], strict=True)
            )
        )
        delta_shuffle_folds.append(
            tuple(
                left - right
                for left, right in zip(
                    values["MV04"], values["MV04-SHUFFLE"], strict=True
                )
            )
        )
        fold_mean = math.fsum(values["MV04"]) / len(values["MV04"])
        positive_folds += fold_mean > 0.0
        mv04_40.extend(row.net_returns[2] for row in models["MV04"].rows)
    bootstrap = FoldBootstrapConfig(
        replicates=evaluation_plan.bootstrap_replicates,
        block_sessions=evaluation_plan.bootstrap_block_sessions,
        seed=evaluation_plan.bootstrap_seed,
        lower_probability=0.025,
    )
    flat_mv04 = tuple(value for fold in mv04_folds for value in fold)
    flat_bars = tuple(value for fold in delta_bars_folds for value in fold)
    flat_shuffle = tuple(value for fold in delta_shuffle_folds for value in fold)
    mv04_10 = tuple(
        row.net_returns[0] for fold in range(4) for row in by_key[(fold, "MV04")].rows
    )
    mv04_40_tuple = tuple(mv04_40)
    daily_mean = math.fsum(flat_mv04) / len(flat_mv04)
    daily_variance = math.fsum((value - daily_mean) ** 2 for value in flat_mv04) / len(
        flat_mv04
    )
    daily_volatility = math.sqrt(daily_variance)
    downside = tuple(min(value, 0.0) for value in flat_mv04)
    downside_deviation = math.sqrt(
        math.fsum(value * value for value in downside) / len(downside)
    )
    mean_turnover = math.fsum(
        (low_cost - high_cost) / (0.004 - 0.001)
        for low_cost, high_cost in zip(mv04_10, mv04_40_tuple, strict=True)
    ) / len(flat_mv04)
    statistics = {
        "mean_mv04_net_20bp": daily_mean,
        "mean_mv04_minus_mv02_net_20bp": math.fsum(flat_bars) / len(flat_bars),
        "mean_mv04_minus_shuffle_net_20bp": math.fsum(flat_shuffle) / len(flat_shuffle),
        "mv04_net_20bp_lcb95": fold_cluster_block_bootstrap_lcb(mv04_folds, bootstrap),
        "mv04_minus_mv02_net_20bp_lcb95": fold_cluster_block_bootstrap_lcb(
            delta_bars_folds, bootstrap
        ),
        "mv04_minus_shuffle_net_20bp_lcb95": fold_cluster_block_bootstrap_lcb(
            delta_shuffle_folds, bootstrap
        ),
        "mean_mv04_net_40bp": math.fsum(mv04_40) / len(mv04_40),
        "positive_mv04_fold_count": positive_folds,
        "annualized_mv04_net_return_20bp": 252.0 * daily_mean,
        "annualized_mv04_net_volatility_20bp": math.sqrt(252.0) * daily_volatility,
        "mv04_net_sharpe_20bp": (
            0.0
            if daily_volatility == 0.0
            else math.sqrt(252.0) * daily_mean / daily_volatility
        ),
        "mv04_net_sortino_20bp": (
            0.0
            if downside_deviation == 0.0
            else math.sqrt(252.0) * daily_mean / downside_deviation
        ),
        "mv04_maximum_drawdown_20bp": _maximum_drawdown(flat_mv04),
        "mv04_hit_rate_20bp": sum(value > 0.0 for value in flat_mv04) / len(flat_mv04),
        "mv04_mean_one_way_turnover": mean_turnover,
        "mv04_break_even_one_way_cost": (
            0.002 + daily_mean / mean_turnover if mean_turnover > 0.0 else 0.0
        ),
    }
    passed = (
        statistics["mv04_net_20bp_lcb95"] > 0.0
        and statistics["mv04_minus_mv02_net_20bp_lcb95"] > 0.0
        and statistics["mv04_minus_shuffle_net_20bp_lcb95"] > 0.0
        and positive_folds >= 3
    )
    ordered_capacity_panels = tuple(
        sorted(capacity_panels, key=lambda row: (row.fold_index, row.setting_id))
    )
    for capacity_panel in ordered_capacity_panels:
        capacity_panel.validate()
    expected_capacity = tuple(
        (fold, setting)
        for fold in range(4)
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    if (
        tuple((row.fold_index, row.setting_id) for row in ordered_capacity_panels)
        != expected_capacity
    ):
        raise MassiveProfitabilityInferenceV1Error(
            "outer evidence lacks one exact capacity panel"
        )
    mv04_capacity_10m = tuple(
        next(
            row.clipped_mean_daily_net_return_20bp
            for row in panel.rows
            if row.capital_usd == 10_000_000.0
        )
        for panel in ordered_capacity_panels
        if panel.setting_id == "MV04"
    )
    statistics["mean_mv04_clipped_10m_net_20bp"] = math.fsum(mv04_capacity_10m) / len(
        mv04_capacity_10m
    )
    passed = (
        passed
        and statistics["mean_mv04_net_40bp"] >= 0.0
        and statistics["mean_mv04_clipped_10m_net_20bp"] > 0.0
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_SCHEMA,
        "evaluation_plan_semantic_receipt_sha256": (
            evaluation_plan.semantic_receipt_sha256
        ),
        "composite_receipts": tuple(row.semantic_receipt_sha256 for row in ordered),
        "capacity_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in ordered_capacity_panels)
        ),
        **statistics,
        "outer_profitability_gate_passed": passed,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_INFERENCE_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_INFERENCE_V1_SOURCE_SHA256,
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
        raise MassiveProfitabilityInferenceV1Error(
            "outer evidence artifact ID is not path safe"
        )
    relative = f"massive-profitability-outer-evidence-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=evaluation_plan.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    result = MassiveProfitabilityOuterEvidenceV1(
        **semantic,  # type: ignore[arg-type]
        loaded_source=loaded,
    )
    result.validate()
    return result


def parse_massive_profitability_outer_evidence_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityOuterEvidenceV1:
    payload = json.loads(
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    )
    payload["composite_receipts"] = tuple(payload["composite_receipts"])
    result = MassiveProfitabilityOuterEvidenceV1(**payload, loaded_source=loaded_source)
    result.validate()
    if canonical_json_file_bytes(
        result.semantic_unsigned()
        | {"semantic_receipt_sha256": result.semantic_receipt_sha256}
    ) != read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source):
        raise MassiveProfitabilityInferenceV1Error(
            "outer evidence canonical bytes differ"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_OUTER_EVIDENCE_V1_SCHEMA",
    "MassiveProfitabilityCompositeDailyRowV1",
    "MassiveProfitabilityCompositePnlV1",
    "MassiveProfitabilityHorizonRiskScalingV1",
    "MassiveProfitabilityInferenceV1Error",
    "MassiveProfitabilityOuterEvidenceV1",
    "build_massive_profitability_composite_pnl_v1",
    "build_massive_profitability_horizon_risk_scaling_v1",
    "materialize_massive_profitability_outer_evidence_v1",
    "parse_massive_profitability_outer_evidence_v1",
]
