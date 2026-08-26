"""Frozen outer-fold evaluation plan for the Massive P0 profitability test."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_profitability_predictions_v1 import (
    MassiveProfitabilityOuterPredictionsV1,
)
from rl_quant.evaluation.massive_profitability_tournament_inputs_v1 import (
    build_massive_profitability_tournament_plan_v1,
)
from rl_quant.features.massive_profitability_data_gate_v2 import (
    MassiveProfitabilityDataGateV2,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityPhasePlanV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    MassiveProfitabilityTournamentPlanV1,
)

MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA = (
    "rl-quant.massive-profitability-evaluation-plan-v1"
)
MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_DATASET = (
    "massive-profitability-evaluation-plan-v1"
)
MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1 = (
    "MV00",
    "MV02",
    "MV04",
    "MV04-SHUFFLE",
)
MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1 = (1, 5, 21, 63)
MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1 = (0.001, 0.002, 0.004)
MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1 = (
    1_000_000.0,
    10_000_000.0,
    50_000_000.0,
)
MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1 = 26082026
MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2 = (0.25, 0.25, 0.25, 0.25)
MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_RECEIPTS_V2 = tuple(
    semantic_sha256(
        {
            "schema": "rl-quant.massive-profitability-fixed-horizon-scaling-v2",
            "fold_index": fold_index,
            "horizons": MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
            "weights": MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
            "source": "pre-registered-constant-no-outcome-input",
        }
    )
    for fold_index in range(4)
)
MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "settings": MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1,
        "score": "predicted-mean",
        "residualization": (
            "ridge-cross-sectional",
            1e-6,
            (
                "intercept",
                "log-source-economic-value",
                "log-trailing-63-session-adv",
                "reversal-5",
                "momentum-21-minus-5",
                "economic-volatility-63",
            ),
        ),
        "tail_fraction": 0.20,
        "horizons": MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
        "costs": MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
        "primary_cost": 0.002,
        "borrow_annual": 0.01,
        "risk": "fit-only-inverse-volatility-equal-risk",
        "bootstrap": (2_000, 63, MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1),
        "capital": MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1,
        "adv_limit": 0.02,
        "outer_only": True,
        "lockbox": False,
        "final_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityEvaluationPlanV1Error(ValueError):
    """The profitability evaluator differs from its pre-registered contract."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityEvaluationPlanV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityEvaluationPredictionV1:
    fold_index: int
    setting_id: str
    prediction_semantic_receipt_sha256: str
    outer_test_inventory_sha256: str
    ensemble: bool
    seed_inventory: tuple[int, ...]
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
            or self.setting_id not in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
            or (
                self.setting_id == "MV00"
                and (self.ensemble or self.seed_inventory != (0,))
            )
            or (
                self.setting_id != "MV00"
                and (
                    not self.ensemble
                    or self.seed_inventory
                    != MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
                )
            )
        ):
            raise MassiveProfitabilityEvaluationPlanV1Error(
                "evaluation prediction inventory differs"
            )
        for value in (
            self.prediction_semantic_receipt_sha256,
            self.outer_test_inventory_sha256,
            self.receipt_sha256,
        ):
            _digest("evaluation prediction", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityEvaluationPlanV1Error(
                "evaluation prediction receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityEvaluationPlanV1:
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    outer_fold_receipts: tuple[str, ...]
    horizon_risk_scaling_receipts: tuple[str, ...]
    predictions: tuple[MassiveProfitabilityEvaluationPredictionV1, ...]
    score_field: str
    residual_ridge_lambda: float
    residual_exposure_fields: tuple[str, ...]
    tail_fraction: float
    horizons: tuple[int, ...]
    cost_rates: tuple[float, ...]
    primary_cost_rate: float
    annual_short_borrow_rate: float
    bootstrap_block_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    capacity_capital_usd: tuple[float, ...]
    primary_capacity_usd: float
    adv_participation_limit: float
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    evaluator_retuning_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "loaded_source"}
        }

    def validate(self) -> None:
        keys = tuple((row.fold_index, row.setting_id) for row in self.predictions)
        expected = tuple(
            (fold, setting)
            for fold in range(4)
            for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
        )
        if (
            self.schema != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SHA256
            or len(self.outer_fold_receipts) != 4
            or len(self.horizon_risk_scaling_receipts) != 4
            or keys != expected
            or self.score_field != "mean"
            or self.residual_ridge_lambda != 1e-6
            or self.residual_exposure_fields
            != (
                "intercept",
                "log_source_economic_value",
                "log_trailing_63_session_adv",
                "reversal_5",
                "momentum_21_minus_5",
                "economic_volatility_63",
            )
            or self.tail_fraction != 0.20
            or self.horizons != MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1
            or self.cost_rates != MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1
            or self.primary_cost_rate != 0.002
            or self.annual_short_borrow_rate != 0.01
            or self.bootstrap_block_sessions != 63
            or self.bootstrap_replicates != 2_000
            or self.bootstrap_seed != MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1
            or self.capacity_capital_usd != MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1
            or self.primary_capacity_usd != 10_000_000.0
            or self.adv_participation_limit != 0.02
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.evaluator_retuning_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityEvaluationPlanV1Error(
                "evaluation plan identity or authorization differs"
            )
        for row in self.predictions:
            row.validate()
        for value in (
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            *self.outer_fold_receipts,
            *self.horizon_risk_scaling_receipts,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("evaluation plan", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityEvaluationPlanV1Error(
                "evaluation plan semantic receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.tournament_plan_receipt_sha256
        ):
            raise MassiveProfitabilityEvaluationPlanV1Error(
                "evaluation plan committed source differs"
            )


def _prediction_inventory(
    *,
    predictions: Sequence[MassiveProfitabilityOuterPredictionsV1],
    phase_plan: MassiveProfitabilityPhasePlanV1,
    tournament_plan: MassiveProfitabilityTournamentPlanV1,
) -> tuple[MassiveProfitabilityEvaluationPredictionV1, ...]:
    ordered = tuple(
        sorted(predictions, key=lambda row: (row.fold_index, row.setting_id))
    )
    if len(ordered) != 16:
        raise MassiveProfitabilityEvaluationPlanV1Error(
            "evaluation requires four settings for all four outer folds"
        )
    result: list[MassiveProfitabilityEvaluationPredictionV1] = []
    for prediction in ordered:
        prediction.validate()
        fold = phase_plan.outer_folds[prediction.fold_index]
        if (
            prediction.tournament_plan_receipt_sha256 != tournament_plan.receipt_sha256
            or prediction.fold_receipt_sha256 != fold.receipt_sha256
            or prediction.outer_test_session_dates != fold.outer_test_session_dates
        ):
            raise MassiveProfitabilityEvaluationPlanV1Error(
                "prediction is detached from the frozen outer fold"
            )
        body = {
            "fold_index": prediction.fold_index,
            "setting_id": prediction.setting_id,
            "prediction_semantic_receipt_sha256": prediction.semantic_receipt_sha256,
            "outer_test_inventory_sha256": fold.outer_test_inventory_sha256,
            "ensemble": prediction.ensemble,
            "seed_inventory": prediction.seed_inventory,
        }
        result.append(
            MassiveProfitabilityEvaluationPredictionV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    return tuple(result)


def materialize_massive_profitability_evaluation_plan_v1(
    *,
    root: str | Path,
    artifact_id: str,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV1,
    tournament_plan: MassiveProfitabilityTournamentPlanV1,
    predictions: Sequence[MassiveProfitabilityOuterPredictionsV1],
    horizon_risk_scaling_receipts: Sequence[str],
    committed_at_ms: int,
) -> MassiveProfitabilityEvaluationPlanV1:
    """Freeze the complete outer-evaluation contract before reading outer P&L."""

    data_gate.validate()
    phase_plan.validate()
    tournament_plan.validate()
    expected_tournament = build_massive_profitability_tournament_plan_v1(
        data_gate=data_gate, phase_plan=phase_plan
    )
    if expected_tournament.receipt_sha256 != tournament_plan.receipt_sha256:
        raise MassiveProfitabilityEvaluationPlanV1Error(
            "evaluation tournament plan is not regenerated from the gate"
        )
    inventory = _prediction_inventory(
        predictions=predictions,
        phase_plan=phase_plan,
        tournament_plan=tournament_plan,
    )
    risk_receipts = tuple(horizon_risk_scaling_receipts)
    if len(risk_receipts) != 4:
        raise MassiveProfitabilityEvaluationPlanV1Error(
            "evaluation plan requires one precomputed risk scale per outer fold"
        )
    for receipt in risk_receipts:
        _digest("horizon risk scaling", receipt)
    semantic = {
        "schema": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA,
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "tournament_plan_receipt_sha256": tournament_plan.receipt_sha256,
        "outer_fold_receipts": tuple(
            row.receipt_sha256 for row in phase_plan.outer_folds
        ),
        "horizon_risk_scaling_receipts": risk_receipts,
        "predictions": tuple(asdict(row) for row in inventory),
        "score_field": "mean",
        "residual_ridge_lambda": 1e-6,
        "residual_exposure_fields": (
            "intercept",
            "log_source_economic_value",
            "log_trailing_63_session_adv",
            "reversal_5",
            "momentum_21_minus_5",
            "economic_volatility_63",
        ),
        "tail_fraction": 0.20,
        "horizons": MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
        "cost_rates": MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
        "primary_cost_rate": 0.002,
        "annual_short_borrow_rate": 0.01,
        "bootstrap_block_sessions": 63,
        "bootstrap_replicates": 2_000,
        "bootstrap_seed": MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1,
        "capacity_capital_usd": MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1,
        "primary_capacity_usd": 10_000_000.0,
        "adv_participation_limit": 0.02,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SHA256
        ),
        "outer_evaluation_authorized": True,
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
        raise MassiveProfitabilityEvaluationPlanV1Error(
            "evaluation plan artifact ID is not path safe"
        )
    relative = f"massive-profitability-evaluation-plan-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=tournament_plan.receipt_sha256,
        committed_at_ms=committed_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    runtime = dict(semantic)
    runtime.pop("predictions")
    result = MassiveProfitabilityEvaluationPlanV1(
        **runtime,  # type: ignore[arg-type]
        predictions=inventory,
        loaded_source=loaded,
    )
    result.validate()
    return result


def parse_massive_profitability_evaluation_plan_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityEvaluationPlanV1:
    """Reload the immutable plan and regenerate its canonical bytes."""

    payload = json.loads(
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    )
    predictions = tuple(
        MassiveProfitabilityEvaluationPredictionV1(
            **{
                **row,
                "seed_inventory": tuple(row["seed_inventory"]),
            }
        )
        for row in payload.pop("predictions")
    )
    for name in (
        "outer_fold_receipts",
        "horizon_risk_scaling_receipts",
        "residual_exposure_fields",
        "horizons",
        "cost_rates",
        "capacity_capital_usd",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveProfitabilityEvaluationPlanV1(
        **payload, predictions=predictions, loaded_source=loaded_source
    )
    result.validate()
    expected = result.semantic_unsigned() | {
        "predictions": tuple(asdict(row) for row in result.predictions),
        "semantic_receipt_sha256": result.semantic_receipt_sha256,
    }
    if canonical_json_file_bytes(expected) != read_loaded_massive_source_bytes(
        root=root, loaded_source=loaded_source
    ):
        raise MassiveProfitabilityEvaluationPlanV1Error(
            "evaluation plan canonical bytes differ"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1",
    "MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1",
    "MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1",
    "MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1",
    "MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_RECEIPTS_V2",
    "MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2",
    "MassiveProfitabilityEvaluationPlanV1",
    "MassiveProfitabilityEvaluationPlanV1Error",
    "MassiveProfitabilityEvaluationPredictionV1",
    "materialize_massive_profitability_evaluation_plan_v1",
    "parse_massive_profitability_evaluation_plan_v1",
]
