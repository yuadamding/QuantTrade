"""Training-replay-bound evaluation plan for the embargoed V2 tournament."""

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
from rl_quant.evaluation.massive_profitability_evaluation_plan_v1 import (
    MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1,
    MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1,
    MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
    MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
    MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1,
    MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_RECEIPTS_V2,
    MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
    MassiveProfitabilityEvaluationPredictionV1,
)
from rl_quant.evaluation.massive_profitability_predictions_v3 import (
    MassiveProfitabilityOuterPredictionsV3,
)
from rl_quant.features.massive_profitability_data_gate_v2 import (
    MassiveProfitabilityDataGateV2,
)
from rl_quant.features.massive_profitability_phase_plan_v2 import (
    MassiveProfitabilityPhasePlanV2,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentPlanV2,
    parse_massive_profitability_tournament_plan_v2,
)
from rl_quant.training.massive_profitability_trained_run_v3 import (
    MassiveProfitabilityModelCheckpointV3,
)

MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SCHEMA = (
    "rl-quant.massive-profitability-evaluation-plan-v4"
)
MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_DATASET = (
    "massive-profitability-evaluation-plan-v4"
)
MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "phase": "embargoed-phase-plan-v2",
        "tournament": "reloaded-tournament-plan-v2",
        "predictions": "runtime-replayed-outer-predictions-v3",
        "score": "predicted-mean",
        "horizon_weights": MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
        "horizon_weight_source": "pre-registered-constant-no-outcome-input",
        "costs": MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
        "bootstrap": (2_000, 63, MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1),
        "capacity": (MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1, 0.02),
        "outer_only": True,
        "lockbox": False,
        "final_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityEvaluationPlanV4Error(ValueError):
    """The embargoed evaluation plan is detached from TournamentPlan V2."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityEvaluationPlanV4Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityEvaluationPlanV4:
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    tournament_plan_source_receipt_sha256: str
    outer_fold_receipts: tuple[str, ...]
    fixed_horizon_scaling_receipts: tuple[str, ...]
    predictions: tuple[MassiveProfitabilityEvaluationPredictionV1, ...]
    training_replay_checkpoint_receipts: tuple[str, ...]
    training_replay_run_receipts: tuple[str, ...]
    training_replay_inventory_sha256: str
    score_field: str
    residual_ridge_lambda: float
    residual_exposure_fields: tuple[str, ...]
    tail_fraction: float
    horizons: tuple[int, ...]
    horizon_weights: tuple[float, ...]
    horizon_weight_source: str
    cost_rates: tuple[float, ...]
    primary_cost_rate: float
    annual_short_borrow_rate: float
    bootstrap_block_sessions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    capacity_capital_usd: tuple[float, ...]
    capacity_capital_scope: str
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
    schema: str = MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "loaded_source"}
        }

    def validate(self) -> None:
        keys = tuple((row.fold_index, row.setting_id) for row in self.predictions)
        expected = tuple(
            (fold_index, setting)
            for fold_index in range(4)
            for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
        )
        if (
            self.schema != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SCHEMA
            or keys != expected
            or len(self.outer_fold_receipts) != 4
            or len(self.training_replay_checkpoint_receipts) != 60
            or len(self.training_replay_run_receipts) != 60
            or self.training_replay_inventory_sha256
            != semantic_sha256(
                tuple(
                    zip(
                        self.training_replay_checkpoint_receipts,
                        self.training_replay_run_receipts,
                        strict=True,
                    )
                )
            )
            or self.fixed_horizon_scaling_receipts
            != MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_RECEIPTS_V2
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
            or self.horizon_weights != MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2
            or self.horizon_weight_source != "pre-registered-constant-no-outcome-input"
            or self.cost_rates != MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1
            or self.primary_cost_rate != 0.002
            or self.annual_short_borrow_rate != 0.01
            or self.bootstrap_block_sessions != 63
            or self.bootstrap_replicates != 2_000
            or self.bootstrap_seed != MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1
            or self.capacity_capital_usd != MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1
            or self.capacity_capital_scope != "total-composite-capital"
            or self.primary_capacity_usd != 10_000_000.0
            or self.adv_participation_limit != 0.02
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SOURCE_SHA256
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.evaluator_retuning_authorized
            or self.reinforcement_learning_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityEvaluationPlanV4Error(
                "evaluation plan V3 identity or authorization differs"
            )
        for row in self.predictions:
            row.validate()
        for value in (
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            self.tournament_plan_source_receipt_sha256,
            *self.outer_fold_receipts,
            *self.training_replay_checkpoint_receipts,
            *self.training_replay_run_receipts,
            self.training_replay_inventory_sha256,
            *self.fixed_horizon_scaling_receipts,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("evaluation plan V3", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.tournament_plan_receipt_sha256
        ):
            raise MassiveProfitabilityEvaluationPlanV4Error(
                "evaluation plan V3 committed source differs"
            )


def materialize_massive_profitability_evaluation_plan_v4(
    *,
    root: str | Path,
    artifact_id: str,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    predictions: Sequence[MassiveProfitabilityOuterPredictionsV3],
    training_checkpoints: Sequence[MassiveProfitabilityModelCheckpointV3],
    committed_at_ms: int,
) -> MassiveProfitabilityEvaluationPlanV4:
    data_gate.validate()
    phase_plan.validate()
    tournament_plan.validate()
    reloaded = parse_massive_profitability_tournament_plan_v2(
        root=root, loaded_source=tournament_plan.loaded_source
    )
    ordered = tuple(sorted(predictions, key=lambda row: (row.fold_index, row.setting_id)))
    if any(not isinstance(row, MassiveProfitabilityOuterPredictionsV3) for row in ordered):
        raise MassiveProfitabilityEvaluationPlanV4Error(
            "EvaluationPlan V4 rejects every prediction generation except V3"
        )
    ordered_checkpoints = tuple(
        sorted(
            training_checkpoints,
            key=lambda value: (
                value.run.run_v2.run_v1.fold_index,
                value.run.run_v2.run_v1.setting_id,
                value.run.run_v2.run_v1.seed,
            ),
        )
    )
    replay_pairs = tuple(
        (
            value.run.checkpoint_v2_source_receipt_sha256,
            value.run.run_v2.run_receipt_sha256,
        )
        for value in ordered_checkpoints
    )
    prediction_pairs = tuple(
        pair
        for prediction in ordered
        if prediction.setting_id != "MV00"
        for pair in zip(
            prediction.checkpoint_source_receipts,
            prediction.trained_run_receipts,
            strict=True,
        )
    )
    expected_keys = tuple(
        (fold_index, setting)
        for fold_index in range(4)
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    feature_by_date = dict(
        zip(data_gate.gated_session_dates, data_gate.feature_receipts, strict=True)
    )
    if (
        not data_gate.data_gate_passed
        or not data_gate.outer_prediction_authorized
        or reloaded.receipt_sha256 != tournament_plan.receipt_sha256
        or tournament_plan.data_gate_semantic_receipt_sha256
        != data_gate.semantic_receipt_sha256
        or tournament_plan.phase_plan_semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
        or tournament_plan.fold_receipts
        != tuple(row.receipt_sha256 for row in phase_plan.outer_folds)
        or tuple((row.fold_index, row.setting_id) for row in ordered) != expected_keys
        or len(ordered_checkpoints) != 60
        or any(
            not value.run.runtime_training_replayed
            or not value.run.outer_prediction_authorized
            for value in ordered_checkpoints
        )
        or replay_pairs != prediction_pairs
    ):
        raise MassiveProfitabilityEvaluationPlanV4Error(
            "evaluation V4 roots do not share one embargoed tournament"
        )
    inventory = []
    for prediction in ordered:
        prediction.validate()
        fold = phase_plan.outer_folds[prediction.fold_index]
        expected_features = tuple(
            feature_by_date.get(value, "") for value in fold.outer_test_session_dates
        )
        if (
            not prediction.runtime_prediction_replayed
            or not prediction.outer_prediction_authorized
            or prediction.tournament_plan_receipt_sha256 != tournament_plan.receipt_sha256
            or prediction.tournament_plan_source_receipt_sha256
            != tournament_plan.loaded_source.receipt_sha256
            or prediction.phase_plan_semantic_receipt_sha256
            != phase_plan.semantic_receipt_sha256
            or prediction.fold_receipt_sha256 != fold.receipt_sha256
            or prediction.outer_test_session_dates != fold.outer_test_session_dates
            or prediction.feature_receipts != expected_features
            or any(not value for value in expected_features)
        ):
            raise MassiveProfitabilityEvaluationPlanV4Error(
                "prediction V3 is detached from the frozen V3 evaluation roots"
            )
        body = {
            "fold_index": prediction.fold_index,
            "setting_id": prediction.setting_id,
            "prediction_semantic_receipt_sha256": prediction.semantic_receipt_sha256,
            "outer_test_inventory_sha256": fold.outer_test_inventory_sha256,
            "ensemble": prediction.ensemble,
            "seed_inventory": prediction.seed_inventory,
        }
        row = MassiveProfitabilityEvaluationPredictionV1(
            **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
        )
        row.validate()
        inventory.append(row)
    semantic = {
        "schema": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SCHEMA,
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "tournament_plan_receipt_sha256": tournament_plan.receipt_sha256,
        "tournament_plan_source_receipt_sha256": tournament_plan.loaded_source.receipt_sha256,
        "outer_fold_receipts": tournament_plan.fold_receipts,
        "fixed_horizon_scaling_receipts": MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_RECEIPTS_V2,
        "predictions": tuple(asdict(row) for row in inventory),
        "training_replay_checkpoint_receipts": tuple(
            value.loaded_source.receipt_sha256 for value in ordered_checkpoints
        ),
        "training_replay_run_receipts": tuple(
            value.run.semantic_receipt_sha256 for value in ordered_checkpoints
        ),
        "training_replay_inventory_sha256": semantic_sha256(
            tuple(
                (
                    value.loaded_source.receipt_sha256,
                    value.run.semantic_receipt_sha256,
                )
                for value in ordered_checkpoints
            )
        ),
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
        "horizon_weights": MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
        "horizon_weight_source": "pre-registered-constant-no-outcome-input",
        "cost_rates": MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
        "primary_cost_rate": 0.002,
        "annual_short_borrow_rate": 0.01,
        "bootstrap_block_sessions": 63,
        "bootstrap_replicates": 2_000,
        "bootstrap_seed": MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1,
        "capacity_capital_usd": MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1,
        "capacity_capital_scope": "total-composite-capital",
        "primary_capacity_usd": 10_000_000.0,
        "adv_participation_limit": 0.02,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SOURCE_SHA256,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "evaluator_retuning_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    if not artifact_id or any(not (value.isalnum() or value in "-_") for value in artifact_id):
        raise MassiveProfitabilityEvaluationPlanV4Error(
            "evaluation plan V3 artifact ID is not path safe"
        )
    relative = f"massive-profitability/evaluation-plan-v4/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=tournament_plan.receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-EVALUATION-PLAN-V3-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return parse_massive_profitability_evaluation_plan_v4(
        root=root, loaded_source=loaded
    )


def parse_massive_profitability_evaluation_plan_v4(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityEvaluationPlanV4:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityEvaluationPlanV4Error(
            "evaluation plan V3 source is not canonical JSON"
        )
    predictions = tuple(
        MassiveProfitabilityEvaluationPredictionV1(
            **{**row, "seed_inventory": tuple(row["seed_inventory"])}
        )
        for row in payload.pop("predictions")
    )
    for name in (
        "outer_fold_receipts",
        "training_replay_checkpoint_receipts",
        "training_replay_run_receipts",
        "fixed_horizon_scaling_receipts",
        "residual_exposure_fields",
        "horizons",
        "horizon_weights",
        "cost_rates",
        "capacity_capital_usd",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveProfitabilityEvaluationPlanV4(
        **payload, predictions=predictions, loaded_source=loaded_source
    )
    result.validate()
    expected = result.semantic_unsigned() | {
        "predictions": tuple(asdict(row) for row in result.predictions),
        "semantic_receipt_sha256": result.semantic_receipt_sha256,
    }
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityEvaluationPlanV4Error(
            "evaluation plan V3 canonical bytes differ"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_DATASET",
    "MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SCHEMA",
    "MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_EVALUATION_PLAN_V4_SPEC_SHA256",
    "MassiveProfitabilityEvaluationPlanV4",
    "MassiveProfitabilityEvaluationPlanV4Error",
    "materialize_massive_profitability_evaluation_plan_v4",
    "parse_massive_profitability_evaluation_plan_v4",
]
