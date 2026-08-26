"""Create-only outer predictions for the minimal Massive P0 tournament."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityOuterFoldPlanV1,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_HORIZONS_V1,
    MassiveProfitabilityTabularModelV1,
    massive_profitability_mv00_scores_v1,
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
    MassiveProfitabilityNormalizationV1,
    MassiveProfitabilityTournamentDatasetV1,
    MassiveProfitabilityTournamentPlanV1,
    MassiveProfitabilityTrainedRunV1,
    fit_massive_profitability_normalization_v1,
    massive_profitability_tape_permutation_v1,
)

MASSIVE_PROFITABILITY_PREDICTIONS_V1_SCHEMA = (
    "rl-quant.massive-profitability-predictions-v1"
)
MASSIVE_PROFITABILITY_PREDICTIONS_V1_DATASET = (
    "massive-profitability-predictions-v1"
)
MASSIVE_PROFITABILITY_PREDICTIONS_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_PREDICTIONS_V1_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_PREDICTIONS_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_PREDICTIONS_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "support": "exact-outer-fold-feature-cross-sections",
        "settings": ("MV00", "MV02", "MV04", "MV04-SHUFFLE"),
        "seed_ensemble": "five-seed-output-space-mean",
        "targets": "not-read-by-prediction",
        "lockbox": "prohibited",
        "publication": "create-only-canonical-json",
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityPredictionsV1Error(ValueError):
    """An outer prediction artifact differs from the frozen tournament."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityPredictionsV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveProfitabilityPredictionsV1Error(
            "prediction artifact ID is not path safe"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityPredictionRowV1:
    decision_session_date: str
    security_id: str
    mean: tuple[float, ...]
    downside_quantile: tuple[float, ...]
    median: tuple[float, ...]
    upside_quantile: tuple[float, ...]
    scale: tuple[float, ...]
    feature_semantic_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        values = (
            self.mean,
            self.downside_quantile,
            self.median,
            self.upside_quantile,
            self.scale,
        )
        if (
            not self.decision_session_date
            or not self.security_id
            or any(
                len(row) != len(MASSIVE_PROFITABILITY_HORIZONS_V1)
                or any(not math.isfinite(value) for value in row)
                for row in values
            )
            or any(value <= 0.0 for value in self.scale)
            or any(
                not lower <= middle <= upper
                for lower, middle, upper in zip(
                    self.downside_quantile,
                    self.median,
                    self.upside_quantile,
                    strict=True,
                )
            )
        ):
            raise MassiveProfitabilityPredictionsV1Error(
                "prediction row values differ"
            )
        _digest("prediction feature", self.feature_semantic_receipt_sha256)
        _digest("prediction row", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityPredictionsV1Error(
                "prediction row receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityOuterPredictionsV1:
    setting_id: str
    fold_index: int
    seed_inventory: tuple[int, ...]
    ensemble: bool
    outer_test_session_dates: tuple[str, ...]
    rows: tuple[MassiveProfitabilityPredictionRowV1, ...]
    feature_receipts: tuple[str, ...]
    model_run_receipts: tuple[str, ...]
    tournament_plan_receipt_sha256: str
    fold_receipt_sha256: str
    row_inventory_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    outer_prediction_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_PREDICTIONS_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "setting_id": self.setting_id,
            "fold_index": self.fold_index,
            "seed_inventory": self.seed_inventory,
            "ensemble": self.ensemble,
            "outer_test_session_dates": self.outer_test_session_dates,
            "rows": tuple(asdict(row) for row in self.rows),
            "feature_receipts": self.feature_receipts,
            "model_run_receipts": self.model_run_receipts,
            "tournament_plan_receipt_sha256": self.tournament_plan_receipt_sha256,
            "fold_receipt_sha256": self.fold_receipt_sha256,
            "row_inventory_sha256": self.row_inventory_sha256,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "outer_prediction_authorized": self.outer_prediction_authorized,
            "profitability_reporting_authorized": self.profitability_reporting_authorized,
            "lockbox_access_authorized": self.lockbox_access_authorized,
            "reinforcement_learning_authorized": self.reinforcement_learning_authorized,
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_PREDICTIONS_V1_SCHEMA
            or self.setting_id not in {"MV00", "MV02", "MV04", "MV04-SHUFFLE"}
            or isinstance(self.fold_index, bool)
            or not 0 <= self.fold_index < 4
            or not isinstance(self.ensemble, bool)
            or self.outer_test_session_dates
            != tuple(sorted(set(self.outer_test_session_dates)))
            or len(self.outer_test_session_dates) != 126
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_PREDICTIONS_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_PREDICTIONS_V1_SOURCE_SHA256
            or not self.outer_prediction_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityPredictionsV1Error(
                "prediction artifact identity or authorization differs"
            )
        if self.ensemble:
            if (
                self.setting_id == "MV00"
                or self.seed_inventory != MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
                or len(self.model_run_receipts)
                != len(MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1)
            ):
                raise MassiveProfitabilityPredictionsV1Error(
                    "prediction ensemble differs"
                )
        elif (
            len(self.seed_inventory) != 1
            or self.seed_inventory[0] not in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
            or len(self.model_run_receipts) != (0 if self.setting_id == "MV00" else 1)
        ):
            raise MassiveProfitabilityPredictionsV1Error(
                "single-run prediction seed inventory differs"
            )
        keys = tuple((row.decision_session_date, row.security_id) for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveProfitabilityPredictionsV1Error(
                "prediction rows are not canonical"
            )
        if tuple(sorted({value[0] for value in keys})) != self.outer_test_session_dates:
            raise MassiveProfitabilityPredictionsV1Error(
                "prediction dates differ from the outer fold"
            )
        if (
            len(self.feature_receipts) != len(self.outer_test_session_dates)
            or len(set(self.model_run_receipts)) != len(self.model_run_receipts)
        ):
            raise MassiveProfitabilityPredictionsV1Error(
                "prediction source inventories differ"
            )
        for row in self.rows:
            row.validate()
        feature_by_date = dict(
            zip(self.outer_test_session_dates, self.feature_receipts, strict=True)
        )
        for session_date in self.outer_test_session_dates:
            date_rows = tuple(
                row for row in self.rows if row.decision_session_date == session_date
            )
            if (
                len(date_rows) < 2
                or {row.feature_semantic_receipt_sha256 for row in date_rows}
                != {feature_by_date[session_date]}
            ):
                raise MassiveProfitabilityPredictionsV1Error(
                    "prediction feature support differs"
                )
        for value in (
            *self.feature_receipts,
            *self.model_run_receipts,
            self.tournament_plan_receipt_sha256,
            self.fold_receipt_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("prediction artifact", value)
        if (
            self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityPredictionsV1Error(
                "prediction artifact receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_PREDICTIONS_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_PREDICTIONS_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.tournament_plan_receipt_sha256
        ):
            raise MassiveProfitabilityPredictionsV1Error(
                "prediction source transaction differs"
            )


def _normalized(
    values: torch.Tensor,
    valid: torch.Tensor,
    mean: tuple[float, ...],
    scale: tuple[float, ...],
) -> torch.Tensor:
    location = torch.tensor(mean, dtype=values.dtype, device=values.device)
    spread = torch.tensor(scale, dtype=values.dtype, device=values.device)
    return torch.where(valid, (values - location) / spread, torch.zeros_like(values))


def _rows_from_model(
    *,
    dataset: MassiveProfitabilityTournamentDatasetV1,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    run: MassiveProfitabilityTrainedRunV1,
    device: torch.device,
) -> tuple[MassiveProfitabilityPredictionRowV1, ...]:
    run.validate()
    model = MassiveProfitabilityTabularModelV1(setting_id=run.setting_id).to(device)
    model.load_state_dict(run.state_dict(), strict=True)
    model.eval()
    mapping = dataset.by_date()
    target_location = torch.tensor(
        run.normalization.target_median, dtype=torch.float32, device=device
    )
    target_scale = torch.tensor(
        run.normalization.target_scale, dtype=torch.float32, device=device
    )
    result: list[MassiveProfitabilityPredictionRowV1] = []
    with torch.no_grad():
        for session_date in fold.outer_test_session_dates:
            row = mapping[session_date]
            bars_valid = row.bars_valid.to(device)
            tape_valid = row.tape_valid.to(device)
            bars = _normalized(
                row.bars_values.to(device),
                bars_valid,
                run.normalization.bars_mean,
                run.normalization.bars_scale,
            )
            tape = _normalized(
                row.tape_values.to(device),
                tape_valid,
                run.normalization.tape_mean,
                run.normalization.tape_scale,
            )
            if run.setting_id == "MV04-SHUFFLE":
                permutation = massive_profitability_tape_permutation_v1(
                    decision_session_date=session_date,
                    security_ids=row.security_ids,
                ).to(device)
                tape = tape.index_select(0, permutation)
                tape_valid = tape_valid.index_select(0, permutation)
            distribution = model(
                bars_values=bars.unsqueeze(0),
                bars_valid=bars_valid.unsqueeze(0),
                tape_values=tape.unsqueeze(0),
                tape_valid=tape_valid.unsqueeze(0),
                source_staleness=torch.zeros(
                    (1, len(row.security_ids), 1), dtype=torch.float32, device=device
                ),
            )
            outputs = (
                distribution.mean[0] * target_scale + target_location,
                distribution.downside_quantile[0] * target_scale + target_location,
                distribution.median[0] * target_scale + target_location,
                distribution.upside_quantile[0] * target_scale + target_location,
                distribution.scale[0] * target_scale,
            )
            for asset_index, security_id in enumerate(row.security_ids):
                body = {
                    "decision_session_date": session_date,
                    "security_id": security_id,
                    "mean": tuple(float(value) for value in outputs[0][asset_index]),
                    "downside_quantile": tuple(
                        float(value) for value in outputs[1][asset_index]
                    ),
                    "median": tuple(float(value) for value in outputs[2][asset_index]),
                    "upside_quantile": tuple(
                        float(value) for value in outputs[3][asset_index]
                    ),
                    "scale": tuple(float(value) for value in outputs[4][asset_index]),
                    "feature_semantic_receipt_sha256": (
                        row.feature_semantic_receipt_sha256
                    ),
                }
                result.append(
                    MassiveProfitabilityPredictionRowV1(
                        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
                    )
                )
    return tuple(result)


def _mv00_rows(
    *,
    dataset: MassiveProfitabilityTournamentDatasetV1,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    normalization: MassiveProfitabilityNormalizationV1,
) -> tuple[MassiveProfitabilityPredictionRowV1, ...]:
    mapping = dataset.by_date()
    result: list[MassiveProfitabilityPredictionRowV1] = []
    for session_date in fold.outer_test_session_dates:
        row = mapping[session_date]
        bars = _normalized(
            row.bars_values,
            row.bars_valid,
            normalization.bars_mean,
            normalization.bars_scale,
        )
        score = massive_profitability_mv00_scores_v1(
            bars_values=bars, bars_valid=row.bars_valid
        )
        for asset_index, security_id in enumerate(row.security_ids):
            mean = tuple(float(value) for value in score[asset_index])
            body = {
                "decision_session_date": session_date,
                "security_id": security_id,
                "mean": mean,
                "downside_quantile": tuple(value - 1.0 for value in mean),
                "median": mean,
                "upside_quantile": tuple(value + 1.0 for value in mean),
                "scale": (1.0,) * len(MASSIVE_PROFITABILITY_HORIZONS_V1),
                "feature_semantic_receipt_sha256": row.feature_semantic_receipt_sha256,
            }
            result.append(
                MassiveProfitabilityPredictionRowV1(
                    **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
                )
            )
    return tuple(result)


def _publish(
    *,
    root: str | Path,
    artifact_id: str,
    setting_id: str,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    seed_inventory: tuple[int, ...],
    ensemble: bool,
    rows: tuple[MassiveProfitabilityPredictionRowV1, ...],
    feature_receipts: tuple[str, ...],
    model_run_receipts: tuple[str, ...],
    tournament_plan: MassiveProfitabilityTournamentPlanV1,
    committed_at_ms: int,
) -> MassiveProfitabilityOuterPredictionsV1:
    artifact = _artifact_id(artifact_id)
    semantic = {
        "schema": MASSIVE_PROFITABILITY_PREDICTIONS_V1_SCHEMA,
        "setting_id": setting_id,
        "fold_index": fold.fold_index,
        "seed_inventory": seed_inventory,
        "ensemble": ensemble,
        "outer_test_session_dates": fold.outer_test_session_dates,
        "rows": tuple(asdict(row) for row in rows),
        "feature_receipts": feature_receipts,
        "model_run_receipts": model_run_receipts,
        "tournament_plan_receipt_sha256": tournament_plan.receipt_sha256,
        "fold_receipt_sha256": fold.receipt_sha256,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_PREDICTIONS_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_PREDICTIONS_V1_SOURCE_SHA256,
        "outer_prediction_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    relative = f"massive-profitability-predictions-v1/{artifact}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_PREDICTIONS_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_PREDICTIONS_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=tournament_plan.receipt_sha256,
        committed_at_ms=committed_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    runtime = dict(semantic)
    runtime.pop("rows")
    result = MassiveProfitabilityOuterPredictionsV1(
        **runtime,  # type: ignore[arg-type]
        rows=rows,
        loaded_source=loaded,
    )
    result.validate()
    return result


def publish_massive_profitability_outer_predictions_v1(
    *,
    root: str | Path,
    artifact_id: str,
    dataset: MassiveProfitabilityTournamentDatasetV1,
    tournament_plan: MassiveProfitabilityTournamentPlanV1,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    run: MassiveProfitabilityTrainedRunV1,
    committed_at_ms: int,
    device: str | torch.device = "cpu",
) -> MassiveProfitabilityOuterPredictionsV1:
    """Publish one seed's target-blind predictions for one outer fold."""

    dataset.validate()
    tournament_plan.validate()
    fold.validate()
    run.validate()
    if (
        not run.outer_prediction_authorized
        or run.fold_index != fold.fold_index
        or run.tournament_plan_receipt_sha256 != tournament_plan.receipt_sha256
        or dataset.data_gate_semantic_receipt_sha256
        != tournament_plan.data_gate_semantic_receipt_sha256
        or dataset.phase_plan_semantic_receipt_sha256
        != tournament_plan.phase_plan_semantic_receipt_sha256
        or fold.receipt_sha256 != tournament_plan.fold_receipts[fold.fold_index]
    ):
        raise MassiveProfitabilityPredictionsV1Error(
            "trained run is not authorized for this outer fold"
        )
    mapping = dataset.by_date()
    rows = _rows_from_model(
        dataset=dataset,
        fold=fold,
        run=run,
        device=torch.device(device),
    )
    return _publish(
        root=root,
        artifact_id=artifact_id,
        setting_id=run.setting_id,
        fold=fold,
        seed_inventory=(run.seed,),
        ensemble=False,
        rows=rows,
        feature_receipts=tuple(
            mapping[value].feature_semantic_receipt_sha256
            for value in fold.outer_test_session_dates
        ),
        model_run_receipts=(run.run_receipt_sha256,),
        tournament_plan=tournament_plan,
        committed_at_ms=committed_at_ms,
    )


def build_massive_profitability_recovery_predictions_for_test_v1(
    *,
    dataset: MassiveProfitabilityTournamentDatasetV1,
    tournament_plan: MassiveProfitabilityTournamentPlanV1,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    run: MassiveProfitabilityTrainedRunV1,
    device: str | torch.device = "cpu",
) -> tuple[MassiveProfitabilityPredictionRowV1, ...]:
    """Return target-blind rows for a deliberately nonauthorizing recovery run.

    The engineering recovery canary uses a shortened training configuration,
    so its run must never be published as an authorized outer prediction.  The
    same model-loading and target-blind inference path is exercised here, but
    no artifact or performance authorization is created.
    """

    dataset.validate()
    tournament_plan.validate()
    fold.validate()
    run.validate()
    if (
        run.outer_prediction_authorized
        or run.fold_index != fold.fold_index
        or run.tournament_plan_receipt_sha256 != tournament_plan.receipt_sha256
        or dataset.data_gate_semantic_receipt_sha256
        != tournament_plan.data_gate_semantic_receipt_sha256
        or dataset.phase_plan_semantic_receipt_sha256
        != tournament_plan.phase_plan_semantic_receipt_sha256
        or fold.receipt_sha256 != tournament_plan.fold_receipts[fold.fold_index]
    ):
        raise MassiveProfitabilityPredictionsV1Error(
            "recovery prediction run is not the expected nonauthorizing canary"
        )
    return _rows_from_model(
        dataset=dataset,
        fold=fold,
        run=run,
        device=torch.device(device),
    )


def publish_massive_profitability_mv00_outer_predictions_v1(
    *,
    root: str | Path,
    artifact_id: str,
    dataset: MassiveProfitabilityTournamentDatasetV1,
    tournament_plan: MassiveProfitabilityTournamentPlanV1,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    committed_at_ms: int,
) -> MassiveProfitabilityOuterPredictionsV1:
    """Publish the fixed bars sanity baseline without fitting model weights."""

    dataset.validate()
    tournament_plan.validate()
    fold.validate()
    if (
        dataset.data_gate_semantic_receipt_sha256
        != tournament_plan.data_gate_semantic_receipt_sha256
        or dataset.phase_plan_semantic_receipt_sha256
        != tournament_plan.phase_plan_semantic_receipt_sha256
        or fold.receipt_sha256 != tournament_plan.fold_receipts[fold.fold_index]
    ):
        raise MassiveProfitabilityPredictionsV1Error(
            "MV00 inputs are detached from the tournament plan"
        )
    normalization = fit_massive_profitability_normalization_v1(
        dataset=dataset, fit_session_dates=fold.fit_session_dates
    )
    mapping = dataset.by_date()
    return _publish(
        root=root,
        artifact_id=artifact_id,
        setting_id="MV00",
        fold=fold,
        seed_inventory=(0,),
        ensemble=False,
        rows=_mv00_rows(dataset=dataset, fold=fold, normalization=normalization),
        feature_receipts=tuple(
            mapping[value].feature_semantic_receipt_sha256
            for value in fold.outer_test_session_dates
        ),
        model_run_receipts=(),
        tournament_plan=tournament_plan,
        committed_at_ms=committed_at_ms,
    )


def publish_massive_profitability_seed_ensemble_v1(
    *,
    root: str | Path,
    artifact_id: str,
    predictions: Sequence[MassiveProfitabilityOuterPredictionsV1],
    tournament_plan: MassiveProfitabilityTournamentPlanV1,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    committed_at_ms: int,
) -> MassiveProfitabilityOuterPredictionsV1:
    """Average five seed outputs into one market-history prediction artifact."""

    ordered = tuple(sorted(predictions, key=lambda value: value.seed_inventory))
    for value in ordered:
        value.validate()
    tournament_plan.validate()
    fold.validate()
    if (
        len(ordered) != len(MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1)
        or tuple(value.seed_inventory[0] for value in ordered)
        != MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
        or any(value.ensemble for value in ordered)
        or any(value.setting_id == "MV00" for value in ordered)
        or len({value.setting_id for value in ordered}) != 1
        or any(value.fold_index != fold.fold_index for value in ordered)
        or any(
            value.tournament_plan_receipt_sha256 != tournament_plan.receipt_sha256
            for value in ordered
        )
    ):
        raise MassiveProfitabilityPredictionsV1Error(
            "seed ensemble inputs differ"
        )
    reference = ordered[0]
    if any(
        value.outer_test_session_dates != reference.outer_test_session_dates
        or value.feature_receipts != reference.feature_receipts
        or tuple(
            (row.decision_session_date, row.security_id) for row in value.rows
        )
        != tuple((row.decision_session_date, row.security_id) for row in reference.rows)
        for value in ordered[1:]
    ):
        raise MassiveProfitabilityPredictionsV1Error(
            "seed predictions do not share one market support"
        )
    rows: list[MassiveProfitabilityPredictionRowV1] = []
    for row_index, first in enumerate(reference.rows):
        seed_rows = tuple(value.rows[row_index] for value in ordered)
        averaged: dict[str, tuple[float, ...]] = {}
        for field in (
            "mean",
            "downside_quantile",
            "median",
            "upside_quantile",
            "scale",
        ):
            averaged[field] = tuple(
                sum(getattr(row, field)[index] for row in seed_rows) / len(seed_rows)
                for index in range(len(MASSIVE_PROFITABILITY_HORIZONS_V1))
            )
        body = {
            "decision_session_date": first.decision_session_date,
            "security_id": first.security_id,
            **averaged,
            "feature_semantic_receipt_sha256": first.feature_semantic_receipt_sha256,
        }
        rows.append(
            MassiveProfitabilityPredictionRowV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    return _publish(
        root=root,
        artifact_id=artifact_id,
        setting_id=reference.setting_id,
        fold=fold,
        seed_inventory=MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
        ensemble=True,
        rows=tuple(rows),
        feature_receipts=reference.feature_receipts,
        model_run_receipts=tuple(
            value.model_run_receipts[0] for value in ordered
        ),
        tournament_plan=tournament_plan,
        committed_at_ms=committed_at_ms,
    )


def parse_massive_profitability_outer_predictions_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityOuterPredictionsV1:
    """Reload one immutable prediction artifact and regenerate its exact bytes."""

    payload = json.loads(
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    )
    rows = tuple(
        MassiveProfitabilityPredictionRowV1(
            **{
                **row,
                "mean": tuple(row["mean"]),
                "downside_quantile": tuple(row["downside_quantile"]),
                "median": tuple(row["median"]),
                "upside_quantile": tuple(row["upside_quantile"]),
                "scale": tuple(row["scale"]),
            }
        )
        for row in payload.pop("rows")
    )
    for name in (
        "seed_inventory",
        "outer_test_session_dates",
        "feature_receipts",
        "model_run_receipts",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveProfitabilityOuterPredictionsV1(
        **payload, rows=rows, loaded_source=loaded_source
    )
    result.validate()
    if canonical_json_file_bytes(result.semantic_unsigned() | {
        "semantic_receipt_sha256": result.semantic_receipt_sha256
    }) != read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source):
        raise MassiveProfitabilityPredictionsV1Error(
            "prediction canonical bytes differ"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_PREDICTIONS_V1_SCHEMA",
    "MassiveProfitabilityOuterPredictionsV1",
    "MassiveProfitabilityPredictionRowV1",
    "MassiveProfitabilityPredictionsV1Error",
    "build_massive_profitability_recovery_predictions_for_test_v1",
    "parse_massive_profitability_outer_predictions_v1",
    "publish_massive_profitability_mv00_outer_predictions_v1",
    "publish_massive_profitability_outer_predictions_v1",
    "publish_massive_profitability_seed_ensemble_v1",
]
