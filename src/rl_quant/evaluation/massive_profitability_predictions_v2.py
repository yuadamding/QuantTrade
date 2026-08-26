"""Create-only outer predictions bound to the embargoed V2 tournament."""

from __future__ import annotations

import json
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
from rl_quant.evaluation.massive_profitability_predictions_v1 import (
    MassiveProfitabilityPredictionRowV1,
    _mv00_rows,
    _rows_from_model,
)
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityOuterFoldPlanV1,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_HORIZONS_V1,
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
    MassiveProfitabilityTrainedRunV1,
    fit_massive_profitability_normalization_v1,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentDatasetV2,
    MassiveProfitabilityTournamentPlanV2,
)

MASSIVE_PROFITABILITY_PREDICTIONS_V2_SCHEMA = (
    "rl-quant.massive-profitability-predictions-v2"
)
MASSIVE_PROFITABILITY_PREDICTIONS_V2_DATASET = "massive-profitability-predictions-v2"
MASSIVE_PROFITABILITY_PREDICTIONS_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_PREDICTIONS_V2_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_PREDICTIONS_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_PREDICTIONS_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "tournament": "massive-profitability-tournament-v2",
        "phase": "embargoed-phase-plan-v2",
        "settings": ("MV00", "MV02", "MV04", "MV04-SHUFFLE"),
        "seed_ensemble": "five-seed-output-space-mean",
        "embargo_entry": "prohibited",
        "targets": "not-read-by-prediction",
        "lockbox": "prohibited",
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityPredictionsV2Error(ValueError):
    """An outer prediction differs from its embargoed tournament."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityPredictionsV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityOuterPredictionsV2:
    setting_id: str
    fold_index: int
    seed_inventory: tuple[int, ...]
    ensemble: bool
    outer_test_session_dates: tuple[str, ...]
    rows: tuple[MassiveProfitabilityPredictionRowV1, ...]
    feature_receipts: tuple[str, ...]
    model_run_receipts: tuple[str, ...]
    tournament_plan_receipt_sha256: str
    tournament_plan_source_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
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
    schema: str = MASSIVE_PROFITABILITY_PREDICTIONS_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "loaded_source"}
        }

    def validate(self) -> None:
        keys = tuple((row.decision_session_date, row.security_id) for row in self.rows)
        if (
            self.schema != MASSIVE_PROFITABILITY_PREDICTIONS_V2_SCHEMA
            or self.setting_id not in {"MV00", "MV02", "MV04", "MV04-SHUFFLE"}
            or not 0 <= self.fold_index < 4
            or self.outer_test_session_dates
            != tuple(sorted(set(self.outer_test_session_dates)))
            or len(self.outer_test_session_dates) != 126
            or not keys
            or keys != tuple(sorted(set(keys)))
            or tuple(sorted({key[0] for key in keys})) != self.outer_test_session_dates
            or len(self.feature_receipts) != len(self.outer_test_session_dates)
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_PROFITABILITY_PREDICTIONS_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_PREDICTIONS_V2_SOURCE_SHA256
            or not self.outer_prediction_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityPredictionsV2Error(
                "prediction V2 identity or authorization differs"
            )
        if self.ensemble:
            if (
                self.setting_id == "MV00"
                or self.seed_inventory != MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
                or len(self.model_run_receipts) != 5
            ):
                raise MassiveProfitabilityPredictionsV2Error(
                    "prediction V2 ensemble differs"
                )
        elif (
            len(self.seed_inventory) != 1
            or self.seed_inventory[0] not in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
            or len(self.model_run_receipts) != (0 if self.setting_id == "MV00" else 1)
        ):
            raise MassiveProfitabilityPredictionsV2Error(
                "prediction V2 seed inventory differs"
            )
        for row in self.rows:
            row.validate()
        by_date = dict(zip(self.outer_test_session_dates, self.feature_receipts, strict=True))
        if any(
            {
                row.feature_semantic_receipt_sha256
                for row in self.rows
                if row.decision_session_date == session_date
            }
            != {by_date[session_date]}
            for session_date in self.outer_test_session_dates
        ):
            raise MassiveProfitabilityPredictionsV2Error(
                "prediction V2 feature support differs"
            )
        for value in (
            *self.feature_receipts,
            *self.model_run_receipts,
            self.tournament_plan_receipt_sha256,
            self.tournament_plan_source_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.fold_receipt_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("prediction V2", value)
        if (
            self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityPredictionsV2Error(
                "prediction V2 receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_PROFITABILITY_PREDICTIONS_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_PREDICTIONS_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.tournament_plan_receipt_sha256
        ):
            raise MassiveProfitabilityPredictionsV2Error(
                "prediction V2 source transaction differs"
            )


def _publish_v2(
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
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    committed_at_ms: int,
) -> MassiveProfitabilityOuterPredictionsV2:
    tournament_plan.validate()
    fold.validate()
    if (
        fold.receipt_sha256 != tournament_plan.fold_receipts[fold.fold_index]
        or set(fold.outer_test_session_dates) & set(tournament_plan.embargo_session_dates)
    ):
        raise MassiveProfitabilityPredictionsV2Error(
            "prediction V2 fold differs from the embargoed tournament"
        )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_PREDICTIONS_V2_SCHEMA,
        "setting_id": setting_id,
        "fold_index": fold.fold_index,
        "seed_inventory": seed_inventory,
        "ensemble": ensemble,
        "outer_test_session_dates": fold.outer_test_session_dates,
        "rows": tuple(asdict(row) for row in rows),
        "feature_receipts": feature_receipts,
        "model_run_receipts": model_run_receipts,
        "tournament_plan_receipt_sha256": tournament_plan.receipt_sha256,
        "tournament_plan_source_receipt_sha256": tournament_plan.loaded_source.receipt_sha256,
        "phase_plan_semantic_receipt_sha256": tournament_plan.phase_plan_semantic_receipt_sha256,
        "fold_receipt_sha256": fold.receipt_sha256,
        "row_inventory_sha256": semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_PREDICTIONS_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_PREDICTIONS_V2_SOURCE_SHA256,
        "outer_prediction_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    if not artifact_id or any(not (value.isalnum() or value in "-_") for value in artifact_id):
        raise MassiveProfitabilityPredictionsV2Error(
            "prediction V2 artifact ID is not path safe"
        )
    relative = f"massive-profitability/predictions-v2/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_PREDICTIONS_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_PREDICTIONS_V2_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=tournament_plan.receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-PREDICTIONS-V2-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return parse_massive_profitability_outer_predictions_v2(
        root=root, loaded_source=loaded
    )


def publish_massive_profitability_outer_predictions_v2(
    *,
    root: str | Path,
    artifact_id: str,
    dataset: MassiveProfitabilityTournamentDatasetV2,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    run: MassiveProfitabilityTrainedRunV1,
    committed_at_ms: int,
    device: str | torch.device = "cpu",
) -> MassiveProfitabilityOuterPredictionsV2:
    dataset.validate()
    tournament_plan.validate()
    run.validate()
    if (
        not run.outer_prediction_authorized
        or run.fold_index != fold.fold_index
        or run.tournament_plan_receipt_sha256 != tournament_plan.receipt_sha256
        or dataset.data_gate_semantic_receipt_sha256
        != tournament_plan.data_gate_semantic_receipt_sha256
        or dataset.phase_plan_semantic_receipt_sha256
        != tournament_plan.phase_plan_semantic_receipt_sha256
    ):
        raise MassiveProfitabilityPredictionsV2Error(
            "trained run is detached from tournament V2"
        )
    mapping = dataset.by_date()
    rows = _rows_from_model(
        dataset=dataset,  # type: ignore[arg-type]
        fold=fold,
        run=run,
        device=torch.device(device),
    )
    return _publish_v2(
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


def publish_massive_profitability_mv00_outer_predictions_v2(
    *,
    root: str | Path,
    artifact_id: str,
    dataset: MassiveProfitabilityTournamentDatasetV2,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    committed_at_ms: int,
) -> MassiveProfitabilityOuterPredictionsV2:
    dataset.validate()
    tournament_plan.validate()
    normalization = fit_massive_profitability_normalization_v1(
        dataset=dataset,  # type: ignore[arg-type]
        fit_session_dates=fold.fit_session_dates,
    )
    mapping = dataset.by_date()
    return _publish_v2(
        root=root,
        artifact_id=artifact_id,
        setting_id="MV00",
        fold=fold,
        seed_inventory=(0,),
        ensemble=False,
        rows=_mv00_rows(
            dataset=dataset,  # type: ignore[arg-type]
            fold=fold,
            normalization=normalization,
        ),
        feature_receipts=tuple(
            mapping[value].feature_semantic_receipt_sha256
            for value in fold.outer_test_session_dates
        ),
        model_run_receipts=(),
        tournament_plan=tournament_plan,
        committed_at_ms=committed_at_ms,
    )


def publish_massive_profitability_seed_ensemble_v2(
    *,
    root: str | Path,
    artifact_id: str,
    predictions: Sequence[MassiveProfitabilityOuterPredictionsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityOuterFoldPlanV1,
    committed_at_ms: int,
) -> MassiveProfitabilityOuterPredictionsV2:
    ordered = tuple(sorted(predictions, key=lambda value: value.seed_inventory))
    for value in ordered:
        value.validate()
    if (
        len(ordered) != 5
        or tuple(row.seed_inventory[0] for row in ordered)
        != MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
        or any(row.ensemble or row.setting_id == "MV00" for row in ordered)
        or len({row.setting_id for row in ordered}) != 1
        or any(row.fold_index != fold.fold_index for row in ordered)
        or any(row.tournament_plan_receipt_sha256 != tournament_plan.receipt_sha256 for row in ordered)
    ):
        raise MassiveProfitabilityPredictionsV2Error(
            "prediction V2 ensemble inputs differ"
        )
    reference = ordered[0]
    if any(
        row.outer_test_session_dates != reference.outer_test_session_dates
        or row.feature_receipts != reference.feature_receipts
        or tuple((item.decision_session_date, item.security_id) for item in row.rows)
        != tuple((item.decision_session_date, item.security_id) for item in reference.rows)
        for row in ordered[1:]
    ):
        raise MassiveProfitabilityPredictionsV2Error(
            "prediction V2 ensemble support differs"
        )
    rows: list[MassiveProfitabilityPredictionRowV1] = []
    for index, first in enumerate(reference.rows):
        seed_rows = tuple(row.rows[index] for row in ordered)
        averaged = {
            field: tuple(
                sum(getattr(row, field)[horizon] for row in seed_rows) / 5
                for horizon in range(len(MASSIVE_PROFITABILITY_HORIZONS_V1))
            )
            for field in ("mean", "downside_quantile", "median", "upside_quantile", "scale")
        }
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
    return _publish_v2(
        root=root,
        artifact_id=artifact_id,
        setting_id=reference.setting_id,
        fold=fold,
        seed_inventory=MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
        ensemble=True,
        rows=tuple(rows),
        feature_receipts=reference.feature_receipts,
        model_run_receipts=tuple(row.model_run_receipts[0] for row in ordered),
        tournament_plan=tournament_plan,
        committed_at_ms=committed_at_ms,
    )


def parse_massive_profitability_outer_predictions_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityOuterPredictionsV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityPredictionsV2Error(
            "prediction V2 source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityPredictionsV2Error(
            "prediction V2 source is not canonical JSON"
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
    result = MassiveProfitabilityOuterPredictionsV2(
        **payload, rows=rows, loaded_source=loaded_source
    )
    result.validate()
    expected = result.semantic_unsigned() | {
        "rows": tuple(asdict(row) for row in result.rows),
        "semantic_receipt_sha256": result.semantic_receipt_sha256,
    }
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityPredictionsV2Error(
            "prediction V2 canonical bytes differ"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_PREDICTIONS_V2_DATASET",
    "MASSIVE_PROFITABILITY_PREDICTIONS_V2_SCHEMA",
    "MASSIVE_PROFITABILITY_PREDICTIONS_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_PREDICTIONS_V2_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_PREDICTIONS_V2_SPEC_SHA256",
    "MassiveProfitabilityOuterPredictionsV2",
    "MassiveProfitabilityPredictionsV2Error",
    "parse_massive_profitability_outer_predictions_v2",
    "publish_massive_profitability_mv00_outer_predictions_v2",
    "publish_massive_profitability_outer_predictions_v2",
    "publish_massive_profitability_seed_ensemble_v2",
]
