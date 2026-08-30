"""Create-only target-free forecast replay for the frozen outer chronology."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v1 import (
    MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256,
    MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MASSIVE_ADAPTIVE_FORECAST_V2_FLOAT_ARRAY_NAMES,
    MassiveAdaptiveForecastRowV2,
    replay_massive_adaptive_forecast_rows_v2,
)
from rl_quant.evaluation.massive_adaptive_outer_inference_plan_v1 import (
    MassiveAdaptiveOuterInferencePlanV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256,
    MassiveAdaptiveAlphaModelSpecV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)

if TYPE_CHECKING:
    from rl_quant.training.massive_adaptive_profit_checkpoint_selection_authority_v2 import (
        MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    )

MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-outer-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_DATASET = (
    "massive-adaptive-outer-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SCHEMA,
        "encoding": "torch-tensor-archive-loaded-weights-only",
        "publication": "create-only-source-transaction",
        "runtime": "withheld-until-selection-and-forecast-replay",
    }
)
MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "checkpoint": "frozen-source-derived-inner-validation-winner",
        "schedule": "complete-selection-gated-outer-inference-plan-v1",
        "targets": "inaccessible",
        "inference": "cpu-float32-eval-no-grad",
        "normalization": "feature-v3-values-no-additional-transform-v1",
        "calibration": "uncalibrated-model-distribution-output-v1",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveOuterForecastArchiveV1Error(ValueError):
    """Outer forecasts differ from the frozen selection or source roots."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveOuterForecastArchiveV1Error(
            "outer forecast artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveOuterForecastArchiveV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOuterForecastArchiveV1:
    origin_session_dates: tuple[str, ...]
    security_ids: tuple[str, ...]
    row_receipts: tuple[str, ...]
    row_inventory_sha256: str
    checkpoint_selection_authority_receipt_sha256: str
    checkpoint_selection_source_receipt_sha256: str
    selected_checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    training_tensor_receipt_sha256: str
    training_full_decision_root_inventory_sha256: str
    training_origin_decision_root_inventory_sha256: str
    training_window_plan_receipt_sha256: str
    outer_tensor_receipt_sha256: str
    outer_full_decision_root_inventory_sha256: str
    outer_origin_decision_root_inventory_sha256: str
    outer_inference_plan_receipt_sha256: str
    split_plan_receipt_sha256: str
    fold_index: int
    model_spec_receipt_sha256: str
    model_source_sha256: str
    normalization_receipt_sha256: str
    calibration_receipt_sha256: str
    committed_source_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_rows: tuple[MassiveAdaptiveForecastRowV2, ...] | None
    runtime_forecasts_replayed: bool
    outer_forecast_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        names = (
            "schema",
            "origin_session_dates",
            "security_ids",
            "row_receipts",
            "row_inventory_sha256",
            "checkpoint_selection_authority_receipt_sha256",
            "checkpoint_selection_source_receipt_sha256",
            "selected_checkpoint_receipt_sha256",
            "checkpoint_source_receipt_sha256",
            "model_state_receipt_sha256",
            "training_tensor_receipt_sha256",
            "training_full_decision_root_inventory_sha256",
            "training_origin_decision_root_inventory_sha256",
            "training_window_plan_receipt_sha256",
            "outer_tensor_receipt_sha256",
            "outer_full_decision_root_inventory_sha256",
            "outer_origin_decision_root_inventory_sha256",
            "outer_inference_plan_receipt_sha256",
            "split_plan_receipt_sha256",
            "fold_index",
            "model_spec_receipt_sha256",
            "model_source_sha256",
            "normalization_receipt_sha256",
            "calibration_receipt_sha256",
            "committed_source_data_qualified",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "profitability_reporting_authorized",
            "lockbox_access_authorized",
            "reinforcement_learning_authorized",
        )
        return {name: getattr(self, name) for name in names}

    def validate(self) -> None:
        runtime_present = self.runtime_rows is not None
        if (
            self.schema != MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SCHEMA
            or not self.origin_session_dates
            or self.origin_session_dates
            != tuple(sorted(set(self.origin_session_dates)))
            or not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or len(self.row_receipts) != len(self.origin_session_dates)
            or self.row_inventory_sha256 != semantic_sha256(self.row_receipts)
            or self.fold_index < 0
            or not self.committed_source_data_qualified
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SHA256
            or self.model_source_sha256
            != MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256
            or self.normalization_receipt_sha256
            != MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256
            or self.calibration_receipt_sha256
            != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
            or self.runtime_forecasts_replayed != runtime_present
            or self.outer_forecast_authorized != runtime_present
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveOuterForecastArchiveV1Error(
                "outer forecast archive identity or authorization differs"
            )
        for value in (
            *self.row_receipts,
            self.row_inventory_sha256,
            self.checkpoint_selection_authority_receipt_sha256,
            self.checkpoint_selection_source_receipt_sha256,
            self.selected_checkpoint_receipt_sha256,
            self.checkpoint_source_receipt_sha256,
            self.model_state_receipt_sha256,
            self.training_tensor_receipt_sha256,
            self.training_full_decision_root_inventory_sha256,
            self.training_origin_decision_root_inventory_sha256,
            self.training_window_plan_receipt_sha256,
            self.outer_tensor_receipt_sha256,
            self.outer_full_decision_root_inventory_sha256,
            self.outer_origin_decision_root_inventory_sha256,
            self.outer_inference_plan_receipt_sha256,
            self.split_plan_receipt_sha256,
            self.model_spec_receipt_sha256,
            self.model_source_sha256,
            self.normalization_receipt_sha256,
            self.calibration_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("outer forecast archive", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.checkpoint_selection_authority_receipt_sha256
        ):
            raise MassiveAdaptiveOuterForecastArchiveV1Error(
                "outer forecast source transaction differs"
            )
        if self.runtime_rows is not None:
            for row in self.runtime_rows:
                row.validate()
            if (
                tuple(row.decision_session_date for row in self.runtime_rows)
                != self.origin_session_dates
                or tuple(row.security_ids for row in self.runtime_rows)
                != (self.security_ids,) * len(self.runtime_rows)
                or tuple(row.receipt_sha256 for row in self.runtime_rows)
                != self.row_receipts
            ):
                raise MassiveAdaptiveOuterForecastArchiveV1Error(
                    "outer forecast runtime inventory differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _validate_and_replay(
    *,
    checkpoint_selection: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    selected_checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    outer_tensor: MassiveAdaptiveDecisionTensorV1,
    outer_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    outer_plan: MassiveAdaptiveOuterInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> tuple[MassiveAdaptiveForecastRowV2, ...]:
    checkpoint_selection.validate()
    selected_checkpoint.validate()
    training_window_plan.validate()
    outer_tensor.validate()
    outer_plan.validate()
    model_spec.validate()
    selection = checkpoint_selection.selection
    if (
        not checkpoint_selection.runtime_selection_replayed
        or not checkpoint_selection.development_checkpoint_selection_authorized
    ):
        raise MassiveAdaptiveOuterForecastArchiveV1Error(
            "outer forecast provenance is not the frozen qualified selection"
        )
    validate_massive_adaptive_outer_fold_binding_v1(
        selected_checkpoint=selected_checkpoint,
        training_window_plan=training_window_plan,
        outer_plan=outer_plan,
    )
    ordered_roots = tuple(
        sorted(outer_decision_roots, key=lambda row: row.decision_session_date)
    )
    for root in ordered_roots:
        root.validate()
    full_receipts = tuple(root.semantic_receipt_sha256 for root in ordered_roots)
    origin_receipts = tuple(row.decision_root_receipt_sha256 for row in outer_plan.rows)
    if (
        selection.selected_checkpoint_receipt_sha256
        != selected_checkpoint.semantic_receipt_sha256
        or outer_plan.checkpoint_selection_receipt_sha256
        != checkpoint_selection.semantic_receipt_sha256
        or outer_plan.selected_checkpoint_receipt_sha256
        != selected_checkpoint.semantic_receipt_sha256
        or selected_checkpoint.runtime_state is None
        or not selected_checkpoint.runtime_checkpoint_replayed
        or not selected_checkpoint.development_training_authorized
        or selected_checkpoint.window_plan_receipt_sha256
        != training_window_plan.semantic_receipt_sha256
        or training_window_plan.split_role != "training"
        or training_window_plan.fold_index != outer_plan.fold_index
        or outer_plan.decision_tensor_receipt_sha256
        != outer_tensor.semantic_receipt_sha256
        or outer_plan.model_spec_receipt_sha256 != model_spec.receipt_sha256
        or selected_checkpoint.model_spec_receipt_sha256 != model_spec.receipt_sha256
        or selected_checkpoint.split_plan_receipt_sha256
        != outer_plan.split_plan_receipt_sha256
        or tuple(root.decision_session_date for root in ordered_roots)
        != outer_tensor.decision_session_dates
        or tuple(root.feature_semantic_receipt_sha256 for root in ordered_roots)
        != outer_tensor.feature_semantic_receipts
        or tuple(root.action_origin_receipt_sha256 for root in ordered_roots)
        != outer_tensor.action_origin_receipts
        or outer_plan.full_decision_root_inventory_sha256
        != semantic_sha256(full_receipts)
        or outer_plan.origin_decision_root_inventory_sha256
        != semantic_sha256(origin_receipts)
        or any(not root.source_data_qualified for root in ordered_roots)
    ):
        raise MassiveAdaptiveOuterForecastArchiveV1Error(
            "outer forecast provenance is not the frozen qualified selection"
        )
    return replay_massive_adaptive_forecast_rows_v2(
        checkpoint=selected_checkpoint,
        decision_tensor=outer_tensor,
        plan_rows=outer_plan.rows,
        model_spec=model_spec,
    )


def validate_massive_adaptive_outer_fold_binding_v1(
    *,
    selected_checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    outer_plan: MassiveAdaptiveOuterInferencePlanV1,
) -> None:
    """Fail closed when a selected checkpoint crosses outer-fold lineage."""

    selected_checkpoint.validate()
    training_window_plan.validate()
    outer_plan.validate()
    if (
        training_window_plan.split_role != "training"
        or selected_checkpoint.window_plan_receipt_sha256
        != training_window_plan.semantic_receipt_sha256
        or training_window_plan.fold_index != outer_plan.fold_index
    ):
        raise MassiveAdaptiveOuterForecastArchiveV1Error(
            "selected checkpoint and outer fold training window differ"
        )


def _metadata(
    *,
    rows: tuple[MassiveAdaptiveForecastRowV2, ...],
    checkpoint_selection: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    selected_checkpoint: MassiveAdaptiveCheckpointV1,
    outer_tensor: MassiveAdaptiveDecisionTensorV1,
    outer_plan: MassiveAdaptiveOuterInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> dict[str, object]:
    row_receipts = tuple(row.receipt_sha256 for row in rows)
    return {
        "schema": MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SCHEMA,
        "origin_session_dates": tuple(row.decision_session_date for row in rows),
        "security_ids": outer_tensor.security_ids,
        "row_receipts": row_receipts,
        "row_inventory_sha256": semantic_sha256(row_receipts),
        "checkpoint_selection_authority_receipt_sha256": (
            checkpoint_selection.semantic_receipt_sha256
        ),
        "checkpoint_selection_source_receipt_sha256": (
            checkpoint_selection.selection_source_receipt_sha256
        ),
        "selected_checkpoint_receipt_sha256": (
            selected_checkpoint.semantic_receipt_sha256
        ),
        "checkpoint_source_receipt_sha256": (
            selected_checkpoint.loaded_source.receipt.receipt_sha256
        ),
        "model_state_receipt_sha256": selected_checkpoint.model_state_receipt_sha256,
        "training_tensor_receipt_sha256": (
            selected_checkpoint.decision_tensor_receipt_sha256
        ),
        "training_full_decision_root_inventory_sha256": (
            selected_checkpoint.full_decision_root_inventory_sha256
        ),
        "training_origin_decision_root_inventory_sha256": (
            selected_checkpoint.origin_decision_root_inventory_sha256
        ),
        "training_window_plan_receipt_sha256": (
            selected_checkpoint.window_plan_receipt_sha256
        ),
        "outer_tensor_receipt_sha256": outer_tensor.semantic_receipt_sha256,
        "outer_full_decision_root_inventory_sha256": (
            outer_plan.full_decision_root_inventory_sha256
        ),
        "outer_origin_decision_root_inventory_sha256": (
            outer_plan.origin_decision_root_inventory_sha256
        ),
        "outer_inference_plan_receipt_sha256": outer_plan.semantic_receipt_sha256,
        "split_plan_receipt_sha256": outer_plan.split_plan_receipt_sha256,
        "fold_index": outer_plan.fold_index,
        "model_spec_receipt_sha256": model_spec.receipt_sha256,
        "model_source_sha256": MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256,
        "normalization_receipt_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256
        ),
        "calibration_receipt_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256
        ),
        "committed_source_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SHA256
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> tuple[dict[str, object], tuple[MassiveAdaptiveForecastRowV2, ...]]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("metadata"), Mapping)
        or not isinstance(payload.get("rows"), Sequence)
    ):
        raise MassiveAdaptiveOuterForecastArchiveV1Error(
            "outer forecast payload is malformed"
        )
    metadata = dict(cast(Mapping[str, object], payload["metadata"]))
    rows: list[MassiveAdaptiveForecastRowV2] = []
    for value in cast(Sequence[object], payload["rows"]):
        if not isinstance(value, Mapping):
            raise MassiveAdaptiveOuterForecastArchiveV1Error(
                "outer forecast row payload is malformed"
            )
        row_payload = dict(cast(Mapping[str, object], value))
        row_payload["security_ids"] = tuple(
            cast(Sequence[str], row_payload["security_ids"])
        )
        row_payload["array_receipts"] = tuple(
            cast(Sequence[str], row_payload["array_receipts"])
        )
        row = MassiveAdaptiveForecastRowV2(**row_payload)  # type: ignore[arg-type]
        row.validate()
        rows.append(row)
    for name in ("origin_session_dates", "security_ids", "row_receipts"):
        metadata[name] = tuple(cast(Sequence[object], metadata[name]))
    return metadata, tuple(rows)


def parse_massive_adaptive_outer_forecast_archive_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveOuterForecastArchiveV1:
    """Load committed outer forecasts while withholding runtime tensors."""

    metadata, _rows = _load_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveOuterForecastArchiveV1(
        **metadata,  # type: ignore[arg-type]
        loaded_source=loaded_source,
        runtime_rows=None,
        runtime_forecasts_replayed=False,
        outer_forecast_authorized=False,
    )
    result.validate()
    return result


def materialize_massive_adaptive_outer_forecast_archive_v1(
    *,
    root: str | Path,
    artifact_id: str,
    checkpoint_selection: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    selected_checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    outer_tensor: MassiveAdaptiveDecisionTensorV1,
    outer_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    outer_plan: MassiveAdaptiveOuterInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
    committed_at_ms: int,
) -> MassiveAdaptiveOuterForecastArchiveV1:
    """Publish outer forecasts only after source-derived selection freezes."""

    identifier = _artifact_id(artifact_id)
    rows = _validate_and_replay(
        checkpoint_selection=checkpoint_selection,
        selected_checkpoint=selected_checkpoint,
        training_window_plan=training_window_plan,
        outer_tensor=outer_tensor,
        outer_decision_roots=outer_decision_roots,
        outer_plan=outer_plan,
        model_spec=model_spec,
    )
    metadata = _metadata(
        rows=rows,
        checkpoint_selection=checkpoint_selection,
        selected_checkpoint=selected_checkpoint,
        outer_tensor=outer_tensor,
        outer_plan=outer_plan,
        model_spec=model_spec,
    )
    receipt = semantic_sha256(metadata)
    committed_metadata = dict(metadata)
    committed_metadata["semantic_receipt_sha256"] = receipt
    stream = BytesIO()
    torch.save(
        {
            "metadata": committed_metadata,
            "rows": tuple(row.payload() for row in rows),
        },
        stream,
    )
    stream.seek(0)
    relative = f"massive-adaptive/outer-forecast-archive-v1/{identifier}.pt"
    publish_massive_source_object(
        stream=stream,
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=checkpoint_selection.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-OUTER-FORECAST-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_outer_forecast_archive_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_outer_forecast_archive_v1(
        root=root,
        archive=generic,
        checkpoint_selection=checkpoint_selection,
        selected_checkpoint=selected_checkpoint,
        training_window_plan=training_window_plan,
        outer_tensor=outer_tensor,
        outer_decision_roots=outer_decision_roots,
        outer_plan=outer_plan,
        model_spec=model_spec,
    )


def authorize_massive_adaptive_outer_forecast_archive_v1(
    *,
    root: str | Path,
    archive: MassiveAdaptiveOuterForecastArchiveV1,
    checkpoint_selection: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    selected_checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    outer_tensor: MassiveAdaptiveDecisionTensorV1,
    outer_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    outer_plan: MassiveAdaptiveOuterInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveOuterForecastArchiveV1:
    """Rerun every outer forecast before restoring runtime tensors."""

    parsed = parse_massive_adaptive_outer_forecast_archive_v1(
        root=root, loaded_source=archive.loaded_source
    )
    committed_metadata, committed_rows = _load_payload(
        root=root, loaded_source=archive.loaded_source
    )
    rebuilt_rows = _validate_and_replay(
        checkpoint_selection=checkpoint_selection,
        selected_checkpoint=selected_checkpoint,
        training_window_plan=training_window_plan,
        outer_tensor=outer_tensor,
        outer_decision_roots=outer_decision_roots,
        outer_plan=outer_plan,
        model_spec=model_spec,
    )
    expected_metadata = _metadata(
        rows=rebuilt_rows,
        checkpoint_selection=checkpoint_selection,
        selected_checkpoint=selected_checkpoint,
        outer_tensor=outer_tensor,
        outer_plan=outer_plan,
        model_spec=model_spec,
    )
    if (
        parsed.semantic_receipt_sha256 != archive.semantic_receipt_sha256
        or committed_metadata
        != {
            **expected_metadata,
            "semantic_receipt_sha256": semantic_sha256(expected_metadata),
        }
        or len(committed_rows) != len(rebuilt_rows)
        or any(
            committed.unsigned() != rebuilt.unsigned()
            or committed.receipt_sha256 != rebuilt.receipt_sha256
            or any(
                not torch.equal(
                    cast(torch.Tensor, getattr(committed, name)),
                    cast(torch.Tensor, getattr(rebuilt, name)),
                )
                for name in (
                    *MASSIVE_ADAPTIVE_FORECAST_V2_FLOAT_ARRAY_NAMES,
                    "valid",
                )
            )
            for committed, rebuilt in zip(committed_rows, rebuilt_rows, strict=True)
        )
    ):
        raise MassiveAdaptiveOuterForecastArchiveV1Error(
            "outer forecast archive does not replay from the frozen selection"
        )
    result = replace(
        parsed,
        runtime_rows=rebuilt_rows,
        runtime_forecasts_replayed=True,
        outer_forecast_authorized=True,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_OUTER_FORECAST_ARCHIVE_V1_SCHEMA",
    "MassiveAdaptiveOuterForecastArchiveV1",
    "MassiveAdaptiveOuterForecastArchiveV1Error",
    "authorize_massive_adaptive_outer_forecast_archive_v1",
    "materialize_massive_adaptive_outer_forecast_archive_v1",
    "parse_massive_adaptive_outer_forecast_archive_v1",
    "validate_massive_adaptive_outer_fold_binding_v1",
]
