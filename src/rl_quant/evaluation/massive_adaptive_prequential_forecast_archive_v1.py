"""Create-only aggregate of leakage-safe adaptive forecast blocks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import cast

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MASSIVE_ADAPTIVE_FORECAST_V2_FLOAT_ARRAY_NAMES,
    MassiveAdaptiveForecastArchiveV2,
    MassiveAdaptiveForecastRowV2,
)
from rl_quant.evaluation.massive_adaptive_prequential_forecast_plan_v1 import (
    MassiveAdaptivePrequentialForecastPlanV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-prequential-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_DATASET = (
    "massive-adaptive-prequential-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SCHEMA,
            "encoding": "torch-tensor-archive-loaded-weights-only",
            "publication": "create-only-source-transaction",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source": "exact-replayed-forecast-archive-v2-fold-prefix",
        "checkpoint_cutoff": "strictly-before-each-forecast-block",
        "target_archive": "inaccessible",
        "generic_reload": "nonauthorizing",
        "checkpoint_selection_authority": False,
        "profitability_reporting": False,
        "outer": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptivePrequentialForecastArchiveV1Error(ValueError):
    """Committed prequential forecasts cannot be replayed from their blocks."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptivePrequentialForecastArchiveV1Error(
            "prequential forecast artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptivePrequentialForecastArchiveV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePrequentialForecastArchiveV1:
    origin_session_dates: tuple[str, ...]
    row_receipts: tuple[str, ...]
    row_inventory_sha256: str
    prequential_plan_receipt_sha256: str
    block_inventory_sha256: str
    source_forecast_archive_receipts: tuple[str, ...]
    source_forecast_archive_inventory_sha256: str
    source_forecast_source_receipts: tuple[str, ...]
    checkpoint_receipts: tuple[str, ...]
    training_window_plan_receipts: tuple[str, ...]
    committed_source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_rows: tuple[MassiveAdaptiveForecastRowV2, ...] | None
    runtime_prequential_forecasts_replayed: bool
    development_prequential_forecast_authorized: bool
    checkpoint_selection_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        names = (
            "schema",
            "origin_session_dates",
            "row_receipts",
            "row_inventory_sha256",
            "prequential_plan_receipt_sha256",
            "block_inventory_sha256",
            "source_forecast_archive_receipts",
            "source_forecast_archive_inventory_sha256",
            "source_forecast_source_receipts",
            "checkpoint_receipts",
            "training_window_plan_receipts",
            "committed_source_data_qualified",
            "checkpoint_selection_authorized",
            "profitability_reporting_authorized",
            "outer_evaluation_authorized",
            "lockbox_access_authorized",
            "reinforcement_learning_authorized",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
        )
        return {name: getattr(self, name) for name in names}

    def validate(self) -> None:
        runtime_present = self.runtime_rows is not None
        if (
            self.schema != MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SCHEMA
            or not self.origin_session_dates
            or self.origin_session_dates
            != tuple(sorted(set(self.origin_session_dates)))
            or len(self.row_receipts) != len(self.origin_session_dates)
            or self.row_inventory_sha256 != semantic_sha256(self.row_receipts)
            or not self.source_forecast_archive_receipts
            or self.source_forecast_archive_inventory_sha256
            != semantic_sha256(self.source_forecast_archive_receipts)
            or len(self.source_forecast_source_receipts)
            != len(self.source_forecast_archive_receipts)
            or len(self.checkpoint_receipts)
            != len(self.source_forecast_archive_receipts)
            or len(self.training_window_plan_receipts)
            != len(self.source_forecast_archive_receipts)
            or not isinstance(self.committed_source_data_qualified, bool)
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
            or self.runtime_prequential_forecasts_replayed != runtime_present
            or self.development_prequential_forecast_authorized
            != (runtime_present and self.committed_source_data_qualified)
            or self.checkpoint_selection_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SOURCE_SHA256
        ):
            raise MassiveAdaptivePrequentialForecastArchiveV1Error(
                "prequential forecast archive identity or authorization differs"
            )
        for value in (
            *self.row_receipts,
            self.row_inventory_sha256,
            self.prequential_plan_receipt_sha256,
            self.block_inventory_sha256,
            *self.source_forecast_archive_receipts,
            self.source_forecast_archive_inventory_sha256,
            *self.source_forecast_source_receipts,
            *self.checkpoint_receipts,
            *self.training_window_plan_receipts,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("prequential forecast archive", value)
        if self.runtime_rows is not None:
            for row in self.runtime_rows:
                row.validate()
            if (
                tuple(row.decision_session_date for row in self.runtime_rows)
                != self.origin_session_dates
                or tuple(row.receipt_sha256 for row in self.runtime_rows)
                != self.row_receipts
            ):
                raise MassiveAdaptivePrequentialForecastArchiveV1Error(
                    "prequential runtime forecast inventory differs"
                )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.prequential_plan_receipt_sha256
        ):
            raise MassiveAdaptivePrequentialForecastArchiveV1Error(
                "prequential forecast source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _reconcile(
    *,
    plan: MassiveAdaptivePrequentialForecastPlanV1,
    forecast_archives: Sequence[MassiveAdaptiveForecastArchiveV2],
) -> tuple[MassiveAdaptiveForecastRowV2, ...]:
    plan.validate()
    archives = tuple(forecast_archives)
    if (
        len(archives) != len(plan.blocks)
        or tuple(archive.semantic_receipt_sha256 for archive in archives)
        != plan.source_forecast_archive_receipts
    ):
        raise MassiveAdaptivePrequentialForecastArchiveV1Error(
            "prequential source forecast archive inventory differs"
        )
    rows: list[MassiveAdaptiveForecastRowV2] = []
    for archive, block in zip(archives, plan.blocks, strict=True):
        archive.validate()
        if (
            archive.runtime_rows is None
            or not archive.runtime_forecasts_replayed
            or archive.fold_index != block.fold_index
            or archive.checkpoint_receipt_sha256
            != block.checkpoint_receipt_sha256
            or archive.training_window_plan_receipt_sha256
            != block.training_window_plan_receipt_sha256
            or archive.origin_session_dates != block.forecast_session_dates
            or archive.row_inventory_sha256
            != block.source_forecast_row_inventory_sha256
            or archive.development_forecast_authorized
            != block.source_data_qualified
        ):
            raise MassiveAdaptivePrequentialForecastArchiveV1Error(
                "prequential source forecast block differs"
            )
        rows.extend(archive.runtime_rows)
    result = tuple(rows)
    if (
        tuple(row.decision_session_date for row in result)
        != plan.origin_session_dates
        or tuple(row.receipt_sha256 for row in result)
        != plan.source_forecast_row_receipts
    ):
        raise MassiveAdaptivePrequentialForecastArchiveV1Error(
            "prequential forecast rows differ from the plan"
        )
    return result


def _metadata(
    *,
    plan: MassiveAdaptivePrequentialForecastPlanV1,
    archives: tuple[MassiveAdaptiveForecastArchiveV2, ...],
    rows: tuple[MassiveAdaptiveForecastRowV2, ...],
) -> dict[str, object]:
    row_receipts = tuple(row.receipt_sha256 for row in rows)
    return {
        "schema": MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SCHEMA,
        "origin_session_dates": plan.origin_session_dates,
        "row_receipts": row_receipts,
        "row_inventory_sha256": semantic_sha256(row_receipts),
        "prequential_plan_receipt_sha256": plan.semantic_receipt_sha256,
        "block_inventory_sha256": plan.block_inventory_sha256,
        "source_forecast_archive_receipts": (
            plan.source_forecast_archive_receipts
        ),
        "source_forecast_archive_inventory_sha256": (
            plan.source_forecast_archive_inventory_sha256
        ),
        "source_forecast_source_receipts": tuple(
            archive.loaded_source.receipt.receipt_sha256 for archive in archives
        ),
        "checkpoint_receipts": tuple(
            archive.checkpoint_receipt_sha256 for archive in archives
        ),
        "training_window_plan_receipts": tuple(
            archive.training_window_plan_receipt_sha256 for archive in archives
        ),
        "committed_source_data_qualified": plan.source_data_qualified,
        "checkpoint_selection_authorized": False,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SOURCE_SHA256
        ),
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
        raise MassiveAdaptivePrequentialForecastArchiveV1Error(
            "prequential forecast payload is malformed"
        )
    metadata = dict(cast(Mapping[str, object], payload["metadata"]))
    for name in (
        "origin_session_dates",
        "row_receipts",
        "source_forecast_archive_receipts",
        "source_forecast_source_receipts",
        "checkpoint_receipts",
        "training_window_plan_receipts",
    ):
        metadata[name] = tuple(cast(Sequence[object], metadata[name]))
    rows: list[MassiveAdaptiveForecastRowV2] = []
    for value in cast(Sequence[object], payload["rows"]):
        if not isinstance(value, Mapping):
            raise MassiveAdaptivePrequentialForecastArchiveV1Error(
                "prequential forecast row payload is malformed"
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
    return metadata, tuple(rows)


def parse_massive_adaptive_prequential_forecast_archive_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptivePrequentialForecastArchiveV1:
    """Reopen committed prequential metadata without runtime forecasts."""

    metadata, _rows = _load_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptivePrequentialForecastArchiveV1(
        **metadata,  # type: ignore[arg-type]
        loaded_source=loaded_source,
        runtime_rows=None,
        runtime_prequential_forecasts_replayed=False,
        development_prequential_forecast_authorized=False,
    )
    result.validate()
    return result


def materialize_massive_adaptive_prequential_forecast_archive_v1(
    *,
    root: str | Path,
    artifact_id: str,
    plan: MassiveAdaptivePrequentialForecastPlanV1,
    forecast_archives: Sequence[MassiveAdaptiveForecastArchiveV2],
    committed_at_ms: int,
) -> MassiveAdaptivePrequentialForecastArchiveV1:
    """Publish and immediately replay one target-free prequential archive."""

    identifier = _artifact_id(artifact_id)
    archives = tuple(forecast_archives)
    rows = _reconcile(plan=plan, forecast_archives=archives)
    metadata = _metadata(plan=plan, archives=archives, rows=rows)
    receipt = semantic_sha256(metadata)
    stream = BytesIO()
    archive_payload: dict[str, object] = {
        "metadata": {**metadata, "semantic_receipt_sha256": receipt},
        "rows": tuple(row.payload() for row in rows),
    }
    torch.save(archive_payload, stream)
    stream.seek(0)
    relative = f"massive-adaptive/prequential-forecast-v1/{identifier}.pt"
    publish_massive_source_object(
        stream=stream,
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=plan.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-PREQUENTIAL-FORECAST-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_prequential_forecast_archive_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_prequential_forecast_archive_v1(
        root=root,
        archive=generic,
        plan=plan,
        forecast_archives=archives,
    )


def authorize_massive_adaptive_prequential_forecast_archive_v1(
    *,
    root: str | Path,
    archive: MassiveAdaptivePrequentialForecastArchiveV1,
    plan: MassiveAdaptivePrequentialForecastPlanV1,
    forecast_archives: Sequence[MassiveAdaptiveForecastArchiveV2],
) -> MassiveAdaptivePrequentialForecastArchiveV1:
    """Reconcile every child archive and committed forecast tensor."""

    parsed = parse_massive_adaptive_prequential_forecast_archive_v1(
        root=root, loaded_source=archive.loaded_source
    )
    committed_metadata, committed_rows = _load_payload(
        root=root, loaded_source=archive.loaded_source
    )
    archives = tuple(forecast_archives)
    rebuilt_rows = _reconcile(plan=plan, forecast_archives=archives)
    expected_metadata = _metadata(plan=plan, archives=archives, rows=rebuilt_rows)
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
                for name in (*MASSIVE_ADAPTIVE_FORECAST_V2_FLOAT_ARRAY_NAMES, "valid")
            )
            for committed, rebuilt in zip(
                committed_rows, rebuilt_rows, strict=True
            )
        )
    ):
        raise MassiveAdaptivePrequentialForecastArchiveV1Error(
            "prequential forecast archive does not replay"
        )
    result = replace(
        parsed,
        runtime_rows=rebuilt_rows,
        runtime_prequential_forecasts_replayed=True,
        development_prequential_forecast_authorized=(
            parsed.committed_source_data_qualified
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_ARCHIVE_V1_SCHEMA",
    "MassiveAdaptivePrequentialForecastArchiveV1",
    "MassiveAdaptivePrequentialForecastArchiveV1Error",
    "authorize_massive_adaptive_prequential_forecast_archive_v1",
    "materialize_massive_adaptive_prequential_forecast_archive_v1",
    "parse_massive_adaptive_prequential_forecast_archive_v1",
]
