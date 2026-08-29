"""Root-bound replay authority for adaptive forecast archives.

The authority is deliberately smaller than the tensor archive.  It records
that the package reopened the immutable archive and reproduced every forecast
from the exact checkpoint, tensor, decision roots, window plan, and model
specification.  Generic reloads are nonauthorizing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v1 import (
    MassiveAdaptiveForecastArchiveV1,
    authorize_massive_adaptive_forecast_archive_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MassiveAdaptiveAlphaModelSpecV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
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


MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-forecast-replay-authority-v1"
)
MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_DATASET = (
    "massive-adaptive-forecast-replay-authority-v1"
)
MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SCHEMA,
            "encoding": "canonical-json",
            "publication": "create-only-source-transaction",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "archive": "immutable-adaptive-forecast-archive-v1",
        "replay": "checkpoint-tensor-root-window-model-reexecution",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveForecastReplayAuthorityV1Error(ValueError):
    """Forecast replay authority or its live root replay differs."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveForecastReplayAuthorityV1Error(
            "adaptive forecast replay artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveForecastReplayAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveForecastReplayAuthorityV1:
    forecast_archive_receipt_sha256: str
    forecast_archive_source_receipt_sha256: str
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    decision_tensor_receipt_sha256: str
    full_decision_root_inventory_sha256: str
    origin_decision_root_inventory_sha256: str
    window_plan_receipt_sha256: str
    model_spec_receipt_sha256: str
    forecast_row_inventory_sha256: str
    origin_session_dates: tuple[str, ...]
    committed_forecasts_replay_qualified: bool
    committed_source_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_forecasts_replayed: bool
    development_forecast_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "forecast_archive_receipt_sha256": (
                self.forecast_archive_receipt_sha256
            ),
            "forecast_archive_source_receipt_sha256": (
                self.forecast_archive_source_receipt_sha256
            ),
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "checkpoint_source_receipt_sha256": (
                self.checkpoint_source_receipt_sha256
            ),
            "model_state_receipt_sha256": self.model_state_receipt_sha256,
            "decision_tensor_receipt_sha256": (
                self.decision_tensor_receipt_sha256
            ),
            "full_decision_root_inventory_sha256": (
                self.full_decision_root_inventory_sha256
            ),
            "origin_decision_root_inventory_sha256": (
                self.origin_decision_root_inventory_sha256
            ),
            "window_plan_receipt_sha256": self.window_plan_receipt_sha256,
            "model_spec_receipt_sha256": self.model_spec_receipt_sha256,
            "forecast_row_inventory_sha256": (
                self.forecast_row_inventory_sha256
            ),
            "origin_session_dates": self.origin_session_dates,
            "committed_forecasts_replay_qualified": (
                self.committed_forecasts_replay_qualified
            ),
            "committed_source_data_qualified": (
                self.committed_source_data_qualified
            ),
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "profitability_reporting_authorized": (
                self.profitability_reporting_authorized
            ),
            "lockbox_access_authorized": self.lockbox_access_authorized,
            "reinforcement_learning_authorized": (
                self.reinforcement_learning_authorized
            ),
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            **self.semantic_unsigned(),
            "semantic_receipt_sha256": self.semantic_receipt_sha256,
        }

    def validate(self) -> None:
        if (
            self.schema
            != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SCHEMA
            or not self.origin_session_dates
            or self.origin_session_dates
            != tuple(sorted(set(self.origin_session_dates)))
            or not self.committed_forecasts_replay_qualified
            or not isinstance(self.committed_source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
            or self.development_forecast_authorized
            != (
                self.runtime_forecasts_replayed
                and self.committed_source_data_qualified
            )
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveForecastReplayAuthorityV1Error(
                "adaptive forecast replay identity or authorization differs"
            )
        for value in (
            self.forecast_archive_receipt_sha256,
            self.forecast_archive_source_receipt_sha256,
            self.checkpoint_receipt_sha256,
            self.checkpoint_source_receipt_sha256,
            self.model_state_receipt_sha256,
            self.decision_tensor_receipt_sha256,
            self.full_decision_root_inventory_sha256,
            self.origin_decision_root_inventory_sha256,
            self.window_plan_receipt_sha256,
            self.model_spec_receipt_sha256,
            self.forecast_row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive forecast replay authority", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.forecast_archive_receipt_sha256
        ):
            raise MassiveAdaptiveForecastReplayAuthorityV1Error(
                "adaptive forecast replay source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _semantic_body(
    archive: MassiveAdaptiveForecastArchiveV1,
) -> dict[str, object]:
    archive.validate()
    if archive.runtime_rows is None or not archive.runtime_forecasts_replayed:
        raise MassiveAdaptiveForecastReplayAuthorityV1Error(
            "adaptive forecast archive has not been replayed"
        )
    return {
        "schema": MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SCHEMA,
        "forecast_archive_receipt_sha256": archive.semantic_receipt_sha256,
        "forecast_archive_source_receipt_sha256": (
            archive.loaded_source.receipt.receipt_sha256
        ),
        "checkpoint_receipt_sha256": archive.checkpoint_receipt_sha256,
        "checkpoint_source_receipt_sha256": (
            archive.checkpoint_source_receipt_sha256
        ),
        "model_state_receipt_sha256": archive.model_state_receipt_sha256,
        "decision_tensor_receipt_sha256": archive.decision_tensor_receipt_sha256,
        "full_decision_root_inventory_sha256": (
            archive.full_decision_root_inventory_sha256
        ),
        "origin_decision_root_inventory_sha256": (
            archive.origin_decision_root_inventory_sha256
        ),
        "window_plan_receipt_sha256": archive.window_plan_receipt_sha256,
        "model_spec_receipt_sha256": archive.model_spec_receipt_sha256,
        "forecast_row_inventory_sha256": archive.row_inventory_sha256,
        "origin_session_dates": archive.origin_session_dates,
        "committed_forecasts_replay_qualified": True,
        "committed_source_data_qualified": (
            archive.committed_source_data_qualified
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SOURCE_SHA256
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }


def _parse_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise MassiveAdaptiveForecastReplayAuthorityV1Error(
            "adaptive forecast replay source is not JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise MassiveAdaptiveForecastReplayAuthorityV1Error(
            "adaptive forecast replay source is malformed"
        )
    payload["origin_session_dates"] = tuple(payload["origin_session_dates"])
    return cast(dict[str, object], payload)


def parse_massive_adaptive_forecast_replay_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveForecastReplayAuthorityV1:
    payload = _parse_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveForecastReplayAuthorityV1(
        **payload,  # type: ignore[arg-type]
        loaded_source=loaded_source,
        runtime_forecasts_replayed=False,
        development_forecast_authorized=False,
    )
    result.validate()
    if canonical_json_file_bytes(result.canonical_payload()) != (
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    ):
        raise MassiveAdaptiveForecastReplayAuthorityV1Error(
            "adaptive forecast replay source is not canonical JSON"
        )
    return result


def materialize_massive_adaptive_forecast_replay_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    archive: MassiveAdaptiveForecastArchiveV1,
    checkpoint: MassiveAdaptiveCheckpointV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: tuple[MassiveAdaptiveDecisionRootV1, ...],
    window_plan: MassiveAdaptiveWindowPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
    committed_at_ms: int,
) -> MassiveAdaptiveForecastReplayAuthorityV1:
    """Publish a receipt and root-replay it before returning authority."""

    identifier = _artifact_id(artifact_id)
    replayed_archive = authorize_massive_adaptive_forecast_archive_v1(
        root=root,
        archive=archive,
        checkpoint=checkpoint,
        decision_tensor=decision_tensor,
        decision_roots=decision_roots,
        window_plan=window_plan,
        model_spec=model_spec,
    )
    body = _semantic_body(replayed_archive)
    payload = {**body, "semantic_receipt_sha256": semantic_sha256(body)}
    relative = f"massive-adaptive/forecast-replay-authority-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=replayed_archive.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-FORECAST-REPLAY-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_forecast_replay_authority_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_forecast_replay_authority_v1(
        root=root,
        authority=generic,
        archive=archive,
        checkpoint=checkpoint,
        decision_tensor=decision_tensor,
        decision_roots=decision_roots,
        window_plan=window_plan,
        model_spec=model_spec,
    )


def authorize_massive_adaptive_forecast_replay_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveForecastReplayAuthorityV1,
    archive: MassiveAdaptiveForecastArchiveV1,
    checkpoint: MassiveAdaptiveCheckpointV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: tuple[MassiveAdaptiveDecisionRootV1, ...],
    window_plan: MassiveAdaptiveWindowPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveForecastReplayAuthorityV1:
    """Reexecute archive inference before promoting the authority."""

    parsed = parse_massive_adaptive_forecast_replay_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    replayed = authorize_massive_adaptive_forecast_archive_v1(
        root=root,
        archive=archive,
        checkpoint=checkpoint,
        decision_tensor=decision_tensor,
        decision_roots=decision_roots,
        window_plan=window_plan,
        model_spec=model_spec,
    )
    expected = _semantic_body(replayed)
    if (
        parsed.semantic_receipt_sha256 != authority.semantic_receipt_sha256
        or parsed.semantic_unsigned() != expected
    ):
        raise MassiveAdaptiveForecastReplayAuthorityV1Error(
            "adaptive forecast replay authority differs from live root replay"
        )
    result = replace(
        parsed,
        runtime_forecasts_replayed=True,
        development_forecast_authorized=(
            parsed.committed_source_data_qualified
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V1_SCHEMA",
    "MassiveAdaptiveForecastReplayAuthorityV1",
    "MassiveAdaptiveForecastReplayAuthorityV1Error",
    "authorize_massive_adaptive_forecast_replay_authority_v1",
    "materialize_massive_adaptive_forecast_replay_authority_v1",
    "parse_massive_adaptive_forecast_replay_authority_v1",
]
