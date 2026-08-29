"""Root-bound replay authority for target-free adaptive forecast archives V2."""

from __future__ import annotations

import json
from collections.abc import Sequence
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
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
    authorize_massive_adaptive_forecast_archive_v2,
)
from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferencePlanV1,
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


MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-forecast-replay-authority-v2"
)
MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_DATASET = (
    "massive-adaptive-forecast-replay-authority-v2"
)
MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "archive": "immutable-target-free-adaptive-forecast-archive-v2",
        "replay": "eligibility-checkpoint-inference-root-model-reexecution",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveForecastReplayAuthorityV2Error(ValueError):
    """A V2 replay receipt differs from live out-of-sample inference."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveForecastReplayAuthorityV2Error(
            "adaptive forecast replay v2 artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveForecastReplayAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveForecastReplayAuthorityV2:
    forecast_archive_receipt_sha256: str
    forecast_archive_source_receipt_sha256: str
    eligibility_authority_receipt_sha256: str
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    training_tensor_receipt_sha256: str
    training_full_decision_root_inventory_sha256: str
    training_origin_decision_root_inventory_sha256: str
    training_window_plan_receipt_sha256: str
    inference_tensor_receipt_sha256: str
    inference_full_decision_root_inventory_sha256: str
    inference_origin_decision_root_inventory_sha256: str
    inference_plan_receipt_sha256: str
    inference_role: str
    fold_index: int
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
    schema: str = MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        names = (
            "schema",
            "forecast_archive_receipt_sha256",
            "forecast_archive_source_receipt_sha256",
            "eligibility_authority_receipt_sha256",
            "checkpoint_receipt_sha256",
            "checkpoint_source_receipt_sha256",
            "model_state_receipt_sha256",
            "training_tensor_receipt_sha256",
            "training_full_decision_root_inventory_sha256",
            "training_origin_decision_root_inventory_sha256",
            "training_window_plan_receipt_sha256",
            "inference_tensor_receipt_sha256",
            "inference_full_decision_root_inventory_sha256",
            "inference_origin_decision_root_inventory_sha256",
            "inference_plan_receipt_sha256",
            "inference_role",
            "fold_index",
            "model_spec_receipt_sha256",
            "forecast_row_inventory_sha256",
            "origin_session_dates",
            "committed_forecasts_replay_qualified",
            "committed_source_data_qualified",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "profitability_reporting_authorized",
            "lockbox_access_authorized",
            "reinforcement_learning_authorized",
        )
        return {name: getattr(self, name) for name in names}

    def canonical_payload(self) -> dict[str, object]:
        return {
            **self.semantic_unsigned(),
            "semantic_receipt_sha256": self.semantic_receipt_sha256,
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SCHEMA
            or self.inference_role != "inner_validation"
            or not self.origin_session_dates
            or self.origin_session_dates
            != tuple(sorted(set(self.origin_session_dates)))
            or not self.committed_forecasts_replay_qualified
            or not isinstance(self.committed_source_data_qualified, bool)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or self.development_forecast_authorized
            != (
                self.runtime_forecasts_replayed and self.committed_source_data_qualified
            )
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveForecastReplayAuthorityV2Error(
                "adaptive forecast replay v2 identity or authorization differs"
            )
        for value in (
            self.forecast_archive_receipt_sha256,
            self.forecast_archive_source_receipt_sha256,
            self.eligibility_authority_receipt_sha256,
            self.checkpoint_receipt_sha256,
            self.checkpoint_source_receipt_sha256,
            self.model_state_receipt_sha256,
            self.training_tensor_receipt_sha256,
            self.training_full_decision_root_inventory_sha256,
            self.training_origin_decision_root_inventory_sha256,
            self.training_window_plan_receipt_sha256,
            self.inference_tensor_receipt_sha256,
            self.inference_full_decision_root_inventory_sha256,
            self.inference_origin_decision_root_inventory_sha256,
            self.inference_plan_receipt_sha256,
            self.model_spec_receipt_sha256,
            self.forecast_row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive forecast replay v2 authority", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.forecast_archive_receipt_sha256
        ):
            raise MassiveAdaptiveForecastReplayAuthorityV2Error(
                "adaptive forecast replay v2 source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _semantic_body(
    archive: MassiveAdaptiveForecastArchiveV2,
) -> dict[str, object]:
    archive.validate()
    if archive.runtime_rows is None or not archive.runtime_forecasts_replayed:
        raise MassiveAdaptiveForecastReplayAuthorityV2Error(
            "adaptive forecast v2 archive has not been replayed"
        )
    return {
        "schema": MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SCHEMA,
        "forecast_archive_receipt_sha256": archive.semantic_receipt_sha256,
        "forecast_archive_source_receipt_sha256": (
            archive.loaded_source.receipt.receipt_sha256
        ),
        "eligibility_authority_receipt_sha256": (
            archive.eligibility_authority_receipt_sha256
        ),
        "checkpoint_receipt_sha256": archive.checkpoint_receipt_sha256,
        "checkpoint_source_receipt_sha256": (archive.checkpoint_source_receipt_sha256),
        "model_state_receipt_sha256": archive.model_state_receipt_sha256,
        "training_tensor_receipt_sha256": archive.training_tensor_receipt_sha256,
        "training_full_decision_root_inventory_sha256": (
            archive.training_full_decision_root_inventory_sha256
        ),
        "training_origin_decision_root_inventory_sha256": (
            archive.training_origin_decision_root_inventory_sha256
        ),
        "training_window_plan_receipt_sha256": (
            archive.training_window_plan_receipt_sha256
        ),
        "inference_tensor_receipt_sha256": (archive.inference_tensor_receipt_sha256),
        "inference_full_decision_root_inventory_sha256": (
            archive.inference_full_decision_root_inventory_sha256
        ),
        "inference_origin_decision_root_inventory_sha256": (
            archive.inference_origin_decision_root_inventory_sha256
        ),
        "inference_plan_receipt_sha256": archive.inference_plan_receipt_sha256,
        "inference_role": archive.inference_role,
        "fold_index": archive.fold_index,
        "model_spec_receipt_sha256": archive.model_spec_receipt_sha256,
        "forecast_row_inventory_sha256": archive.row_inventory_sha256,
        "origin_session_dates": archive.origin_session_dates,
        "committed_forecasts_replay_qualified": True,
        "committed_source_data_qualified": (archive.committed_source_data_qualified),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SOURCE_SHA256
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
        raise MassiveAdaptiveForecastReplayAuthorityV2Error(
            "adaptive forecast replay v2 source is not JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise MassiveAdaptiveForecastReplayAuthorityV2Error(
            "adaptive forecast replay v2 source is malformed"
        )
    payload["origin_session_dates"] = tuple(payload["origin_session_dates"])
    return cast(dict[str, object], payload)


def parse_massive_adaptive_forecast_replay_authority_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveForecastReplayAuthorityV2:
    payload = _parse_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveForecastReplayAuthorityV2(
        **payload,  # type: ignore[arg-type]
        loaded_source=loaded_source,
        runtime_forecasts_replayed=False,
        development_forecast_authorized=False,
    )
    result.validate()
    if canonical_json_file_bytes(result.canonical_payload()) != (
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    ):
        raise MassiveAdaptiveForecastReplayAuthorityV2Error(
            "adaptive forecast replay v2 source is not canonical JSON"
        )
    return result


def materialize_massive_adaptive_forecast_replay_authority_v2(
    *,
    root: str | Path,
    artifact_id: str,
    archive: MassiveAdaptiveForecastArchiveV2,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    inference_tensor: MassiveAdaptiveDecisionTensorV1,
    inference_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    inference_plan: MassiveAdaptiveInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
    committed_at_ms: int,
) -> MassiveAdaptiveForecastReplayAuthorityV2:
    """Publish a replay receipt only after target-free model reexecution."""

    identifier = _artifact_id(artifact_id)
    replayed_archive = authorize_massive_adaptive_forecast_archive_v2(
        root=root,
        archive=archive,
        checkpoint=checkpoint,
        training_window_plan=training_window_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=inference_decision_roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
    )
    body = _semantic_body(replayed_archive)
    payload = {**body, "semantic_receipt_sha256": semantic_sha256(body)}
    relative = f"massive-adaptive/forecast-replay-authority-v2/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=replayed_archive.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-FORECAST-REPLAY-V2-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_forecast_replay_authority_v2(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_forecast_replay_authority_v2(
        root=root,
        authority=generic,
        archive=archive,
        checkpoint=checkpoint,
        training_window_plan=training_window_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=inference_decision_roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
    )


def authorize_massive_adaptive_forecast_replay_authority_v2(
    *,
    root: str | Path,
    authority: MassiveAdaptiveForecastReplayAuthorityV2,
    archive: MassiveAdaptiveForecastArchiveV2,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    inference_tensor: MassiveAdaptiveDecisionTensorV1,
    inference_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    inference_plan: MassiveAdaptiveInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveForecastReplayAuthorityV2:
    """Reexecute eligible inner-validation inference before promotion."""

    parsed = parse_massive_adaptive_forecast_replay_authority_v2(
        root=root, loaded_source=authority.loaded_source
    )
    replayed = authorize_massive_adaptive_forecast_archive_v2(
        root=root,
        archive=archive,
        checkpoint=checkpoint,
        training_window_plan=training_window_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=inference_decision_roots,
        inference_plan=inference_plan,
        model_spec=model_spec,
    )
    expected = _semantic_body(replayed)
    if (
        parsed.semantic_receipt_sha256 != authority.semantic_receipt_sha256
        or parsed.semantic_unsigned() != expected
    ):
        raise MassiveAdaptiveForecastReplayAuthorityV2Error(
            "adaptive forecast replay v2 differs from live root replay"
        )
    result = replace(
        parsed,
        runtime_forecasts_replayed=True,
        development_forecast_authorized=(parsed.committed_source_data_qualified),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_FORECAST_REPLAY_AUTHORITY_V2_SCHEMA",
    "MassiveAdaptiveForecastReplayAuthorityV2",
    "MassiveAdaptiveForecastReplayAuthorityV2Error",
    "authorize_massive_adaptive_forecast_replay_authority_v2",
    "materialize_massive_adaptive_forecast_replay_authority_v2",
    "parse_massive_adaptive_forecast_replay_authority_v2",
]
