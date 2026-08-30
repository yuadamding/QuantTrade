"""RL outer forecasts that require a replayed pre-access commitment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v1 import (
    MassiveAdaptiveOuterAccessCommitmentV1,
)
from rl_quant.evaluation.massive_adaptive_outer_forecast_archive_v1 import (
    MassiveAdaptiveOuterForecastArchiveV1,
    authorize_massive_adaptive_outer_forecast_archive_v1,
    materialize_massive_adaptive_outer_forecast_archive_v1,
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
    MassiveAdaptiveAlphaModelSpecV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import MassiveAdaptiveCheckpointV1
from rl_quant.training.massive_adaptive_profit_checkpoint_selection_authority_v2 import (
    MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import MassiveAdaptiveWindowPlanV1


MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_DATASET = (
    "massive-adaptive-rl-outer-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "prerequisite": "replayed-outer-access-commitment-v1",
        "forecast": "exact-outer-forecast-archive-v1",
        "bypass_authorizing": False,
        "profitability_reporting": False,
        "lockbox": False,
    }
)
MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SCHEMA,
        "payload": "commitment-and-forecast-receipts",
        "generic_reload": "nonauthorizing",
        "promotion": "reopen-commitment-and-rerun-outer-forecast",
    }
)


class MassiveAdaptiveRLOuterForecastArchiveV1Error(ValueError):
    """An RL outer forecast bypassed or differed from its commitment."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLOuterForecastArchiveV1Error(
            "RL outer forecast ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterForecastArchiveV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterForecastArchiveV1:
    fold_index: int
    outer_access_commitment_receipt_sha256: str
    outer_access_commitment_source_receipt_sha256: str
    outer_inference_plan_receipt_sha256: str
    outer_origin_inventory_sha256: str
    selected_checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    raw_outer_forecast_archive_receipt_sha256: str
    raw_outer_forecast_source_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_outer_forecast_archive: MassiveAdaptiveOuterForecastArchiveV1 | None
    runtime_forecast_replayed: bool
    outer_forecast_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "outer_access_commitment_receipt_sha256": (
                self.outer_access_commitment_receipt_sha256
            ),
            "outer_access_commitment_source_receipt_sha256": (
                self.outer_access_commitment_source_receipt_sha256
            ),
            "outer_inference_plan_receipt_sha256": (
                self.outer_inference_plan_receipt_sha256
            ),
            "outer_origin_inventory_sha256": self.outer_origin_inventory_sha256,
            "selected_checkpoint_receipt_sha256": (
                self.selected_checkpoint_receipt_sha256
            ),
            "model_state_receipt_sha256": self.model_state_receipt_sha256,
            "raw_outer_forecast_archive_receipt_sha256": (
                self.raw_outer_forecast_archive_receipt_sha256
            ),
            "raw_outer_forecast_source_receipt_sha256": (
                self.raw_outer_forecast_source_receipt_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        runtime = self.runtime_outer_forecast_archive is not None
        expected = runtime and self.runtime_forecast_replayed and self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SCHEMA
            or self.fold_index < 0
            or self.runtime_forecast_replayed != runtime
            or self.outer_forecast_authorized != expected
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterForecastArchiveV1Error(
                "RL outer forecast archive differs"
            )
        for value in (
            self.outer_access_commitment_receipt_sha256,
            self.outer_access_commitment_source_receipt_sha256,
            self.outer_inference_plan_receipt_sha256,
            self.outer_origin_inventory_sha256,
            self.selected_checkpoint_receipt_sha256,
            self.model_state_receipt_sha256,
            self.raw_outer_forecast_archive_receipt_sha256,
            self.raw_outer_forecast_source_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("RL outer forecast archive", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.outer_access_commitment_receipt_sha256
        ):
            raise MassiveAdaptiveRLOuterForecastArchiveV1Error(
                "RL outer forecast source transaction differs"
            )
        if runtime and self.runtime_outer_forecast_archive is not None:
            self.runtime_outer_forecast_archive.validate()
            if (
                not self.runtime_outer_forecast_archive.outer_forecast_authorized
                or self.runtime_outer_forecast_archive.semantic_receipt_sha256
                != self.raw_outer_forecast_archive_receipt_sha256
            ):
                raise MassiveAdaptiveRLOuterForecastArchiveV1Error(
                    "RL outer forecast runtime archive differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _metadata(
    *,
    commitment: MassiveAdaptiveOuterAccessCommitmentV1,
    archive: MassiveAdaptiveOuterForecastArchiveV1,
) -> dict[str, object]:
    commitment.validate()
    archive.validate()
    if (
        not commitment.outer_forecast_access_authorized
        or not commitment.runtime_commitment_replayed
        or not archive.outer_forecast_authorized
        or archive.fold_index != commitment.fold_index
        or archive.outer_inference_plan_receipt_sha256
        != commitment.outer_inference_plan_receipt_sha256
        or archive.outer_origin_decision_root_inventory_sha256
        != commitment.outer_origin_inventory_sha256
        or archive.selected_checkpoint_receipt_sha256
        != commitment.supervised_checkpoint_receipt_sha256
        or archive.model_state_receipt_sha256
        != commitment.supervised_model_state_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterForecastArchiveV1Error(
            "RL outer forecast does not match its pre-access commitment"
        )
    return {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SCHEMA,
        "fold_index": commitment.fold_index,
        "outer_access_commitment_receipt_sha256": (
            commitment.semantic_receipt_sha256
        ),
        "outer_access_commitment_source_receipt_sha256": (
            commitment.loaded_source.receipt.receipt_sha256
        ),
        "outer_inference_plan_receipt_sha256": (
            commitment.outer_inference_plan_receipt_sha256
        ),
        "outer_origin_inventory_sha256": commitment.outer_origin_inventory_sha256,
        "selected_checkpoint_receipt_sha256": (
            commitment.supervised_checkpoint_receipt_sha256
        ),
        "model_state_receipt_sha256": commitment.supervised_model_state_receipt_sha256,
        "raw_outer_forecast_archive_receipt_sha256": archive.semantic_receipt_sha256,
        "raw_outer_forecast_source_receipt_sha256": (
            archive.loaded_source.receipt.receipt_sha256
        ),
        "source_data_qualified": bool(
            commitment.source_data_qualified and archive.committed_source_data_qualified
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SHA256
        ),
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLOuterForecastArchiveV1Error(
            "RL outer forecast gate is not canonical JSON"
        )
    return dict(value)


def parse_massive_adaptive_rl_outer_forecast_archive_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLOuterForecastArchiveV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLOuterForecastArchiveV1(
        **payload,  # type: ignore[arg-type]
        loaded_source=loaded_source,
        runtime_outer_forecast_archive=None,
        runtime_forecast_replayed=False,
        outer_forecast_authorized=False,
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_outer_forecast_archive_v1(
    *,
    root: str | Path,
    gated_archive: MassiveAdaptiveRLOuterForecastArchiveV1,
    commitment: MassiveAdaptiveOuterAccessCommitmentV1,
    raw_archive: MassiveAdaptiveOuterForecastArchiveV1,
    checkpoint_selection: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    selected_checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    outer_tensor: MassiveAdaptiveDecisionTensorV1,
    outer_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    outer_plan: MassiveAdaptiveOuterInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveRLOuterForecastArchiveV1:
    parsed = parse_massive_adaptive_rl_outer_forecast_archive_v1(
        root=root, loaded_source=gated_archive.loaded_source
    )
    replayed = authorize_massive_adaptive_outer_forecast_archive_v1(
        root=root,
        archive=raw_archive,
        checkpoint_selection=checkpoint_selection,
        selected_checkpoint=selected_checkpoint,
        training_window_plan=training_window_plan,
        outer_tensor=outer_tensor,
        outer_decision_roots=outer_decision_roots,
        outer_plan=outer_plan,
        model_spec=model_spec,
    )
    metadata = _metadata(commitment=commitment, archive=replayed)
    expected = {**metadata, "semantic_receipt_sha256": semantic_sha256(metadata)}
    if (
        parsed.semantic_receipt_sha256 != gated_archive.semantic_receipt_sha256
        or _load_payload(root=root, loaded_source=gated_archive.loaded_source)
        != expected
    ):
        raise MassiveAdaptiveRLOuterForecastArchiveV1Error(
            "RL outer forecast gate does not replay"
        )
    result = replace(
        parsed,
        runtime_outer_forecast_archive=replayed,
        runtime_forecast_replayed=True,
        outer_forecast_authorized=parsed.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_outer_forecast_archive_v1(
    *,
    root: str | Path,
    artifact_id: str,
    commitment: MassiveAdaptiveOuterAccessCommitmentV1,
    checkpoint_selection: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    selected_checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    outer_tensor: MassiveAdaptiveDecisionTensorV1,
    outer_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    outer_plan: MassiveAdaptiveOuterInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLOuterForecastArchiveV1:
    """Materialize the raw forecast only through a prior commitment receipt."""

    identifier = _artifact_id(artifact_id)
    commitment.validate()
    if (
        not commitment.runtime_commitment_replayed
        or not commitment.outer_forecast_access_authorized
        or commitment.outer_inference_plan_receipt_sha256
        != outer_plan.semantic_receipt_sha256
        or commitment.loaded_source.commit.committed_at_ms >= committed_at_ms
    ):
        raise MassiveAdaptiveRLOuterForecastArchiveV1Error(
            "outer-access commitment is absent or not prior to forecast publication"
        )
    raw_archive = materialize_massive_adaptive_outer_forecast_archive_v1(
        root=root,
        artifact_id=f"{identifier}-raw",
        checkpoint_selection=checkpoint_selection,
        selected_checkpoint=selected_checkpoint,
        training_window_plan=training_window_plan,
        outer_tensor=outer_tensor,
        outer_decision_roots=outer_decision_roots,
        outer_plan=outer_plan,
        model_spec=model_spec,
        committed_at_ms=committed_at_ms,
    )
    metadata = _metadata(commitment=commitment, archive=raw_archive)
    payload = {**metadata, "semantic_receipt_sha256": semantic_sha256(metadata)}
    relative = f"massive-adaptive/rl-outer-forecast-archive-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms + 1,
        downloaded_at_ms=committed_at_ms + 1,
        schema_sha256=MASSIVE_ADAPTIVE_RL_OUTER_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=commitment.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms + 1,
        request_id=f"ADAPTIVE-RL-OUTER-FORECAST-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms + 1,
    )
    generic = parse_massive_adaptive_rl_outer_forecast_archive_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_rl_outer_forecast_archive_v1(
        root=root,
        gated_archive=generic,
        commitment=commitment,
        raw_archive=raw_archive,
        checkpoint_selection=checkpoint_selection,
        selected_checkpoint=selected_checkpoint,
        training_window_plan=training_window_plan,
        outer_tensor=outer_tensor,
        outer_decision_roots=outer_decision_roots,
        outer_plan=outer_plan,
        model_spec=model_spec,
    )


__all__ = [
    "MassiveAdaptiveRLOuterForecastArchiveV1",
    "MassiveAdaptiveRLOuterForecastArchiveV1Error",
    "authorize_massive_adaptive_rl_outer_forecast_archive_v1",
    "materialize_massive_adaptive_rl_outer_forecast_archive_v1",
    "parse_massive_adaptive_rl_outer_forecast_archive_v1",
]
