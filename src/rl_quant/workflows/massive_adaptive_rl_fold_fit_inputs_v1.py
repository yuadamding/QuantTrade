"""Persisted Manifest-V3-owned inputs for one adaptive-RL fit fold."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from io import BytesIO
import json
from pathlib import Path
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_fit_environment_registry_v1 import (
    MassiveAdaptiveRLFitEnvironmentRegistryV1,
    build_massive_adaptive_rl_fit_environment_registry_v1,
)
from rl_quant.training.massive_adaptive_rl_fold_fit_chronology_authority_v1 import (
    MassiveAdaptiveRLFoldFitChronologyAuthorityV1,
    build_massive_adaptive_rl_fold_fit_chronology_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v2 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV2,
    build_massive_adaptive_rl_training_forecast_authority_v2,
)
from rl_quant.workflows.massive_adaptive_rl_execution_environment_v1 import (
    MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLRuntimeSourcesV1,
)


MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fold-fit-inputs-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-fold-fit-inputs-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA,
        "encoding": "canonical-json-receipt-envelope",
        "runtime_replay": "rebuild-all-inputs-from-manifest-and-runtime-sources",
    }
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "caller_inputs": "manifest-v3-and-witnessed-runtime-sources-only",
        "training_forecast": "package-built-authority-v2",
        "chronology": "fit-only-date-role-authority",
        "environments": "package-built-source-qualified-registry",
        "execution_environment": "persisted-clean-deterministic-runtime-authority-v1",
        "publication": "create-only-source-transaction",
        "profitability_reporting": False,
        "outer_access": False,
    }
)


class MassiveAdaptiveRLFoldFitInputsV1Error(ValueError):
    """Persisted fit inputs did not replay from their exact source graph."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFoldFitInputsV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLFoldFitInputsV1Error(
            "adaptive RL fold-fit input artifact ID is not path safe"
        )
    return value


def fold_fit_inputs_relative_path_v1(*, experiment_id: str, fold_index: int) -> str:
    if fold_index not in range(4):
        raise MassiveAdaptiveRLFoldFitInputsV1Error(
            "adaptive RL fold-fit input fold index differs"
        )
    return (
        "massive-adaptive/rl-fold-fit-inputs-v1/"
        f"{_artifact_id(experiment_id)}-fold{fold_index}.json"
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFoldFitInputsAuthorityV1:
    experiment_id: str
    outer_fold_index: int
    manifest_v3_receipt_sha256: str
    base_manifest_v2_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    execution_environment_authority: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1
    training_forecast_authority: MassiveAdaptiveRLTrainingForecastAuthorityV2
    fit_chronology_authority: MassiveAdaptiveRLFoldFitChronologyAuthorityV1
    fit_environment_registry: MassiveAdaptiveRLFitEnvironmentRegistryV1
    source_data_qualified: bool
    runtime_inputs_replayed: bool
    semantic_receipt_sha256: str
    development_rl_training_inputs_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA
    _manifest: MassiveAdaptiveRLExperimentManifestV3 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "outer_fold_index": self.outer_fold_index,
            "manifest_v3_receipt_sha256": self.manifest_v3_receipt_sha256,
            "base_manifest_v2_receipt_sha256": (self.base_manifest_v2_receipt_sha256),
            "runtime_sources_receipt_sha256": self.runtime_sources_receipt_sha256,
            "runtime_graph_witness_receipt_sha256": (
                self.runtime_graph_witness_receipt_sha256
            ),
            "execution_environment_authority_receipt_sha256": (
                self.execution_environment_authority.semantic_receipt_sha256
            ),
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority.semantic_receipt_sha256
            ),
            "fit_chronology_authority_receipt_sha256": (
                self.fit_chronology_authority.semantic_receipt_sha256
            ),
            "fit_environment_registry_receipt_sha256": (
                self.fit_environment_registry.semantic_receipt_sha256
            ),
            "fit_environment_mapping_receipt_sha256": (
                self.fit_environment_registry.environment_registry_receipt_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.execution_environment_authority.validate()
        self.training_forecast_authority.validate()
        self.fit_chronology_authority.validate()
        self.fit_environment_registry.validate()
        runtime_present = (
            self._manifest is not None and self._runtime_sources is not None
        )
        if runtime_present:
            assert self._manifest is not None
            assert self._runtime_sources is not None
            self._manifest.validate()
            self._runtime_sources.validate()
        runtime_receipt = (
            None
            if self._runtime_sources is None
            else self._runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
        )
        expected_qualified = bool(
            runtime_present
            and self.execution_environment_authority.development_execution_authorized
            and self.training_forecast_authority.source_data_qualified
            and self.training_forecast_authority.reinforcement_learning_authorized
            and self.fit_chronology_authority.source_data_qualified
            and self.fit_chronology_authority.development_rl_training_authorized
            and self.fit_environment_registry.source_data_qualified
        )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA
            or not runtime_present
            or self._manifest is None
            or self._runtime_sources is None
            or self.outer_fold_index not in range(4)
            or self.experiment_id != self._manifest.experiment_id
            or self.experiment_id != self._runtime_sources.experiment_id
            or self.manifest_v3_receipt_sha256 != self._manifest.semantic_receipt_sha256
            or self.base_manifest_v2_receipt_sha256
            != self._manifest.base_manifest.semantic_receipt_sha256
            or self.runtime_sources_receipt_sha256
            != self._runtime_sources.semantic_receipt_sha256
            or self.runtime_graph_witness_receipt_sha256 != runtime_receipt
            or self.execution_environment_authority.experiment_id != self.experiment_id
            or self.execution_environment_authority.manifest_v3_receipt_sha256
            != self.manifest_v3_receipt_sha256
            or self.execution_environment_authority.execution_device_specification
            != self._manifest.execution_device_specification
            or self.training_forecast_authority.outer_fold_index
            != self.outer_fold_index
            or self.fit_chronology_authority.fold_index != self.outer_fold_index
            or self.fit_environment_registry.outer_fold_index != self.outer_fold_index
            or self.fit_chronology_authority.training_forecast_authority_receipt_sha256
            != self.training_forecast_authority.semantic_receipt_sha256
            or self.fit_environment_registry.forecast_archive_receipts
            != self.training_forecast_authority.source_forecast_archive_receipts
            or self.source_data_qualified != expected_qualified
            or self.runtime_inputs_replayed != expected_qualified
            or self.development_rl_training_inputs_authorized != expected_qualified
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFoldFitInputsV1Error(
                "adaptive RL fold-fit input authority differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLFoldFitInputsV1Error(
                "adaptive RL fold-fit input source transaction differs"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.base_manifest_v2_receipt_sha256,
            self.runtime_sources_receipt_sha256,
            self.runtime_graph_witness_receipt_sha256,
            self.execution_environment_authority.semantic_receipt_sha256,
            self.training_forecast_authority.semantic_receipt_sha256,
            self.fit_chronology_authority.semantic_receipt_sha256,
            self.fit_environment_registry.semantic_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL fold-fit inputs", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_fold_fit_inputs_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    execution_environment_authority: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
    outer_fold_index: int,
) -> MassiveAdaptiveRLFoldFitInputsAuthorityV1:
    manifest.validate()
    runtime_sources.validate()
    execution_environment_authority.validate()
    if (
        outer_fold_index not in manifest.base_manifest.fold_indices
        or manifest.experiment_id != runtime_sources.experiment_id
        or manifest.semantic_receipt_sha256
        != runtime_sources.manifest_v3_receipt_sha256
        or not execution_environment_authority.development_execution_authorized
    ):
        raise MassiveAdaptiveRLFoldFitInputsV1Error(
            "adaptive RL fold-fit input roots differ"
        )
    fold = runtime_sources.fold(outer_fold_index)
    training = build_massive_adaptive_rl_training_forecast_authority_v2(
        outer_fold_index=outer_fold_index,
        block_sessions=manifest.base_manifest.prequential_block_sessions,
        split_plan=runtime_sources.split_plan,
        forecast_archives=fold.fit_forecast_archives,
        training_window_plans=fold.training_windows,
        checkpoint_choices=fold.checkpoint_choices,
        calibrations=fold.calibrations,
    )
    registry = build_massive_adaptive_rl_fit_environment_registry_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        outer_fold_index=outer_fold_index,
    )
    chronology = build_massive_adaptive_rl_fold_fit_chronology_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        training_forecast_authority=training,
    )
    runtime_receipt = (
        runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    )
    if runtime_receipt is None:
        raise MassiveAdaptiveRLFoldFitInputsV1Error(
            "adaptive RL runtime graph witness receipt is absent"
        )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "outer_fold_index": outer_fold_index,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "base_manifest_v2_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": runtime_receipt,
        "execution_environment_authority": execution_environment_authority,
        "training_forecast_authority": training,
        "fit_chronology_authority": chronology,
        "fit_environment_registry": registry,
        "source_data_qualified": True,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFoldFitInputsAuthorityV1(
        **body,  # type: ignore[arg-type]
        runtime_inputs_replayed=True,
        semantic_receipt_sha256="0" * 64,
        development_rl_training_inputs_authorized=True,
        _manifest=manifest,
        _runtime_sources=runtime_sources,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFoldFitInputsV1Error(
            "adaptive RL fold-fit inputs are not canonical JSON"
        )
    return dict(cast(Mapping[str, object], value))


def authorize_massive_adaptive_rl_fold_fit_inputs_authority_v1(
    *,
    root: str | Path,
    loaded_source: LoadedMassiveSourceObject,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    execution_environment_authority: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
    outer_fold_index: int,
) -> MassiveAdaptiveRLFoldFitInputsAuthorityV1:
    rebuilt = build_massive_adaptive_rl_fold_fit_inputs_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        execution_environment_authority=execution_environment_authority,
        outer_fold_index=outer_fold_index,
    )
    expected = {
        **rebuilt.semantic_unsigned(),
        "semantic_receipt_sha256": rebuilt.semantic_receipt_sha256,
    }
    if canonical_json_file_bytes(
        _load_payload(root=root, loaded_source=loaded_source)
    ) != canonical_json_file_bytes(expected):
        raise MassiveAdaptiveRLFoldFitInputsV1Error(
            "adaptive RL fold-fit inputs did not replay"
        )
    result = replace(rebuilt, _loaded_source=loaded_source)
    result.validate()
    return result


def materialize_massive_adaptive_rl_fold_fit_inputs_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    execution_environment_authority: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
    outer_fold_index: int,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFoldFitInputsAuthorityV1:
    resolved_root = Path(root)
    try:
        resolved_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MassiveAdaptiveRLFoldFitInputsV1Error(
            "adaptive RL fold-fit input root is unavailable"
        ) from error
    authority = build_massive_adaptive_rl_fold_fit_inputs_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        execution_environment_authority=execution_environment_authority,
        outer_fold_index=outer_fold_index,
    )
    relative = fold_fit_inputs_relative_path_v1(
        experiment_id=manifest.experiment_id,
        fold_index=outer_fold_index,
    )
    payload = {
        **authority.semantic_unsigned(),
        "semantic_receipt_sha256": authority.semantic_receipt_sha256,
    }
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=resolved_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            "ADAPTIVE-RL-FOLD-FIT-INPUTS-V1-"
            f"{manifest.experiment_id}-fold{outer_fold_index}"
        ),
    )
    loaded = load_massive_source_bundle(
        root=resolved_root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_fold_fit_inputs_authority_v1(
        root=resolved_root,
        loaded_source=loaded,
        manifest=manifest,
        runtime_sources=runtime_sources,
        execution_environment_authority=execution_environment_authority,
        outer_fold_index=outer_fold_index,
    )


def load_massive_adaptive_rl_fold_fit_inputs_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    execution_environment_authority: MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
    outer_fold_index: int,
    verified_at_ms: int,
) -> MassiveAdaptiveRLFoldFitInputsAuthorityV1:
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=fold_fit_inputs_relative_path_v1(
            experiment_id=manifest.experiment_id,
            fold_index=outer_fold_index,
        ),
        verified_at_ms=verified_at_ms,
    )
    return authorize_massive_adaptive_rl_fold_fit_inputs_authority_v1(
        root=root,
        loaded_source=loaded,
        manifest=manifest,
        runtime_sources=runtime_sources,
        execution_environment_authority=execution_environment_authority,
        outer_fold_index=outer_fold_index,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA",
    "MassiveAdaptiveRLFoldFitInputsAuthorityV1",
    "MassiveAdaptiveRLFoldFitInputsV1Error",
    "authorize_massive_adaptive_rl_fold_fit_inputs_authority_v1",
    "build_massive_adaptive_rl_fold_fit_inputs_authority_v1",
    "fold_fit_inputs_relative_path_v1",
    "load_massive_adaptive_rl_fold_fit_inputs_authority_v1",
    "materialize_massive_adaptive_rl_fold_fit_inputs_authority_v1",
]
