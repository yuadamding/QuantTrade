"""V2-backed canonical inner-validation inputs for adaptive RL.

The V1 validation inputs remain the execution witnesses used by the existing
forecast and economic-environment implementations.  This module adds the
generation boundary required by the validation-complete source graph: every
authorizing V2 input binds the exact SourceBundle V2, RuntimeSourceGraph V2,
ReplayDependencyIndex V2, RuntimeSources V2, and the exact persisted V1 child
that was rebuilt from that runtime.

Generic reload is deliberately nonauthorizing.  A V2 authority is promoted
only by exact replay against RuntimeSources V2 and the completed four-fold fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
)
from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferencePlanV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveRLValidationEnvironmentAuthorityV1,
    MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    MassiveAdaptiveRLValidationSourcesAuthorityV1,
    authorize_massive_adaptive_rl_validation_environment_registry_v1,
    authorize_massive_adaptive_rl_validation_sources_authority_v1,
    load_massive_adaptive_rl_validation_environment_registry_v1,
    load_massive_adaptive_rl_validation_sources_authority_v1,
    massive_adaptive_rl_validation_downstream_evidence_exists_v1,
    prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v1,
    prepare_or_resume_massive_adaptive_rl_validation_sources_v1,
    validation_environment_registry_relative_path_v1,
    validation_sources_authority_relative_path_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MassiveAdaptiveRLExperimentManifestV4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility,
)


MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-sources-authority-v2"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_DATASET = (
    "massive-adaptive-rl-validation-sources-authority-v2"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-environment-registry-v2"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_DATASET = (
    "massive-adaptive-rl-validation-environment-registry-v2"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SCHEMA,
            "encoding": "canonical-json-v2-backed-validation-sources",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SCHEMA,
            "encoding": "canonical-json-v2-backed-validation-environment-registry",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "runtime_sources": "exact-validation-complete-runtime-sources-v2",
        "training_compatibility": "exact-v1-runtime-and-graph-witness",
        "base_v1_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SPEC_SHA256
        ),
        "base_v1_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V1_SOURCE_SHA256
        ),
        "base_child": "exact-persisted-runtime-replayed-validation-sources-v1",
        "source_generation": "bundle-graph-index-runtime-v2-receipts",
        "projections": "exact-training-and-validation-source-projections",
        "publication": "manifest-and-fold-derived-create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "validation_sources": "exact-runtime-replayed-v2-source-authority",
        "runtime_sources": "exact-validation-complete-runtime-sources-v2",
        "base_v1_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256
        ),
        "base_v1_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256
        ),
        "base_child": "exact-persisted-runtime-replayed-registry-v1",
        "costs": (10.0, 20.0, 40.0),
        "publication": "manifest-and-fold-derived-create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)

_FOLD_INDICES = (0, 1, 2, 3)


class MassiveAdaptiveRLValidationInputsV2Error(ValueError):
    """A V2-backed validation input is absent, mixed, or inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _transaction_exists(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "V2 validation-input source transaction is incomplete"
        )
    return all(present)


def massive_adaptive_rl_validation_downstream_evidence_exists_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
) -> bool:
    """Detect V2 barrier or outcome evidence that forbids upstream repair."""

    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in _FOLD_INDICES:
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation downstream-evidence V2 fold differs"
        )
    resolved = Path(root)
    key = f"v4-{manifest.semantic_receipt_sha256}-fold-{fold_index}"
    exact = (
        resolved
        / "massive-adaptive"
        / "rl-four-fold-validation-inputs-authority-v2"
        / f"v4-{manifest.semantic_receipt_sha256}.json",
        resolved
        / "massive-adaptive"
        / "rl-fold-validation-authority-v2"
        / f"{key}.json",
    )
    for payload in exact:
        if any(
            path.exists() or path.is_symlink()
            for path in (
                payload,
                payload.with_name(payload.name + ".receipt.json"),
                payload.with_name(payload.name + ".commit.json"),
            )
        ):
            return True
    directory = resolved / "massive-adaptive" / "rl-validation-outcome-authority-v2"
    return (
        directory.is_dir() and next(directory.glob(f"{key}-*.json*"), None) is not None
    )


def validation_sources_authority_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in _FOLD_INDICES:
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation-source V2 fold differs"
        )
    return (
        "massive-adaptive/rl-validation-sources-authority-v2/"
        f"v4-{manifest.semantic_receipt_sha256}-fold-{fold_index}.json"
    )


def validation_environment_registry_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in _FOLD_INDICES:
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation-registry V2 fold differs"
        )
    return (
        "massive-adaptive/rl-validation-environment-registry-v2/"
        f"v4-{manifest.semantic_receipt_sha256}-fold-{fold_index}.json"
    )


def _runtime_generation_facts(
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV2,
) -> dict[str, object]:
    runtime_sources.validate()
    bundle = runtime_sources.source_bundle_v2
    graph = runtime_sources.runtime_source_graph_authority_v2
    index = runtime_sources.replay_dependency_index_v2
    values: dict[str, object] = {
        "source_bundle_v2_receipt_sha256": bundle.semantic_receipt_sha256,
        "source_bundle_v2_source_receipt_sha256": bundle.source_receipt_sha256,
        "source_bundle_v2_commit_receipt_sha256": (
            bundle.source_transaction_receipt_sha256
        ),
        "source_bundle_v2_committed_at_ms": bundle.source_transaction_committed_at_ms,
        "runtime_source_graph_v2_receipt_sha256": graph.semantic_receipt_sha256,
        "runtime_source_graph_v2_source_receipt_sha256": graph.source_receipt_sha256,
        "runtime_source_graph_v2_commit_receipt_sha256": (
            graph.source_transaction_receipt_sha256
        ),
        "runtime_source_graph_v2_committed_at_ms": (
            graph.source_transaction_committed_at_ms
        ),
        "runtime_source_graph_v2_witness_receipt_sha256": (
            graph.runtime_authority_receipt_sha256
        ),
        "replay_dependency_index_v2_receipt_sha256": index.semantic_receipt_sha256,
        "replay_dependency_index_v2_source_receipt_sha256": (
            index.source_receipt_sha256
        ),
        "replay_dependency_index_v2_commit_receipt_sha256": (
            index.source_transaction_receipt_sha256
        ),
        "replay_dependency_index_v2_committed_at_ms": (
            index.source_transaction_committed_at_ms
        ),
        "runtime_sources_v2_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "base_runtime_sources_v1_receipt_sha256": (
            runtime_sources.base_runtime_sources_v1_receipt_sha256
        ),
        "base_runtime_source_graph_v1_witness_receipt_sha256": (
            runtime_sources.base_runtime_source_graph_v1_witness_receipt_sha256
        ),
        "training_source_projection_sha256": (
            runtime_sources.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            runtime_sources.validation_source_projection_sha256
        ),
    }
    if any(value is None for value in values.values()):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "runtime source generation V2 lacks a persisted transaction"
        )
    bundle_time = cast(int, values["source_bundle_v2_committed_at_ms"])
    graph_time = cast(int, values["runtime_source_graph_v2_committed_at_ms"])
    index_time = cast(int, values["replay_dependency_index_v2_committed_at_ms"])
    if not bundle_time < graph_time < index_time:
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "runtime source generation V2 chronology differs"
        )
    return values


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationSourcesAuthorityV2:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    fold_index: int
    source_bundle_v2_receipt_sha256: str
    source_bundle_v2_source_receipt_sha256: str
    source_bundle_v2_commit_receipt_sha256: str
    source_bundle_v2_committed_at_ms: int
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_source_receipt_sha256: str
    runtime_source_graph_v2_commit_receipt_sha256: str
    runtime_source_graph_v2_committed_at_ms: int
    runtime_source_graph_v2_witness_receipt_sha256: str
    replay_dependency_index_v2_receipt_sha256: str
    replay_dependency_index_v2_source_receipt_sha256: str
    replay_dependency_index_v2_commit_receipt_sha256: str
    replay_dependency_index_v2_committed_at_ms: int
    runtime_sources_v2_receipt_sha256: str
    base_runtime_sources_v1_receipt_sha256: str
    base_runtime_source_graph_v1_witness_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    base_validation_sources_v1_receipt_sha256: str
    base_validation_sources_v1_source_receipt_sha256: str
    base_validation_sources_v1_commit_receipt_sha256: str
    base_validation_sources_v1_committed_at_ms: int
    validation_origin_inputs_receipt_sha256: str
    validation_tensor_session_dates: tuple[str, ...]
    validation_decision_session_dates: tuple[str, ...]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_inputs_replayed: bool = False
    development_validation_inputs_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SCHEMA
    _runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2 | None = field(
        default=None, compare=False, repr=False
    )
    _four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _base_authority_v1: MassiveAdaptiveRLValidationSourcesAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
            and name
            not in {
                "semantic_receipt_sha256",
                "runtime_inputs_replayed",
                "development_validation_inputs_authorized",
            }
        }

    @property
    def source_transaction_verified(self) -> bool:
        return self._loaded_source is not None

    @property
    def source_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.receipt.receipt_sha256
        )

    @property
    def source_transaction_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.receipt_sha256
        )

    @property
    def source_transaction_committed_at_ms(self) -> int | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.committed_at_ms
        )

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_inputs_replayed
            and self.development_validation_inputs_authorized
            and self.source_data_qualified
        )

    @property
    def base_authority_v1(self) -> MassiveAdaptiveRLValidationSourcesAuthorityV1:
        self.validate()
        if self._base_authority_v1 is None:
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation source V2 has no V1 runtime witness"
            )
        return self._base_authority_v1

    @property
    def runtime_forecast_archive(self) -> MassiveAdaptiveForecastArchiveV2:
        return self.base_authority_v1.runtime_forecast_archive

    @property
    def runtime_inference_plan(self) -> MassiveAdaptiveInferencePlanV1:
        return self.base_authority_v1.runtime_inference_plan

    @property
    def runtime_chronology_authority(self) -> MassiveAdaptiveRLChronologyAuthorityV1:
        return self.base_authority_v1.runtime_chronology_authority

    def validate(self) -> None:
        runtime_parts = (
            self._runtime_sources_v2,
            self._four_fold_fit_authority,
            self._base_authority_v1,
        )
        runtime_present = any(value is not None for value in runtime_parts)
        if runtime_present and any(value is None for value in runtime_parts):
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation source V2 runtime witness is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        for value in runtime_parts:
            if value is not None:
                value.validate()
        expected_authorized = bool(runtime_present and self.source_data_qualified)
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in _FOLD_INDICES
            or not self.validation_tensor_session_dates
            or not self.validation_decision_session_dates
            or self.validation_tensor_session_dates
            != tuple(sorted(set(self.validation_tensor_session_dates)))
            or self.validation_decision_session_dates
            != tuple(sorted(set(self.validation_decision_session_dates)))
            or not set(self.validation_decision_session_dates).issubset(
                self.validation_tensor_session_dates
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_inputs_replayed != runtime_present
            or self.development_validation_inputs_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation source authority V2 differs"
            )
        times = (
            self.source_bundle_v2_committed_at_ms,
            self.runtime_source_graph_v2_committed_at_ms,
            self.replay_dependency_index_v2_committed_at_ms,
            self.base_validation_sources_v1_committed_at_ms,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in times
            )
            or not times[0] < times[1] < times[2]
        ):
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation source V2 generation chronology differs"
            )
        if runtime_present:
            assert self._runtime_sources_v2 is not None
            assert self._four_fold_fit_authority is not None
            assert self._base_authority_v1 is not None
            runtime = self._runtime_sources_v2
            fit = self._four_fold_fit_authority
            base = self._base_authority_v1
            validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
                runtime_sources_v2=runtime,
                four_fold_fit_authority=fit,
            )
            facts = _runtime_generation_facts(runtime)
            base_source = base.source_receipt_sha256
            base_commit = base.source_transaction_receipt_sha256
            base_time = base.source_transaction_committed_at_ms
            if (
                any(getattr(self, name) != value for name, value in facts.items())
                or base_source is None
                or base_commit is None
                or base_time is None
                or self.experiment_id != runtime.experiment_id
                or self.manifest_v4_receipt_sha256 != base.manifest_v4_receipt_sha256
                or self.training_manifest_v3_receipt_sha256
                != runtime.manifest_v3_receipt_sha256
                or self.four_fold_fit_authority_receipt_sha256
                != fit.semantic_receipt_sha256
                or base.fold_index != self.fold_index
                or base.runtime_sources_receipt_sha256
                != runtime.base_runtime_sources_v1_receipt_sha256
                or base.runtime_graph_witness_receipt_sha256
                != runtime.base_runtime_source_graph_v1_witness_receipt_sha256
                or self.base_validation_sources_v1_receipt_sha256
                != base.semantic_receipt_sha256
                or self.base_validation_sources_v1_source_receipt_sha256 != base_source
                or self.base_validation_sources_v1_commit_receipt_sha256 != base_commit
                or self.base_validation_sources_v1_committed_at_ms != base_time
                or self.validation_origin_inputs_receipt_sha256
                != base.validation_origin_inputs_receipt_sha256
                or self.validation_tensor_session_dates
                != base.validation_tensor_session_dates
                or self.validation_decision_session_dates
                != base.validation_decision_session_dates
                or not base.development_stage_authorized
            ):
                raise MassiveAdaptiveRLValidationInputsV2Error(
                    "validation source V2 contains a mixed generation"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= max(
                self.replay_dependency_index_v2_committed_at_ms,
                self.base_validation_sources_v1_committed_at_ms,
            )
        ):
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation source V2 transaction differs"
            )
        for name, digest_value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest("validation source V2", digest_value)
        _digest("validation source V2", self.semantic_receipt_sha256)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_validation_sources_authority_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    base_authority_v1: MassiveAdaptiveRLValidationSourcesAuthorityV1,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(four_fold_fit_authority) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
        or type(runtime_sources_v2) is not MassiveAdaptiveRLRuntimeSourcesV2
        or type(base_authority_v1) is not MassiveAdaptiveRLValidationSourcesAuthorityV1
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation source V2 requires exact authority types"
        )
    manifest.validate()
    runtime_sources_v2.validate()
    four_fold_fit_authority.validate()
    base_authority_v1.validate()
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    base_source = base_authority_v1.source_receipt_sha256
    base_commit = base_authority_v1.source_transaction_receipt_sha256
    base_time = base_authority_v1.source_transaction_committed_at_ms
    if (
        base_source is None
        or base_commit is None
        or base_time is None
        or not base_authority_v1.development_stage_authorized
        or base_authority_v1.experiment_id != manifest.experiment_id
        or base_authority_v1.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or base_authority_v1.runtime_sources_receipt_sha256
        != runtime_sources_v2.base_runtime_sources_v1_receipt_sha256
        or base_authority_v1.runtime_graph_witness_receipt_sha256
        != runtime_sources_v2.base_runtime_source_graph_v1_witness_receipt_sha256
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation source V1 is not the V2 runtime witness"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "four_fold_fit_authority_receipt_sha256": (
            four_fold_fit_authority.semantic_receipt_sha256
        ),
        "fold_index": base_authority_v1.fold_index,
        **_runtime_generation_facts(runtime_sources_v2),
        "base_validation_sources_v1_receipt_sha256": (
            base_authority_v1.semantic_receipt_sha256
        ),
        "base_validation_sources_v1_source_receipt_sha256": base_source,
        "base_validation_sources_v1_commit_receipt_sha256": base_commit,
        "base_validation_sources_v1_committed_at_ms": base_time,
        "validation_origin_inputs_receipt_sha256": (
            base_authority_v1.validation_origin_inputs_receipt_sha256
        ),
        "validation_tensor_session_dates": (
            base_authority_v1.validation_tensor_session_dates
        ),
        "validation_decision_session_dates": (
            base_authority_v1.validation_decision_session_dates
        ),
        "source_data_qualified": bool(
            runtime_sources_v2.source_data_qualified
            and base_authority_v1.source_data_qualified
        ),
    }
    provisional = MassiveAdaptiveRLValidationSourcesAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_inputs_replayed=True,
        development_validation_inputs_authorized=bool(body["source_data_qualified"]),
        _runtime_sources_v2=runtime_sources_v2,
        _four_fold_fit_authority=four_fold_fit_authority,
        _base_authority_v1=base_authority_v1,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse_sources_body(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation source V2 is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "validation_tensor_session_dates",
        "validation_decision_session_dates",
    ):
        rows = body.get(name)
        if not isinstance(rows, list):
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation source V2 date inventory is malformed"
            )
        body[name] = tuple(rows)
    return body


def parse_massive_adaptive_rl_validation_sources_authority_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
    body = _parse_sources_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLValidationSourcesAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_validation_sources_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    verified_at_ms: int,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
    return parse_massive_adaptive_rl_validation_sources_authority_v2(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=validation_sources_authority_relative_path_v2(
                manifest=manifest,
                fold_index=fold_index,
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_validation_sources_authority_v2(
    *,
    authority: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    base_authority_v1: MassiveAdaptiveRLValidationSourcesAuthorityV1,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
    authority.validate()
    expected = build_massive_adaptive_rl_validation_sources_authority_v2(
        manifest=manifest,
        four_fold_fit_authority=four_fold_fit_authority,
        runtime_sources_v2=runtime_sources_v2,
        base_authority_v1=base_authority_v1,
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != validation_sources_authority_relative_path_v2(
            manifest=manifest,
            fold_index=authority.fold_index,
        )
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation source authority V2 does not replay"
        )
    result = replace(
        authority,
        runtime_inputs_replayed=True,
        development_validation_inputs_authorized=authority.source_data_qualified,
        _runtime_sources_v2=runtime_sources_v2,
        _four_fold_fit_authority=four_fold_fit_authority,
        _base_authority_v1=base_authority_v1,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_validation_sources_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    authority: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    committed_at_ms: int,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
    manifest.validate()
    authority.validate()
    relative = validation_sources_authority_relative_path_v2(
        manifest=manifest,
        fold_index=authority.fold_index,
    )
    if _transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation source authority V2 already exists"
        )
    if any(
        checker(
            root=root,
            manifest=manifest,
            fold_index=authority.fold_index,
        )
        for checker in (
            massive_adaptive_rl_validation_downstream_evidence_exists_v1,
            massive_adaptive_rl_validation_downstream_evidence_exists_v2,
        )
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "missing validation source V2 cannot be created after validation evidence"
        )
    if (
        not authority.runtime_inputs_replayed
        or isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms
        <= max(
            authority.replay_dependency_index_v2_committed_at_ms,
            authority.base_validation_sources_v1_committed_at_ms,
        )
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation source V2 was not committed after its source generation"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-VALIDATION-SOURCES-V2-{authority.experiment_id}-"
            f"FOLD{authority.fold_index}"
        ),
    )
    result = replace(
        authority,
        _loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=committed_at_ms,
        ),
    )
    result.validate()
    return result


def prepare_or_resume_massive_adaptive_rl_validation_sources_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    fold_index: int,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
    runtime_sources_v2.validate()
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    base_runtime = runtime_sources_v2.base_runtime_sources_v1
    base_relative = validation_sources_authority_relative_path_v1(
        manifest=manifest,
        fold_index=fold_index,
    )
    base_exists = _transaction_exists(root=root, relative=base_relative)
    if not base_exists and not allow_materialize:
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "base validation source V1 is absent"
        )
    if base_exists:
        base = authorize_massive_adaptive_rl_validation_sources_authority_v1(
            root=root,
            authority=load_massive_adaptive_rl_validation_sources_authority_v1(
                root=root,
                manifest=manifest,
                fold_index=fold_index,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            four_fold_fit_authority=four_fold_fit_authority,
            runtime_sources=base_runtime,
        )
    else:
        base = prepare_or_resume_massive_adaptive_rl_validation_sources_v1(
            root=root,
            manifest=manifest,
            four_fold_fit_authority=four_fold_fit_authority,
            runtime_sources=base_runtime,
            fold_index=fold_index,
            committed_at_ms=committed_at_ms,
        )
    relative = validation_sources_authority_relative_path_v2(
        manifest=manifest,
        fold_index=fold_index,
    )
    if _transaction_exists(root=root, relative=relative):
        return authorize_massive_adaptive_rl_validation_sources_authority_v2(
            authority=load_massive_adaptive_rl_validation_sources_authority_v2(
                root=root,
                manifest=manifest,
                fold_index=fold_index,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            four_fold_fit_authority=four_fold_fit_authority,
            runtime_sources_v2=runtime_sources_v2,
            base_authority_v1=base,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation source authority V2 is absent"
        )
    base_time = cast(int, base.source_transaction_committed_at_ms)
    index_time = cast(
        int,
        runtime_sources_v2.replay_dependency_index_v2.source_transaction_committed_at_ms,
    )
    return materialize_massive_adaptive_rl_validation_sources_authority_v2(
        root=root,
        manifest=manifest,
        authority=build_massive_adaptive_rl_validation_sources_authority_v2(
            manifest=manifest,
            four_fold_fit_authority=four_fold_fit_authority,
            runtime_sources_v2=runtime_sources_v2,
            base_authority_v1=base,
        ),
        committed_at_ms=max(committed_at_ms + 1, base_time + 1, index_time + 1),
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationEnvironmentRegistryV2:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_index: int
    runtime_sources_v2_receipt_sha256: str
    source_bundle_v2_receipt_sha256: str
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_witness_receipt_sha256: str
    replay_dependency_index_v2_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    validation_sources_v2_receipt_sha256: str
    validation_sources_v2_source_receipt_sha256: str
    validation_sources_v2_commit_receipt_sha256: str
    validation_sources_v2_committed_at_ms: int
    base_validation_registry_v1_receipt_sha256: str
    base_validation_registry_v1_source_receipt_sha256: str
    base_validation_registry_v1_commit_receipt_sha256: str
    base_validation_registry_v1_committed_at_ms: int
    cost_basis_points: tuple[float, ...]
    environment_authority_receipts: tuple[str, ...]
    environment_authority_inventory_sha256: str
    validation_context_receipt_sha256: str
    initial_capital: float
    maximum_fill_participation: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_environments_replayed: bool = False
    development_validation_environments_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SCHEMA
    _runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2 | None = field(
        default=None, compare=False, repr=False
    )
    _validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2 | None = (
        field(default=None, compare=False, repr=False)
    )
    _base_registry_v1: MassiveAdaptiveRLValidationEnvironmentRegistryV1 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
            and name
            not in {
                "semantic_receipt_sha256",
                "runtime_environments_replayed",
                "development_validation_environments_authorized",
            }
        }

    @property
    def source_transaction_verified(self) -> bool:
        return self._loaded_source is not None

    @property
    def source_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.receipt.receipt_sha256
        )

    @property
    def source_transaction_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.receipt_sha256
        )

    @property
    def source_transaction_committed_at_ms(self) -> int | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.committed_at_ms
        )

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_environments_replayed
            and self.development_validation_environments_authorized
            and self.source_data_qualified
        )

    @property
    def base_registry_v1(self) -> MassiveAdaptiveRLValidationEnvironmentRegistryV1:
        self.validate()
        if self._base_registry_v1 is None:
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation registry V2 has no V1 runtime witness"
            )
        return self._base_registry_v1

    def build_environments(self) -> Mapping[float, MassiveAdaptiveProfitabilityEnvV1]:
        return self.base_registry_v1.build_environments()

    def environment_authority(
        self, transaction_cost_basis_points: float
    ) -> MassiveAdaptiveRLValidationEnvironmentAuthorityV1:
        return self.base_registry_v1.environment_authority(
            transaction_cost_basis_points
        )

    def validate(self) -> None:
        runtime_parts = (
            self._runtime_sources_v2,
            self._validation_sources_v2,
            self._base_registry_v1,
        )
        runtime_present = any(value is not None for value in runtime_parts)
        if runtime_present and any(value is None for value in runtime_parts):
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation registry V2 runtime witness is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        for value in runtime_parts:
            if value is not None:
                value.validate()
        expected_authorized = bool(runtime_present and self.source_data_qualified)
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in _FOLD_INDICES
            or self.cost_basis_points != (10.0, 20.0, 40.0)
            or len(self.environment_authority_receipts) != 3
            or self.environment_authority_inventory_sha256
            != semantic_sha256(self.environment_authority_receipts)
            or type(self.initial_capital) is not float
            or self.initial_capital <= 0.0
            or type(self.maximum_fill_participation) is not float
            or not 0.0 < self.maximum_fill_participation <= 1.0
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_environments_replayed != runtime_present
            or self.development_validation_environments_authorized
            != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation environment registry V2 differs"
            )
        if (
            isinstance(self.validation_sources_v2_committed_at_ms, bool)
            or isinstance(self.base_validation_registry_v1_committed_at_ms, bool)
            or self.validation_sources_v2_committed_at_ms < 0
            or self.base_validation_registry_v1_committed_at_ms < 0
            or self.validation_sources_v2_committed_at_ms
            >= self.base_validation_registry_v1_committed_at_ms
        ):
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation registry V2 commit chronology differs"
            )
        if runtime_present:
            assert self._runtime_sources_v2 is not None
            assert self._validation_sources_v2 is not None
            assert self._base_registry_v1 is not None
            runtime = self._runtime_sources_v2
            source = self._validation_sources_v2
            base = self._base_registry_v1
            base_source = base.source_receipt_sha256
            base_commit = base.source_transaction_receipt_sha256
            base_time = base.source_transaction_committed_at_ms
            if (
                not source.development_stage_authorized
                or not base.development_stage_authorized
                or base_source is None
                or base_commit is None
                or base_time is None
                or self.experiment_id != runtime.experiment_id
                or self.manifest_v4_receipt_sha256 != source.manifest_v4_receipt_sha256
                or self.training_manifest_v3_receipt_sha256
                != runtime.manifest_v3_receipt_sha256
                or self.fold_index != source.fold_index
                or self.fold_index != base.fold_index
                or self.runtime_sources_v2_receipt_sha256
                != runtime.semantic_receipt_sha256
                or self.source_bundle_v2_receipt_sha256
                != runtime.source_bundle_v2_receipt_sha256
                or self.runtime_source_graph_v2_receipt_sha256
                != runtime.runtime_source_graph_v2_receipt_sha256
                or self.runtime_source_graph_v2_witness_receipt_sha256
                != runtime.runtime_source_graph_v2_witness_receipt_sha256
                or self.replay_dependency_index_v2_receipt_sha256
                != runtime.replay_dependency_index_v2_receipt_sha256
                or self.training_source_projection_sha256
                != runtime.training_source_projection_sha256
                or self.validation_source_projection_sha256
                != runtime.validation_source_projection_sha256
                or self.validation_sources_v2_receipt_sha256
                != source.semantic_receipt_sha256
                or self.validation_sources_v2_source_receipt_sha256
                != source.source_receipt_sha256
                or self.validation_sources_v2_commit_receipt_sha256
                != source.source_transaction_receipt_sha256
                or self.validation_sources_v2_committed_at_ms
                != source.source_transaction_committed_at_ms
                or base.validation_sources_authority_receipt_sha256
                != source.base_validation_sources_v1_receipt_sha256
                or base.runtime_sources_receipt_sha256
                != runtime.base_runtime_sources_v1_receipt_sha256
                or self.base_validation_registry_v1_receipt_sha256
                != base.semantic_receipt_sha256
                or self.base_validation_registry_v1_source_receipt_sha256 != base_source
                or self.base_validation_registry_v1_commit_receipt_sha256 != base_commit
                or self.base_validation_registry_v1_committed_at_ms != base_time
                or self.cost_basis_points != base.cost_basis_points
                or self.environment_authority_receipts
                != base.environment_authority_receipts
                or self.environment_authority_inventory_sha256
                != base.environment_authority_inventory_sha256
                or self.validation_context_receipt_sha256
                != base.validation_context_receipt_sha256
                or self.initial_capital != base.initial_capital
                or self.maximum_fill_participation != base.maximum_fill_participation
            ):
                raise MassiveAdaptiveRLValidationInputsV2Error(
                    "validation registry V2 contains a mixed generation"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= max(
                self.validation_sources_v2_committed_at_ms,
                self.base_validation_registry_v1_committed_at_ms,
            )
        ):
            raise MassiveAdaptiveRLValidationInputsV2Error(
                "validation registry V2 transaction differs"
            )
        for name, digest_value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest("validation registry V2", digest_value)
        for environment_receipt in self.environment_authority_receipts:
            _digest("validation registry V2 environment", environment_receipt)
        _digest("validation registry V2", self.semantic_receipt_sha256)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_validation_environment_registry_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    base_registry_v1: MassiveAdaptiveRLValidationEnvironmentRegistryV1,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(runtime_sources_v2) is not MassiveAdaptiveRLRuntimeSourcesV2
        or type(validation_sources_v2)
        is not MassiveAdaptiveRLValidationSourcesAuthorityV2
        or type(base_registry_v1)
        is not MassiveAdaptiveRLValidationEnvironmentRegistryV1
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation registry V2 requires exact authority types"
        )
    manifest.validate()
    runtime_sources_v2.validate()
    validation_sources_v2.validate()
    base_registry_v1.validate()
    source_receipt = validation_sources_v2.source_receipt_sha256
    source_commit = validation_sources_v2.source_transaction_receipt_sha256
    source_time = validation_sources_v2.source_transaction_committed_at_ms
    base_source = base_registry_v1.source_receipt_sha256
    base_commit = base_registry_v1.source_transaction_receipt_sha256
    base_time = base_registry_v1.source_transaction_committed_at_ms
    if any(
        value is None
        for value in (
            source_receipt,
            source_commit,
            source_time,
            base_source,
            base_commit,
            base_time,
        )
    ) or not (
        validation_sources_v2.development_stage_authorized
        and base_registry_v1.development_stage_authorized
        and base_registry_v1.validation_sources_authority_receipt_sha256
        == validation_sources_v2.base_validation_sources_v1_receipt_sha256
        and base_registry_v1.runtime_sources_receipt_sha256
        == runtime_sources_v2.base_runtime_sources_v1_receipt_sha256
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation registry V1 is not the V2 runtime witness"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_index": validation_sources_v2.fold_index,
        "runtime_sources_v2_receipt_sha256": runtime_sources_v2.semantic_receipt_sha256,
        "source_bundle_v2_receipt_sha256": (
            runtime_sources_v2.source_bundle_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_receipt_sha256": (
            runtime_sources_v2.runtime_source_graph_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_witness_receipt_sha256": (
            runtime_sources_v2.runtime_source_graph_v2_witness_receipt_sha256
        ),
        "replay_dependency_index_v2_receipt_sha256": (
            runtime_sources_v2.replay_dependency_index_v2_receipt_sha256
        ),
        "training_source_projection_sha256": (
            runtime_sources_v2.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            runtime_sources_v2.validation_source_projection_sha256
        ),
        "validation_sources_v2_receipt_sha256": (
            validation_sources_v2.semantic_receipt_sha256
        ),
        "validation_sources_v2_source_receipt_sha256": cast(str, source_receipt),
        "validation_sources_v2_commit_receipt_sha256": cast(str, source_commit),
        "validation_sources_v2_committed_at_ms": cast(int, source_time),
        "base_validation_registry_v1_receipt_sha256": (
            base_registry_v1.semantic_receipt_sha256
        ),
        "base_validation_registry_v1_source_receipt_sha256": cast(str, base_source),
        "base_validation_registry_v1_commit_receipt_sha256": cast(str, base_commit),
        "base_validation_registry_v1_committed_at_ms": cast(int, base_time),
        "cost_basis_points": base_registry_v1.cost_basis_points,
        "environment_authority_receipts": (
            base_registry_v1.environment_authority_receipts
        ),
        "environment_authority_inventory_sha256": (
            base_registry_v1.environment_authority_inventory_sha256
        ),
        "validation_context_receipt_sha256": (
            base_registry_v1.validation_context_receipt_sha256
        ),
        "initial_capital": base_registry_v1.initial_capital,
        "maximum_fill_participation": (base_registry_v1.maximum_fill_participation),
        "source_data_qualified": bool(
            runtime_sources_v2.source_data_qualified
            and validation_sources_v2.source_data_qualified
            and base_registry_v1.source_data_qualified
        ),
    }
    provisional = MassiveAdaptiveRLValidationEnvironmentRegistryV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_environments_replayed=True,
        development_validation_environments_authorized=bool(
            body["source_data_qualified"]
        ),
        _runtime_sources_v2=runtime_sources_v2,
        _validation_sources_v2=validation_sources_v2,
        _base_registry_v1=base_registry_v1,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse_registry_body(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation registry V2 is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    costs = cast(Sequence[float | int], body["cost_basis_points"])
    body["cost_basis_points"] = tuple(float(row) for row in costs)
    body["environment_authority_receipts"] = tuple(
        cast(Sequence[str], body["environment_authority_receipts"])
    )
    body["initial_capital"] = float(cast(float | int, body["initial_capital"]))
    body["maximum_fill_participation"] = float(
        cast(float | int, body["maximum_fill_participation"])
    )
    return body


def parse_massive_adaptive_rl_validation_environment_registry_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
    body = _parse_registry_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLValidationEnvironmentRegistryV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_validation_environment_registry_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    verified_at_ms: int,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
    return parse_massive_adaptive_rl_validation_environment_registry_v2(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=validation_environment_registry_relative_path_v2(
                manifest=manifest,
                fold_index=fold_index,
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_validation_environment_registry_v2(
    *,
    registry: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    base_registry_v1: MassiveAdaptiveRLValidationEnvironmentRegistryV1,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
    registry.validate()
    expected = build_massive_adaptive_rl_validation_environment_registry_v2(
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        validation_sources_v2=validation_sources_v2,
        base_registry_v1=base_registry_v1,
    )
    if (
        registry._loaded_source is None
        or registry._loaded_source.payload_relative_path
        != validation_environment_registry_relative_path_v2(
            manifest=manifest,
            fold_index=registry.fold_index,
        )
        or registry.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation registry V2 does not replay"
        )
    result = replace(
        registry,
        runtime_environments_replayed=True,
        development_validation_environments_authorized=registry.source_data_qualified,
        _runtime_sources_v2=runtime_sources_v2,
        _validation_sources_v2=validation_sources_v2,
        _base_registry_v1=base_registry_v1,
    )
    result.validate()
    environments = result.build_environments()
    for authority in result.base_registry_v1.environment_authorities:
        authority.validate_environment(
            environments[authority.transaction_cost_basis_points]
        )
    return result


def materialize_massive_adaptive_rl_validation_environment_registry_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    registry: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    committed_at_ms: int,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
    manifest.validate()
    registry.validate()
    relative = validation_environment_registry_relative_path_v2(
        manifest=manifest,
        fold_index=registry.fold_index,
    )
    if _transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation registry V2 already exists"
        )
    if any(
        checker(
            root=root,
            manifest=manifest,
            fold_index=registry.fold_index,
        )
        for checker in (
            massive_adaptive_rl_validation_downstream_evidence_exists_v1,
            massive_adaptive_rl_validation_downstream_evidence_exists_v2,
        )
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "missing validation registry V2 cannot be created after validation evidence"
        )
    if (
        not registry.runtime_environments_replayed
        or isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms
        <= max(
            registry.validation_sources_v2_committed_at_ms,
            registry.base_validation_registry_v1_committed_at_ms,
        )
    ):
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation registry V2 was not committed after its inputs"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(registry.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=registry.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-VALIDATION-REGISTRY-V2-{registry.experiment_id}-"
            f"FOLD{registry.fold_index}"
        ),
    )
    result = replace(
        registry,
        _loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=committed_at_ms,
        ),
    )
    result.validate()
    return result


def prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
    base_runtime = runtime_sources_v2.base_runtime_sources_v1
    base_source = validation_sources_v2.base_authority_v1
    base_relative = validation_environment_registry_relative_path_v1(
        manifest=manifest,
        fold_index=validation_sources_v2.fold_index,
    )
    base_exists = _transaction_exists(root=root, relative=base_relative)
    if not base_exists and not allow_materialize:
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "base validation registry V1 is absent"
        )
    if base_exists:
        base = authorize_massive_adaptive_rl_validation_environment_registry_v1(
            root=root,
            registry=load_massive_adaptive_rl_validation_environment_registry_v1(
                root=root,
                manifest=manifest,
                fold_index=validation_sources_v2.fold_index,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            validation_sources=base_source,
            runtime_sources=base_runtime,
        )
    else:
        base = prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v1(
            root=root,
            manifest=manifest,
            validation_sources=base_source,
            runtime_sources=base_runtime,
            committed_at_ms=committed_at_ms,
        )
    relative = validation_environment_registry_relative_path_v2(
        manifest=manifest,
        fold_index=validation_sources_v2.fold_index,
    )
    if _transaction_exists(root=root, relative=relative):
        return authorize_massive_adaptive_rl_validation_environment_registry_v2(
            registry=load_massive_adaptive_rl_validation_environment_registry_v2(
                root=root,
                manifest=manifest,
                fold_index=validation_sources_v2.fold_index,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            runtime_sources_v2=runtime_sources_v2,
            validation_sources_v2=validation_sources_v2,
            base_registry_v1=base,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLValidationInputsV2Error(
            "validation registry V2 is absent"
        )
    source_time = cast(int, validation_sources_v2.source_transaction_committed_at_ms)
    base_time = cast(int, base.source_transaction_committed_at_ms)
    return materialize_massive_adaptive_rl_validation_environment_registry_v2(
        root=root,
        manifest=manifest,
        registry=build_massive_adaptive_rl_validation_environment_registry_v2(
            manifest=manifest,
            runtime_sources_v2=runtime_sources_v2,
            validation_sources_v2=validation_sources_v2,
            base_registry_v1=base,
        ),
        committed_at_ms=max(committed_at_ms + 1, source_time + 1, base_time + 1),
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_DATASET",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_DATASET",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SPEC_SHA256",
    "MassiveAdaptiveRLValidationEnvironmentRegistryV2",
    "MassiveAdaptiveRLValidationInputsV2Error",
    "MassiveAdaptiveRLValidationSourcesAuthorityV2",
    "authorize_massive_adaptive_rl_validation_environment_registry_v2",
    "authorize_massive_adaptive_rl_validation_sources_authority_v2",
    "build_massive_adaptive_rl_validation_environment_registry_v2",
    "build_massive_adaptive_rl_validation_sources_authority_v2",
    "load_massive_adaptive_rl_validation_environment_registry_v2",
    "load_massive_adaptive_rl_validation_sources_authority_v2",
    "massive_adaptive_rl_validation_downstream_evidence_exists_v2",
    "materialize_massive_adaptive_rl_validation_environment_registry_v2",
    "materialize_massive_adaptive_rl_validation_sources_authority_v2",
    "parse_massive_adaptive_rl_validation_environment_registry_v2",
    "parse_massive_adaptive_rl_validation_sources_authority_v2",
    "prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v2",
    "prepare_or_resume_massive_adaptive_rl_validation_sources_v2",
    "validation_environment_registry_relative_path_v2",
    "validation_sources_authority_relative_path_v2",
]
