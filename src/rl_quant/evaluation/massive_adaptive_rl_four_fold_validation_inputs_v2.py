"""Experiment-wide V2 source-generation barrier for inner validation.

All four V2 validation-source authorities and all four V2 environment
registries are committed before this create-only aggregate.  The aggregate
also binds the exact transitional V1 barrier used by the existing evaluators,
but only the V2 authority represents the validation-complete generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import fcntl
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from typing import Iterator, cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v1 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V1_SPEC_SHA256,
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
    run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    massive_adaptive_rl_validation_downstream_evidence_exists_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v2 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256,
    MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    MassiveAdaptiveRLValidationSourcesAuthorityV2,
    massive_adaptive_rl_validation_downstream_evidence_exists_v2,
    prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v2,
    prepare_or_resume_massive_adaptive_rl_validation_sources_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
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


MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-four-fold-validation-inputs-authority-v2"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_DATASET = (
    "massive-adaptive-rl-four-fold-validation-inputs-authority-v2"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SHA256 = (
    file_sha256(Path(__file__))
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": (
                MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SCHEMA
            ),
            "encoding": "canonical-json-v2-source-generation-validation-barrier",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V2_SPEC_SHA256 = semantic_sha256(
    {
        "runtime_sources": "exact-validation-complete-runtime-sources-v2",
        "training_compatibility": "exact-base-v1-runtime-and-graph-witness",
        "validation_sources": "four-exact-v2-source-authorities",
        "environment_registries": "four-exact-v2-registries",
        "candidate_population": "exact-completed-four-fold-fit-inventories",
        "base_v1_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V1_SPEC_SHA256
        ),
        "base_v1_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SHA256
        ),
        "v2_input_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256
        ),
        "ordering": "all-v2-inputs-before-v2-barrier-before-every-outcome",
        "publication": "manifest-derived-create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)

_FOLD_INDICES = (0, 1, 2, 3)


class MassiveAdaptiveRLFourFoldValidationInputsV2Error(ValueError):
    """The V2 four-fold validation-input authority differs or is absent."""


class MassiveAdaptiveRLFourFoldValidationInputsV2LeaseUnavailable(
    MassiveAdaptiveRLFourFoldValidationInputsV2Error
):
    """Another process owns V2 validation-input publication."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
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
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "four-fold validation-input V2 transaction is incomplete"
        )
    return all(present)


def four_fold_validation_inputs_authority_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> str:
    manifest.validate()
    return (
        "massive-adaptive/rl-four-fold-validation-inputs-authority-v2/"
        f"v4-{manifest.semantic_receipt_sha256}.json"
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    source_bundle_v2_receipt_sha256: str
    source_bundle_v2_source_receipt_sha256: str
    source_bundle_v2_commit_receipt_sha256: str
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_witness_receipt_sha256: str
    replay_dependency_index_v2_receipt_sha256: str
    runtime_sources_v2_receipt_sha256: str
    base_runtime_sources_v1_receipt_sha256: str
    base_runtime_source_graph_v1_witness_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    fold_indices: tuple[int, ...]
    validation_sources_v2_receipts: tuple[str, ...]
    validation_sources_v2_source_receipts: tuple[str, ...]
    validation_sources_v2_commit_receipts: tuple[str, ...]
    validation_sources_v2_committed_at_ms: tuple[int, ...]
    validation_environment_registry_v2_receipts: tuple[str, ...]
    validation_registry_v2_source_receipts: tuple[str, ...]
    validation_registry_v2_commit_receipts: tuple[str, ...]
    validation_registry_v2_committed_at_ms: tuple[int, ...]
    base_validation_sources_v1_receipts: tuple[str, ...]
    base_validation_registry_v1_receipts: tuple[str, ...]
    validation_context_receipts: tuple[str, ...]
    validation_decision_session_date_inventories: tuple[tuple[str, ...], ...]
    expected_candidate_checkpoint_authority_receipt_inventories: tuple[
        tuple[str, ...], ...
    ]
    base_four_fold_validation_inputs_v1_receipt_sha256: str
    base_four_fold_validation_inputs_v1_source_receipt_sha256: str
    base_four_fold_validation_inputs_v1_commit_receipt_sha256: str
    base_four_fold_validation_inputs_v1_committed_at_ms: int
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_inputs_replayed: bool = False
    development_validation_execution_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V2_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SCHEMA
    _runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2 | None = field(
        default=None, compare=False, repr=False
    )
    _four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _validation_sources_v2: tuple[
        MassiveAdaptiveRLValidationSourcesAuthorityV2, ...
    ] = field(default=(), compare=False, repr=False)
    _validation_registries_v2: tuple[
        MassiveAdaptiveRLValidationEnvironmentRegistryV2, ...
    ] = field(default=(), compare=False, repr=False)
    _base_authority_v1: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1 | None = (
        field(default=None, compare=False, repr=False)
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
                "development_validation_execution_authorized",
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
            and self.development_validation_execution_authorized
            and self.source_data_qualified
        )

    @property
    def base_authority_v1(self) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1:
        self.validate()
        if self._base_authority_v1 is None:
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation-input V2 has no V1 witness"
            )
        return self._base_authority_v1

    def validation_sources(
        self, fold_index: int
    ) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
        self.validate()
        if fold_index not in self.fold_indices or not self._validation_sources_v2:
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation source V2 is unavailable"
            )
        return self._validation_sources_v2[self.fold_indices.index(fold_index)]

    def validation_registry(
        self, fold_index: int
    ) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
        self.validate()
        if fold_index not in self.fold_indices or not self._validation_registries_v2:
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation registry V2 is unavailable"
            )
        return self._validation_registries_v2[self.fold_indices.index(fold_index)]

    def validate(self) -> None:
        runtime_parts = (
            self._runtime_sources_v2,
            self._four_fold_fit_authority,
            self._base_authority_v1,
        )
        runtime_present = any(value is not None for value in runtime_parts) or bool(
            self._validation_sources_v2 or self._validation_registries_v2
        )
        if runtime_present and (
            any(value is None for value in runtime_parts)
            or len(self._validation_sources_v2) != 4
            or len(self._validation_registries_v2) != 4
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation-input V2 runtime is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        if self._runtime_sources_v2 is not None:
            self._runtime_sources_v2.validate()
        if self._four_fold_fit_authority is not None:
            self._four_fold_fit_authority.validate()
        if self._base_authority_v1 is not None:
            self._base_authority_v1.validate()
        for source_authority in self._validation_sources_v2:
            source_authority.validate()
        for registry_authority in self._validation_registries_v2:
            registry_authority.validate()
        expected_authorized = bool(
            runtime_present
            and all(
                row.development_stage_authorized for row in self._validation_sources_v2
            )
            and all(
                row.development_stage_authorized
                for row in self._validation_registries_v2
            )
        )
        inventories: tuple[Sequence[object], ...] = (
            self.validation_sources_v2_receipts,
            self.validation_sources_v2_source_receipts,
            self.validation_sources_v2_commit_receipts,
            self.validation_sources_v2_committed_at_ms,
            self.validation_environment_registry_v2_receipts,
            self.validation_registry_v2_source_receipts,
            self.validation_registry_v2_commit_receipts,
            self.validation_registry_v2_committed_at_ms,
            self.base_validation_sources_v1_receipts,
            self.base_validation_registry_v1_receipts,
            self.validation_context_receipts,
            self.validation_decision_session_date_inventories,
            self.expected_candidate_checkpoint_authority_receipt_inventories,
        )
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SCHEMA
            or not self.experiment_id
            or self.fold_indices != _FOLD_INDICES
            or any(len(rows) != 4 for rows in inventories)
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_inputs_replayed != runtime_present
            or self.development_validation_execution_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation-input authority V2 differs"
            )
        commit_times = (
            *self.validation_sources_v2_committed_at_ms,
            *self.validation_registry_v2_committed_at_ms,
            self.base_four_fold_validation_inputs_v1_committed_at_ms,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in commit_times
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation-input V2 commit time differs"
            )
        if any(
            source_time >= registry_time
            for source_time, registry_time in zip(
                self.validation_sources_v2_committed_at_ms,
                self.validation_registry_v2_committed_at_ms,
                strict=True,
            )
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "validation source V2 was not committed before its registry"
            )
        if runtime_present:
            assert self._runtime_sources_v2 is not None
            assert self._four_fold_fit_authority is not None
            assert self._base_authority_v1 is not None
            runtime = self._runtime_sources_v2
            fit = self._four_fold_fit_authority
            base = self._base_authority_v1
            sources = self._validation_sources_v2
            registries = self._validation_registries_v2
            validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
                runtime_sources_v2=runtime,
                four_fold_fit_authority=fit,
            )
            bundle = runtime.source_bundle_v2
            base_source_receipt = base.source_receipt_sha256
            base_commit_receipt = base.source_transaction_receipt_sha256
            base_time = base.source_transaction_committed_at_ms
            if (
                base_source_receipt is None
                or base_commit_receipt is None
                or base_time is None
                or not base.development_stage_authorized
                or self.experiment_id != runtime.experiment_id
                or self.manifest_v4_receipt_sha256
                != sources[0].manifest_v4_receipt_sha256
                or self.training_manifest_v3_receipt_sha256
                != runtime.manifest_v3_receipt_sha256
                or self.four_fold_fit_authority_receipt_sha256
                != fit.semantic_receipt_sha256
                or self.source_bundle_v2_receipt_sha256
                != runtime.source_bundle_v2_receipt_sha256
                or self.source_bundle_v2_source_receipt_sha256
                != bundle.source_receipt_sha256
                or self.source_bundle_v2_commit_receipt_sha256
                != bundle.source_transaction_receipt_sha256
                or self.runtime_source_graph_v2_receipt_sha256
                != runtime.runtime_source_graph_v2_receipt_sha256
                or self.runtime_source_graph_v2_witness_receipt_sha256
                != runtime.runtime_source_graph_v2_witness_receipt_sha256
                or self.replay_dependency_index_v2_receipt_sha256
                != runtime.replay_dependency_index_v2_receipt_sha256
                or self.runtime_sources_v2_receipt_sha256
                != runtime.semantic_receipt_sha256
                or self.base_runtime_sources_v1_receipt_sha256
                != runtime.base_runtime_sources_v1_receipt_sha256
                or self.base_runtime_source_graph_v1_witness_receipt_sha256
                != runtime.base_runtime_source_graph_v1_witness_receipt_sha256
                or self.training_source_projection_sha256
                != runtime.training_source_projection_sha256
                or self.validation_source_projection_sha256
                != runtime.validation_source_projection_sha256
                or tuple(row.fold_index for row in sources) != _FOLD_INDICES
                or tuple(row.fold_index for row in registries) != _FOLD_INDICES
                or self.validation_sources_v2_receipts
                != tuple(row.semantic_receipt_sha256 for row in sources)
                or self.validation_sources_v2_source_receipts
                != tuple(cast(str, row.source_receipt_sha256) for row in sources)
                or self.validation_sources_v2_commit_receipts
                != tuple(
                    cast(str, row.source_transaction_receipt_sha256) for row in sources
                )
                or self.validation_sources_v2_committed_at_ms
                != tuple(
                    cast(int, row.source_transaction_committed_at_ms) for row in sources
                )
                or self.validation_environment_registry_v2_receipts
                != tuple(row.semantic_receipt_sha256 for row in registries)
                or self.validation_registry_v2_source_receipts
                != tuple(cast(str, row.source_receipt_sha256) for row in registries)
                or self.validation_registry_v2_commit_receipts
                != tuple(
                    cast(str, row.source_transaction_receipt_sha256)
                    for row in registries
                )
                or self.validation_registry_v2_committed_at_ms
                != tuple(
                    cast(int, row.source_transaction_committed_at_ms)
                    for row in registries
                )
                or self.base_validation_sources_v1_receipts
                != tuple(
                    row.base_validation_sources_v1_receipt_sha256 for row in sources
                )
                or self.base_validation_registry_v1_receipts
                != tuple(
                    row.base_validation_registry_v1_receipt_sha256 for row in registries
                )
                or self.validation_context_receipts
                != tuple(row.validation_context_receipt_sha256 for row in registries)
                or self.validation_decision_session_date_inventories
                != tuple(row.validation_decision_session_dates for row in sources)
                or self.expected_candidate_checkpoint_authority_receipt_inventories
                != base.expected_candidate_checkpoint_authority_receipt_inventories
                or self.base_four_fold_validation_inputs_v1_receipt_sha256
                != base.semantic_receipt_sha256
                or self.base_four_fold_validation_inputs_v1_source_receipt_sha256
                != base_source_receipt
                or self.base_four_fold_validation_inputs_v1_commit_receipt_sha256
                != base_commit_receipt
                or self.base_four_fold_validation_inputs_v1_committed_at_ms != base_time
                or base.runtime_sources_receipt_sha256
                != runtime.base_runtime_sources_v1_receipt_sha256
                or base.runtime_graph_witness_receipt_sha256
                != runtime.base_runtime_source_graph_v1_witness_receipt_sha256
                or any(
                    registry.validation_sources_v2_receipt_sha256
                    != source.semantic_receipt_sha256
                    for source, registry in zip(sources, registries, strict=True)
                )
            ):
                raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                    "four-fold validation-input V2 contains a mixed generation"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms <= max(commit_times)
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation-input V2 transaction differs"
            )
        for name, digest_value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest("four-fold validation inputs V2", digest_value)
        for inventory in (
            self.validation_sources_v2_receipts,
            self.validation_sources_v2_source_receipts,
            self.validation_sources_v2_commit_receipts,
            self.validation_environment_registry_v2_receipts,
            self.validation_registry_v2_source_receipts,
            self.validation_registry_v2_commit_receipts,
            self.base_validation_sources_v1_receipts,
            self.base_validation_registry_v1_receipts,
            self.validation_context_receipts,
            *self.expected_candidate_checkpoint_authority_receipt_inventories,
        ):
            for inventory_receipt in inventory:
                _digest("four-fold validation inputs V2 inventory", inventory_receipt)
        _digest("four-fold validation inputs V2", self.semantic_receipt_sha256)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    validation_sources_v2: Sequence[MassiveAdaptiveRLValidationSourcesAuthorityV2],
    validation_environment_registries_v2: Sequence[
        MassiveAdaptiveRLValidationEnvironmentRegistryV2
    ],
    base_authority_v1: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(four_fold_fit_authority) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
        or type(runtime_sources_v2) is not MassiveAdaptiveRLRuntimeSourcesV2
        or type(base_authority_v1)
        is not MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "four-fold validation-input V2 requires exact root types"
        )
    sources = tuple(validation_sources_v2)
    registries = tuple(validation_environment_registries_v2)
    if (
        len(sources) != 4
        or len(registries) != 4
        or any(
            type(row) is not MassiveAdaptiveRLValidationSourcesAuthorityV2
            for row in sources
        )
        or any(
            type(row) is not MassiveAdaptiveRLValidationEnvironmentRegistryV2
            for row in registries
        )
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "four-fold validation-input V2 child inventory differs"
        )
    manifest.validate()
    runtime_sources_v2.validate()
    four_fold_fit_authority.validate()
    base_authority_v1.validate()
    for row in (*sources, *registries):
        row.validate()
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    bundle = runtime_sources_v2.source_bundle_v2
    base_source = base_authority_v1.source_receipt_sha256
    base_commit = base_authority_v1.source_transaction_receipt_sha256
    base_time = base_authority_v1.source_transaction_committed_at_ms
    if any(
        value is None
        for value in (
            bundle.source_receipt_sha256,
            bundle.source_transaction_receipt_sha256,
            base_source,
            base_commit,
            base_time,
        )
    ) or not (
        base_authority_v1.development_stage_authorized
        and tuple(row.fold_index for row in sources) == _FOLD_INDICES
        and tuple(row.fold_index for row in registries) == _FOLD_INDICES
        and all(row.development_stage_authorized for row in sources)
        and all(row.development_stage_authorized for row in registries)
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "four-fold validation-input V2 runtime is not authorized"
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
        "source_bundle_v2_receipt_sha256": (
            runtime_sources_v2.source_bundle_v2_receipt_sha256
        ),
        "source_bundle_v2_source_receipt_sha256": cast(
            str, bundle.source_receipt_sha256
        ),
        "source_bundle_v2_commit_receipt_sha256": cast(
            str, bundle.source_transaction_receipt_sha256
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
        "runtime_sources_v2_receipt_sha256": (
            runtime_sources_v2.semantic_receipt_sha256
        ),
        "base_runtime_sources_v1_receipt_sha256": (
            runtime_sources_v2.base_runtime_sources_v1_receipt_sha256
        ),
        "base_runtime_source_graph_v1_witness_receipt_sha256": (
            runtime_sources_v2.base_runtime_source_graph_v1_witness_receipt_sha256
        ),
        "training_source_projection_sha256": (
            runtime_sources_v2.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            runtime_sources_v2.validation_source_projection_sha256
        ),
        "fold_indices": _FOLD_INDICES,
        "validation_sources_v2_receipts": tuple(
            row.semantic_receipt_sha256 for row in sources
        ),
        "validation_sources_v2_source_receipts": tuple(
            cast(str, row.source_receipt_sha256) for row in sources
        ),
        "validation_sources_v2_commit_receipts": tuple(
            cast(str, row.source_transaction_receipt_sha256) for row in sources
        ),
        "validation_sources_v2_committed_at_ms": tuple(
            cast(int, row.source_transaction_committed_at_ms) for row in sources
        ),
        "validation_environment_registry_v2_receipts": tuple(
            row.semantic_receipt_sha256 for row in registries
        ),
        "validation_registry_v2_source_receipts": tuple(
            cast(str, row.source_receipt_sha256) for row in registries
        ),
        "validation_registry_v2_commit_receipts": tuple(
            cast(str, row.source_transaction_receipt_sha256) for row in registries
        ),
        "validation_registry_v2_committed_at_ms": tuple(
            cast(int, row.source_transaction_committed_at_ms) for row in registries
        ),
        "base_validation_sources_v1_receipts": tuple(
            row.base_validation_sources_v1_receipt_sha256 for row in sources
        ),
        "base_validation_registry_v1_receipts": tuple(
            row.base_validation_registry_v1_receipt_sha256 for row in registries
        ),
        "validation_context_receipts": tuple(
            row.validation_context_receipt_sha256 for row in registries
        ),
        "validation_decision_session_date_inventories": tuple(
            row.validation_decision_session_dates for row in sources
        ),
        "expected_candidate_checkpoint_authority_receipt_inventories": (
            base_authority_v1.expected_candidate_checkpoint_authority_receipt_inventories
        ),
        "base_four_fold_validation_inputs_v1_receipt_sha256": (
            base_authority_v1.semantic_receipt_sha256
        ),
        "base_four_fold_validation_inputs_v1_source_receipt_sha256": cast(
            str, base_source
        ),
        "base_four_fold_validation_inputs_v1_commit_receipt_sha256": cast(
            str, base_commit
        ),
        "base_four_fold_validation_inputs_v1_committed_at_ms": cast(int, base_time),
        "source_data_qualified": bool(
            runtime_sources_v2.source_data_qualified
            and four_fold_fit_authority.source_data_qualified
            and all(row.source_data_qualified for row in sources)
            and all(row.source_data_qualified for row in registries)
            and base_authority_v1.source_data_qualified
        ),
    }
    provisional = MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_inputs_replayed=True,
        development_validation_execution_authorized=bool(body["source_data_qualified"]),
        _runtime_sources_v2=runtime_sources_v2,
        _four_fold_fit_authority=four_fold_fit_authority,
        _validation_sources_v2=sources,
        _validation_registries_v2=registries,
        _base_authority_v1=base_authority_v1,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse_body(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "four-fold validation-input V2 is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "fold_indices",
        "validation_sources_v2_receipts",
        "validation_sources_v2_source_receipts",
        "validation_sources_v2_commit_receipts",
        "validation_sources_v2_committed_at_ms",
        "validation_environment_registry_v2_receipts",
        "validation_registry_v2_source_receipts",
        "validation_registry_v2_commit_receipts",
        "validation_registry_v2_committed_at_ms",
        "base_validation_sources_v1_receipts",
        "base_validation_registry_v1_receipts",
        "validation_context_receipts",
    ):
        body[name] = tuple(cast(Sequence[object], body[name]))
    for name in (
        "validation_decision_session_date_inventories",
        "expected_candidate_checkpoint_authority_receipt_inventories",
    ):
        body[name] = tuple(
            tuple(cast(Sequence[str], row))
            for row in cast(Sequence[object], body[name])
        )
    return body


def parse_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2:
    body = _parse_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    verified_at_ms: int,
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2:
    return parse_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=(
                four_fold_validation_inputs_authority_relative_path_v2(
                    manifest=manifest
                )
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
    *,
    authority: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    validation_sources_v2: Sequence[MassiveAdaptiveRLValidationSourcesAuthorityV2],
    validation_environment_registries_v2: Sequence[
        MassiveAdaptiveRLValidationEnvironmentRegistryV2
    ],
    base_authority_v1: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2:
    authority.validate()
    expected = build_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
        manifest=manifest,
        four_fold_fit_authority=four_fold_fit_authority,
        runtime_sources_v2=runtime_sources_v2,
        validation_sources_v2=validation_sources_v2,
        validation_environment_registries_v2=(validation_environment_registries_v2),
        base_authority_v1=base_authority_v1,
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != four_fold_validation_inputs_authority_relative_path_v2(manifest=manifest)
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "four-fold validation-input authority V2 does not replay"
        )
    result = replace(
        authority,
        runtime_inputs_replayed=True,
        development_validation_execution_authorized=authority.source_data_qualified,
        _runtime_sources_v2=runtime_sources_v2,
        _four_fold_fit_authority=four_fold_fit_authority,
        _validation_sources_v2=tuple(validation_sources_v2),
        _validation_registries_v2=tuple(validation_environment_registries_v2),
        _base_authority_v1=base_authority_v1,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    authority: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2:
    manifest.validate()
    authority.validate()
    relative = four_fold_validation_inputs_authority_relative_path_v2(manifest=manifest)
    if _transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "four-fold validation-input authority V2 already exists"
        )
    if any(
        checker(root=root, manifest=manifest, fold_index=fold_index)
        for checker in (
            massive_adaptive_rl_validation_downstream_evidence_exists_v1,
            massive_adaptive_rl_validation_downstream_evidence_exists_v2,
        )
        for fold_index in _FOLD_INDICES
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "missing V2 validation barrier cannot be created after outcome evidence"
        )
    prerequisite_time = max(
        *authority.validation_sources_v2_committed_at_ms,
        *authority.validation_registry_v2_committed_at_ms,
        authority.base_four_fold_validation_inputs_v1_committed_at_ms,
    )
    if (
        not authority.runtime_inputs_replayed
        or isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms <= prerequisite_time
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "four-fold validation-input V2 must follow every input"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=(
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_DATASET
        ),
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-FOUR-FOLD-VALIDATION-INPUTS-V2-{authority.experiment_id}"
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


def validate_massive_adaptive_rl_validation_outcome_barrier_v2(
    *,
    authority: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    validation_environment_registry: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    fold_index: int,
    outcome_committed_at_ms: int,
    checkpoint_authority_receipt_sha256: str | None = None,
) -> None:
    """Require the V2 all-fold barrier and exact V2 registry before an outcome."""

    if (
        type(authority) is not MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
        or type(validation_environment_registry)
        is not MassiveAdaptiveRLValidationEnvironmentRegistryV2
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "inner-validation outcome requires exact V2 input authorities"
        )
    authority.validate()
    validation_environment_registry.validate()
    barrier_time = authority.source_transaction_committed_at_ms
    if (
        not authority.development_stage_authorized
        or not validation_environment_registry.development_stage_authorized
        or barrier_time is None
        or barrier_time >= outcome_committed_at_ms
        or fold_index not in authority.fold_indices
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "V2 validation-input barrier is not precommitted and authorized"
        )
    index = authority.fold_indices.index(fold_index)
    if (
        validation_environment_registry.fold_index != fold_index
        or validation_environment_registry.semantic_receipt_sha256
        != authority.validation_environment_registry_v2_receipts[index]
        or validation_environment_registry.source_receipt_sha256
        != authority.validation_registry_v2_source_receipts[index]
        or validation_environment_registry.source_transaction_receipt_sha256
        != authority.validation_registry_v2_commit_receipts[index]
        or validation_environment_registry.source_transaction_committed_at_ms
        != authority.validation_registry_v2_committed_at_ms[index]
        or validation_environment_registry.validation_context_receipt_sha256
        != authority.validation_context_receipts[index]
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "validation registry V2 is not a child of the V2 barrier"
        )
    if checkpoint_authority_receipt_sha256 is not None:
        _digest("validation checkpoint", checkpoint_authority_receipt_sha256)
        if (
            checkpoint_authority_receipt_sha256
            not in (
                authority.expected_candidate_checkpoint_authority_receipt_inventories[
                    index
                ]
            )
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "validation checkpoint is outside the preregistered population"
            )


@contextmanager
def _four_fold_validation_inputs_v2_lease(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> Iterator[None]:
    manifest.validate()
    directory = (
        Path(root) / "massive-adaptive" / "rl-four-fold-validation-input-leases-v2"
    )
    try:
        Path(root).mkdir(parents=True, exist_ok=True)
        if Path(root).is_symlink():
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation-input V2 root is a symlink"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation-input V2 lease directory is a symlink"
            )
        descriptor = os.open(
            directory / f"v4-{manifest.semantic_receipt_sha256}.lock",
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
    except OSError as error:
        raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
            "four-fold validation-input V2 lease is unavailable"
        ) from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation-input V2 lease identity differs"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MassiveAdaptiveRLFourFoldValidationInputsV2LeaseUnavailable(
                "four-fold validation-input V2 lease is already held"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2:
    """Commit or strictly replay all V2 validation inputs as one generation."""

    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    relative = four_fold_validation_inputs_authority_relative_path_v2(manifest=manifest)
    with _four_fold_validation_inputs_v2_lease(root=root, manifest=manifest):
        aggregate_exists = _transaction_exists(root=root, relative=relative)
        sources = tuple(
            prepare_or_resume_massive_adaptive_rl_validation_sources_v2(
                root=root,
                manifest=manifest,
                four_fold_fit_authority=four_fold_fit_authority,
                runtime_sources_v2=runtime_sources_v2,
                fold_index=fold_index,
                committed_at_ms=committed_at_ms + 2 * fold_index,
                allow_materialize=allow_materialize,
            )
            for fold_index in _FOLD_INDICES
        )
        registries = tuple(
            prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v2(
                root=root,
                manifest=manifest,
                runtime_sources_v2=runtime_sources_v2,
                validation_sources_v2=sources[fold_index],
                committed_at_ms=committed_at_ms + 8 + 2 * fold_index,
                allow_materialize=allow_materialize,
            )
            for fold_index in _FOLD_INDICES
        )
        base = run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v1(
            root=root,
            manifest=manifest,
            four_fold_fit_authority=four_fold_fit_authority,
            runtime_sources=runtime_sources_v2.base_runtime_sources_v1,
            committed_at_ms=committed_at_ms + 16,
            allow_materialize=allow_materialize,
        )
        if aggregate_exists:
            return authorize_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
                authority=load_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
                    root=root,
                    manifest=manifest,
                    verified_at_ms=committed_at_ms,
                ),
                manifest=manifest,
                four_fold_fit_authority=four_fold_fit_authority,
                runtime_sources_v2=runtime_sources_v2,
                validation_sources_v2=sources,
                validation_environment_registries_v2=registries,
                base_authority_v1=base,
            )
        if not allow_materialize:
            raise MassiveAdaptiveRLFourFoldValidationInputsV2Error(
                "four-fold validation-input authority V2 is absent"
            )
        authority = build_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
            manifest=manifest,
            four_fold_fit_authority=four_fold_fit_authority,
            runtime_sources_v2=runtime_sources_v2,
            validation_sources_v2=sources,
            validation_environment_registries_v2=registries,
            base_authority_v1=base,
        )
        prerequisite = max(
            *authority.validation_registry_v2_committed_at_ms,
            authority.base_four_fold_validation_inputs_v1_committed_at_ms,
        )
        return materialize_massive_adaptive_rl_four_fold_validation_inputs_authority_v2(
            root=root,
            manifest=manifest,
            authority=authority,
            committed_at_ms=max(committed_at_ms + 25, prerequisite + 1),
        )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_DATASET",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V2_SPEC_SHA256",
    "MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2",
    "MassiveAdaptiveRLFourFoldValidationInputsV2Error",
    "MassiveAdaptiveRLFourFoldValidationInputsV2LeaseUnavailable",
    "authorize_massive_adaptive_rl_four_fold_validation_inputs_authority_v2",
    "build_massive_adaptive_rl_four_fold_validation_inputs_authority_v2",
    "four_fold_validation_inputs_authority_relative_path_v2",
    "load_massive_adaptive_rl_four_fold_validation_inputs_authority_v2",
    "materialize_massive_adaptive_rl_four_fold_validation_inputs_authority_v2",
    "parse_massive_adaptive_rl_four_fold_validation_inputs_authority_v2",
    "run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v2",
    "validate_massive_adaptive_rl_validation_outcome_barrier_v2",
]
