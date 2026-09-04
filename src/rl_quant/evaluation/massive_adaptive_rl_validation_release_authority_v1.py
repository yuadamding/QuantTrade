"""Manifest-V5 initial and authenticated delayed validation releases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v2 import (
    MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    MassiveAdaptiveRLValidationSourcesAuthorityV2,
    prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v2,
    prepare_or_resume_massive_adaptive_rl_validation_sources_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    MassiveAdaptiveRLExperimentLockV1Error,
    MassiveAdaptiveRLExperimentLockV1Unavailable,
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_delayed_validation_release_capability_v1,
    issue_massive_adaptive_rl_manifest_v5_initial_validation_release_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)

if TYPE_CHECKING:
    from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
    )


MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-validation-release-authority-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA,
            "encoding": "canonical-json-exact-v5-validation-release",
            "generic_reload": "nonauthorizing",
        }
    )
)
_INITIAL_RELEASE_KIND = "initial-folds-0-1"
_POST_OUTER_ZERO_RELEASE_KIND = "post-outer-0-fold-2"
_POST_OUTER_ONE_RELEASE_KIND = "post-outer-1-fold-3"


class MassiveAdaptiveRLValidationReleaseAuthorityV1Error(ValueError):
    """The V5 validation release or its exact chronology differs."""


class MassiveAdaptiveRLValidationReleaseAuthorityV1LeaseUnavailable(RuntimeError):
    """Another process owns the validation-release publication boundary."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: str | None) -> str:
    if value is None:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(f"{name} is absent")
    return _digest(name, value)


def _required_timestamp(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            f"{name} is absent or invalid"
        )
    return value


def _wall_clock_ms() -> int:
    return _required_timestamp("validation release clock", time.time_ns() // 1_000_000)


def validation_release_authority_relative_path_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    released_fold_index: int | None = None,
) -> str:
    manifest.validate()
    if released_fold_index is None:
        filename = "initial.json"
    elif released_fold_index == 2:
        filename = "post-outer-0-fold-2.json"
    elif released_fold_index == 3:
        filename = "post-outer-1-fold-3.json"
    else:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release fold differs"
        )
    return f"adaptive-rl/{manifest.experiment_id}/validation-release-v1/{filename}"


def _transaction_state(*, root: str | Path, relative: str) -> tuple[bool, bool]:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    return all(present), any(present) and not all(present)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationReleaseAuthorityV1:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    base_manifest_v4_receipt_sha256: str
    manifest_v5_registration_authority_receipt_sha256: str
    manifest_v5_registration_source_receipt_sha256: str
    manifest_v5_registration_commit_receipt_sha256: str
    manifest_v5_registration_committed_at_ms: int
    execution_implementation_registration_authority_receipt_sha256: str
    execution_implementation_registration_source_receipt_sha256: str
    execution_implementation_registration_commit_receipt_sha256: str
    execution_implementation_registration_committed_at_ms: int
    scientific_execution_fingerprint_sha256: str
    training_state_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    release_kind: str
    released_validation_fold_indices: tuple[int, ...]
    withheld_validation_fold_indices: tuple[int, ...]
    predecessor_outer_fold_index: int | None
    predecessor_outer_fold_seal_receipt_sha256: str | None
    initial_validation_inputs_authority_receipt_sha256: str
    initial_validation_inputs_source_receipt_sha256: str
    initial_validation_inputs_commit_receipt_sha256: str
    initial_validation_inputs_committed_at_ms: int
    prequential_validation_plan_receipt_sha256: str
    runtime_sources_v2_receipt_sha256: str
    source_bundle_v2_receipt_sha256: str
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_witness_receipt_sha256: str
    replay_dependency_index_v2_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    validation_sources_v2_receipts: tuple[str, ...]
    validation_sources_v2_source_receipts: tuple[str, ...]
    validation_sources_v2_commit_receipts: tuple[str, ...]
    validation_sources_v2_committed_at_ms: tuple[int, ...]
    validation_environment_registry_v2_receipts: tuple[str, ...]
    validation_registry_v2_source_receipts: tuple[str, ...]
    validation_registry_v2_commit_receipts: tuple[str, ...]
    validation_registry_v2_committed_at_ms: tuple[int, ...]
    validation_context_receipts: tuple[str, ...]
    validation_decision_session_date_inventories: tuple[tuple[str, ...], ...]
    expected_candidate_checkpoint_authority_receipt_inventories: tuple[
        tuple[str, ...], ...
    ]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    predecessor_outer_fold_seal_source_receipt_sha256: str | None = None
    predecessor_outer_fold_seal_commit_receipt_sha256: str | None = None
    predecessor_outer_fold_seal_committed_at_ms: int | None = None
    runtime_lineage_replayed: bool = False
    development_validation_release_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV5 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_manifest_registration: (
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_initial_inputs: (
        MassiveAdaptiveRLInitialValidationInputsAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_validation_sources: tuple[
        MassiveAdaptiveRLValidationSourcesAuthorityV2, ...
    ] = field(default=(), compare=False, repr=False)
    _runtime_validation_registries: tuple[
        MassiveAdaptiveRLValidationEnvironmentRegistryV2, ...
    ] = field(default=(), compare=False, repr=False)
    _runtime_predecessor_outer_seal: (
        MassiveAdaptiveRLOuterFoldSealAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name
            not in {
                "semantic_receipt_sha256",
                "runtime_lineage_replayed",
                "development_validation_release_authorized",
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
            and self.runtime_lineage_replayed
            and self.development_validation_release_authorized
            and self.source_data_qualified
        )

    @property
    def initial_validation_inputs(
        self,
    ) -> MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
        """Return the exact replayed initial input anchor for every release."""

        self.validate()
        if (
            not self.development_stage_authorized
            or self._runtime_initial_inputs is None
        ):
            raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                "validation release inputs have not been exactly replayed"
            )
        return self._runtime_initial_inputs

    def validation_sources(
        self, fold_index: int
    ) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
        """Return one causally released V2 validation-source authority."""

        if fold_index not in self.released_validation_fold_indices:
            raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                "validation fold is not present in this release"
            )
        self.validate()
        if not self.development_stage_authorized:
            raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                "validation release has not been exactly replayed"
            )
        if self.release_kind == _INITIAL_RELEASE_KIND:
            return self.initial_validation_inputs.validation_sources(fold_index)
        return self._runtime_validation_sources[0]

    def validation_registry(
        self, fold_index: int
    ) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
        """Return one causally released V2 validation-environment registry."""

        if fold_index not in self.released_validation_fold_indices:
            raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                "validation fold is not present in this release"
            )
        self.validate()
        if not self.development_stage_authorized:
            raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                "validation release has not been exactly replayed"
            )
        if self.release_kind == _INITIAL_RELEASE_KIND:
            return self.initial_validation_inputs.validation_registry(fold_index)
        return self._runtime_validation_registries[0]

    @property
    def four_fold_fit_authority(self) -> MassiveAdaptiveRLFourFoldFitAuthorityV1:
        """Return the completed fit aggregate that supplies candidate checkpoints."""

        return self.initial_validation_inputs.four_fold_fit_authority

    def validate(self) -> None:
        base_runtime_roots = (
            self._runtime_manifest,
            self._runtime_manifest_registration,
            self._runtime_execution_registration,
            self._runtime_initial_inputs,
        )
        release_specs = {
            _INITIAL_RELEASE_KIND: ((0, 1), (2, 3), None, (1, 2)),
            _POST_OUTER_ZERO_RELEASE_KIND: ((2,), (3,), 0, (3,)),
            _POST_OUTER_ONE_RELEASE_KIND: ((3,), (), 1, (4,)),
        }
        release_spec = release_specs.get(self.release_kind)
        expected_count = 0 if release_spec is None else len(release_spec[0])
        delayed = bool(release_spec is not None and release_spec[2] is not None)
        rows_runtime = bool(
            not delayed
            and not self._runtime_validation_sources
            and not self._runtime_validation_registries
            or delayed
            and len(self._runtime_validation_sources) == expected_count
            and len(self._runtime_validation_registries) == expected_count
        )
        predecessor_runtime = self._runtime_predecessor_outer_seal is not None
        runtime_present = bool(
            all(value is not None for value in base_runtime_roots)
            and rows_runtime
            and predecessor_runtime == delayed
        )
        any_runtime = bool(
            any(value is not None for value in base_runtime_roots)
            or self._runtime_validation_sources
            or self._runtime_validation_registries
            or predecessor_runtime
        )
        if any_runtime != runtime_present:
            raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                "validation release runtime lineage is partial"
            )
        lengths = tuple(
            len(value)
            for value in (
                self.validation_sources_v2_receipts,
                self.validation_sources_v2_source_receipts,
                self.validation_sources_v2_commit_receipts,
                self.validation_sources_v2_committed_at_ms,
                self.validation_environment_registry_v2_receipts,
                self.validation_registry_v2_source_receipts,
                self.validation_registry_v2_commit_receipts,
                self.validation_registry_v2_committed_at_ms,
                self.validation_context_receipts,
                self.validation_decision_session_date_inventories,
                self.expected_candidate_checkpoint_authority_receipt_inventories,
            )
        )
        expected_runtime_authorized = bool(
            runtime_present and self.source_data_qualified
        )
        predecessor_fields = (
            self.predecessor_outer_fold_seal_receipt_sha256,
            self.predecessor_outer_fold_seal_source_receipt_sha256,
            self.predecessor_outer_fold_seal_commit_receipt_sha256,
            self.predecessor_outer_fold_seal_committed_at_ms,
        )
        initial_chronology = bool(
            release_spec is not None
            and not delayed
            and self.execution_implementation_registration_committed_at_ms
            < min(self.validation_sources_v2_committed_at_ms, default=-1)
            and max(self.validation_sources_v2_committed_at_ms, default=-1)
            < min(self.validation_registry_v2_committed_at_ms, default=-1)
            and max(self.validation_registry_v2_committed_at_ms, default=-1)
            < self.initial_validation_inputs_committed_at_ms
        )
        predecessor_time = self.predecessor_outer_fold_seal_committed_at_ms
        delayed_chronology = bool(
            release_spec is not None
            and delayed
            and isinstance(predecessor_time, int)
            and not isinstance(predecessor_time, bool)
            and self.initial_validation_inputs_committed_at_ms < predecessor_time
            and predecessor_time
            < min(self.validation_sources_v2_committed_at_ms, default=-1)
            and max(self.validation_sources_v2_committed_at_ms, default=-1)
            < min(self.validation_registry_v2_committed_at_ms, default=-1)
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or release_spec is None
            or self.released_validation_fold_indices != release_spec[0]
            or self.withheld_validation_fold_indices != release_spec[1]
            or self.predecessor_outer_fold_index != release_spec[2]
            or (delayed and any(value is None for value in predecessor_fields))
            or (not delayed and any(value is not None for value in predecessor_fields))
            or set(lengths) != {expected_count}
            or tuple(
                len(row)
                for row in self.expected_candidate_checkpoint_authority_receipt_inventories
            )
            != release_spec[3]
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    self.manifest_v5_registration_committed_at_ms,
                    self.execution_implementation_registration_committed_at_ms,
                    self.initial_validation_inputs_committed_at_ms,
                    *self.validation_sources_v2_committed_at_ms,
                    *self.validation_registry_v2_committed_at_ms,
                )
            )
            or self.manifest_v5_registration_committed_at_ms
            >= self.execution_implementation_registration_committed_at_ms
            or not (initial_chronology or delayed_chronology)
            or any(
                len(set(inventory)) != len(inventory)
                for inventory in (
                    self.validation_sources_v2_receipts,
                    self.validation_sources_v2_source_receipts,
                    self.validation_sources_v2_commit_receipts,
                    self.validation_environment_registry_v2_receipts,
                    self.validation_registry_v2_source_receipts,
                    self.validation_registry_v2_commit_receipts,
                    self.validation_context_receipts,
                )
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_lineage_replayed != runtime_present
            or self.development_validation_release_authorized
            != expected_runtime_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                "initial validation release differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256") and value is not None:
                _digest(name, value)
        for inventory in (
            self.validation_sources_v2_receipts,
            self.validation_sources_v2_source_receipts,
            self.validation_sources_v2_commit_receipts,
            self.validation_environment_registry_v2_receipts,
            self.validation_registry_v2_source_receipts,
            self.validation_registry_v2_commit_receipts,
            self.validation_context_receipts,
            *self.expected_candidate_checkpoint_authority_receipt_inventories,
        ):
            for value in inventory:
                _digest("validation release inventory", value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.semantic_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= max(
                    self.initial_validation_inputs_committed_at_ms,
                    max(self.validation_registry_v2_committed_at_ms),
                    self.predecessor_outer_fold_seal_committed_at_ms or 0,
                )
            ):
                raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                    "validation release source transaction differs"
                )
        if runtime_present:
            manifest = cast(
                MassiveAdaptiveRLExperimentManifestV5, self._runtime_manifest
            )
            registration = cast(
                MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
                self._runtime_manifest_registration,
            )
            implementation = cast(
                MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
                self._runtime_execution_registration,
            )
            initial = cast(
                MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
                self._runtime_initial_inputs,
            )
            for authority in (manifest, registration, implementation, initial):
                authority.validate()
            if delayed:
                for source, registry, fold_index in zip(
                    self._runtime_validation_sources,
                    self._runtime_validation_registries,
                    self.released_validation_fold_indices,
                    strict=True,
                ):
                    source.validate()
                    registry.validate()
                    position = self.released_validation_fold_indices.index(fold_index)
                    if (
                        source.fold_index != fold_index
                        or registry.fold_index != fold_index
                        or source.semantic_receipt_sha256
                        != self.validation_sources_v2_receipts[position]
                        or registry.semantic_receipt_sha256
                        != self.validation_environment_registry_v2_receipts[position]
                        or not source.development_validation_inputs_authorized
                        or not registry.development_validation_environments_authorized
                    ):
                        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                            "validation release runtime inputs differ"
                        )
            if (
                manifest.semantic_receipt_sha256 != self.manifest_v5_receipt_sha256
                or manifest.scientific_protocol_projection_sha256
                != self.scientific_protocol_projection_sha256
                or manifest.base_manifest.semantic_receipt_sha256
                != self.base_manifest_v4_receipt_sha256
                or registration.semantic_receipt_sha256
                != self.manifest_v5_registration_authority_receipt_sha256
                or registration.source_receipt_sha256
                != self.manifest_v5_registration_source_receipt_sha256
                or registration.source_transaction_receipt_sha256
                != self.manifest_v5_registration_commit_receipt_sha256
                or registration.source_transaction_committed_at_ms
                != self.manifest_v5_registration_committed_at_ms
                or implementation.semantic_receipt_sha256
                != self.execution_implementation_registration_authority_receipt_sha256
                or implementation.source_receipt_sha256
                != self.execution_implementation_registration_source_receipt_sha256
                or implementation.source_transaction_receipt_sha256
                != self.execution_implementation_registration_commit_receipt_sha256
                or implementation.source_transaction_committed_at_ms
                != self.execution_implementation_registration_committed_at_ms
                or implementation.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or implementation.training_state_receipt_sha256
                != self.training_state_receipt_sha256
                or implementation.four_fold_fit_authority_receipt_sha256
                != self.four_fold_fit_authority_receipt_sha256
                or not implementation.development_execution_registered
                or initial.semantic_receipt_sha256
                != self.initial_validation_inputs_authority_receipt_sha256
                or initial.source_receipt_sha256
                != self.initial_validation_inputs_source_receipt_sha256
                or initial.source_transaction_receipt_sha256
                != self.initial_validation_inputs_commit_receipt_sha256
                or initial.source_transaction_committed_at_ms
                != self.initial_validation_inputs_committed_at_ms
                or initial.four_fold_fit_authority_receipt_sha256
                != self.four_fold_fit_authority_receipt_sha256
                or not initial.development_stage_authorized
            ):
                raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                    "validation release runtime lineage differs"
                )
            if delayed:
                from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
                    MassiveAdaptiveRLOuterFoldSealAuthorityV1,
                )

                seal = self._runtime_predecessor_outer_seal
                if type(seal) is not MassiveAdaptiveRLOuterFoldSealAuthorityV1:
                    raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                        "validation release predecessor seal type differs"
                    )
                seal.validate()
                if (
                    seal.fold_index != self.predecessor_outer_fold_index
                    or seal.semantic_receipt_sha256
                    != self.predecessor_outer_fold_seal_receipt_sha256
                    or seal.source_receipt_sha256
                    != self.predecessor_outer_fold_seal_source_receipt_sha256
                    or seal.source_transaction_receipt_sha256
                    != self.predecessor_outer_fold_seal_commit_receipt_sha256
                    or seal.source_transaction_committed_at_ms
                    != self.predecessor_outer_fold_seal_committed_at_ms
                    or seal.manifest_v5_receipt_sha256
                    != self.manifest_v5_receipt_sha256
                    or seal.execution_implementation_registration_receipt_sha256
                    != self.execution_implementation_registration_authority_receipt_sha256
                    or seal.scientific_execution_fingerprint_sha256
                    != self.scientific_execution_fingerprint_sha256
                    or not seal.development_outer_fold_sealed
                    or not seal.validation_release_authorized
                ):
                    raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                        "validation release predecessor seal differs"
                    )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_initial_validation_release_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV5:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release requires exact Manifest V5"
        )
    if (
        type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(execution_registration)
        is not MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
        or type(initial_inputs)
        is not MassiveAdaptiveRLInitialValidationInputsAuthorityV1
    ):
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release requires exact authority generations"
        )
    for authority in (
        manifest,
        manifest_registration,
        execution_registration,
        initial_inputs,
    ):
        authority.validate()
    manifest_registration_source = _required_digest(
        "Manifest V5 registration source", manifest_registration.source_receipt_sha256
    )
    manifest_registration_commit = _required_digest(
        "Manifest V5 registration commit",
        manifest_registration.source_transaction_receipt_sha256,
    )
    manifest_registration_time = _required_timestamp(
        "Manifest V5 registration time",
        manifest_registration.source_transaction_committed_at_ms,
    )
    execution_source = _required_digest(
        "execution implementation source", execution_registration.source_receipt_sha256
    )
    execution_commit = _required_digest(
        "execution implementation commit",
        execution_registration.source_transaction_receipt_sha256,
    )
    execution_time = _required_timestamp(
        "execution implementation time",
        execution_registration.source_transaction_committed_at_ms,
    )
    initial_source = _required_digest(
        "initial validation-input source", initial_inputs.source_receipt_sha256
    )
    initial_commit = _required_digest(
        "initial validation-input commit",
        initial_inputs.source_transaction_receipt_sha256,
    )
    initial_time = _required_timestamp(
        "initial validation-input time",
        initial_inputs.source_transaction_committed_at_ms,
    )
    if (
        not manifest_registration.development_protocol_registered
        or not execution_registration.development_execution_registered
        or not initial_inputs.development_stage_authorized
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or execution_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or execution_registration.manifest_v5_registration_authority_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
        or initial_inputs.manifest_v4_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or execution_registration.four_fold_fit_authority_receipt_sha256
        != initial_inputs.four_fold_fit_authority_receipt_sha256
        or not (
            manifest_registration_time
            < execution_time
            < min(initial_inputs.validation_sources_v2_committed_at_ms)
        )
        or initial_time <= max(initial_inputs.validation_registry_v2_committed_at_ms)
    ):
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release roots or chronology differ"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": (
            manifest.scientific_protocol_projection_sha256
        ),
        "base_manifest_v4_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "manifest_v5_registration_authority_receipt_sha256": (
            manifest_registration.semantic_receipt_sha256
        ),
        "manifest_v5_registration_source_receipt_sha256": (
            manifest_registration_source
        ),
        "manifest_v5_registration_commit_receipt_sha256": (
            manifest_registration_commit
        ),
        "manifest_v5_registration_committed_at_ms": manifest_registration_time,
        "execution_implementation_registration_authority_receipt_sha256": (
            execution_registration.semantic_receipt_sha256
        ),
        "execution_implementation_registration_source_receipt_sha256": (
            execution_source
        ),
        "execution_implementation_registration_commit_receipt_sha256": (
            execution_commit
        ),
        "execution_implementation_registration_committed_at_ms": execution_time,
        "scientific_execution_fingerprint_sha256": (
            execution_registration.scientific_execution_fingerprint_sha256
        ),
        "training_state_receipt_sha256": (
            execution_registration.training_state_receipt_sha256
        ),
        "four_fold_fit_authority_receipt_sha256": (
            initial_inputs.four_fold_fit_authority_receipt_sha256
        ),
        "release_kind": _INITIAL_RELEASE_KIND,
        "released_validation_fold_indices": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1
        ),
        "withheld_validation_fold_indices": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1
        ),
        "predecessor_outer_fold_index": None,
        "predecessor_outer_fold_seal_receipt_sha256": None,
        "predecessor_outer_fold_seal_source_receipt_sha256": None,
        "predecessor_outer_fold_seal_commit_receipt_sha256": None,
        "predecessor_outer_fold_seal_committed_at_ms": None,
        "initial_validation_inputs_authority_receipt_sha256": (
            initial_inputs.semantic_receipt_sha256
        ),
        "initial_validation_inputs_source_receipt_sha256": initial_source,
        "initial_validation_inputs_commit_receipt_sha256": initial_commit,
        "initial_validation_inputs_committed_at_ms": initial_time,
        "prequential_validation_plan_receipt_sha256": (
            initial_inputs.prequential_validation_plan_receipt_sha256
        ),
        "runtime_sources_v2_receipt_sha256": (
            initial_inputs.runtime_sources_v2_receipt_sha256
        ),
        "source_bundle_v2_receipt_sha256": (
            initial_inputs.source_bundle_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_receipt_sha256": (
            initial_inputs.runtime_source_graph_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_witness_receipt_sha256": (
            initial_inputs.runtime_source_graph_v2_witness_receipt_sha256
        ),
        "replay_dependency_index_v2_receipt_sha256": (
            initial_inputs.replay_dependency_index_v2_receipt_sha256
        ),
        "training_source_projection_sha256": (
            initial_inputs.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            initial_inputs.validation_source_projection_sha256
        ),
        "validation_sources_v2_receipts": initial_inputs.validation_sources_v2_receipts,
        "validation_sources_v2_source_receipts": (
            initial_inputs.validation_sources_v2_source_receipts
        ),
        "validation_sources_v2_commit_receipts": (
            initial_inputs.validation_sources_v2_commit_receipts
        ),
        "validation_sources_v2_committed_at_ms": (
            initial_inputs.validation_sources_v2_committed_at_ms
        ),
        "validation_environment_registry_v2_receipts": (
            initial_inputs.validation_environment_registry_v2_receipts
        ),
        "validation_registry_v2_source_receipts": (
            initial_inputs.validation_registry_v2_source_receipts
        ),
        "validation_registry_v2_commit_receipts": (
            initial_inputs.validation_registry_v2_commit_receipts
        ),
        "validation_registry_v2_committed_at_ms": (
            initial_inputs.validation_registry_v2_committed_at_ms
        ),
        "validation_context_receipts": initial_inputs.validation_context_receipts,
        "validation_decision_session_date_inventories": (
            initial_inputs.validation_decision_session_date_inventories
        ),
        "expected_candidate_checkpoint_authority_receipt_inventories": (
            initial_inputs.expected_candidate_checkpoint_authority_receipt_inventories
        ),
        "source_data_qualified": bool(
            execution_registration.source_data_qualified
            and initial_inputs.source_data_qualified
        ),
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA,
    }
    provisional = MassiveAdaptiveRLValidationReleaseAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_lineage_replayed=True,
        development_validation_release_authorized=bool(body["source_data_qualified"]),
        _runtime_manifest=manifest,
        _runtime_manifest_registration=manifest_registration,
        _runtime_execution_registration=execution_registration,
        _runtime_initial_inputs=initial_inputs,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def build_massive_adaptive_rl_delayed_validation_release_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    predecessor_outer_fold_seal: MassiveAdaptiveRLOuterFoldSealAuthorityV1,
    validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_registry: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
    """Bind one delayed tape only after its exact prior outer fold is sealed."""

    from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
        MassiveAdaptiveRLOuterFoldSealAuthorityV1 as OuterSeal,
    )

    authorities = (
        manifest,
        manifest_registration,
        execution_registration,
        initial_inputs,
        predecessor_outer_fold_seal,
        validation_sources,
        validation_registry,
    )
    for authority in authorities:
        authority.validate()
    predecessor = predecessor_outer_fold_seal.fold_index
    released_fold = predecessor + 2
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(execution_registration)
        is not MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
        or type(initial_inputs)
        is not MassiveAdaptiveRLInitialValidationInputsAuthorityV1
        or type(predecessor_outer_fold_seal) is not OuterSeal
        or type(validation_sources) is not MassiveAdaptiveRLValidationSourcesAuthorityV2
        or type(validation_registry)
        is not MassiveAdaptiveRLValidationEnvironmentRegistryV2
        or predecessor not in (0, 1)
        or released_fold not in (2, 3)
        or validation_sources.fold_index != released_fold
        or validation_registry.fold_index != released_fold
        or not predecessor_outer_fold_seal.validation_release_authorized
        or predecessor_outer_fold_seal.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or validation_sources.manifest_v4_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or validation_registry.manifest_v4_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or validation_registry.validation_sources_v2_receipt_sha256
        != validation_sources.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "delayed validation release roots differ"
        )
    manifest_registration_source = _required_digest(
        "Manifest V5 registration source", manifest_registration.source_receipt_sha256
    )
    manifest_registration_commit = _required_digest(
        "Manifest V5 registration commit",
        manifest_registration.source_transaction_receipt_sha256,
    )
    execution_source = _required_digest(
        "execution implementation source", execution_registration.source_receipt_sha256
    )
    execution_commit = _required_digest(
        "execution implementation commit",
        execution_registration.source_transaction_receipt_sha256,
    )
    initial_source = _required_digest(
        "initial validation-input source", initial_inputs.source_receipt_sha256
    )
    initial_commit = _required_digest(
        "initial validation-input commit",
        initial_inputs.source_transaction_receipt_sha256,
    )
    predecessor_source = _required_digest(
        "predecessor outer-fold seal source",
        predecessor_outer_fold_seal.source_receipt_sha256,
    )
    predecessor_commit = _required_digest(
        "predecessor outer-fold seal commit",
        predecessor_outer_fold_seal.source_transaction_receipt_sha256,
    )
    predecessor_time = _required_timestamp(
        "predecessor outer-fold seal time",
        predecessor_outer_fold_seal.source_transaction_committed_at_ms,
    )
    validation_source_receipt = _required_digest(
        "delayed validation source", validation_sources.source_receipt_sha256
    )
    validation_source_commit = _required_digest(
        "delayed validation source commit",
        validation_sources.source_transaction_receipt_sha256,
    )
    validation_source_time = _required_timestamp(
        "delayed validation source time",
        validation_sources.source_transaction_committed_at_ms,
    )
    registry_source = _required_digest(
        "delayed validation registry source", validation_registry.source_receipt_sha256
    )
    registry_commit = _required_digest(
        "delayed validation registry commit",
        validation_registry.source_transaction_receipt_sha256,
    )
    registry_time = _required_timestamp(
        "delayed validation registry time",
        validation_registry.source_transaction_committed_at_ms,
    )
    initial_time = _required_timestamp(
        "initial validation-input time",
        initial_inputs.source_transaction_committed_at_ms,
    )
    if not initial_time < predecessor_time < validation_source_time < registry_time:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "delayed validation release chronology differs"
        )
    fold_fit = initial_inputs.four_fold_fit_authority.fold_fit(released_fold)
    release_kind = (
        _POST_OUTER_ZERO_RELEASE_KIND
        if released_fold == 2
        else _POST_OUTER_ONE_RELEASE_KIND
    )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": (
            manifest.scientific_protocol_projection_sha256
        ),
        "base_manifest_v4_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "manifest_v5_registration_authority_receipt_sha256": (
            manifest_registration.semantic_receipt_sha256
        ),
        "manifest_v5_registration_source_receipt_sha256": (
            manifest_registration_source
        ),
        "manifest_v5_registration_commit_receipt_sha256": (
            manifest_registration_commit
        ),
        "manifest_v5_registration_committed_at_ms": _required_timestamp(
            "Manifest V5 registration time",
            manifest_registration.source_transaction_committed_at_ms,
        ),
        "execution_implementation_registration_authority_receipt_sha256": (
            execution_registration.semantic_receipt_sha256
        ),
        "execution_implementation_registration_source_receipt_sha256": (
            execution_source
        ),
        "execution_implementation_registration_commit_receipt_sha256": (
            execution_commit
        ),
        "execution_implementation_registration_committed_at_ms": (
            _required_timestamp(
                "execution implementation time",
                execution_registration.source_transaction_committed_at_ms,
            )
        ),
        "scientific_execution_fingerprint_sha256": (
            execution_registration.scientific_execution_fingerprint_sha256
        ),
        "training_state_receipt_sha256": (
            execution_registration.training_state_receipt_sha256
        ),
        "four_fold_fit_authority_receipt_sha256": (
            initial_inputs.four_fold_fit_authority_receipt_sha256
        ),
        "release_kind": release_kind,
        "released_validation_fold_indices": (released_fold,),
        "withheld_validation_fold_indices": (3,) if released_fold == 2 else (),
        "predecessor_outer_fold_index": predecessor,
        "predecessor_outer_fold_seal_receipt_sha256": (
            predecessor_outer_fold_seal.semantic_receipt_sha256
        ),
        "initial_validation_inputs_authority_receipt_sha256": (
            initial_inputs.semantic_receipt_sha256
        ),
        "initial_validation_inputs_source_receipt_sha256": initial_source,
        "initial_validation_inputs_commit_receipt_sha256": initial_commit,
        "initial_validation_inputs_committed_at_ms": initial_time,
        "prequential_validation_plan_receipt_sha256": (
            initial_inputs.prequential_validation_plan_receipt_sha256
        ),
        "runtime_sources_v2_receipt_sha256": (
            initial_inputs.runtime_sources_v2_receipt_sha256
        ),
        "source_bundle_v2_receipt_sha256": (
            initial_inputs.source_bundle_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_receipt_sha256": (
            initial_inputs.runtime_source_graph_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_witness_receipt_sha256": (
            initial_inputs.runtime_source_graph_v2_witness_receipt_sha256
        ),
        "replay_dependency_index_v2_receipt_sha256": (
            initial_inputs.replay_dependency_index_v2_receipt_sha256
        ),
        "training_source_projection_sha256": (
            initial_inputs.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            initial_inputs.validation_source_projection_sha256
        ),
        "validation_sources_v2_receipts": (validation_sources.semantic_receipt_sha256,),
        "validation_sources_v2_source_receipts": (validation_source_receipt,),
        "validation_sources_v2_commit_receipts": (validation_source_commit,),
        "validation_sources_v2_committed_at_ms": (validation_source_time,),
        "validation_environment_registry_v2_receipts": (
            validation_registry.semantic_receipt_sha256,
        ),
        "validation_registry_v2_source_receipts": (registry_source,),
        "validation_registry_v2_commit_receipts": (registry_commit,),
        "validation_registry_v2_committed_at_ms": (registry_time,),
        "validation_context_receipts": (
            validation_registry.validation_context_receipt_sha256,
        ),
        "validation_decision_session_date_inventories": (
            validation_sources.validation_decision_session_dates,
        ),
        "expected_candidate_checkpoint_authority_receipt_inventories": (
            fold_fit.candidate_checkpoint_authority_receipts,
        ),
        "source_data_qualified": bool(
            execution_registration.source_data_qualified
            and initial_inputs.source_data_qualified
            and predecessor_outer_fold_seal.source_data_qualified
            and validation_sources.source_data_qualified
            and validation_registry.source_data_qualified
        ),
        "predecessor_outer_fold_seal_source_receipt_sha256": predecessor_source,
        "predecessor_outer_fold_seal_commit_receipt_sha256": predecessor_commit,
        "predecessor_outer_fold_seal_committed_at_ms": predecessor_time,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA,
    }
    provisional = MassiveAdaptiveRLValidationReleaseAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_lineage_replayed=True,
        development_validation_release_authorized=bool(body["source_data_qualified"]),
        _runtime_manifest=manifest,
        _runtime_manifest_registration=manifest_registration,
        _runtime_execution_registration=execution_registration,
        _runtime_initial_inputs=initial_inputs,
        _runtime_validation_sources=(validation_sources,),
        _runtime_validation_registries=(validation_registry,),
        _runtime_predecessor_outer_seal=predecessor_outer_fold_seal,
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
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release source is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "released_validation_fold_indices",
        "withheld_validation_fold_indices",
        "validation_sources_v2_receipts",
        "validation_sources_v2_source_receipts",
        "validation_sources_v2_commit_receipts",
        "validation_sources_v2_committed_at_ms",
        "validation_environment_registry_v2_receipts",
        "validation_registry_v2_source_receipts",
        "validation_registry_v2_commit_receipts",
        "validation_registry_v2_committed_at_ms",
        "validation_context_receipts",
    ):
        body[name] = tuple(cast(Sequence[object], body[name]))
    for name in (
        "validation_decision_session_date_inventories",
        "expected_candidate_checkpoint_authority_receipt_inventories",
    ):
        body[name] = tuple(
            tuple(cast(Sequence[str], row))
            for row in cast(Sequence[Sequence[str]], body[name])
        )
    return body


def parse_massive_adaptive_rl_validation_release_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
    body = _parse_body(root=root, loaded_source=loaded_source)
    try:
        result = MassiveAdaptiveRLValidationReleaseAuthorityV1(
            **body,  # type: ignore[arg-type]
            semantic_receipt_sha256=semantic_sha256(body),
            _loaded_source=loaded_source,
        )
    except TypeError as error:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release source fields differ"
        ) from error
    result.validate()
    return result


def load_massive_adaptive_rl_validation_release_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    verified_at_ms: int,
    released_fold_index: int | None = None,
) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
    return parse_massive_adaptive_rl_validation_release_authority_v1(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=validation_release_authority_relative_path_v1(
                manifest=manifest, released_fold_index=released_fold_index
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_validation_release_authority_v1(
    *,
    authority: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
    authority.validate()
    expected = build_massive_adaptive_rl_initial_validation_release_authority_v1(
        manifest=manifest,
        manifest_registration=manifest_registration,
        execution_registration=execution_registration,
        initial_inputs=initial_inputs,
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != validation_release_authority_relative_path_v1(manifest=manifest)
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release did not replay"
        )
    result = replace(
        authority,
        runtime_lineage_replayed=True,
        development_validation_release_authorized=authority.source_data_qualified,
        _runtime_manifest=manifest,
        _runtime_manifest_registration=manifest_registration,
        _runtime_execution_registration=execution_registration,
        _runtime_initial_inputs=initial_inputs,
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_delayed_validation_release_authority_v1(
    *,
    authority: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    predecessor_outer_fold_seal: MassiveAdaptiveRLOuterFoldSealAuthorityV1,
    validation_sources: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_registry: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
    authority.validate()
    expected = build_massive_adaptive_rl_delayed_validation_release_authority_v1(
        manifest=manifest,
        manifest_registration=manifest_registration,
        execution_registration=execution_registration,
        initial_inputs=initial_inputs,
        predecessor_outer_fold_seal=predecessor_outer_fold_seal,
        validation_sources=validation_sources,
        validation_registry=validation_registry,
    )
    released_fold = predecessor_outer_fold_seal.fold_index + 2
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != validation_release_authority_relative_path_v1(
            manifest=manifest, released_fold_index=released_fold
        )
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "delayed validation release did not replay"
        )
    result = replace(
        authority,
        runtime_lineage_replayed=True,
        development_validation_release_authorized=authority.source_data_qualified,
        _runtime_manifest=manifest,
        _runtime_manifest_registration=manifest_registration,
        _runtime_execution_registration=execution_registration,
        _runtime_initial_inputs=initial_inputs,
        _runtime_validation_sources=(validation_sources,),
        _runtime_validation_registries=(validation_registry,),
        _runtime_predecessor_outer_seal=predecessor_outer_fold_seal,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_rl_delayed_validation_release_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    predecessor_outer_fold_seal: MassiveAdaptiveRLOuterFoldSealAuthorityV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
    """Release V2 or V3 from the exact authenticated O0 or O1 seal."""

    predecessor_outer_fold_seal.validate()
    if (
        type(allow_materialize) is not bool
        or predecessor_outer_fold_seal.fold_index not in (0, 1)
        or not predecessor_outer_fold_seal.validation_release_authorized
    ):
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "delayed validation release predecessor differs"
        )
    released_fold = predecessor_outer_fold_seal.fold_index + 2
    relative = validation_release_authority_relative_path_v1(
        manifest=manifest, released_fold_index=released_fold
    )
    try:
        with massive_adaptive_rl_experiment_materialization_lock_v1(
            artifact_root=root, experiment_id=manifest.experiment_id
        ):
            complete, partial = _transaction_state(root=root, relative=relative)
            if partial:
                raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                    "delayed validation release transaction is incomplete"
                )
            seal_time = _required_timestamp(
                "predecessor outer-fold seal time",
                predecessor_outer_fold_seal.source_transaction_committed_at_ms,
            )
            capability = issue_massive_adaptive_rl_manifest_v5_delayed_validation_release_capability_v1(
                root=root,
                authority=manifest_registration,
                fold_index=released_fold,
            )
            with massive_adaptive_rl_manifest_v5_writer_scope_v1(
                root=root, capability=capability
            ):
                validation_sources = (
                    prepare_or_resume_massive_adaptive_rl_validation_sources_v2(
                        root=root,
                        manifest=manifest.base_manifest,
                        four_fold_fit_authority=initial_inputs.four_fold_fit_authority,
                        runtime_sources_v2=initial_inputs.runtime_sources_v2,
                        fold_index=released_fold,
                        committed_at_ms=max(_wall_clock_ms(), seal_time) + 1,
                        allow_materialize=allow_materialize,
                        v5_writer_capability=capability,
                    )
                )
                source_time = _required_timestamp(
                    "delayed validation-source time",
                    validation_sources.source_transaction_committed_at_ms,
                )
                validation_registry = prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v2(
                    root=root,
                    manifest=manifest.base_manifest,
                    runtime_sources_v2=initial_inputs.runtime_sources_v2,
                    validation_sources_v2=validation_sources,
                    committed_at_ms=source_time + 1,
                    allow_materialize=allow_materialize,
                    v5_writer_capability=capability,
                )
                expected = (
                    build_massive_adaptive_rl_delayed_validation_release_authority_v1(
                        manifest=manifest,
                        manifest_registration=manifest_registration,
                        execution_registration=execution_registration,
                        initial_inputs=initial_inputs,
                        predecessor_outer_fold_seal=predecessor_outer_fold_seal,
                        validation_sources=validation_sources,
                        validation_registry=validation_registry,
                    )
                )
                if not complete:
                    if not allow_materialize:
                        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
                            "delayed validation release is absent"
                        )
                    registry_time = _required_timestamp(
                        "delayed validation-registry time",
                        validation_registry.source_transaction_committed_at_ms,
                    )
                    committed_at_ms = max(_wall_clock_ms(), registry_time) + 1
                    publish_massive_source_object(
                        stream=BytesIO(
                            canonical_json_file_bytes(expected.semantic_unsigned())
                        ),
                        root=root,
                        relative_payload_path=relative,
                        dataset_id=MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_DATASET,
                        source_object_key=relative,
                        requested_at_ms=committed_at_ms,
                        downloaded_at_ms=committed_at_ms,
                        schema_sha256=MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
                        entitlement_receipt_sha256=expected.semantic_receipt_sha256,
                        committed_at_ms=committed_at_ms,
                        request_id=(
                            f"ADAPTIVE-RL-VALIDATION-RELEASE-{manifest.experiment_id}-"
                            f"FOLD{released_fold}"
                        ),
                    )
            loaded = load_massive_source_bundle(
                root=root,
                relative_payload_path=relative,
                verified_at_ms=_wall_clock_ms(),
            )
            return (
                authorize_massive_adaptive_rl_delayed_validation_release_authority_v1(
                    authority=parse_massive_adaptive_rl_validation_release_authority_v1(
                        root=root, loaded_source=loaded
                    ),
                    manifest=manifest,
                    manifest_registration=manifest_registration,
                    execution_registration=execution_registration,
                    initial_inputs=initial_inputs,
                    predecessor_outer_fold_seal=predecessor_outer_fold_seal,
                    validation_sources=validation_sources,
                    validation_registry=validation_registry,
                )
            )
    except MassiveAdaptiveRLExperimentLockV1Unavailable as error:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1LeaseUnavailable(
            "delayed validation release publication is already owned"
        ) from error
    except MassiveAdaptiveRLExperimentLockV1Error as error:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "delayed validation release publication lock is invalid"
        ) from error


def _run_or_resume_initial_validation_release_unlocked_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    allow_materialize: bool,
) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
    relative = validation_release_authority_relative_path_v1(manifest=manifest)
    complete, partial = _transaction_state(root=root, relative=relative)
    if partial:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release transaction is incomplete"
        )
    verified_at_ms = _wall_clock_ms()
    if complete:
        return authorize_massive_adaptive_rl_validation_release_authority_v1(
            authority=load_massive_adaptive_rl_validation_release_authority_v1(
                root=root,
                manifest=manifest,
                verified_at_ms=verified_at_ms,
            ),
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            initial_inputs=initial_inputs,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release is absent"
        )
    authority = build_massive_adaptive_rl_initial_validation_release_authority_v1(
        manifest=manifest,
        manifest_registration=manifest_registration,
        execution_registration=execution_registration,
        initial_inputs=initial_inputs,
    )
    committed_at_ms = max(
        verified_at_ms,
        authority.initial_validation_inputs_committed_at_ms + 1,
    )
    capability = (
        issue_massive_adaptive_rl_manifest_v5_initial_validation_release_capability_v1(
            root=root,
            authority=manifest_registration,
        )
    )
    with massive_adaptive_rl_manifest_v5_writer_scope_v1(
        root=root,
        capability=capability,
    ):
        publish_massive_source_object(
            stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
            root=root,
            relative_payload_path=relative,
            dataset_id=MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_DATASET,
            source_object_key=relative,
            requested_at_ms=committed_at_ms,
            downloaded_at_ms=committed_at_ms,
            schema_sha256=(
                MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            ),
            entitlement_receipt_sha256=authority.semantic_receipt_sha256,
            committed_at_ms=committed_at_ms,
            request_id=f"ADAPTIVE-RL-VALIDATION-RELEASE-{manifest.experiment_id}",
        )
    return authorize_massive_adaptive_rl_validation_release_authority_v1(
        authority=load_massive_adaptive_rl_validation_release_authority_v1(
            root=root,
            manifest=manifest,
            verified_at_ms=committed_at_ms,
        ),
        manifest=manifest,
        manifest_registration=manifest_registration,
        execution_registration=execution_registration,
        initial_inputs=initial_inputs,
    )


def run_or_resume_massive_adaptive_rl_initial_validation_release_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
    initial_inputs: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
    """Bind the initial tapes to the exact frozen implementation, create-only."""

    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release materialization mode differs"
        )
    if not allow_materialize:
        return _run_or_resume_initial_validation_release_unlocked_v1(
            root=root,
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            initial_inputs=initial_inputs,
            allow_materialize=False,
        )
    try:
        with massive_adaptive_rl_experiment_materialization_lock_v1(
            artifact_root=root,
            experiment_id=manifest.experiment_id,
        ):
            return _run_or_resume_initial_validation_release_unlocked_v1(
                root=root,
                manifest=manifest,
                manifest_registration=manifest_registration,
                execution_registration=execution_registration,
                initial_inputs=initial_inputs,
                allow_materialize=True,
            )
    except MassiveAdaptiveRLExperimentLockV1Unavailable as error:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1LeaseUnavailable(
            "validation release publication is already owned"
        ) from error
    except MassiveAdaptiveRLExperimentLockV1Error as error:
        raise MassiveAdaptiveRLValidationReleaseAuthorityV1Error(
            "validation release publication lock is invalid"
        ) from error


__all__ = [
    "MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SOURCE_SHA256",
    "MassiveAdaptiveRLValidationReleaseAuthorityV1",
    "MassiveAdaptiveRLValidationReleaseAuthorityV1Error",
    "MassiveAdaptiveRLValidationReleaseAuthorityV1LeaseUnavailable",
    "authorize_massive_adaptive_rl_validation_release_authority_v1",
    "authorize_massive_adaptive_rl_delayed_validation_release_authority_v1",
    "build_massive_adaptive_rl_delayed_validation_release_authority_v1",
    "build_massive_adaptive_rl_initial_validation_release_authority_v1",
    "load_massive_adaptive_rl_validation_release_authority_v1",
    "parse_massive_adaptive_rl_validation_release_authority_v1",
    "run_or_resume_massive_adaptive_rl_delayed_validation_release_v1",
    "run_or_resume_massive_adaptive_rl_initial_validation_release_v1",
    "validation_release_authority_relative_path_v1",
]
