"""Commit every adaptive-RL validation input before any validation outcome.

The fold-local validation source and environment authorities establish one
canonical tape per fold.  This module adds the experiment-wide temporal
barrier: all four tapes and their candidate checkpoint populations are
persisted in one create-only authority before a PPO or FC06 evaluator may run.
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
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    MassiveAdaptiveRLValidationSourcesAuthorityV1,
    authorize_massive_adaptive_rl_validation_environment_registry_v1,
    authorize_massive_adaptive_rl_validation_sources_authority_v1,
    load_massive_adaptive_rl_validation_environment_registry_v1,
    load_massive_adaptive_rl_validation_sources_authority_v1,
    massive_adaptive_rl_validation_downstream_evidence_exists_v1,
    prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v1,
    prepare_or_resume_massive_adaptive_rl_validation_sources_v1,
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
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLRuntimeSourcesV1,
)


MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-four-fold-validation-inputs-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-four-fold-validation-inputs-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SHA256 = (
    file_sha256(Path(__file__))
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": (
                MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA
            ),
            "encoding": "canonical-json-four-fold-validation-input-barrier",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V1_SPEC_SHA256 = semantic_sha256(
    {
        "fold_inventory": (0, 1, 2, 3),
        "source": "four-canonical-validation-source-authorities",
        "environment": "four-canonical-10-20-40bp-environment-registries",
        "candidate_population": "exact-completed-fold-fit-checkpoint-inventories",
        "ordering": "all-four-registries-before-aggregate-before-any-outcome",
        "execution": "leaf-outcomes-require-replayed-aggregate",
        "persistence": "manifest-derived-create-only-source-transaction",
        "concurrency": "one-nonblocking-experiment-generation-lease",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)

_FOLD_INDICES = (0, 1, 2, 3)


class MassiveAdaptiveRLFourFoldValidationInputsV1Error(ValueError):
    """The four-fold validation input barrier is absent or inconsistent."""


class MassiveAdaptiveRLFourFoldValidationInputsLeaseUnavailable(
    MassiveAdaptiveRLFourFoldValidationInputsV1Error
):
    """Another process owns four-fold validation-input publication."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _source_transaction_exists(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input source transaction is incomplete"
        )
    return all(present)


def four_fold_validation_inputs_authority_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> str:
    manifest.validate()
    return (
        "massive-adaptive/rl-four-fold-validation-inputs-authority-v1/"
        f"v4-{manifest.semantic_receipt_sha256}.json"
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    fold_indices: tuple[int, ...]
    validation_sources_authority_receipts: tuple[str, ...]
    validation_sources_source_receipts: tuple[str, ...]
    validation_sources_commit_receipts: tuple[str, ...]
    validation_sources_committed_at_ms: tuple[int, ...]
    validation_environment_registry_receipts: tuple[str, ...]
    validation_registry_source_receipts: tuple[str, ...]
    validation_registry_commit_receipts: tuple[str, ...]
    validation_registry_committed_at_ms: tuple[int, ...]
    validation_context_receipts: tuple[str, ...]
    validation_decision_session_date_inventories: tuple[tuple[str, ...], ...]
    expected_candidate_checkpoint_authority_receipt_inventories: tuple[
        tuple[str, ...], ...
    ]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_inputs_replayed: bool = False
    development_validation_execution_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA
    _validation_sources: tuple[MassiveAdaptiveRLValidationSourcesAuthorityV1, ...] = (
        field(default=(), compare=False, repr=False)
    )
    _validation_registries: tuple[
        MassiveAdaptiveRLValidationEnvironmentRegistryV1, ...
    ] = field(default=(), compare=False, repr=False)
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "manifest_v4_receipt_sha256": self.manifest_v4_receipt_sha256,
            "training_manifest_v3_receipt_sha256": (
                self.training_manifest_v3_receipt_sha256
            ),
            "four_fold_fit_authority_receipt_sha256": (
                self.four_fold_fit_authority_receipt_sha256
            ),
            "runtime_sources_receipt_sha256": self.runtime_sources_receipt_sha256,
            "runtime_graph_witness_receipt_sha256": (
                self.runtime_graph_witness_receipt_sha256
            ),
            "fold_indices": self.fold_indices,
            "validation_sources_authority_receipts": (
                self.validation_sources_authority_receipts
            ),
            "validation_sources_source_receipts": (
                self.validation_sources_source_receipts
            ),
            "validation_sources_commit_receipts": (
                self.validation_sources_commit_receipts
            ),
            "validation_sources_committed_at_ms": (
                self.validation_sources_committed_at_ms
            ),
            "validation_environment_registry_receipts": (
                self.validation_environment_registry_receipts
            ),
            "validation_registry_source_receipts": (
                self.validation_registry_source_receipts
            ),
            "validation_registry_commit_receipts": (
                self.validation_registry_commit_receipts
            ),
            "validation_registry_committed_at_ms": (
                self.validation_registry_committed_at_ms
            ),
            "validation_context_receipts": self.validation_context_receipts,
            "validation_decision_session_date_inventories": (
                self.validation_decision_session_date_inventories
            ),
            "expected_candidate_checkpoint_authority_receipt_inventories": (
                self.expected_candidate_checkpoint_authority_receipt_inventories
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
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

    def validate(self) -> None:
        runtime_present = bool(self._validation_sources or self._validation_registries)
        if runtime_present and (
            len(self._validation_sources) != 4 or len(self._validation_registries) != 4
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
                "four-fold validation-input runtime is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        for source_authority in self._validation_sources:
            source_authority.validate()
        for registry_authority in self._validation_registries:
            registry_authority.validate()
        expected_authorized = bool(
            runtime_present
            and all(
                row.development_stage_authorized for row in self._validation_sources
            )
            and all(
                row.development_stage_authorized for row in self._validation_registries
            )
        )
        inventories = (
            self.validation_sources_authority_receipts,
            self.validation_sources_source_receipts,
            self.validation_sources_commit_receipts,
            self.validation_sources_committed_at_ms,
            self.validation_environment_registry_receipts,
            self.validation_registry_source_receipts,
            self.validation_registry_commit_receipts,
            self.validation_registry_committed_at_ms,
            self.validation_context_receipts,
            self.validation_decision_session_date_inventories,
            self.expected_candidate_checkpoint_authority_receipt_inventories,
        )
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA
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
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
                "four-fold validation-input authority differs"
            )
        if runtime_present:
            sources = self._validation_sources
            registries = self._validation_registries
            if (
                tuple(row.fold_index for row in sources) != _FOLD_INDICES
                or tuple(row.fold_index for row in registries) != _FOLD_INDICES
                or self.validation_sources_authority_receipts
                != tuple(row.semantic_receipt_sha256 for row in sources)
                or self.validation_sources_source_receipts
                != tuple(cast(str, row.source_receipt_sha256) for row in sources)
                or self.validation_sources_commit_receipts
                != tuple(
                    cast(str, row.source_transaction_receipt_sha256) for row in sources
                )
                or self.validation_sources_committed_at_ms
                != tuple(
                    cast(int, row.source_transaction_committed_at_ms) for row in sources
                )
                or self.validation_environment_registry_receipts
                != tuple(row.semantic_receipt_sha256 for row in registries)
                or self.validation_registry_source_receipts
                != tuple(cast(str, row.source_receipt_sha256) for row in registries)
                or self.validation_registry_commit_receipts
                != tuple(
                    cast(str, row.source_transaction_receipt_sha256)
                    for row in registries
                )
                or self.validation_registry_committed_at_ms
                != tuple(
                    cast(int, row.source_transaction_committed_at_ms)
                    for row in registries
                )
                or self.validation_context_receipts
                != tuple(row.validation_context_receipt_sha256 for row in registries)
                or self.validation_decision_session_date_inventories
                != tuple(row.validation_decision_session_dates for row in sources)
                or any(
                    registry.validation_sources_authority_receipt_sha256
                    != source.semantic_receipt_sha256
                    or source.source_transaction_committed_at_ms is None
                    or registry.source_transaction_committed_at_ms is None
                    or source.source_transaction_committed_at_ms
                    >= registry.source_transaction_committed_at_ms
                    for source, registry in zip(sources, registries, strict=True)
                )
            ):
                raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
                    "four-fold validation-input runtime differs"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
                "four-fold validation-input source transaction differs"
            )
        for value in (
            self.manifest_v4_receipt_sha256,
            self.training_manifest_v3_receipt_sha256,
            self.four_fold_fit_authority_receipt_sha256,
            self.runtime_sources_receipt_sha256,
            self.runtime_graph_witness_receipt_sha256,
            *self.validation_sources_authority_receipts,
            *self.validation_sources_source_receipts,
            *self.validation_sources_commit_receipts,
            *self.validation_environment_registry_receipts,
            *self.validation_registry_source_receipts,
            *self.validation_registry_commit_receipts,
            *self.validation_context_receipts,
            *(
                receipt
                for inventory in (
                    self.expected_candidate_checkpoint_authority_receipt_inventories
                )
                for receipt in inventory
            ),
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("four-fold validation inputs", value)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                *self.validation_sources_committed_at_ms,
                *self.validation_registry_committed_at_ms,
            )
        ):
            raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
                "four-fold validation-input commit time differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    validation_sources: Sequence[MassiveAdaptiveRLValidationSourcesAuthorityV1],
    validation_environment_registries: Sequence[
        MassiveAdaptiveRLValidationEnvironmentRegistryV1
    ],
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(four_fold_fit_authority) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
        or type(runtime_sources) is not MassiveAdaptiveRLRuntimeSourcesV1
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input root type differs"
        )
    manifest.validate()
    four_fold_fit_authority.validate()
    runtime_sources.validate()
    sources = tuple(validation_sources)
    registries = tuple(validation_environment_registries)
    if (
        len(sources) != 4
        or len(registries) != 4
        or any(
            type(row) is not MassiveAdaptiveRLValidationSourcesAuthorityV1
            for row in sources
        )
        or any(
            type(row) is not MassiveAdaptiveRLValidationEnvironmentRegistryV1
            for row in registries
        )
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input child inventory differs"
        )
    for authority in (*sources, *registries):
        authority.validate()
    runtime_witness = (
        runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    )
    if (
        runtime_witness is None
        or not four_fold_fit_authority.development_stage_authorized
        or not four_fold_fit_authority.source_transaction_verified
        or not runtime_sources.source_data_qualified
        or manifest.experiment_id != four_fold_fit_authority.experiment_id
        or manifest.experiment_id != runtime_sources.experiment_id
        or manifest.base_manifest.semantic_receipt_sha256
        != four_fold_fit_authority.manifest_v3_receipt_sha256
        or manifest.base_manifest.semantic_receipt_sha256
        != runtime_sources.manifest_v3_receipt_sha256
        or four_fold_fit_authority.runtime_sources_receipt_sha256
        != runtime_sources.semantic_receipt_sha256
        or four_fold_fit_authority.runtime_graph_witness_receipt_sha256
        != runtime_witness
        or tuple(row.fold_index for row in sources) != _FOLD_INDICES
        or tuple(row.fold_index for row in registries) != _FOLD_INDICES
        or any(not row.development_stage_authorized for row in sources)
        or any(not row.development_stage_authorized for row in registries)
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input lineage is not authorized"
        )
    fold_fits = tuple(
        four_fold_fit_authority.fold_fit(fold_index) for fold_index in _FOLD_INDICES
    )
    if any(
        source.experiment_id != manifest.experiment_id
        or source.manifest_v4_receipt_sha256 != manifest.semantic_receipt_sha256
        or source.training_manifest_v3_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or source.four_fold_fit_authority_receipt_sha256
        != four_fold_fit_authority.semantic_receipt_sha256
        or source.fold_fit_authority_receipt_sha256 != fold_fit.semantic_receipt_sha256
        or source.runtime_sources_receipt_sha256
        != runtime_sources.semantic_receipt_sha256
        or source.runtime_graph_witness_receipt_sha256 != runtime_witness
        or registry.experiment_id != manifest.experiment_id
        or registry.manifest_v4_receipt_sha256 != manifest.semantic_receipt_sha256
        or registry.training_manifest_v3_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or registry.validation_sources_authority_receipt_sha256
        != source.semantic_receipt_sha256
        or registry.runtime_sources_receipt_sha256
        != runtime_sources.semantic_receipt_sha256
        or registry.runtime_graph_witness_receipt_sha256 != runtime_witness
        or source.source_receipt_sha256 is None
        or source.source_transaction_receipt_sha256 is None
        or source.source_transaction_committed_at_ms is None
        or registry.source_receipt_sha256 is None
        or registry.source_transaction_receipt_sha256 is None
        or registry.source_transaction_committed_at_ms is None
        or source.source_transaction_committed_at_ms
        >= registry.source_transaction_committed_at_ms
        for source, registry, fold_fit in zip(
            sources, registries, fold_fits, strict=True
        )
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input child lineage differs"
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
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": runtime_witness,
        "fold_indices": _FOLD_INDICES,
        "validation_sources_authority_receipts": tuple(
            row.semantic_receipt_sha256 for row in sources
        ),
        "validation_sources_source_receipts": tuple(
            cast(str, row.source_receipt_sha256) for row in sources
        ),
        "validation_sources_commit_receipts": tuple(
            cast(str, row.source_transaction_receipt_sha256) for row in sources
        ),
        "validation_sources_committed_at_ms": tuple(
            cast(int, row.source_transaction_committed_at_ms) for row in sources
        ),
        "validation_environment_registry_receipts": tuple(
            row.semantic_receipt_sha256 for row in registries
        ),
        "validation_registry_source_receipts": tuple(
            cast(str, row.source_receipt_sha256) for row in registries
        ),
        "validation_registry_commit_receipts": tuple(
            cast(str, row.source_transaction_receipt_sha256) for row in registries
        ),
        "validation_registry_committed_at_ms": tuple(
            cast(int, row.source_transaction_committed_at_ms) for row in registries
        ),
        "validation_context_receipts": tuple(
            row.validation_context_receipt_sha256 for row in registries
        ),
        "validation_decision_session_date_inventories": tuple(
            row.validation_decision_session_dates for row in sources
        ),
        "expected_candidate_checkpoint_authority_receipt_inventories": tuple(
            row.candidate_checkpoint_authority_receipts for row in fold_fits
        ),
        "source_data_qualified": bool(
            four_fold_fit_authority.source_data_qualified
            and runtime_sources.source_data_qualified
            and all(row.source_data_qualified for row in sources)
            and all(row.source_data_qualified for row in registries)
        ),
    }
    provisional = MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_inputs_replayed=True,
        development_validation_execution_authorized=True,
        _validation_sources=sources,
        _validation_registries=registries,
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
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input payload is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "fold_indices",
        "validation_sources_authority_receipts",
        "validation_sources_source_receipts",
        "validation_sources_commit_receipts",
        "validation_sources_committed_at_ms",
        "validation_environment_registry_receipts",
        "validation_registry_source_receipts",
        "validation_registry_commit_receipts",
        "validation_registry_committed_at_ms",
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


def parse_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1:
    body = _parse_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_inputs_replayed=False,
        development_validation_execution_authorized=False,
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    verified_at_ms: int,
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1:
    relative = four_fold_validation_inputs_authority_relative_path_v1(manifest=manifest)
    return parse_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    validation_sources: Sequence[MassiveAdaptiveRLValidationSourcesAuthorityV1],
    validation_environment_registries: Sequence[
        MassiveAdaptiveRLValidationEnvironmentRegistryV1
    ],
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1:
    del root
    authority.validate()
    expected_path = four_fold_validation_inputs_authority_relative_path_v1(
        manifest=manifest
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path != expected_path
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input authority path differs"
        )
    expected = build_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
        manifest=manifest,
        four_fold_fit_authority=four_fold_fit_authority,
        runtime_sources=runtime_sources,
        validation_sources=validation_sources,
        validation_environment_registries=validation_environment_registries,
    )
    if authority.semantic_unsigned() != expected.semantic_unsigned():
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input authority does not replay"
        )
    result = replace(
        authority,
        runtime_inputs_replayed=True,
        development_validation_execution_authorized=authority.source_data_qualified,
        _validation_sources=tuple(validation_sources),
        _validation_registries=tuple(validation_environment_registries),
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    authority: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1:
    manifest.validate()
    authority.validate()
    if (
        authority.runtime_inputs_replayed is not True
        or authority.manifest_v4_receipt_sha256 != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation inputs are not runtime replayed"
        )
    manifest_receipt = authority.manifest_v4_receipt_sha256
    relative = (
        "massive-adaptive/rl-four-fold-validation-inputs-authority-v1/"
        f"v4-{manifest_receipt}.json"
    )
    if _source_transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input authority already exists"
        )
    if any(
        massive_adaptive_rl_validation_downstream_evidence_exists_v1(
            root=root,
            manifest=manifest,
            fold_index=fold_index,
        )
        for fold_index in _FOLD_INDICES
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "missing four-fold validation-input barrier cannot be created after outcome evidence"
        )
    if (
        isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms <= max(authority.validation_registry_committed_at_ms)
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input barrier must follow every registry commit"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=(
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_DATASET
        ),
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-FOUR-FOLD-VALIDATION-INPUTS-V1-{authority.experiment_id}"
        ),
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    result = replace(authority, _loaded_source=loaded)
    result.validate()
    return result


def validate_massive_adaptive_rl_validation_outcome_barrier_v1(
    *,
    authority: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1,
    validation_environment_registry: MassiveAdaptiveRLValidationEnvironmentRegistryV1,
    fold_index: int,
    outcome_committed_at_ms: int,
) -> None:
    """Require the all-fold barrier and requested registry to predate an outcome."""

    if type(authority) is not MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1:
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "inner-validation outcome requires the four-fold input authority"
        )
    authority.validate()
    if (
        not authority.development_stage_authorized
        or authority.source_receipt_sha256 is None
        or authority.source_transaction_receipt_sha256 is None
        or authority.source_transaction_committed_at_ms is None
        or authority.source_transaction_committed_at_ms >= outcome_committed_at_ms
        or fold_index not in authority.fold_indices
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input barrier is not precommitted and authorized"
        )
    index = authority.fold_indices.index(fold_index)
    if (
        validation_environment_registry.fold_index != fold_index
        or validation_environment_registry.semantic_receipt_sha256
        != authority.validation_environment_registry_receipts[index]
        or validation_environment_registry.source_receipt_sha256
        != authority.validation_registry_source_receipts[index]
        or validation_environment_registry.source_transaction_receipt_sha256
        != authority.validation_registry_commit_receipts[index]
        or validation_environment_registry.source_transaction_committed_at_ms
        != authority.validation_registry_committed_at_ms[index]
        or validation_environment_registry.validation_context_receipt_sha256
        != authority.validation_context_receipts[index]
    ):
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "validation registry is not a child of the four-fold input authority"
        )


@contextmanager
def _four_fold_validation_inputs_lease(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> Iterator[None]:
    manifest.validate()
    lease_directory = (
        Path(root) / "massive-adaptive" / "rl-four-fold-validation-input-leases-v1"
    )
    try:
        Path(root).mkdir(parents=True, exist_ok=True)
        if Path(root).is_symlink():
            raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
                "four-fold validation-input root is a symlink"
            )
        lease_directory.mkdir(parents=True, exist_ok=True)
        if lease_directory.is_symlink():
            raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
                "four-fold validation-input lease directory is a symlink"
            )
        descriptor = os.open(
            lease_directory / f"v4-{manifest.semantic_receipt_sha256}.lock",
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
    except OSError as error:
        raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
            "four-fold validation-input lease is unavailable"
        ) from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
                "four-fold validation-input lease identity differs"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MassiveAdaptiveRLFourFoldValidationInputsLeaseUnavailable(
                "four-fold validation-input lease is already held"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1:
    """Commit all four validation tapes before returning execution authority."""

    relative = four_fold_validation_inputs_authority_relative_path_v1(manifest=manifest)
    with _four_fold_validation_inputs_lease(root=root, manifest=manifest):
        aggregate_exists = _source_transaction_exists(root=root, relative=relative)
        if aggregate_exists:
            sources = tuple(
                authorize_massive_adaptive_rl_validation_sources_authority_v1(
                    root=root,
                    authority=load_massive_adaptive_rl_validation_sources_authority_v1(
                        root=root,
                        manifest=manifest,
                        fold_index=fold_index,
                        verified_at_ms=committed_at_ms,
                    ),
                    manifest=manifest,
                    four_fold_fit_authority=four_fold_fit_authority,
                    runtime_sources=runtime_sources,
                )
                for fold_index in _FOLD_INDICES
            )
            registries = tuple(
                authorize_massive_adaptive_rl_validation_environment_registry_v1(
                    root=root,
                    registry=(
                        load_massive_adaptive_rl_validation_environment_registry_v1(
                            root=root,
                            manifest=manifest,
                            fold_index=fold_index,
                            verified_at_ms=committed_at_ms,
                        )
                    ),
                    manifest=manifest,
                    validation_sources=sources[fold_index],
                    runtime_sources=runtime_sources,
                )
                for fold_index in _FOLD_INDICES
            )
            return authorize_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
                root=root,
                authority=load_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
                    root=root,
                    manifest=manifest,
                    verified_at_ms=committed_at_ms,
                ),
                manifest=manifest,
                four_fold_fit_authority=four_fold_fit_authority,
                runtime_sources=runtime_sources,
                validation_sources=sources,
                validation_environment_registries=registries,
            )
        if not allow_materialize:
            raise MassiveAdaptiveRLFourFoldValidationInputsV1Error(
                "completed four-fold validation-input authority is absent"
            )
        sources = tuple(
            prepare_or_resume_massive_adaptive_rl_validation_sources_v1(
                root=root,
                manifest=manifest,
                four_fold_fit_authority=four_fold_fit_authority,
                runtime_sources=runtime_sources,
                fold_index=fold_index,
                committed_at_ms=committed_at_ms + fold_index,
            )
            for fold_index in _FOLD_INDICES
        )
        registries = tuple(
            prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v1(
                root=root,
                manifest=manifest,
                validation_sources=sources[fold_index],
                runtime_sources=runtime_sources,
                committed_at_ms=committed_at_ms + 4 + fold_index,
            )
            for fold_index in _FOLD_INDICES
        )
        authority = build_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
            manifest=manifest,
            four_fold_fit_authority=four_fold_fit_authority,
            runtime_sources=runtime_sources,
            validation_sources=sources,
            validation_environment_registries=registries,
        )
        barrier_committed_at_ms = max(
            committed_at_ms + 8,
            max(authority.validation_registry_committed_at_ms) + 1,
        )
        return materialize_massive_adaptive_rl_four_fold_validation_inputs_authority_v1(
            root=root,
            manifest=manifest,
            authority=authority,
            committed_at_ms=barrier_committed_at_ms,
        )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFourFoldValidationInputsAuthorityV1",
    "MassiveAdaptiveRLFourFoldValidationInputsLeaseUnavailable",
    "MassiveAdaptiveRLFourFoldValidationInputsV1Error",
    "authorize_massive_adaptive_rl_four_fold_validation_inputs_authority_v1",
    "build_massive_adaptive_rl_four_fold_validation_inputs_authority_v1",
    "four_fold_validation_inputs_authority_relative_path_v1",
    "load_massive_adaptive_rl_four_fold_validation_inputs_authority_v1",
    "materialize_massive_adaptive_rl_four_fold_validation_inputs_authority_v1",
    "parse_massive_adaptive_rl_four_fold_validation_inputs_authority_v1",
    "run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v1",
    "validate_massive_adaptive_rl_validation_outcome_barrier_v1",
]
