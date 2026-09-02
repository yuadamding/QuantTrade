"""Aggregate and execute the four Manifest-V3 adaptive-RL fit folds."""

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
from typing import cast, Iterator

import torch

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
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MassiveAdaptiveRLExperimentStageV2,
    MassiveAdaptiveRLExperimentStateV2,
    advance_massive_adaptive_rl_experiment_state_v2,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_inputs_v1 import (
    MassiveAdaptiveRLFoldFitInputsAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_v1 import (
    MassiveAdaptiveRLFoldFitAuthorityV1,
    load_massive_adaptive_rl_fold_fit_authority_v1,
    prepare_or_resume_massive_adaptive_rl_fold_fit_inputs_v1,
    run_or_resume_massive_adaptive_rl_fold_fit_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLRuntimeSourcesV1,
)


MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-four-fold-fit-inputs-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-four-fold-fit-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-four-fold-fit-inputs-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-four-fold-fit-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA,
            "encoding": "canonical-json-receipt-envelope",
        }
    )
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA,
            "encoding": "canonical-json-receipt-envelope",
        }
    )
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "fold_inventory": (0, 1, 2, 3),
        "input_ordering": "all-inputs-authorized-before-any-fold-execution",
        "execution": "package-owned-run-or-resume-per-fold",
        "scientific_environment": "one-cross-worker-fingerprint",
        "physical_worker_compatibility": "one-device-class-and-driver-fingerprint",
        "nested_witnesses": "exact-types-and-source-transaction-verified",
        "replay_discovery": "canonical-package-owned-fold-paths",
        "persistence": "create-only-input-and-completed-aggregate-authorities",
        "state_binding": (
            "fit-forecasts-authorized-then-ppo-and-fixed-controls-trained"
        ),
        "profitability_reporting": False,
        "outer_access": False,
    }
)
_FOUR_FOLD_INDICES = (0, 1, 2, 3)


class MassiveAdaptiveRLFourFoldFitV1Error(ValueError):
    """The four-fold fit inventory or its persisted lineage differs."""


class MassiveAdaptiveRLFourFoldFitExecutionLeaseUnavailable(
    MassiveAdaptiveRLFourFoldFitV1Error
):
    """Another root process owns this four-fold execution stage."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _runtime_witness_receipt(
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
) -> str:
    receipt = (
        runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    )
    return _digest("adaptive RL runtime graph witness", receipt)


def _source_transaction_exists(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    receipt = payload.with_name(payload.name + ".receipt.json")
    commit = payload.with_name(payload.name + ".commit.json")
    present = tuple(
        path.exists() or path.is_symlink() for path in (payload, receipt, commit)
    )
    if any(present) and not all(present):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit source transaction is incomplete"
        )
    return all(present)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFourFoldFitInputsAuthorityV1:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    fold_indices: tuple[int, ...]
    fold_fit_input_authority_receipts: tuple[str, ...]
    execution_environment_authority_receipts: tuple[str, ...]
    scientific_execution_fingerprint_sha256: str
    physical_worker_compatibility_sha256: str
    source_data_qualified: bool
    runtime_inputs_replayed: bool
    semantic_receipt_sha256: str
    development_rl_training_inputs_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA
    _fold_fit_inputs: tuple[MassiveAdaptiveRLFoldFitInputsAuthorityV1, ...] = field(
        default=(), repr=False, compare=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, repr=False, compare=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "manifest_v3_receipt_sha256": self.manifest_v3_receipt_sha256,
            "runtime_sources_receipt_sha256": self.runtime_sources_receipt_sha256,
            "runtime_graph_witness_receipt_sha256": (
                self.runtime_graph_witness_receipt_sha256
            ),
            "fold_indices": self.fold_indices,
            "fold_fit_input_authority_receipts": (
                self.fold_fit_input_authority_receipts
            ),
            "execution_environment_authority_receipts": (
                self.execution_environment_authority_receipts
            ),
            "scientific_execution_fingerprint_sha256": (
                self.scientific_execution_fingerprint_sha256
            ),
            "physical_worker_compatibility_sha256": (
                self.physical_worker_compatibility_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "runtime_inputs_replayed": self.runtime_inputs_replayed,
            "development_rl_training_inputs_authorized": (
                self.development_rl_training_inputs_authorized
            ),
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
    def development_stage_authorized(self) -> bool:
        return bool(
            self.development_rl_training_inputs_authorized
            and self.source_data_qualified
            and self.runtime_inputs_replayed
            and self.source_transaction_verified
        )

    def validate(self) -> None:
        if any(
            type(authority) is not MassiveAdaptiveRLFoldFitInputsAuthorityV1
            for authority in self._fold_fit_inputs
        ):
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit inputs require exact persisted fold-input authorities"
            )
        for authority in self._fold_fit_inputs:
            authority.validate()
        environments = tuple(
            authority.execution_environment_authority
            for authority in self._fold_fit_inputs
        )
        expected_authorized = bool(
            len(self._fold_fit_inputs) == 4
            and all(
                authority.development_stage_authorized
                for authority in self._fold_fit_inputs
            )
        )
        fingerprints = tuple(
            environment.scientific_execution_fingerprint_sha256
            for environment in environments
        )
        worker_compatibility = tuple(
            environment.physical_worker_compatibility_sha256
            for environment in environments
        )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or self.fold_indices != _FOUR_FOLD_INDICES
            or tuple(row.outer_fold_index for row in self._fold_fit_inputs)
            != self.fold_indices
            or tuple(row.experiment_id for row in self._fold_fit_inputs)
            != (self.experiment_id,) * 4
            or tuple(
                row.manifest_v3_receipt_sha256 for row in self._fold_fit_inputs
            )
            != (self.manifest_v3_receipt_sha256,) * 4
            or tuple(
                row.runtime_sources_receipt_sha256 for row in self._fold_fit_inputs
            )
            != (self.runtime_sources_receipt_sha256,) * 4
            or tuple(
                row.runtime_graph_witness_receipt_sha256
                for row in self._fold_fit_inputs
            )
            != (self.runtime_graph_witness_receipt_sha256,) * 4
            or self.fold_fit_input_authority_receipts
            != tuple(row.semantic_receipt_sha256 for row in self._fold_fit_inputs)
            or self.execution_environment_authority_receipts
            != tuple(row.semantic_receipt_sha256 for row in environments)
            or not fingerprints
            or len(set(fingerprints)) != 1
            or self.scientific_execution_fingerprint_sha256 != fingerprints[0]
            or not worker_compatibility
            or len(set(worker_compatibility)) != 1
            or self.physical_worker_compatibility_sha256
            != worker_compatibility[0]
            or not expected_authorized
            or self.source_data_qualified != expected_authorized
            or self.runtime_inputs_replayed != expected_authorized
            or self.development_rl_training_inputs_authorized
            != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit inputs authority differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit-input source transaction differs"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.runtime_sources_receipt_sha256,
            self.runtime_graph_witness_receipt_sha256,
            *self.fold_fit_input_authority_receipts,
            *self.execution_environment_authority_receipts,
            self.scientific_execution_fingerprint_sha256,
            self.physical_worker_compatibility_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL four-fold fit inputs", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFourFoldFitAuthorityV1:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    four_fold_fit_inputs_authority_receipt_sha256: str
    fold_indices: tuple[int, ...]
    fold_fit_authority_receipts: tuple[str, ...]
    fold_fit_input_authority_receipts: tuple[str, ...]
    execution_environment_authority_receipts: tuple[str, ...]
    scientific_execution_fingerprint_sha256: str
    physical_worker_compatibility_sha256: str
    source_data_qualified: bool
    runtime_fit_replayed: bool
    semantic_receipt_sha256: str
    development_rl_training_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA
    _fit_inputs_authority: MassiveAdaptiveRLFourFoldFitInputsAuthorityV1 | None = (
        field(default=None, repr=False, compare=False)
    )
    _fold_fits: tuple[MassiveAdaptiveRLFoldFitAuthorityV1, ...] = field(
        default=(), repr=False, compare=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, repr=False, compare=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "manifest_v3_receipt_sha256": self.manifest_v3_receipt_sha256,
            "runtime_sources_receipt_sha256": self.runtime_sources_receipt_sha256,
            "runtime_graph_witness_receipt_sha256": (
                self.runtime_graph_witness_receipt_sha256
            ),
            "four_fold_fit_inputs_authority_receipt_sha256": (
                self.four_fold_fit_inputs_authority_receipt_sha256
            ),
            "fold_indices": self.fold_indices,
            "fold_fit_authority_receipts": self.fold_fit_authority_receipts,
            "fold_fit_input_authority_receipts": (
                self.fold_fit_input_authority_receipts
            ),
            "execution_environment_authority_receipts": (
                self.execution_environment_authority_receipts
            ),
            "scientific_execution_fingerprint_sha256": (
                self.scientific_execution_fingerprint_sha256
            ),
            "physical_worker_compatibility_sha256": (
                self.physical_worker_compatibility_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "runtime_fit_replayed": self.runtime_fit_replayed,
            "development_rl_training_authorized": (
                self.development_rl_training_authorized
            ),
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
    def development_stage_authorized(self) -> bool:
        return bool(
            self.development_rl_training_authorized
            and self.source_data_qualified
            and self.runtime_fit_replayed
            and self.source_transaction_verified
        )

    def fold_fit(self, fold_index: int) -> MassiveAdaptiveRLFoldFitAuthorityV1:
        """Return one canonical persisted fold from the replayed aggregate."""

        self.validate()
        if fold_index not in self.fold_indices:
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit child is absent"
            )
        result = self._fold_fits[fold_index]
        if (
            result.outer_fold_index != fold_index
            or result.semantic_receipt_sha256
            != self.fold_fit_authority_receipts[fold_index]
        ):
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit child differs"
            )
        return result

    def validate(self) -> None:
        if type(self._fit_inputs_authority) is not (
            MassiveAdaptiveRLFourFoldFitInputsAuthorityV1
        ) or any(
            type(authority) is not MassiveAdaptiveRLFoldFitAuthorityV1
            for authority in self._fold_fits
        ):
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit requires exact persisted fold authorities"
            )
        if self._fit_inputs_authority is not None:
            self._fit_inputs_authority.validate()
        for authority in self._fold_fits:
            authority.validate()
        if self._loaded_source is not None:
            self._loaded_source.validate()
        expected_authorized = bool(
            self._fit_inputs_authority is not None
            and self._fit_inputs_authority.development_stage_authorized
            and len(self._fold_fits) == 4
            and all(
                authority.development_stage_authorized
                for authority in self._fold_fits
            )
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA
            or self._fit_inputs_authority is None
            or self._fit_inputs_authority.experiment_id != self.experiment_id
            or self._fit_inputs_authority.manifest_v3_receipt_sha256
            != self.manifest_v3_receipt_sha256
            or self._fit_inputs_authority.runtime_sources_receipt_sha256
            != self.runtime_sources_receipt_sha256
            or self._fit_inputs_authority.runtime_graph_witness_receipt_sha256
            != self.runtime_graph_witness_receipt_sha256
            or self.fold_indices != _FOUR_FOLD_INDICES
            or tuple(row.outer_fold_index for row in self._fold_fits)
            != self.fold_indices
            or tuple(row.experiment_id for row in self._fold_fits)
            != (self.experiment_id,) * 4
            or tuple(row.manifest_v3_receipt_sha256 for row in self._fold_fits)
            != (self.manifest_v3_receipt_sha256,) * 4
            or tuple(row.runtime_sources_receipt_sha256 for row in self._fold_fits)
            != (self.runtime_sources_receipt_sha256,) * 4
            or tuple(
                row.runtime_graph_witness_receipt_sha256 for row in self._fold_fits
            )
            != (self.runtime_graph_witness_receipt_sha256,) * 4
            or self.four_fold_fit_inputs_authority_receipt_sha256
            != self._fit_inputs_authority.semantic_receipt_sha256
            or self.fold_fit_authority_receipts
            != tuple(row.semantic_receipt_sha256 for row in self._fold_fits)
            or self.fold_fit_input_authority_receipts
            != tuple(
                row.fit_inputs_authority.semantic_receipt_sha256
                for row in self._fold_fits
            )
            or self.fold_fit_input_authority_receipts
            != self._fit_inputs_authority.fold_fit_input_authority_receipts
            or self.execution_environment_authority_receipts
            != tuple(
                row.execution_environment_authority.semantic_receipt_sha256
                for row in self._fold_fits
            )
            or self.execution_environment_authority_receipts
            != self._fit_inputs_authority.execution_environment_authority_receipts
            or self.scientific_execution_fingerprint_sha256
            != self._fit_inputs_authority.scientific_execution_fingerprint_sha256
            or self.physical_worker_compatibility_sha256
            != self._fit_inputs_authority.physical_worker_compatibility_sha256
            or not expected_authorized
            or self.source_data_qualified != expected_authorized
            or self.runtime_fit_replayed != expected_authorized
            or self.development_rl_training_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit authority differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit source transaction differs"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.runtime_sources_receipt_sha256,
            self.runtime_graph_witness_receipt_sha256,
            self.four_fold_fit_inputs_authority_receipt_sha256,
            *self.fold_fit_authority_receipts,
            *self.fold_fit_input_authority_receipts,
            *self.execution_environment_authority_receipts,
            self.scientific_execution_fingerprint_sha256,
            self.physical_worker_compatibility_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL four-fold fit", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    fold_fit_inputs: Sequence[MassiveAdaptiveRLFoldFitInputsAuthorityV1],
) -> MassiveAdaptiveRLFourFoldFitInputsAuthorityV1:
    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV3 or type(
        runtime_sources
    ) is not MassiveAdaptiveRLRuntimeSourcesV1:
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit inputs require exact manifest and runtime sources"
        )
    manifest.validate()
    runtime_sources.validate()
    rows = tuple(fold_fit_inputs)
    if any(
        type(row) is not MassiveAdaptiveRLFoldFitInputsAuthorityV1 for row in rows
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit inputs require exact persisted fold-input authorities"
        )
    environments = tuple(row.execution_environment_authority for row in rows)
    fingerprint = (
        environments[0].scientific_execution_fingerprint_sha256
        if environments
        else "0" * 64
    )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": _runtime_witness_receipt(
            runtime_sources
        ),
        "fold_indices": tuple(row.outer_fold_index for row in rows),
        "fold_fit_input_authority_receipts": tuple(
            row.semantic_receipt_sha256 for row in rows
        ),
        "execution_environment_authority_receipts": tuple(
            row.semantic_receipt_sha256 for row in environments
        ),
        "scientific_execution_fingerprint_sha256": fingerprint,
        "physical_worker_compatibility_sha256": (
            environments[0].physical_worker_compatibility_sha256
            if environments
            else "0" * 64
        ),
        "source_data_qualified": True,
        "runtime_inputs_replayed": True,
        "development_rl_training_inputs_authorized": True,
    }
    provisional = MassiveAdaptiveRLFourFoldFitInputsAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        _fold_fit_inputs=rows,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def build_massive_adaptive_rl_four_fold_fit_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    fit_inputs_authority: MassiveAdaptiveRLFourFoldFitInputsAuthorityV1,
    fold_fits: Sequence[MassiveAdaptiveRLFoldFitAuthorityV1],
) -> MassiveAdaptiveRLFourFoldFitAuthorityV1:
    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV3 or type(
        runtime_sources
    ) is not MassiveAdaptiveRLRuntimeSourcesV1:
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit requires exact manifest and runtime sources"
        )
    manifest.validate()
    runtime_sources.validate()
    rows = tuple(fold_fits)
    if type(fit_inputs_authority) is not (
        MassiveAdaptiveRLFourFoldFitInputsAuthorityV1
    ) or any(type(row) is not MassiveAdaptiveRLFoldFitAuthorityV1 for row in rows):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit requires exact persisted fold authorities"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": _runtime_witness_receipt(
            runtime_sources
        ),
        "four_fold_fit_inputs_authority_receipt_sha256": (
            fit_inputs_authority.semantic_receipt_sha256
        ),
        "fold_indices": tuple(row.outer_fold_index for row in rows),
        "fold_fit_authority_receipts": tuple(
            row.semantic_receipt_sha256 for row in rows
        ),
        "fold_fit_input_authority_receipts": tuple(
            row.fit_inputs_authority.semantic_receipt_sha256 for row in rows
        ),
        "execution_environment_authority_receipts": tuple(
            row.execution_environment_authority.semantic_receipt_sha256
            for row in rows
        ),
        "scientific_execution_fingerprint_sha256": (
            fit_inputs_authority.scientific_execution_fingerprint_sha256
        ),
        "physical_worker_compatibility_sha256": (
            fit_inputs_authority.physical_worker_compatibility_sha256
        ),
        "source_data_qualified": True,
        "runtime_fit_replayed": True,
        "development_rl_training_authorized": True,
    }
    provisional = MassiveAdaptiveRLFourFoldFitAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        _fit_inputs_authority=fit_inputs_authority,
        _fold_fits=rows,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def four_fold_fit_inputs_authority_relative_path_v1(*, experiment_id: str) -> str:
    if not experiment_id or any(
        not (character.isalnum() or character in "-_") for character in experiment_id
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit experiment ID is not path safe"
        )
    return (
        "massive-adaptive/rl-four-fold-fit-inputs-authority-v1/"
        f"{experiment_id}.json"
    )


def four_fold_fit_authority_relative_path_v1(*, experiment_id: str) -> str:
    four_fold_fit_inputs_authority_relative_path_v1(experiment_id=experiment_id)
    return (
        "massive-adaptive/rl-four-fold-fit-authority-v1/"
        f"{experiment_id}.json"
    )


def _payload(authority: object) -> dict[str, object]:
    validate = getattr(authority, "validate")
    validate()
    unsigned = getattr(authority, "semantic_unsigned")()
    return {
        **unsigned,
        "semantic_receipt_sha256": getattr(authority, "semantic_receipt_sha256"),
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit aggregate is not canonical JSON"
        )
    return dict(cast(Mapping[str, object], value))


def _materialize(
    *,
    root: str | Path,
    relative: str,
    authority: object,
    dataset_id: str,
    source_schema_sha256: str,
    committed_at_ms: int,
    request_id: str,
) -> LoadedMassiveSourceObject:
    resolved = Path(root)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit artifact root is unavailable"
        ) from error
    semantic_receipt = cast(str, getattr(authority, "semantic_receipt_sha256"))
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(authority))),
        root=resolved,
        relative_payload_path=relative,
        dataset_id=dataset_id,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=source_schema_sha256,
        entitlement_receipt_sha256=semantic_receipt,
        committed_at_ms=committed_at_ms,
        request_id=request_id,
    )
    return load_massive_source_bundle(
        root=resolved,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )


def materialize_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFourFoldFitInputsAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFourFoldFitInputsAuthorityV1:
    relative = four_fold_fit_inputs_authority_relative_path_v1(
        experiment_id=authority.experiment_id
    )
    loaded = _materialize(
        root=root,
        relative=relative,
        authority=authority,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_DATASET,
        source_schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-FOUR-FOLD-FIT-INPUTS-V1-{authority.experiment_id}",
    )
    if canonical_json_file_bytes(_load_payload(root=root, loaded_source=loaded)) != (
        canonical_json_file_bytes(_payload(authority))
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "published adaptive RL four-fold fit inputs differ"
        )
    result = replace(authority, _loaded_source=loaded)
    result.validate()
    return result


def load_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    verified_at_ms: int,
    device: torch.device | str | None = None,
) -> MassiveAdaptiveRLFourFoldFitInputsAuthorityV1:
    fold_fit_inputs = tuple(
        prepare_or_resume_massive_adaptive_rl_fold_fit_inputs_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            outer_fold_index=fold_index,
            artifact_root=root,
            committed_at_ms=verified_at_ms + fold_index,
            device=device,
            allow_materialize=False,
        )
        for fold_index in _FOUR_FOLD_INDICES
    )
    rebuilt = build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        fold_fit_inputs=fold_fit_inputs,
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=four_fold_fit_inputs_authority_relative_path_v1(
            experiment_id=manifest.experiment_id
        ),
        verified_at_ms=verified_at_ms,
    )
    if canonical_json_file_bytes(_load_payload(root=root, loaded_source=loaded)) != (
        canonical_json_file_bytes(_payload(rebuilt))
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit inputs did not replay"
        )
    result = replace(rebuilt, _loaded_source=loaded)
    result.validate()
    return result


def materialize_massive_adaptive_rl_four_fold_fit_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFourFoldFitAuthorityV1:
    relative = four_fold_fit_authority_relative_path_v1(
        experiment_id=authority.experiment_id
    )
    loaded = _materialize(
        root=root,
        relative=relative,
        authority=authority,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_DATASET,
        source_schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-FOUR-FOLD-FIT-V1-{authority.experiment_id}",
    )
    if canonical_json_file_bytes(_load_payload(root=root, loaded_source=loaded)) != (
        canonical_json_file_bytes(_payload(authority))
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "published adaptive RL four-fold fit authority differs"
        )
    result = replace(authority, _loaded_source=loaded)
    result.validate()
    return result


def load_massive_adaptive_rl_four_fold_fit_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    verified_at_ms: int,
    device: torch.device | str | None = None,
) -> MassiveAdaptiveRLFourFoldFitAuthorityV1:
    fit_inputs_authority = (
        load_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            root=root,
            manifest=manifest,
            runtime_sources=runtime_sources,
            verified_at_ms=verified_at_ms,
            device=device,
        )
    )
    fold_fits = tuple(
        load_massive_adaptive_rl_fold_fit_authority_v1(
            root=root,
            manifest=manifest,
            runtime_sources=runtime_sources,
            outer_fold_index=fold_index,
            committed_at_ms=verified_at_ms + fold_index,
            device=device,
        )
        for fold_index in _FOUR_FOLD_INDICES
    )
    rebuilt = build_massive_adaptive_rl_four_fold_fit_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        fit_inputs_authority=fit_inputs_authority,
        fold_fits=fold_fits,
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=four_fold_fit_authority_relative_path_v1(
            experiment_id=manifest.experiment_id
        ),
        verified_at_ms=verified_at_ms,
    )
    if canonical_json_file_bytes(_load_payload(root=root, loaded_source=loaded)) != (
        canonical_json_file_bytes(_payload(rebuilt))
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit authority did not replay"
        )
    result = replace(rebuilt, _loaded_source=loaded)
    result.validate()
    return result


@contextmanager
def _four_fold_fit_execution_lease(
    *, root: str | Path, experiment_id: str, stage: str
) -> Iterator[None]:
    four_fold_fit_authority_relative_path_v1(experiment_id=experiment_id)
    if stage not in {"inputs", "fit"}:
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit lease stage differs"
        )
    resolved_root = Path(root)
    lease_directory = (
        resolved_root / "massive-adaptive" / "rl-four-fold-fit-leases-v1"
    )
    try:
        resolved_root.mkdir(parents=True, exist_ok=True)
        if resolved_root.is_symlink():
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit artifact root is a symlink"
            )
        lease_directory.mkdir(parents=True, exist_ok=True)
        if lease_directory.is_symlink():
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit lease directory is a symlink"
            )
        descriptor = os.open(
            lease_directory / f"{experiment_id}-{stage}.lock",
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
    except OSError as error:
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit execution lease is unavailable"
        ) from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit execution lease identity differs"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MassiveAdaptiveRLFourFoldFitExecutionLeaseUnavailable(
                "adaptive RL four-fold fit execution lease is already held"
            ) from error
        except OSError as error:
            raise MassiveAdaptiveRLFourFoldFitV1Error(
                "adaptive RL four-fold fit execution lease failed"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _run_or_resume_massive_adaptive_rl_four_fold_fit_inputs_v1_unlocked(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFourFoldFitInputsAuthorityV1:
    """Authorize every fold input before any PPO fold is allowed to run."""

    relative = four_fold_fit_inputs_authority_relative_path_v1(
        experiment_id=manifest.experiment_id
    )
    aggregate_exists = _source_transaction_exists(
        root=artifact_root, relative=relative
    )
    if not allow_materialize and not aggregate_exists:
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "completed adaptive RL four-fold fit-input aggregate is absent"
        )
    if aggregate_exists:
        return load_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            root=artifact_root,
            manifest=manifest,
            runtime_sources=runtime_sources,
            verified_at_ms=committed_at_ms + 4,
            device=device,
        )
    fold_inputs = tuple(
        prepare_or_resume_massive_adaptive_rl_fold_fit_inputs_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            outer_fold_index=fold_index,
            artifact_root=artifact_root,
            committed_at_ms=committed_at_ms + fold_index,
            device=device,
            allow_materialize=True,
        )
        for fold_index in _FOUR_FOLD_INDICES
    )
    authority = build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        fold_fit_inputs=fold_inputs,
    )
    return materialize_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
        root=artifact_root,
        authority=authority,
        committed_at_ms=committed_at_ms + 4,
    )


def run_or_resume_massive_adaptive_rl_four_fold_fit_inputs_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFourFoldFitInputsAuthorityV1:
    """Authorize all fold inputs under one cross-process root-stage lease."""

    with _four_fold_fit_execution_lease(
        root=artifact_root,
        experiment_id=manifest.experiment_id,
        stage="inputs",
    ):
        return _run_or_resume_massive_adaptive_rl_four_fold_fit_inputs_v1_unlocked(
            manifest=manifest,
            runtime_sources=runtime_sources,
            artifact_root=artifact_root,
            committed_at_ms=committed_at_ms,
            device=device,
            allow_materialize=allow_materialize,
        )


def _run_or_resume_massive_adaptive_rl_four_fold_fit_v1_unlocked(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    fit_inputs_authority: MassiveAdaptiveRLFourFoldFitInputsAuthorityV1,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFourFoldFitAuthorityV1:
    """Run or strictly replay folds 0-3 and publish their completed aggregate."""

    if type(fit_inputs_authority) is not (
        MassiveAdaptiveRLFourFoldFitInputsAuthorityV1
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit requires an exact persisted input aggregate"
        )
    fit_inputs_authority.validate()
    if (
        not fit_inputs_authority.development_stage_authorized
        or fit_inputs_authority.experiment_id != manifest.experiment_id
        or fit_inputs_authority.manifest_v3_receipt_sha256
        != manifest.semantic_receipt_sha256
        or fit_inputs_authority.runtime_sources_receipt_sha256
        != runtime_sources.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit input roots differ"
        )

    canonical_fit_inputs = (
        load_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            root=artifact_root,
            manifest=manifest,
            runtime_sources=runtime_sources,
            verified_at_ms=committed_at_ms,
            device=device,
        )
    )
    if (
        canonical_fit_inputs.semantic_receipt_sha256
        != fit_inputs_authority.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit canonical input aggregate differs"
        )
    fit_inputs_authority = canonical_fit_inputs

    relative = four_fold_fit_authority_relative_path_v1(
        experiment_id=manifest.experiment_id
    )
    aggregate_exists = _source_transaction_exists(
        root=artifact_root, relative=relative
    )
    if not allow_materialize and not aggregate_exists:
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "completed adaptive RL four-fold fit aggregate is absent"
        )
    if aggregate_exists:
        return load_massive_adaptive_rl_four_fold_fit_authority_v1(
            root=artifact_root,
            manifest=manifest,
            runtime_sources=runtime_sources,
            verified_at_ms=committed_at_ms + 401,
            device=device,
        )
    if allow_materialize:
        fold_fits = tuple(
            run_or_resume_massive_adaptive_rl_fold_fit_v1(
                manifest=manifest,
                runtime_sources=runtime_sources,
                outer_fold_index=fold_index,
                artifact_root=artifact_root,
                committed_at_ms=committed_at_ms + fold_index * 100,
                device=device,
            )
            for fold_index in _FOUR_FOLD_INDICES
        )
    else:
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "completed adaptive RL four-fold fit aggregate is absent"
        )
    authority = build_massive_adaptive_rl_four_fold_fit_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        fit_inputs_authority=fit_inputs_authority,
        fold_fits=fold_fits,
    )
    return materialize_massive_adaptive_rl_four_fold_fit_authority_v1(
        root=artifact_root,
        authority=authority,
        committed_at_ms=committed_at_ms + 401,
    )


def run_or_resume_massive_adaptive_rl_four_fold_fit_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    fit_inputs_authority: MassiveAdaptiveRLFourFoldFitInputsAuthorityV1,
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str | None = None,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFourFoldFitAuthorityV1:
    """Run or replay all folds under one cross-process root-stage lease."""

    with _four_fold_fit_execution_lease(
        root=artifact_root,
        experiment_id=manifest.experiment_id,
        stage="fit",
    ):
        return _run_or_resume_massive_adaptive_rl_four_fold_fit_v1_unlocked(
            manifest=manifest,
            runtime_sources=runtime_sources,
            fit_inputs_authority=fit_inputs_authority,
            artifact_root=artifact_root,
            committed_at_ms=committed_at_ms,
            device=device,
            allow_materialize=allow_materialize,
        )


def advance_massive_adaptive_rl_four_fold_fit_inputs_state_v1(
    *,
    artifact_root: str | Path,
    previous: MassiveAdaptiveRLExperimentStateV2,
    authority: MassiveAdaptiveRLFourFoldFitInputsAuthorityV1,
) -> MassiveAdaptiveRLExperimentStateV2:
    authority.validate()
    if (
        not authority.development_stage_authorized
        or previous.experiment_id != authority.experiment_id
        or previous.manifest_receipt_sha256
        != authority.manifest_v3_receipt_sha256
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold fit-input state roots differ"
        )
    return advance_massive_adaptive_rl_experiment_state_v2(
        artifact_root=artifact_root,
        previous=previous,
        stage=MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED,
        stage_artifact_receipt_sha256=authority.semantic_receipt_sha256,
    )


def advance_massive_adaptive_rl_four_fold_fit_state_v1(
    *,
    artifact_root: str | Path,
    previous: MassiveAdaptiveRLExperimentStateV2,
    authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
) -> MassiveAdaptiveRLExperimentStateV2:
    authority.validate()
    if (
        not authority.development_stage_authorized
        or previous.experiment_id != authority.experiment_id
        or previous.manifest_receipt_sha256
        != authority.manifest_v3_receipt_sha256
        or previous.last_completed_stage
        is not MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED
        or previous.last_completed_stage_artifact_receipt_sha256
        != authority.four_fold_fit_inputs_authority_receipt_sha256
    ):
        raise MassiveAdaptiveRLFourFoldFitV1Error(
            "adaptive RL four-fold completed-fit state roots differ"
        )
    return advance_massive_adaptive_rl_experiment_state_v2(
        artifact_root=artifact_root,
        previous=previous,
        stage=MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED,
        stage_artifact_receipt_sha256=authority.semantic_receipt_sha256,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_INPUTS_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFourFoldFitAuthorityV1",
    "MassiveAdaptiveRLFourFoldFitExecutionLeaseUnavailable",
    "MassiveAdaptiveRLFourFoldFitInputsAuthorityV1",
    "MassiveAdaptiveRLFourFoldFitV1Error",
    "advance_massive_adaptive_rl_four_fold_fit_inputs_state_v1",
    "advance_massive_adaptive_rl_four_fold_fit_state_v1",
    "build_massive_adaptive_rl_four_fold_fit_authority_v1",
    "build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1",
    "four_fold_fit_authority_relative_path_v1",
    "four_fold_fit_inputs_authority_relative_path_v1",
    "load_massive_adaptive_rl_four_fold_fit_authority_v1",
    "load_massive_adaptive_rl_four_fold_fit_inputs_authority_v1",
    "materialize_massive_adaptive_rl_four_fold_fit_authority_v1",
    "materialize_massive_adaptive_rl_four_fold_fit_inputs_authority_v1",
    "run_or_resume_massive_adaptive_rl_four_fold_fit_inputs_v1",
    "run_or_resume_massive_adaptive_rl_four_fold_fit_v1",
]
