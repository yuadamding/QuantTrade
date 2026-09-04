"""Prequential validation plan and initial fold-0/1 input commitment.

The registered split is not an all-folds-before-outer design.  Fold-2
validation is exactly outer fold 0 and fold-3 validation is exactly outer
fold 1.  Consequently, only validation folds 0 and 1 may be materialized at
the initial development boundary.  Later validation-input generations must
be downstream of sealed outer-fold evidence and receive distinct schemas.

This module deliberately does not adapt the legacy all-four barrier.  It
publishes one create-only initial authority containing only the two causally
available validation tapes and rejects any artifact tree in which fold 2 or
3 inputs, legacy all-four barriers, or validation outcomes already exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from io import BytesIO
import fcntl
import json
import os
from pathlib import Path
import stat
import time
from typing import Iterator, cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v1 import (
    four_fold_validation_inputs_authority_relative_path_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    four_fold_validation_inputs_authority_relative_path_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    massive_adaptive_rl_validation_downstream_evidence_exists_v1,
    validation_decision_tensor_relative_path_v1,
    validation_environment_registry_relative_path_v1,
    validation_forecast_archive_relative_path_v1,
    validation_sources_authority_relative_path_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v2 import (
    MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    MassiveAdaptiveRLValidationSourcesAuthorityV2,
    massive_adaptive_rl_validation_downstream_evidence_exists_v2,
    prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v2,
    prepare_or_resume_massive_adaptive_rl_validation_sources_v2,
    validation_environment_registry_relative_path_v2,
    validation_sources_authority_relative_path_v2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_INNER_VALIDATION_SESSIONS_V1,
    MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1,
    MassiveAdaptiveSplitPlanV1,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1,
    MassiveAdaptiveRLExperimentManifestV4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    MassiveAdaptiveRLManifestV5WriterCapabilityV1,
    manifest_v5_compatibility_writer_guard_v1,
)


MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-prequential-validation-plan-v1"
)
MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-initial-validation-inputs-authority-v1"
)
MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-initial-validation-inputs-authority-v1"
)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_INPUTS_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA,
            "encoding": "canonical-json-prequential-initial-fold-zero-one-inputs",
            "generic_reload": "nonauthorizing",
        }
    )
)

_FOLD_INDICES = (0, 1, 2, 3)
_INITIAL_FOLD_INDICES = (0, 1)
_WITHHELD_FOLD_INDICES = (2, 3)
_VALIDATION_RELEASE_PREREQUISITES: tuple[int | None, ...] = (None, None, 0, 1)
_OUTER_TO_VALIDATION_OVERLAP_EDGES = ((0, 2), (1, 3))
_PREQUENTIAL_STAGE_NAMES = (
    "initial-validation-inputs-folds-0-1-committed",
    "selection-0-frozen",
    "selection-1-frozen",
    "outer-0-sealed",
    "validation-inputs-fold-2-committed",
    "selection-2-frozen",
    "outer-1-sealed",
    "validation-inputs-fold-3-committed",
    "selection-3-frozen",
    "outer-2-sealed",
    "outer-3-sealed",
    "development-profitability-report-published",
)


class MassiveAdaptiveRLPolicyScheduleDispositionV1(str, Enum):
    FOUR_FOLD_POLICY_SCHEDULE_QUALIFIED = "four-fold-policy-schedule-qualified"
    DIAGNOSTIC_ONLY_POLICY_SCHEDULE = "diagnostic-only-policy-schedule"
    INVALID_FOUR_FOLD_EVIDENCE = "invalid-four-fold-evidence"


class MassiveAdaptiveRLPrequentialValidationInputsV1Error(ValueError):
    """The prequential validation geometry or initial input boundary differs."""


class MassiveAdaptiveRLInitialValidationInputsLeaseUnavailable(
    MassiveAdaptiveRLPrequentialValidationInputsV1Error
):
    """Another process owns the initial prequential input generation."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _wall_clock_ms() -> int:
    value = time.time_ns() // 1_000_000
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "prequential validation publication clock differs"
        )
    return value


def _next_publication_ms(previous: int) -> int:
    return max(_wall_clock_ms(), previous + 1)


def policy_schedule_disposition_v1(
    selected_candidate_validation_eligible: Sequence[bool],
) -> MassiveAdaptiveRLPolicyScheduleDispositionV1:
    """Apply Manifest V4's diagnostic continuation to a complete schedule."""

    values = tuple(selected_candidate_validation_eligible)
    if len(values) != 4 or any(type(value) is not bool for value in values):
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "policy-schedule eligibility inventory differs"
        )
    if all(values):
        return (
            MassiveAdaptiveRLPolicyScheduleDispositionV1.FOUR_FOLD_POLICY_SCHEDULE_QUALIFIED
        )
    return MassiveAdaptiveRLPolicyScheduleDispositionV1.DIAGNOSTIC_ONLY_POLICY_SCHEDULE


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPrequentialValidationPlanV1:
    manifest_v4_receipt_sha256: str
    split_plan_receipt_sha256: str
    fold_indices: tuple[int, ...]
    initial_validation_fold_indices: tuple[int, ...]
    withheld_validation_fold_indices: tuple[int, ...]
    validation_release_prerequisite_outer_fold_indices: tuple[int | None, ...]
    outer_to_validation_overlap_edges: tuple[tuple[int, int], ...]
    validation_session_date_inventories: tuple[tuple[str, ...], ...]
    outer_session_date_inventories: tuple[tuple[str, ...], ...]
    execution_stage_names: tuple[str, ...]
    no_eligible_candidate_policy: str
    qualified_schedule_disposition: str
    diagnostic_schedule_disposition: str
    invalid_evidence_disposition: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if descriptor.name != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        validation = self.validation_session_date_inventories
        outer = self.outer_session_date_inventories
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SCHEMA
            or self.fold_indices != _FOLD_INDICES
            or self.initial_validation_fold_indices != _INITIAL_FOLD_INDICES
            or self.withheld_validation_fold_indices != _WITHHELD_FOLD_INDICES
            or self.validation_release_prerequisite_outer_fold_indices
            != _VALIDATION_RELEASE_PREREQUISITES
            or self.outer_to_validation_overlap_edges
            != _OUTER_TO_VALIDATION_OVERLAP_EDGES
            or len(validation) != 4
            or len(outer) != 4
            or any(
                len(row) != MASSIVE_ADAPTIVE_INNER_VALIDATION_SESSIONS_V1
                or row != tuple(sorted(set(row)))
                for row in validation
            )
            or any(
                len(row) != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
                or row != tuple(sorted(set(row)))
                for row in outer
            )
            or validation[2] != outer[0]
            or validation[3] != outer[1]
            or any(set(validation[index]).intersection(outer[0]) for index in (0, 1))
            or any(set(validation[index]).intersection(outer[1]) for index in (0, 1))
            or any(set(left).intersection(right) for left, right in zip(outer, outer[1:]))
            or self.execution_stage_names != _PREQUENTIAL_STAGE_NAMES
            or self.no_eligible_candidate_policy
            != MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
            or self.qualified_schedule_disposition
            != MassiveAdaptiveRLPolicyScheduleDispositionV1.FOUR_FOLD_POLICY_SCHEDULE_QUALIFIED.value
            or self.diagnostic_schedule_disposition
            != MassiveAdaptiveRLPolicyScheduleDispositionV1.DIAGNOSTIC_ONLY_POLICY_SCHEDULE.value
            or self.invalid_evidence_disposition
            != MassiveAdaptiveRLPolicyScheduleDispositionV1.INVALID_FOUR_FOLD_EVIDENCE.value
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "prequential validation plan differs"
            )
        for value in (
            self.manifest_v4_receipt_sha256,
            self.split_plan_receipt_sha256,
            self.protocol_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("prequential validation plan", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())

    def released_validation_folds(
        self, *, sealed_outer_fold_indices: Sequence[int]
    ) -> tuple[int, ...]:
        """Return the validation folds causally released by a sealed prefix."""

        sealed = tuple(sealed_outer_fold_indices)
        if (
            any(isinstance(value, bool) or value not in _FOLD_INDICES for value in sealed)
            or sealed != tuple(range(len(sealed)))
        ):
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "sealed outer-fold inventory is not a canonical prefix"
            )
        return tuple(
            fold_index
            for fold_index, prerequisite in zip(
                self.fold_indices,
                self.validation_release_prerequisite_outer_fold_indices,
                strict=True,
            )
            if prerequisite is None or prerequisite in sealed
        )


def build_massive_adaptive_rl_prequential_validation_plan_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    split_plan: MassiveAdaptiveSplitPlanV1,
) -> MassiveAdaptiveRLPrequentialValidationPlanV1:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(split_plan) is not MassiveAdaptiveSplitPlanV1
    ):
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "prequential validation plan requires exact root types"
        )
    manifest.validate()
    split_plan.validate()
    folds = split_plan.outer_folds
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SCHEMA,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "fold_indices": _FOLD_INDICES,
        "initial_validation_fold_indices": _INITIAL_FOLD_INDICES,
        "withheld_validation_fold_indices": _WITHHELD_FOLD_INDICES,
        "validation_release_prerequisite_outer_fold_indices": (
            _VALIDATION_RELEASE_PREREQUISITES
        ),
        "outer_to_validation_overlap_edges": _OUTER_TO_VALIDATION_OVERLAP_EDGES,
        "validation_session_date_inventories": tuple(
            fold.inner_validation_session_dates for fold in folds
        ),
        "outer_session_date_inventories": tuple(
            fold.outer_test_session_dates for fold in folds
        ),
        "execution_stage_names": _PREQUENTIAL_STAGE_NAMES,
        "no_eligible_candidate_policy": manifest.no_eligible_candidate_policy,
        "qualified_schedule_disposition": (
            MassiveAdaptiveRLPolicyScheduleDispositionV1.FOUR_FOLD_POLICY_SCHEDULE_QUALIFIED.value
        ),
        "diagnostic_schedule_disposition": (
            MassiveAdaptiveRLPolicyScheduleDispositionV1.DIAGNOSTIC_ONLY_POLICY_SCHEDULE.value
        ),
        "invalid_evidence_disposition": (
            MassiveAdaptiveRLPolicyScheduleDispositionV1.INVALID_FOUR_FOLD_EVIDENCE.value
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLPrequentialValidationPlanV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "folds": _FOLD_INDICES,
        "initial_validation_folds": _INITIAL_FOLD_INDICES,
        "release_prerequisites": _VALIDATION_RELEASE_PREREQUISITES,
        "overlap_edges": _OUTER_TO_VALIDATION_OVERLAP_EDGES,
        "execution_stage_names": _PREQUENTIAL_STAGE_NAMES,
        "ineligible_schedule": (
            MassiveAdaptiveRLPolicyScheduleDispositionV1.DIAGNOSTIC_ONLY_POLICY_SCHEDULE.value
        ),
        "positive_authorization": "all-four-selections-validation-eligible",
        "selective_stopping": False,
    }
)
MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256 = semantic_sha256(
    {
        "plan_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256
        ),
        "input_generation": "validation-complete-runtime-sources-v2",
        "initial_folds": _INITIAL_FOLD_INDICES,
        "withheld_folds": _WITHHELD_FOLD_INDICES,
        "ordering": "both-sources-before-both-registries-before-initial-barrier",
        "legacy_all_four_barrier": "rejected",
        "legacy_validation_outcomes": "rejected",
        "later_fold_inputs": "absent-until-sealed-outer-prerequisite",
        "publication": "manifest-derived-create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "outer_access": False,
        "profitability_reporting": False,
    }
)


def initial_validation_inputs_authority_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> str:
    manifest.validate()
    return (
        "massive-adaptive/rl-prequential-initial-validation-inputs-authority-v1/"
        f"v4-{manifest.semantic_receipt_sha256}.json"
    )


def _transaction_state(*, root: str | Path, relative: str) -> tuple[bool, bool]:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    return all(present), any(present) and not all(present)


def _transaction_exists(*, root: str | Path, relative: str) -> bool:
    complete, partial = _transaction_state(root=root, relative=relative)
    if partial:
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "prequential validation source transaction is incomplete"
        )
    return complete


def _forbidden_prequential_artifacts(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> tuple[str, ...]:
    """Return artifacts proving the all-at-once or later-fold path was opened."""

    candidates = [
        four_fold_validation_inputs_authority_relative_path_v1(manifest=manifest),
        four_fold_validation_inputs_authority_relative_path_v2(manifest=manifest),
    ]
    for fold_index in _WITHHELD_FOLD_INDICES:
        candidates.extend(
            (
                validation_decision_tensor_relative_path_v1(
                    manifest=manifest, fold_index=fold_index
                ),
                validation_forecast_archive_relative_path_v1(
                    manifest=manifest, fold_index=fold_index
                ),
                validation_sources_authority_relative_path_v1(
                    manifest=manifest, fold_index=fold_index
                ),
                validation_environment_registry_relative_path_v1(
                    manifest=manifest, fold_index=fold_index
                ),
                validation_sources_authority_relative_path_v2(
                    manifest=manifest, fold_index=fold_index
                ),
                validation_environment_registry_relative_path_v2(
                    manifest=manifest, fold_index=fold_index
                ),
            )
        )
    found = []
    for relative in candidates:
        complete, partial = _transaction_state(root=root, relative=relative)
        if complete or partial:
            found.append(relative)
    for fold_index in _FOLD_INDICES:
        if massive_adaptive_rl_validation_downstream_evidence_exists_v1(
            root=root, manifest=manifest, fold_index=fold_index
        ) or massive_adaptive_rl_validation_downstream_evidence_exists_v2(
            root=root, manifest=manifest, fold_index=fold_index
        ):
            found.append(f"legacy-validation-outcome-fold-{fold_index}")
    return tuple(sorted(set(found)))


def massive_adaptive_rl_forbidden_prequential_artifacts_v1(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> tuple[str, ...]:
    """Expose the fail-closed legacy/later-evidence inventory to root protocols."""

    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV4:
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "prequential artifact scan requires exact Manifest V4"
        )
    manifest.validate()
    return _forbidden_prequential_artifacts(root=root, manifest=manifest)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    prequential_validation_plan_receipt_sha256: str
    split_plan_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    runtime_sources_v2_receipt_sha256: str
    source_bundle_v2_receipt_sha256: str
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_witness_receipt_sha256: str
    replay_dependency_index_v2_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    validation_fold_indices: tuple[int, ...]
    withheld_validation_fold_indices: tuple[int, ...]
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
    runtime_inputs_replayed: bool = False
    development_validation_inputs_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_INPUTS_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV4 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_four_fold_fit: MassiveAdaptiveRLFourFoldFitAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_plan: MassiveAdaptiveRLPrequentialValidationPlanV1 | None = field(
        default=None, compare=False, repr=False
    )
    _validation_sources_v2: tuple[
        MassiveAdaptiveRLValidationSourcesAuthorityV2, ...
    ] = field(default=(), compare=False, repr=False)
    _validation_registries_v2: tuple[
        MassiveAdaptiveRLValidationEnvironmentRegistryV2, ...
    ] = field(default=(), compare=False, repr=False)
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
    def prequential_validation_plan(
        self,
    ) -> MassiveAdaptiveRLPrequentialValidationPlanV1:
        self.validate()
        if not self.development_stage_authorized or self._runtime_plan is None:
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "prequential validation plan has not been exactly replayed"
            )
        return self._runtime_plan

    @property
    def four_fold_fit_authority(
        self,
    ) -> MassiveAdaptiveRLFourFoldFitAuthorityV1:
        """Return the exact replayed fit aggregate bound by this authority."""

        self.validate()
        if (
            not self.development_stage_authorized
            or self._runtime_four_fold_fit is None
        ):
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "four-fold fit authority has not been exactly replayed"
            )
        return self._runtime_four_fold_fit

    def validation_sources(
        self, fold_index: int
    ) -> MassiveAdaptiveRLValidationSourcesAuthorityV2:
        self.validate()
        if (
            fold_index not in self.validation_fold_indices
            or not self.development_stage_authorized
        ):
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "validation fold is not released by the initial authority"
            )
        return self._validation_sources_v2[self.validation_fold_indices.index(fold_index)]

    def validation_registry(
        self, fold_index: int
    ) -> MassiveAdaptiveRLValidationEnvironmentRegistryV2:
        self.validate()
        if (
            fold_index not in self.validation_fold_indices
            or not self.development_stage_authorized
        ):
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "validation fold is not released by the initial authority"
            )
        return self._validation_registries_v2[
            self.validation_fold_indices.index(fold_index)
        ]

    def validate(self) -> None:
        runtime_present = all(
            value is not None
            for value in (
                self._runtime_manifest,
                self._runtime_sources_v2,
                self._runtime_four_fold_fit,
                self._runtime_plan,
            )
        ) and len(self._validation_sources_v2) == len(
            self.validation_fold_indices
        ) and len(self._validation_registries_v2) == len(self.validation_fold_indices)
        expected_runtime_flags = bool(runtime_present and self.source_data_qualified)
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
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or self.validation_fold_indices != _INITIAL_FOLD_INDICES
            or self.withheld_validation_fold_indices != _WITHHELD_FOLD_INDICES
            or set(lengths) != {2}
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in (
                    *self.validation_sources_v2_committed_at_ms,
                    *self.validation_registry_v2_committed_at_ms,
                )
            )
            or max(self.validation_sources_v2_committed_at_ms)
            >= min(self.validation_registry_v2_committed_at_ms)
            or any(
                len(row) != MASSIVE_ADAPTIVE_INNER_VALIDATION_SESSIONS_V1
                or row != tuple(sorted(set(row)))
                for row in self.validation_decision_session_date_inventories
            )
            or tuple(
                len(row)
                for row in self.expected_candidate_checkpoint_authority_receipt_inventories
            )
            != (1, 2)
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
            or self.runtime_inputs_replayed != runtime_present
            or self.development_validation_inputs_authorized != expected_runtime_flags
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_INPUTS_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "initial prequential validation-input authority differs"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.semantic_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= max(self.validation_registry_v2_committed_at_ms)
            ):
                raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                    "initial validation-input source transaction differs"
                )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest("initial prequential validation inputs", value)
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
                _digest("initial validation-input inventory", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _required_transaction(
    value: str | int | None, *, name: str
) -> str | int:
    if value is None:
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            f"{name} source transaction is absent"
        )
    return value


def build_massive_adaptive_rl_initial_validation_inputs_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    validation_sources_v2: Sequence[MassiveAdaptiveRLValidationSourcesAuthorityV2],
    validation_environment_registries_v2: Sequence[
        MassiveAdaptiveRLValidationEnvironmentRegistryV2
    ],
) -> MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(runtime_sources_v2) is not MassiveAdaptiveRLRuntimeSourcesV2
        or type(four_fold_fit_authority) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
    ):
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "initial validation inputs require exact root generations"
        )
    sources = tuple(validation_sources_v2)
    registries = tuple(validation_environment_registries_v2)
    if (
        len(sources) != 2
        or len(registries) != 2
        or any(type(row) is not MassiveAdaptiveRLValidationSourcesAuthorityV2 for row in sources)
        or any(
            type(row) is not MassiveAdaptiveRLValidationEnvironmentRegistryV2
            for row in registries
        )
    ):
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "initial validation child inventory differs"
        )
    manifest.validate()
    runtime_sources_v2.validate()
    four_fold_fit_authority.validate()
    for row in (*sources, *registries):
        row.validate()
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    split_plan = runtime_sources_v2.base_runtime_sources_v1.split_plan
    plan = build_massive_adaptive_rl_prequential_validation_plan_v1(
        manifest=manifest, split_plan=split_plan
    )
    if (
        tuple(row.fold_index for row in sources) != _INITIAL_FOLD_INDICES
        or tuple(row.fold_index for row in registries) != _INITIAL_FOLD_INDICES
        or any(not row.development_stage_authorized for row in (*sources, *registries))
        or any(
            registry.validation_sources_v2_receipt_sha256
            != source.semantic_receipt_sha256
            for source, registry in zip(sources, registries, strict=True)
        )
        or tuple(row.validation_decision_session_dates for row in sources)
        != tuple(plan.validation_session_date_inventories[index] for index in _INITIAL_FOLD_INDICES)
    ):
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "initial validation inputs do not match the prequential plan"
        )
    expected_candidates = tuple(
        four_fold_fit_authority.fold_fit(index).candidate_checkpoint_authority_receipts
        for index in _INITIAL_FOLD_INDICES
    )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": manifest.base_manifest.semantic_receipt_sha256,
        "prequential_validation_plan_receipt_sha256": plan.semantic_receipt_sha256,
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "four_fold_fit_authority_receipt_sha256": four_fold_fit_authority.semantic_receipt_sha256,
        "runtime_sources_v2_receipt_sha256": runtime_sources_v2.semantic_receipt_sha256,
        "source_bundle_v2_receipt_sha256": runtime_sources_v2.source_bundle_v2_receipt_sha256,
        "runtime_source_graph_v2_receipt_sha256": runtime_sources_v2.runtime_source_graph_v2_receipt_sha256,
        "runtime_source_graph_v2_witness_receipt_sha256": runtime_sources_v2.runtime_source_graph_v2_witness_receipt_sha256,
        "replay_dependency_index_v2_receipt_sha256": runtime_sources_v2.replay_dependency_index_v2_receipt_sha256,
        "training_source_projection_sha256": runtime_sources_v2.training_source_projection_sha256,
        "validation_source_projection_sha256": runtime_sources_v2.validation_source_projection_sha256,
        "validation_fold_indices": _INITIAL_FOLD_INDICES,
        "withheld_validation_fold_indices": _WITHHELD_FOLD_INDICES,
        "validation_sources_v2_receipts": tuple(row.semantic_receipt_sha256 for row in sources),
        "validation_sources_v2_source_receipts": tuple(
            cast(str, _required_transaction(row.source_receipt_sha256, name="validation source"))
            for row in sources
        ),
        "validation_sources_v2_commit_receipts": tuple(
            cast(str, _required_transaction(row.source_transaction_receipt_sha256, name="validation source"))
            for row in sources
        ),
        "validation_sources_v2_committed_at_ms": tuple(
            cast(int, _required_transaction(row.source_transaction_committed_at_ms, name="validation source"))
            for row in sources
        ),
        "validation_environment_registry_v2_receipts": tuple(
            row.semantic_receipt_sha256 for row in registries
        ),
        "validation_registry_v2_source_receipts": tuple(
            cast(str, _required_transaction(row.source_receipt_sha256, name="validation registry"))
            for row in registries
        ),
        "validation_registry_v2_commit_receipts": tuple(
            cast(str, _required_transaction(row.source_transaction_receipt_sha256, name="validation registry"))
            for row in registries
        ),
        "validation_registry_v2_committed_at_ms": tuple(
            cast(int, _required_transaction(row.source_transaction_committed_at_ms, name="validation registry"))
            for row in registries
        ),
        "validation_context_receipts": tuple(row.validation_context_receipt_sha256 for row in registries),
        "validation_decision_session_date_inventories": tuple(
            row.validation_decision_session_dates for row in sources
        ),
        "expected_candidate_checkpoint_authority_receipt_inventories": expected_candidates,
        "source_data_qualified": bool(
            runtime_sources_v2.source_data_qualified
            and four_fold_fit_authority.source_data_qualified
            and all(row.source_data_qualified for row in (*sources, *registries))
        ),
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_INPUTS_V1_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA,
    }
    provisional = MassiveAdaptiveRLInitialValidationInputsAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_inputs_replayed=True,
        development_validation_inputs_authorized=bool(body["source_data_qualified"]),
        _runtime_manifest=manifest,
        _runtime_sources_v2=runtime_sources_v2,
        _runtime_four_fold_fit=four_fold_fit_authority,
        _runtime_plan=plan,
        _validation_sources_v2=sources,
        _validation_registries_v2=registries,
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
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "initial validation-input source is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "validation_fold_indices",
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


def parse_massive_adaptive_rl_initial_validation_inputs_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
    body = _parse_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLInitialValidationInputsAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_initial_validation_inputs_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    verified_at_ms: int,
) -> MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
    return parse_massive_adaptive_rl_initial_validation_inputs_authority_v1(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=initial_validation_inputs_authority_relative_path_v1(
                manifest=manifest
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_initial_validation_inputs_authority_v1(
    *,
    authority: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    validation_sources_v2: Sequence[MassiveAdaptiveRLValidationSourcesAuthorityV2],
    validation_environment_registries_v2: Sequence[
        MassiveAdaptiveRLValidationEnvironmentRegistryV2
    ],
) -> MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
    authority.validate()
    expected = build_massive_adaptive_rl_initial_validation_inputs_authority_v1(
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
        validation_sources_v2=validation_sources_v2,
        validation_environment_registries_v2=(
            validation_environment_registries_v2
        ),
    )
    relative = initial_validation_inputs_authority_relative_path_v1(manifest=manifest)
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path != relative
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "initial validation-input authority did not replay"
        )
    result = replace(
        authority,
        runtime_inputs_replayed=True,
        development_validation_inputs_authorized=authority.source_data_qualified,
        _runtime_manifest=manifest,
        _runtime_sources_v2=runtime_sources_v2,
        _runtime_four_fold_fit=four_fold_fit_authority,
        _runtime_plan=expected._runtime_plan,
        _validation_sources_v2=tuple(validation_sources_v2),
        _validation_registries_v2=tuple(validation_environment_registries_v2),
    )
    result.validate()
    return result


@manifest_v5_compatibility_writer_guard_v1(writer_role="initial-validation-inputs")
def materialize_massive_adaptive_rl_initial_validation_inputs_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    authority: MassiveAdaptiveRLInitialValidationInputsAuthorityV1,
    committed_at_ms: int,
    v5_writer_capability: MassiveAdaptiveRLManifestV5WriterCapabilityV1 | None = None,
) -> MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
    authority.validate()
    relative = initial_validation_inputs_authority_relative_path_v1(manifest=manifest)
    if _transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "initial validation-input authority already exists"
        )
    forbidden = _forbidden_prequential_artifacts(root=root, manifest=manifest)
    if forbidden:
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "prequential inputs cannot follow opened later-fold or legacy evidence"
        )
    if (
        not authority.runtime_inputs_replayed
        or isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms <= max(authority.validation_registry_v2_committed_at_ms)
    ):
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "initial validation-input barrier must follow both registries"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-PREQUENTIAL-INITIAL-INPUTS-{manifest.experiment_id}",
    )
    generic = load_massive_adaptive_rl_initial_validation_inputs_authority_v1(
        root=root, manifest=manifest, verified_at_ms=committed_at_ms
    )
    return authorize_massive_adaptive_rl_initial_validation_inputs_authority_v1(
        authority=generic,
        manifest=manifest,
        runtime_sources_v2=cast(
            MassiveAdaptiveRLRuntimeSourcesV2, authority._runtime_sources_v2
        ),
        four_fold_fit_authority=cast(
            MassiveAdaptiveRLFourFoldFitAuthorityV1, authority._runtime_four_fold_fit
        ),
        validation_sources_v2=authority._validation_sources_v2,
        validation_environment_registries_v2=authority._validation_registries_v2,
    )


@contextmanager
def _initial_inputs_lease(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> Iterator[None]:
    manifest.validate()
    directory = Path(root) / "massive-adaptive" / "rl-prequential-input-leases-v1"
    descriptor = -1
    try:
        Path(root).mkdir(parents=True, exist_ok=True)
        if Path(root).is_symlink():
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "prequential validation root is a symlink"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "prequential validation lease directory is a symlink"
            )
        descriptor = os.open(
            directory / f"v4-{manifest.semantic_receipt_sha256}.lock",
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "prequential validation lease identity differs"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MassiveAdaptiveRLInitialValidationInputsLeaseUnavailable(
                "prequential validation inputs are already owned"
            ) from error
    except (MassiveAdaptiveRLPrequentialValidationInputsV1Error, OSError):
        if descriptor >= 0:
            os.close(descriptor)
        raise
    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@manifest_v5_compatibility_writer_guard_v1(
    writer_role="initial-validation-inputs",
    materialize_parameter="allow_materialize",
)
def _run_or_resume_massive_adaptive_rl_initial_validation_inputs_with_capability_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    allow_materialize: bool = True,
    v5_writer_capability: MassiveAdaptiveRLManifestV5WriterCapabilityV1 | None = None,
) -> MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
    """Commit only folds 0 and 1; never materialize folds 2 or 3."""

    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
            "initial validation materialization mode differs"
        )
    manifest.validate()
    runtime_sources_v2.validate()
    four_fold_fit_authority.validate()
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    relative = initial_validation_inputs_authority_relative_path_v1(manifest=manifest)
    lease = _initial_inputs_lease(root=root, manifest=manifest) if allow_materialize else nullcontext()
    with lease:
        forbidden = _forbidden_prequential_artifacts(root=root, manifest=manifest)
        if forbidden:
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "legacy all-at-once or causally unavailable validation evidence exists"
            )
        exists = _transaction_exists(root=root, relative=relative)
        cursor = _wall_clock_ms()
        sources = []
        for fold_index in _INITIAL_FOLD_INDICES:
            source = prepare_or_resume_massive_adaptive_rl_validation_sources_v2(
                root=root,
                manifest=manifest,
                four_fold_fit_authority=four_fold_fit_authority,
                runtime_sources_v2=runtime_sources_v2,
                fold_index=fold_index,
                committed_at_ms=cursor,
                allow_materialize=allow_materialize,
                v5_writer_capability=v5_writer_capability,
            )
            sources.append(source)
            cursor = _next_publication_ms(
                cast(int, source.source_transaction_committed_at_ms)
            )
        registries = []
        for source in sources:
            registry = prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v2(
                root=root,
                manifest=manifest,
                runtime_sources_v2=runtime_sources_v2,
                validation_sources_v2=source,
                committed_at_ms=cursor,
                allow_materialize=allow_materialize,
                v5_writer_capability=v5_writer_capability,
            )
            registries.append(registry)
            cursor = _next_publication_ms(
                cast(int, registry.source_transaction_committed_at_ms)
            )
        if exists:
            return authorize_massive_adaptive_rl_initial_validation_inputs_authority_v1(
                authority=load_massive_adaptive_rl_initial_validation_inputs_authority_v1(
                    root=root, manifest=manifest, verified_at_ms=cursor
                ),
                manifest=manifest,
                runtime_sources_v2=runtime_sources_v2,
                four_fold_fit_authority=four_fold_fit_authority,
                validation_sources_v2=sources,
                validation_environment_registries_v2=registries,
            )
        if not allow_materialize:
            raise MassiveAdaptiveRLPrequentialValidationInputsV1Error(
                "initial prequential validation-input authority is absent"
            )
        authority = build_massive_adaptive_rl_initial_validation_inputs_authority_v1(
            manifest=manifest,
            runtime_sources_v2=runtime_sources_v2,
            four_fold_fit_authority=four_fold_fit_authority,
            validation_sources_v2=sources,
            validation_environment_registries_v2=registries,
        )
        return materialize_massive_adaptive_rl_initial_validation_inputs_authority_v1(
            root=root,
            manifest=manifest,
            authority=authority,
            committed_at_ms=max(
                cursor,
                max(authority.validation_registry_v2_committed_at_ms) + 1,
            ),
            v5_writer_capability=v5_writer_capability,
        )


def run_or_resume_massive_adaptive_rl_initial_validation_inputs_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLInitialValidationInputsAuthorityV1:
    """Run the legacy initial boundary without a V5 writer capability."""

    return _run_or_resume_massive_adaptive_rl_initial_validation_inputs_with_capability_v1(
        root=root,
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
        allow_materialize=allow_materialize,
        v5_writer_capability=None,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_INPUTS_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256",
    "MassiveAdaptiveRLInitialValidationInputsAuthorityV1",
    "MassiveAdaptiveRLInitialValidationInputsLeaseUnavailable",
    "MassiveAdaptiveRLPolicyScheduleDispositionV1",
    "MassiveAdaptiveRLPrequentialValidationInputsV1Error",
    "MassiveAdaptiveRLPrequentialValidationPlanV1",
    "authorize_massive_adaptive_rl_initial_validation_inputs_authority_v1",
    "build_massive_adaptive_rl_initial_validation_inputs_authority_v1",
    "build_massive_adaptive_rl_prequential_validation_plan_v1",
    "initial_validation_inputs_authority_relative_path_v1",
    "load_massive_adaptive_rl_initial_validation_inputs_authority_v1",
    "massive_adaptive_rl_forbidden_prequential_artifacts_v1",
    "materialize_massive_adaptive_rl_initial_validation_inputs_authority_v1",
    "parse_massive_adaptive_rl_initial_validation_inputs_authority_v1",
    "policy_schedule_disposition_v1",
    "run_or_resume_massive_adaptive_rl_initial_validation_inputs_v1",
]
