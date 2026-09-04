"""Create-only aggregate for four attested walk-forward RL selections.

The adaptive split plan is walk-forward: each inner-validation fold selects
the checkpoint used by its corresponding later outer fold.  This authority
therefore records four distinct Selection-V3 decisions; it does not invent a
post-hoc global checkpoint or refit rule.  A complete but validation-ineligible
set is persisted as ``NO_QUALIFIED_POLICY`` and cannot authorize freezing or
outer access in this generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from enum import Enum
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
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_execution_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_V1_SPEC_SHA256,
    MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
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
    MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1,
    MassiveAdaptiveRLExperimentManifestV4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    legacy_manifest_v5_rejecting_writer_guard_v1,
)


MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-four-fold-policy-selection-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-four-fold-policy-selection-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": (
                MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SCHEMA
            ),
            "encoding": "canonical-json-four-attested-walk-forward-selections",
            "generic_reload": "integrity-only",
        }
    )
)
MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-four-fold-validation-execution-authority-v1-in-fold-order",
        "fold_execution_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_V1_SPEC_SHA256
        ),
        "selection_role": "four-distinct-walk-forward-outer-fold-policies",
        "global_checkpoint_or_refit_selection": False,
        "eligibility": "all-four-selection-v3-candidates-validation-eligible",
        "qualified_disposition": "four-fold-selections-qualified",
        "ineligible_disposition": "no-qualified-policy",
        "invalid_evidence": "raise-without-publication",
        "publication": "manifest-derived-create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "final_policy_freezing": False,
        "outer_access": False,
        "profitability_reporting": False,
        "lockbox_access": False,
    }
)

_FOLD_INDICES = (0, 1, 2, 3)


class MassiveAdaptiveRLFourFoldSelectionDispositionV1(str, Enum):
    FOUR_FOLD_SELECTIONS_QUALIFIED = "four-fold-selections-qualified"
    NO_QUALIFIED_POLICY = "no-qualified-policy"


class MassiveAdaptiveRLFourFoldPolicySelectionV1Error(ValueError):
    """The four-fold selection evidence is absent, mixed, or inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
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
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            "four-fold selection source transaction is incomplete"
        )
    return all(present)


def four_fold_policy_selection_authority_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> str:
    manifest.validate()
    return (
        "massive-adaptive/rl-four-fold-policy-selection-authority-v1/"
        f"v4-{manifest.semantic_receipt_sha256}.json"
    )


def _required(value: str | int | None, *, name: str) -> str | int:
    if value is None:
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            f"{name} source transaction is absent"
        )
    return value


def _selection_disposition(
    eligible: Sequence[bool],
) -> MassiveAdaptiveRLFourFoldSelectionDispositionV1:
    return (
        MassiveAdaptiveRLFourFoldSelectionDispositionV1.FOUR_FOLD_SELECTIONS_QUALIFIED
        if all(eligible)
        else MassiveAdaptiveRLFourFoldSelectionDispositionV1.NO_QUALIFIED_POLICY
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    runtime_sources_v2_receipt_sha256: str
    source_bundle_v2_receipt_sha256: str
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_witness_receipt_sha256: str
    replay_dependency_index_v2_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    four_fold_validation_inputs_v2_receipt_sha256: str
    four_fold_validation_inputs_v2_source_receipt_sha256: str
    four_fold_validation_inputs_v2_commit_receipt_sha256: str
    four_fold_validation_inputs_v2_committed_at_ms: int
    fold_indices: tuple[int, ...]
    fold_execution_authority_receipts: tuple[str, ...]
    fold_execution_source_receipts: tuple[str, ...]
    fold_execution_commit_receipts: tuple[str, ...]
    fold_execution_committed_at_ms: tuple[int, ...]
    policy_selection_v3_receipts: tuple[str, ...]
    policy_selection_v3_source_receipts: tuple[str, ...]
    policy_selection_v3_commit_receipts: tuple[str, ...]
    selected_checkpoint_authority_receipts: tuple[str, ...]
    selected_checkpoint_receipts: tuple[str, ...]
    selected_model_state_receipts: tuple[str, ...]
    selected_update_indices: tuple[int, ...]
    selected_candidate_validation_eligible: tuple[bool, ...]
    validation_eligibility_failure_inventories: tuple[tuple[str, ...], ...]
    validation_execution_environment_receipt_sha256: str
    validation_execution_environment_source_receipt_sha256: str
    validation_execution_environment_commit_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    selection_disposition: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_four_fold_selection_replayed: bool = False
    four_fold_selection_recording_authorized: bool = False
    final_policy_freezing_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV4 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_four_fold_fit: MassiveAdaptiveRLFourFoldFitAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_fold_executions: tuple[
        MassiveAdaptiveRLFoldValidationExecutionAuthorityV1, ...
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
                "runtime_four_fold_selection_replayed",
                "four_fold_selection_recording_authorized",
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
    def all_selected_policies_validation_eligible(self) -> bool:
        return all(self.selected_candidate_validation_eligible)

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_four_fold_selection_replayed
            and self.four_fold_selection_recording_authorized
            and self.source_data_qualified
        )

    @property
    def positive_profitability_authorization_eligible(self) -> bool:
        return bool(
            self.development_stage_authorized
            and self.all_selected_policies_validation_eligible
            and self.selection_disposition
            == MassiveAdaptiveRLFourFoldSelectionDispositionV1.FOUR_FOLD_SELECTIONS_QUALIFIED.value
        )

    def fold_execution(
        self, fold_index: int
    ) -> MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
        self.validate()
        if fold_index not in self.fold_indices or not self._runtime_fold_executions:
            raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
                "four-fold selection runtime fold is unavailable"
            )
        return self._runtime_fold_executions[self.fold_indices.index(fold_index)]

    def validate(self) -> None:
        runtime_scalars = (
            self._runtime_manifest,
            self._runtime_sources_v2,
            self._runtime_four_fold_fit,
            self._runtime_validation_inputs_v2,
        )
        runtime_present = any(value is not None for value in runtime_scalars) or bool(
            self._runtime_fold_executions
        )
        if runtime_present and (
            any(value is None for value in runtime_scalars)
            or len(self._runtime_fold_executions) != 4
        ):
            raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
                "four-fold selection runtime is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        if runtime_present:
            assert self._runtime_manifest is not None
            assert self._runtime_sources_v2 is not None
            assert self._runtime_four_fold_fit is not None
            assert self._runtime_validation_inputs_v2 is not None
            expected = _four_fold_selection_body(
                manifest=self._runtime_manifest,
                runtime_sources_v2=self._runtime_sources_v2,
                four_fold_fit_authority=self._runtime_four_fold_fit,
                four_fold_validation_inputs_v2=self._runtime_validation_inputs_v2,
                fold_executions=self._runtime_fold_executions,
            )
            expected.update(
                {
                    "final_policy_freezing_authorized": False,
                    "outer_access_authorized": False,
                    "profitability_reporting_authorized": False,
                    "lockbox_access_authorized": False,
                    "protocol_receipt_sha256": (
                        MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
                    ),
                    "specification_sha256": (
                        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_V1_SPEC_SHA256
                    ),
                    "implementation_source_sha256": (
                        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SHA256
                    ),
                    "schema": (
                        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SCHEMA
                    ),
                }
            )
            if self.semantic_unsigned() != expected:
                raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
                    "four-fold selection runtime replay differs"
                )
        inventories: tuple[Sequence[object], ...] = (
            self.fold_execution_authority_receipts,
            self.fold_execution_source_receipts,
            self.fold_execution_commit_receipts,
            self.fold_execution_committed_at_ms,
            self.policy_selection_v3_receipts,
            self.policy_selection_v3_source_receipts,
            self.policy_selection_v3_commit_receipts,
            self.selected_checkpoint_authority_receipts,
            self.selected_checkpoint_receipts,
            self.selected_model_state_receipts,
            self.selected_update_indices,
            self.selected_candidate_validation_eligible,
            self.validation_eligibility_failure_inventories,
        )
        expected_runtime_flags = bool(
            runtime_present
            and self.source_transaction_verified
            and self.source_data_qualified
            and all(
                row.development_stage_authorized
                for row in self._runtime_fold_executions
            )
        )
        expected_disposition = _selection_disposition(
            self.selected_candidate_validation_eligible
        ).value
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or self.fold_indices != _FOLD_INDICES
            or any(len(inventory) != 4 for inventory in inventories)
            or len(set(self.fold_execution_authority_receipts)) != 4
            or len(set(self.fold_execution_source_receipts)) != 4
            or len(set(self.fold_execution_commit_receipts)) != 4
            or len(set(self.policy_selection_v3_receipts)) != 4
            or len(set(self.policy_selection_v3_source_receipts)) != 4
            or len(set(self.policy_selection_v3_commit_receipts)) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    self.four_fold_validation_inputs_v2_committed_at_ms,
                    *self.fold_execution_committed_at_ms,
                    *self.selected_update_indices,
                )
            )
            or any(
                type(value) is not bool
                for value in self.selected_candidate_validation_eligible
            )
            or any(
                (eligible and failures) or (not eligible and not failures)
                for eligible, failures in zip(
                    self.selected_candidate_validation_eligible,
                    self.validation_eligibility_failure_inventories,
                    strict=True,
                )
            )
            or any(
                tuple(sorted(set(failures))) != failures
                or not set(failures).issubset(
                    MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1
                )
                for failures in self.validation_eligibility_failure_inventories
            )
            or any(
                committed_at_ms <= self.four_fold_validation_inputs_v2_committed_at_ms
                for committed_at_ms in self.fold_execution_committed_at_ms
            )
            or self.selection_disposition != expected_disposition
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_four_fold_selection_replayed != runtime_present
            or self.four_fold_selection_recording_authorized != expected_runtime_flags
            or self.final_policy_freezing_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
                "four-fold policy selection authority differs"
            )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= max(self.fold_execution_committed_at_ms)
        ):
            raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
                "four-fold selection source transaction differs"
            )
        for descriptor in fields(self):
            if descriptor.name.endswith("_sha256"):
                _digest("four-fold selection", getattr(self, descriptor.name))
        for inventory in (
            self.fold_execution_authority_receipts,
            self.fold_execution_source_receipts,
            self.fold_execution_commit_receipts,
            self.policy_selection_v3_receipts,
            self.policy_selection_v3_source_receipts,
            self.policy_selection_v3_commit_receipts,
            self.selected_checkpoint_authority_receipts,
            self.selected_checkpoint_receipts,
            self.selected_model_state_receipts,
        ):
            for value in inventory:
                _digest("four-fold selection inventory", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _four_fold_selection_body(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    fold_executions: Sequence[MassiveAdaptiveRLFoldValidationExecutionAuthorityV1],
) -> dict[str, object]:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(runtime_sources_v2) is not MassiveAdaptiveRLRuntimeSourcesV2
        or type(four_fold_fit_authority) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
        or type(four_fold_validation_inputs_v2)
        is not MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ):
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            "four-fold selection requires exact root generations"
        )
    executions = tuple(fold_executions)
    if len(executions) != 4 or any(
        type(row) is not MassiveAdaptiveRLFoldValidationExecutionAuthorityV1
        for row in executions
    ):
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            "four-fold selection requires four exact fold executions"
        )
    manifest.validate()
    runtime_sources_v2.validate()
    four_fold_fit_authority.validate()
    four_fold_validation_inputs_v2.validate()
    for row in executions:
        row.validate()
    validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
    )
    barrier_source = _required(
        four_fold_validation_inputs_v2.source_receipt_sha256,
        name="four-fold validation inputs",
    )
    barrier_commit = _required(
        four_fold_validation_inputs_v2.source_transaction_receipt_sha256,
        name="four-fold validation inputs",
    )
    barrier_time = _required(
        four_fold_validation_inputs_v2.source_transaction_committed_at_ms,
        name="four-fold validation inputs",
    )
    selections = tuple(row.policy_selection_v3 for row in executions)
    execution_sources = tuple(
        _required(row.source_receipt_sha256, name="fold execution")
        for row in executions
    )
    execution_commits = tuple(
        _required(row.source_transaction_receipt_sha256, name="fold execution")
        for row in executions
    )
    execution_times = tuple(
        _required(row.source_transaction_committed_at_ms, name="fold execution")
        for row in executions
    )
    expected_checkpoints = four_fold_validation_inputs_v2.expected_candidate_checkpoint_authority_receipt_inventories
    if (
        not four_fold_validation_inputs_v2.development_stage_authorized
        or not four_fold_fit_authority.development_stage_authorized
        or tuple(row.fold_index for row in executions) != _FOLD_INDICES
        or not all(row.development_stage_authorized for row in executions)
        or not all(selection.development_stage_authorized for selection in selections)
        or any(
            row.experiment_id != manifest.experiment_id
            or row.manifest_v4_receipt_sha256 != manifest.semantic_receipt_sha256
            or row.training_manifest_v3_receipt_sha256
            != manifest.base_manifest.semantic_receipt_sha256
            or row.runtime_sources_v2_receipt_sha256
            != runtime_sources_v2.semantic_receipt_sha256
            or row.four_fold_fit_authority_receipt_sha256
            != four_fold_fit_authority.semantic_receipt_sha256
            or row.four_fold_validation_inputs_v2_receipt_sha256
            != four_fold_validation_inputs_v2.semantic_receipt_sha256
            or row.checkpoint_authority_receipts != expected_checkpoints[index]
            for index, row in enumerate(executions)
        )
        or len(
            {row.validation_execution_environment_receipt_sha256 for row in executions}
        )
        != 1
        or len(
            {
                row.validation_execution_environment_source_receipt_sha256
                for row in executions
            }
        )
        != 1
        or len(
            {
                row.validation_execution_environment_commit_receipt_sha256
                for row in executions
            }
        )
        != 1
        or len({row.scientific_execution_fingerprint_sha256 for row in executions}) != 1
        or any(
            selection.fold_index != index
            or selection.expected_candidate_checkpoint_authority_receipts
            != expected_checkpoints[index]
            for index, selection in enumerate(selections)
        )
    ):
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            "four-fold selection evidence is mixed or incomplete"
        )
    eligible = tuple(
        selection.selected_candidate_validation_eligible for selection in selections
    )
    return {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "runtime_sources_v2_receipt_sha256": (
            runtime_sources_v2.semantic_receipt_sha256
        ),
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
        "four_fold_fit_authority_receipt_sha256": (
            four_fold_fit_authority.semantic_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_receipt_sha256": (
            four_fold_validation_inputs_v2.semantic_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_source_receipt_sha256": cast(
            str, barrier_source
        ),
        "four_fold_validation_inputs_v2_commit_receipt_sha256": cast(
            str, barrier_commit
        ),
        "four_fold_validation_inputs_v2_committed_at_ms": cast(int, barrier_time),
        "fold_indices": _FOLD_INDICES,
        "fold_execution_authority_receipts": tuple(
            row.semantic_receipt_sha256 for row in executions
        ),
        "fold_execution_source_receipts": cast(tuple[str, ...], execution_sources),
        "fold_execution_commit_receipts": cast(tuple[str, ...], execution_commits),
        "fold_execution_committed_at_ms": cast(tuple[int, ...], execution_times),
        "policy_selection_v3_receipts": tuple(
            selection.semantic_receipt_sha256 for selection in selections
        ),
        "policy_selection_v3_source_receipts": tuple(
            selection.policy_selection_v3_source_receipt_sha256
            for selection in executions
        ),
        "policy_selection_v3_commit_receipts": tuple(
            selection.policy_selection_v3_commit_receipt_sha256
            for selection in executions
        ),
        "selected_checkpoint_authority_receipts": tuple(
            selection.selected_checkpoint_authority_receipt_sha256
            for selection in selections
        ),
        "selected_checkpoint_receipts": tuple(
            selection.selected_checkpoint_receipt_sha256 for selection in selections
        ),
        "selected_model_state_receipts": tuple(
            selection.selected_model_state_receipt_sha256 for selection in selections
        ),
        "selected_update_indices": tuple(
            selection.selected_update_index for selection in selections
        ),
        "selected_candidate_validation_eligible": eligible,
        "validation_eligibility_failure_inventories": tuple(
            selection.validation_eligibility_failures for selection in selections
        ),
        "validation_execution_environment_receipt_sha256": executions[
            0
        ].validation_execution_environment_receipt_sha256,
        "validation_execution_environment_source_receipt_sha256": executions[
            0
        ].validation_execution_environment_source_receipt_sha256,
        "validation_execution_environment_commit_receipt_sha256": executions[
            0
        ].validation_execution_environment_commit_receipt_sha256,
        "scientific_execution_fingerprint_sha256": executions[
            0
        ].scientific_execution_fingerprint_sha256,
        "selection_disposition": _selection_disposition(eligible).value,
        "source_data_qualified": bool(
            runtime_sources_v2.source_data_qualified
            and four_fold_fit_authority.source_data_qualified
            and four_fold_validation_inputs_v2.source_data_qualified
            and all(row.source_data_qualified for row in executions)
            and all(selection.source_data_qualified for selection in selections)
        ),
    }


def build_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    fold_executions: Sequence[MassiveAdaptiveRLFoldValidationExecutionAuthorityV1],
) -> MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1:
    executions = tuple(fold_executions)
    body = _four_fold_selection_body(
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        fold_executions=executions,
    )
    provisional = MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_four_fold_selection_replayed=True,
        four_fold_selection_recording_authorized=False,
        _runtime_manifest=manifest,
        _runtime_sources_v2=runtime_sources_v2,
        _runtime_four_fold_fit=four_fold_fit_authority,
        _runtime_validation_inputs_v2=four_fold_validation_inputs_v2,
        _runtime_fold_executions=executions,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            "four-fold selection authority is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    tuple_fields = (
        "fold_indices",
        "fold_execution_authority_receipts",
        "fold_execution_source_receipts",
        "fold_execution_commit_receipts",
        "fold_execution_committed_at_ms",
        "policy_selection_v3_receipts",
        "policy_selection_v3_source_receipts",
        "policy_selection_v3_commit_receipts",
        "selected_checkpoint_authority_receipts",
        "selected_checkpoint_receipts",
        "selected_model_state_receipts",
        "selected_update_indices",
        "selected_candidate_validation_eligible",
    )
    for name in tuple_fields:
        rows = body.get(name)
        if not isinstance(rows, list):
            raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
                "four-fold selection inventory is malformed"
            )
        body[name] = tuple(rows)
    failures = body.get("validation_eligibility_failure_inventories")
    if not isinstance(failures, list) or any(
        not isinstance(row, list) for row in failures
    ):
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            "four-fold selection failure inventory is malformed"
        )
    body["validation_eligibility_failure_inventories"] = tuple(
        tuple(row) for row in failures
    )
    try:
        result = MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1(
            **body,  # type: ignore[arg-type]
            semantic_receipt_sha256=semantic_sha256(body),
            _loaded_source=loaded_source,
        )
    except TypeError as error:
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            "four-fold selection authority fields differ"
        ) from error
    result.validate()
    return result


def load_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    verified_at_ms: int,
) -> MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1:
    return _parse(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=(
                four_fold_policy_selection_authority_relative_path_v1(manifest=manifest)
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


@legacy_manifest_v5_rejecting_writer_guard_v1()
def materialize_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    authority: MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1:
    manifest.validate()
    authority.validate()
    relative = four_fold_policy_selection_authority_relative_path_v1(manifest=manifest)
    if (
        authority.source_transaction_verified
        or not authority.runtime_four_fold_selection_replayed
        or not authority.source_data_qualified
        or isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms <= max(authority.fold_execution_committed_at_ms)
        or authority.manifest_v4_receipt_sha256 != manifest.semantic_receipt_sha256
        or _transaction_exists(root=root, relative=relative)
    ):
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            "four-fold selection materialization is not authorized"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            "ADAPTIVE-RL-FOUR-FOLD-POLICY-SELECTION-AUTHORITY-V1-"
            f"{manifest.semantic_receipt_sha256}"
        ),
    )
    return load_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
        root=root,
        manifest=manifest,
        verified_at_ms=committed_at_ms,
    )


def authorize_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
    *,
    authority: MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    fold_executions: Sequence[MassiveAdaptiveRLFoldValidationExecutionAuthorityV1],
) -> MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1:
    authority.validate()
    expected = build_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        fold_executions=fold_executions,
    )
    if (
        not authority.source_transaction_verified
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
            "four-fold selection authority did not replay"
        )
    result = replace(
        authority,
        runtime_four_fold_selection_replayed=True,
        four_fold_selection_recording_authorized=authority.source_data_qualified,
        _runtime_manifest=manifest,
        _runtime_sources_v2=runtime_sources_v2,
        _runtime_four_fold_fit=four_fold_fit_authority,
        _runtime_validation_inputs_v2=four_fold_validation_inputs_v2,
        _runtime_fold_executions=tuple(fold_executions),
    )
    result.validate()
    return result


@legacy_manifest_v5_rejecting_writer_guard_v1(
    materialize_parameter="allow_materialize"
)
def run_or_resume_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
    four_fold_validation_inputs_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    ),
    fold_executions: Sequence[MassiveAdaptiveRLFoldValidationExecutionAuthorityV1],
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1:
    relative = four_fold_policy_selection_authority_relative_path_v1(manifest=manifest)
    if _transaction_exists(root=root, relative=relative):
        generic = load_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
            root=root, manifest=manifest, verified_at_ms=committed_at_ms
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLFourFoldPolicySelectionV1Error(
                "canonical four-fold selection authority is absent"
            )
        active = build_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
            manifest=manifest,
            runtime_sources_v2=runtime_sources_v2,
            four_fold_fit_authority=four_fold_fit_authority,
            four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
            fold_executions=fold_executions,
        )
        generic = (
            materialize_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
                root=root,
                manifest=manifest,
                authority=active,
                committed_at_ms=committed_at_ms,
            )
        )
    return authorize_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
        authority=generic,
        manifest=manifest,
        runtime_sources_v2=runtime_sources_v2,
        four_fold_fit_authority=four_fold_fit_authority,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        fold_executions=fold_executions,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFourFoldPolicySelectionAuthorityV1",
    "MassiveAdaptiveRLFourFoldPolicySelectionV1Error",
    "MassiveAdaptiveRLFourFoldSelectionDispositionV1",
    "authorize_massive_adaptive_rl_four_fold_policy_selection_authority_v1",
    "build_massive_adaptive_rl_four_fold_policy_selection_authority_v1",
    "four_fold_policy_selection_authority_relative_path_v1",
    "load_massive_adaptive_rl_four_fold_policy_selection_authority_v1",
    "materialize_massive_adaptive_rl_four_fold_policy_selection_authority_v1",
    "run_or_resume_massive_adaptive_rl_four_fold_policy_selection_authority_v1",
]
