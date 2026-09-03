"""Source-generation-V2-bound adaptive-RL policy selection authority.

PolicySelection V2 remains the exact Manifest-V4 ranking computation.  This
module gives that computation a new authority generation whose only accepted
validation input is a replayed FoldValidationAuthority V2.  The persisted V2
selection is retained as an exact computational witness; it cannot authorize
this generation without the complete V2 source, registry, barrier, and
validation-outcome lineage.

Generic reload is integrity-only.  Runtime authorization requires exact
replay of both the V2 fold evidence and the canonical Selection-V2 witness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
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
from rl_quant.evaluation.massive_adaptive_rl_validation_evidence_v2 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V2_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256,
    MassiveAdaptiveRLFoldValidationAuthorityV2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v2 import (
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_DATASET,
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256,
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256,
    MassiveAdaptiveRLPolicyCandidateV2,
    MassiveAdaptiveRLPolicySelectionAuthorityV2,
    MassiveAdaptiveRLPolicySelectionV2,
    authorize_massive_adaptive_rl_policy_selection_authority_v2,
    materialize_massive_adaptive_rl_policy_selection_authority_v2,
    parse_massive_adaptive_rl_policy_selection_authority_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256,
    MassiveAdaptiveRLExperimentManifestV4,
)


MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-selection-authority-v3"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_DATASET = (
    "massive-adaptive-rl-policy-selection-authority-v3"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SCHEMA,
            "encoding": "canonical-json-source-generation-v2-policy-selection",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SPEC_SHA256 = semantic_sha256(
    {
        "validation_input": "exact-persisted-runtime-replayed-fold-validation-v2",
        "fold_validation_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V2_SPEC_SHA256
        ),
        "fold_validation_v2_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256
        ),
        "selection_computation": "exact-policy-selection-v2-witness",
        "selection_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SPEC_SHA256
        ),
        "selection_v2_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
        ),
        "source_generation": "exact-source-bundle-graph-index-runtime-v2",
        "validation_inputs": "exact-v2-source-registry-and-four-fold-barrier",
        "validation_outcomes": "exact-v2-primary-ladder-and-fc06-inventories",
        "candidate_metrics": "derived-only-by-selection-v2-from-base-v1-fold-witness",
        "ordering": "fold-validation-v2-before-selection-v2-before-selection-v3",
        "publication": "manifest-and-fold-derived-create-only",
        "v1_fold_direct_authorization": False,
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_evaluation": False,
        "lockbox_access": False,
    }
)


class MassiveAdaptiveRLPolicySelectionV3Error(ValueError):
    """Policy selection lacks exact V2 validation lineage or replay."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
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
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection source transaction is incomplete"
        )
    return all(present)


def policy_selection_v2_witness_artifact_id_v3(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 fold differs"
        )
    return f"v3-witness-v4-{manifest.semantic_receipt_sha256}-fold-{fold_index}"


def policy_selection_v2_witness_relative_path_v3(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    return (
        "massive-adaptive/rl-policy-selection-v2/"
        f"{policy_selection_v2_witness_artifact_id_v3(manifest=manifest, fold_index=fold_index)}.json"
    )


def policy_selection_authority_relative_path_v3(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 fold differs"
        )
    return (
        "massive-adaptive/rl-policy-selection-authority-v3/"
        f"v4-{manifest.semantic_receipt_sha256}-fold-{fold_index}.json"
    )


def _runtime_selection_v2(
    authority: MassiveAdaptiveRLPolicySelectionAuthorityV2,
) -> tuple[
    MassiveAdaptiveRLPolicySelectionV2,
    tuple[MassiveAdaptiveRLPolicyCandidateV2, ...],
]:
    selection = authority.runtime_selection
    candidates = authority.runtime_candidates
    if selection is None or candidates is None:
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V2 computational witness is absent"
        )
    selection.validate()
    for candidate in candidates:
        candidate.validate()
    return selection, candidates


def _selection_v3_body(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV2,
    base_selection_authority_v2: MassiveAdaptiveRLPolicySelectionAuthorityV2,
) -> dict[str, object]:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV4
        or type(validation_authority) is not MassiveAdaptiveRLFoldValidationAuthorityV2
        or type(base_selection_authority_v2)
        is not MassiveAdaptiveRLPolicySelectionAuthorityV2
    ):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 requires exact V2 authority types"
        )
    manifest.validate()
    validation_authority.validate()
    base_selection_authority_v2.validate()
    if (
        not validation_authority.development_stage_authorized
        or not base_selection_authority_v2.development_policy_selection_authorized
        or not base_selection_authority_v2.policy_freezing_authorized
    ):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 evidence is not replay-authorized"
        )
    fold_source = validation_authority.source_receipt_sha256
    fold_commit = validation_authority.source_transaction_receipt_sha256
    fold_time = validation_authority.source_transaction_committed_at_ms
    # This is the already-validated runtime witness attached by FoldValidation V2.
    # Reading it here avoids changing that persisted authority generation merely
    # to expose a downstream convenience accessor.
    barrier = validation_authority._runtime_four_fold_barrier_v2
    base_fold = validation_authority.base_fold_validation_v1
    base_loaded = base_selection_authority_v2.loaded_source
    runtime_base_fold = base_selection_authority_v2.runtime_fold_validation_authority
    base_loaded.validate()
    selection, candidates = _runtime_selection_v2(base_selection_authority_v2)
    if barrier is None:
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 has no runtime V2 input barrier"
        )
    barrier.validate()
    if (
        fold_source is None
        or fold_commit is None
        or fold_time is None
        or not barrier.development_stage_authorized
        or barrier.source_receipt_sha256 is None
        or barrier.source_transaction_receipt_sha256 is None
        or barrier.source_transaction_committed_at_ms is None
    ):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 lineage is not persisted"
        )
    fold_index = validation_authority.fold_index
    expected_base_path = policy_selection_v2_witness_relative_path_v3(
        manifest=manifest,
        fold_index=fold_index,
    )
    if (
        validation_authority.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or validation_authority.training_manifest_v3_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or base_selection_authority_v2.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or base_selection_authority_v2.training_manifest_v3_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or runtime_base_fold is None
        or runtime_base_fold.semantic_receipt_sha256
        != base_fold.semantic_receipt_sha256
        or runtime_base_fold.loaded_source.receipt.receipt_sha256
        != base_fold.loaded_source.receipt.receipt_sha256
        or runtime_base_fold.loaded_source.commit.receipt_sha256
        != base_fold.loaded_source.commit.receipt_sha256
        or base_selection_authority_v2.fold_validation_authority_receipt_sha256
        != base_fold.semantic_receipt_sha256
        or base_selection_authority_v2.fold_fit_authority_receipt_sha256
        != validation_authority.fold_fit_authority_receipt_sha256
        or base_selection_authority_v2.candidate_checkpoint_inventory_sha256
        != semantic_sha256(validation_authority.expected_checkpoint_authority_receipts)
        or selection.expected_candidate_checkpoint_authority_receipts
        != validation_authority.expected_checkpoint_authority_receipts
        or tuple(row.semantic_receipt_sha256 for row in candidates)
        != selection.candidate_receipts
        or base_loaded.payload_relative_path != expected_base_path
        or base_loaded.receipt.dataset_id
        != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_DATASET
        or base_loaded.receipt.schema_sha256
        != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        or base_loaded.commit.committed_at_ms <= fold_time
    ):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 computational witness differs"
        )
    source_data_qualified = bool(
        validation_authority.source_data_qualified
        and base_selection_authority_v2.source_data_qualified
        and selection.source_data_qualified
    )
    return {
        "experiment_id": validation_authority.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_index": fold_index,
        "fold_fit_authority_receipt_sha256": (
            validation_authority.fold_fit_authority_receipt_sha256
        ),
        "fold_validation_v2_receipt_sha256": (
            validation_authority.semantic_receipt_sha256
        ),
        "fold_validation_v2_source_receipt_sha256": fold_source,
        "fold_validation_v2_commit_receipt_sha256": fold_commit,
        "fold_validation_v2_committed_at_ms": fold_time,
        "four_fold_validation_inputs_v2_receipt_sha256": (
            validation_authority.four_fold_validation_inputs_v2_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_source_receipt_sha256": (
            validation_authority.four_fold_validation_inputs_v2_source_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_commit_receipt_sha256": (
            validation_authority.four_fold_validation_inputs_v2_commit_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_committed_at_ms": (
            validation_authority.four_fold_validation_inputs_v2_committed_at_ms
        ),
        "source_bundle_v2_receipt_sha256": barrier.source_bundle_v2_receipt_sha256,
        "source_bundle_v2_source_receipt_sha256": (
            barrier.source_bundle_v2_source_receipt_sha256
        ),
        "source_bundle_v2_commit_receipt_sha256": (
            barrier.source_bundle_v2_commit_receipt_sha256
        ),
        "runtime_source_graph_v2_receipt_sha256": (
            barrier.runtime_source_graph_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_witness_receipt_sha256": (
            barrier.runtime_source_graph_v2_witness_receipt_sha256
        ),
        "replay_dependency_index_v2_receipt_sha256": (
            barrier.replay_dependency_index_v2_receipt_sha256
        ),
        "runtime_sources_v2_receipt_sha256": barrier.runtime_sources_v2_receipt_sha256,
        "training_source_projection_sha256": (
            validation_authority.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            validation_authority.validation_source_projection_sha256
        ),
        "validation_sources_v2_receipt_sha256": (
            validation_authority.validation_sources_v2_receipt_sha256
        ),
        "validation_sources_v2_source_receipt_sha256": (
            validation_authority.validation_sources_v2_source_receipt_sha256
        ),
        "validation_sources_v2_commit_receipt_sha256": (
            validation_authority.validation_sources_v2_commit_receipt_sha256
        ),
        "validation_registry_v2_receipt_sha256": (
            validation_authority.validation_registry_v2_receipt_sha256
        ),
        "validation_registry_v2_source_receipt_sha256": (
            validation_authority.validation_registry_v2_source_receipt_sha256
        ),
        "validation_registry_v2_commit_receipt_sha256": (
            validation_authority.validation_registry_v2_commit_receipt_sha256
        ),
        "primary_outcome_v2_receipts": (
            validation_authority.primary_outcome_v2_receipts
        ),
        "primary_outcome_v2_source_receipts": (
            validation_authority.primary_outcome_v2_source_receipts
        ),
        "primary_outcome_v2_commit_receipts": (
            validation_authority.primary_outcome_v2_commit_receipts
        ),
        "ladder_outcome_v2_receipts": (validation_authority.ladder_outcome_v2_receipts),
        "ladder_outcome_v2_source_receipts": (
            validation_authority.ladder_outcome_v2_source_receipts
        ),
        "ladder_outcome_v2_commit_receipts": (
            validation_authority.ladder_outcome_v2_commit_receipts
        ),
        "fixed_control_outcome_v2_receipt_sha256": (
            validation_authority.fixed_control_outcome_v2_receipt_sha256
        ),
        "fixed_control_outcome_v2_source_receipt_sha256": (
            validation_authority.fixed_control_outcome_v2_source_receipt_sha256
        ),
        "fixed_control_outcome_v2_commit_receipt_sha256": (
            validation_authority.fixed_control_outcome_v2_commit_receipt_sha256
        ),
        "base_fold_validation_v1_receipt_sha256": (
            validation_authority.base_fold_validation_v1_receipt_sha256
        ),
        "base_policy_selection_authority_v2_receipt_sha256": (
            base_selection_authority_v2.semantic_receipt_sha256
        ),
        "base_policy_selection_authority_v2_source_receipt_sha256": (
            base_loaded.receipt.receipt_sha256
        ),
        "base_policy_selection_authority_v2_commit_receipt_sha256": (
            base_loaded.commit.receipt_sha256
        ),
        "base_policy_selection_authority_v2_committed_at_ms": (
            base_loaded.commit.committed_at_ms
        ),
        "selection_v2_receipt_sha256": selection.semantic_receipt_sha256,
        "expected_candidate_checkpoint_authority_receipts": (
            selection.expected_candidate_checkpoint_authority_receipts
        ),
        "candidate_receipts": selection.candidate_receipts,
        "candidate_inventory_sha256": selection.candidate_inventory_sha256,
        "ranked_candidate_receipts": selection.ranked_candidate_receipts,
        "ranked_candidate_inventory_sha256": (
            selection.ranked_candidate_inventory_sha256
        ),
        "candidate_checkpoint_inventory_sha256": semantic_sha256(
            selection.expected_candidate_checkpoint_authority_receipts
        ),
        "selected_checkpoint_authority_receipt_sha256": (
            selection.selected_checkpoint_authority_receipt_sha256
        ),
        "selected_checkpoint_receipt_sha256": (
            selection.selected_checkpoint_receipt_sha256
        ),
        "selected_model_state_receipt_sha256": (
            selection.selected_model_state_receipt_sha256
        ),
        "selected_update_index": selection.selected_update_index,
        "selected_candidate_receipt_sha256": (
            selection.selected_candidate_receipt_sha256
        ),
        "selected_candidate_validation_eligible": (
            selection.selected_candidate_validation_eligible
        ),
        "validation_eligibility_failures": (selection.validation_eligibility_failures),
        "selection_pool_kind": selection.selection_pool_kind,
        "source_data_qualified": source_data_qualified,
        "positive_profitability_authorization_eligible": bool(
            source_data_qualified and selection.selected_candidate_validation_eligible
        ),
        "validation_selection_specification_sha256": (
            selection.validation_selection_specification_sha256
        ),
        "numerical_comparison_specification_sha256": (
            selection.numerical_comparison_specification_sha256
        ),
    }


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicySelectionAuthorityV3:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_index: int
    fold_fit_authority_receipt_sha256: str
    fold_validation_v2_receipt_sha256: str
    fold_validation_v2_source_receipt_sha256: str
    fold_validation_v2_commit_receipt_sha256: str
    fold_validation_v2_committed_at_ms: int
    four_fold_validation_inputs_v2_receipt_sha256: str
    four_fold_validation_inputs_v2_source_receipt_sha256: str
    four_fold_validation_inputs_v2_commit_receipt_sha256: str
    four_fold_validation_inputs_v2_committed_at_ms: int
    source_bundle_v2_receipt_sha256: str
    source_bundle_v2_source_receipt_sha256: str
    source_bundle_v2_commit_receipt_sha256: str
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_witness_receipt_sha256: str
    replay_dependency_index_v2_receipt_sha256: str
    runtime_sources_v2_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    validation_sources_v2_receipt_sha256: str
    validation_sources_v2_source_receipt_sha256: str
    validation_sources_v2_commit_receipt_sha256: str
    validation_registry_v2_receipt_sha256: str
    validation_registry_v2_source_receipt_sha256: str
    validation_registry_v2_commit_receipt_sha256: str
    primary_outcome_v2_receipts: tuple[str, ...]
    primary_outcome_v2_source_receipts: tuple[str, ...]
    primary_outcome_v2_commit_receipts: tuple[str, ...]
    ladder_outcome_v2_receipts: tuple[str, ...]
    ladder_outcome_v2_source_receipts: tuple[str, ...]
    ladder_outcome_v2_commit_receipts: tuple[str, ...]
    fixed_control_outcome_v2_receipt_sha256: str
    fixed_control_outcome_v2_source_receipt_sha256: str
    fixed_control_outcome_v2_commit_receipt_sha256: str
    base_fold_validation_v1_receipt_sha256: str
    base_policy_selection_authority_v2_receipt_sha256: str
    base_policy_selection_authority_v2_source_receipt_sha256: str
    base_policy_selection_authority_v2_commit_receipt_sha256: str
    base_policy_selection_authority_v2_committed_at_ms: int
    selection_v2_receipt_sha256: str
    expected_candidate_checkpoint_authority_receipts: tuple[str, ...]
    candidate_receipts: tuple[str, ...]
    candidate_inventory_sha256: str
    ranked_candidate_receipts: tuple[str, ...]
    ranked_candidate_inventory_sha256: str
    candidate_checkpoint_inventory_sha256: str
    selected_checkpoint_authority_receipt_sha256: str
    selected_checkpoint_receipt_sha256: str
    selected_model_state_receipt_sha256: str
    selected_update_index: int
    selected_candidate_receipt_sha256: str
    selected_candidate_validation_eligible: bool
    validation_eligibility_failures: tuple[str, ...]
    selection_pool_kind: str
    source_data_qualified: bool
    positive_profitability_authorization_eligible: bool
    validation_selection_specification_sha256: str
    numerical_comparison_specification_sha256: str
    semantic_receipt_sha256: str
    runtime_selection_replayed: bool = False
    development_policy_selection_authorized: bool = False
    policy_freezing_authorized: bool = False
    outer_diagnostic_preparation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV4 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV2 | None = (
        field(default=None, compare=False, repr=False)
    )
    _runtime_base_selection_authority_v2: (
        MassiveAdaptiveRLPolicySelectionAuthorityV2 | None
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
                "runtime_selection_replayed",
                "development_policy_selection_authorized",
                "policy_freezing_authorized",
                "outer_diagnostic_preparation_authorized",
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
            and self.runtime_selection_replayed
            and self.development_policy_selection_authorized
            and self.policy_freezing_authorized
            and self.outer_diagnostic_preparation_authorized
            and self.source_data_qualified
        )

    @property
    def runtime_selection(self) -> MassiveAdaptiveRLPolicySelectionV2:
        if self._runtime_base_selection_authority_v2 is None:
            raise MassiveAdaptiveRLPolicySelectionV3Error(
                "policy selection V3 has no runtime selection"
            )
        selection, _candidates = _runtime_selection_v2(
            self._runtime_base_selection_authority_v2
        )
        return selection

    @property
    def runtime_candidates(self) -> tuple[MassiveAdaptiveRLPolicyCandidateV2, ...]:
        if self._runtime_base_selection_authority_v2 is None:
            raise MassiveAdaptiveRLPolicySelectionV3Error(
                "policy selection V3 has no runtime candidates"
            )
        _selection, candidates = _runtime_selection_v2(
            self._runtime_base_selection_authority_v2
        )
        return candidates

    def validate(self) -> None:
        runtime_values = (
            self._runtime_manifest,
            self._runtime_validation_authority,
            self._runtime_base_selection_authority_v2,
        )
        runtime = all(value is not None for value in runtime_values)
        if any(value is not None for value in runtime_values) != runtime:
            raise MassiveAdaptiveRLPolicySelectionV3Error(
                "policy selection V3 runtime is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        expected_authorized = bool(
            runtime
            and self.source_transaction_verified
            and self.source_data_qualified
            and self._runtime_validation_authority is not None
            and self._runtime_validation_authority.development_stage_authorized
            and self._runtime_base_selection_authority_v2 is not None
            and self._runtime_base_selection_authority_v2.development_policy_selection_authorized
            and self._runtime_base_selection_authority_v2.policy_freezing_authorized
        )
        candidate_count = self.fold_index + 1
        inventories: tuple[Sequence[object], ...] = (
            self.primary_outcome_v2_receipts,
            self.primary_outcome_v2_source_receipts,
            self.primary_outcome_v2_commit_receipts,
            self.ladder_outcome_v2_receipts,
            self.ladder_outcome_v2_source_receipts,
            self.ladder_outcome_v2_commit_receipts,
            self.expected_candidate_checkpoint_authority_receipts,
            self.candidate_receipts,
            self.ranked_candidate_receipts,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or any(len(inventory) != candidate_count for inventory in inventories)
            or len(set(self.expected_candidate_checkpoint_authority_receipts))
            != candidate_count
            or len(set(self.candidate_receipts)) != candidate_count
            or set(self.ranked_candidate_receipts) != set(self.candidate_receipts)
            or self.candidate_inventory_sha256
            != semantic_sha256(self.candidate_receipts)
            or self.candidate_checkpoint_inventory_sha256
            != semantic_sha256(self.expected_candidate_checkpoint_authority_receipts)
            or self.ranked_candidate_inventory_sha256
            != semantic_sha256(self.ranked_candidate_receipts)
            or self.selected_candidate_receipt_sha256 not in self.candidate_receipts
            or isinstance(self.selected_update_index, bool)
            or not isinstance(self.selected_update_index, int)
            or self.selected_update_index < 0
            or self.selection_pool_kind not in {"eligible", "all-no-eligible"}
            or (self.selection_pool_kind == "eligible")
            != self.selected_candidate_validation_eligible
            or (not self.validation_eligibility_failures)
            != self.selected_candidate_validation_eligible
            or tuple(sorted(set(self.validation_eligibility_failures)))
            != self.validation_eligibility_failures
            or not set(self.validation_eligibility_failures).issubset(
                MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1
            )
            or self.positive_profitability_authorization_eligible
            != bool(
                self.source_data_qualified
                and self.selected_candidate_validation_eligible
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_selection_replayed != runtime
            or self.development_policy_selection_authorized != expected_authorized
            or self.policy_freezing_authorized != expected_authorized
            or self.outer_diagnostic_preparation_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.validation_selection_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
            or self.numerical_comparison_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicySelectionV3Error(
                "policy selection authority V3 differs"
            )
        committed_times = (
            self.four_fold_validation_inputs_v2_committed_at_ms,
            self.fold_validation_v2_committed_at_ms,
            self.base_policy_selection_authority_v2_committed_at_ms,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in committed_times
            )
            or not committed_times[0] < committed_times[1] < committed_times[2]
        ):
            raise MassiveAdaptiveRLPolicySelectionV3Error(
                "policy selection V3 chronology differs"
            )
        if runtime:
            assert self._runtime_manifest is not None
            assert self._runtime_validation_authority is not None
            assert self._runtime_base_selection_authority_v2 is not None
            expected = _selection_v3_body(
                manifest=self._runtime_manifest,
                validation_authority=self._runtime_validation_authority,
                base_selection_authority_v2=(self._runtime_base_selection_authority_v2),
            )
            if self.semantic_unsigned() != {
                **expected,
                "profitability_reporting_authorized": False,
                "outer_evaluation_authorized": False,
                "lockbox_access_authorized": False,
                "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
                "specification_sha256": (
                    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SPEC_SHA256
                ),
                "implementation_source_sha256": (
                    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256
                ),
                "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SCHEMA,
            }:
                raise MassiveAdaptiveRLPolicySelectionV3Error(
                    "policy selection V3 runtime replay differs"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= self.base_policy_selection_authority_v2_committed_at_ms
        ):
            raise MassiveAdaptiveRLPolicySelectionV3Error(
                "policy selection V3 source transaction differs"
            )
        for descriptor in fields(self):
            if descriptor.name.endswith("_sha256"):
                _digest("policy selection V3", getattr(self, descriptor.name))
        for inventory in inventories:
            for value in inventory:
                _digest("policy selection V3 inventory", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_policy_selection_authority_v3(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV2,
    base_selection_authority_v2: MassiveAdaptiveRLPolicySelectionAuthorityV2,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
    body = _selection_v3_body(
        manifest=manifest,
        validation_authority=validation_authority,
        base_selection_authority_v2=base_selection_authority_v2,
    )
    provisional = MassiveAdaptiveRLPolicySelectionAuthorityV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_selection_replayed=True,
        development_policy_selection_authorized=False,
        policy_freezing_authorized=False,
        outer_diagnostic_preparation_authorized=False,
        _runtime_manifest=manifest,
        _runtime_validation_authority=validation_authority,
        _runtime_base_selection_authority_v2=base_selection_authority_v2,
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
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "primary_outcome_v2_receipts",
        "primary_outcome_v2_source_receipts",
        "primary_outcome_v2_commit_receipts",
        "ladder_outcome_v2_receipts",
        "ladder_outcome_v2_source_receipts",
        "ladder_outcome_v2_commit_receipts",
        "expected_candidate_checkpoint_authority_receipts",
        "candidate_receipts",
        "ranked_candidate_receipts",
        "validation_eligibility_failures",
    ):
        rows = body.get(name)
        if not isinstance(rows, list):
            raise MassiveAdaptiveRLPolicySelectionV3Error(
                "policy selection V3 inventory is malformed"
            )
        body[name] = tuple(rows)
    return body


def parse_massive_adaptive_rl_policy_selection_authority_v3(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
    body = _parse_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLPolicySelectionAuthorityV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_policy_selection_authority_v3(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    verified_at_ms: int,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
    return parse_massive_adaptive_rl_policy_selection_authority_v3(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=policy_selection_authority_relative_path_v3(
                manifest=manifest,
                fold_index=fold_index,
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_policy_selection_authority_v3(
    *,
    authority: MassiveAdaptiveRLPolicySelectionAuthorityV3,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV2,
    base_selection_authority_v2: MassiveAdaptiveRLPolicySelectionAuthorityV2,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
    authority.validate()
    expected = build_massive_adaptive_rl_policy_selection_authority_v3(
        manifest=manifest,
        validation_authority=validation_authority,
        base_selection_authority_v2=base_selection_authority_v2,
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != policy_selection_authority_relative_path_v3(
            manifest=manifest,
            fold_index=validation_authority.fold_index,
        )
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection authority V3 does not replay"
        )
    result = replace(
        authority,
        runtime_selection_replayed=True,
        development_policy_selection_authorized=authority.source_data_qualified,
        policy_freezing_authorized=authority.source_data_qualified,
        outer_diagnostic_preparation_authorized=authority.source_data_qualified,
        _runtime_manifest=manifest,
        _runtime_validation_authority=validation_authority,
        _runtime_base_selection_authority_v2=base_selection_authority_v2,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_policy_selection_authority_v3(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV2,
    base_selection_authority_v2: MassiveAdaptiveRLPolicySelectionAuthorityV2,
    committed_at_ms: int,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
    authority = build_massive_adaptive_rl_policy_selection_authority_v3(
        manifest=manifest,
        validation_authority=validation_authority,
        base_selection_authority_v2=base_selection_authority_v2,
    )
    relative = policy_selection_authority_relative_path_v3(
        manifest=manifest,
        fold_index=authority.fold_index,
    )
    if _transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection authority V3 already exists"
        )
    if (
        isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms
        <= authority.base_policy_selection_authority_v2_committed_at_ms
    ):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 must follow its V2 computational witness"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-POLICY-SELECTION-V3-{authority.experiment_id}-"
            f"FOLD{authority.fold_index}"
        ),
    )
    return authorize_massive_adaptive_rl_policy_selection_authority_v3(
        authority=load_massive_adaptive_rl_policy_selection_authority_v3(
            root=root,
            manifest=manifest,
            fold_index=authority.fold_index,
            verified_at_ms=committed_at_ms,
        ),
        manifest=manifest,
        validation_authority=validation_authority,
        base_selection_authority_v2=base_selection_authority_v2,
    )


def _load_and_authorize_base_selection_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV2,
    verified_at_ms: int,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV2:
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=policy_selection_v2_witness_relative_path_v3(
            manifest=manifest,
            fold_index=validation_authority.fold_index,
        ),
        verified_at_ms=verified_at_ms,
    )
    return authorize_massive_adaptive_rl_policy_selection_authority_v2(
        root=root,
        authority=parse_massive_adaptive_rl_policy_selection_authority_v2(
            root=root,
            loaded_source=loaded,
        ),
        manifest=manifest,
        validation_authority=validation_authority.base_fold_validation_v1,
    )


def run_or_resume_massive_adaptive_rl_policy_selection_authority_v3(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    validation_authority: MassiveAdaptiveRLFoldValidationAuthorityV2,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
    """Create or strictly replay the canonical V2-backed fold selection."""

    if type(validation_authority) is not MassiveAdaptiveRLFoldValidationAuthorityV2:
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 requires FoldValidationAuthority V2"
        )
    if (
        isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms < 0
        or not isinstance(allow_materialize, bool)
    ):
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 execution arguments differ"
        )
    manifest.validate()
    validation_authority.validate()
    if not validation_authority.development_stage_authorized:
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 validation evidence is not authorized"
        )
    fold_index = validation_authority.fold_index
    base_relative = policy_selection_v2_witness_relative_path_v3(
        manifest=manifest,
        fold_index=fold_index,
    )
    relative = policy_selection_authority_relative_path_v3(
        manifest=manifest,
        fold_index=fold_index,
    )
    base_exists = _transaction_exists(root=root, relative=base_relative)
    authority_exists = _transaction_exists(root=root, relative=relative)
    if authority_exists and not base_exists:
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection V3 cannot repair its missing V2 witness"
        )
    if base_exists:
        base = _load_and_authorize_base_selection_v2(
            root=root,
            manifest=manifest,
            validation_authority=validation_authority,
            verified_at_ms=committed_at_ms,
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLPolicySelectionV3Error(
                "canonical policy selection V2 witness is absent"
            )
        fold_time = cast(int, validation_authority.source_transaction_committed_at_ms)
        base = materialize_massive_adaptive_rl_policy_selection_authority_v2(
            root=root,
            artifact_id=policy_selection_v2_witness_artifact_id_v3(
                manifest=manifest,
                fold_index=fold_index,
            ),
            manifest=manifest,
            validation_authority=validation_authority.base_fold_validation_v1,
            committed_at_ms=max(committed_at_ms, fold_time + 1),
        )
    if authority_exists:
        return authorize_massive_adaptive_rl_policy_selection_authority_v3(
            authority=load_massive_adaptive_rl_policy_selection_authority_v3(
                root=root,
                manifest=manifest,
                fold_index=fold_index,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            validation_authority=validation_authority,
            base_selection_authority_v2=base,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLPolicySelectionV3Error(
            "policy selection authority V3 is absent"
        )
    return materialize_massive_adaptive_rl_policy_selection_authority_v3(
        root=root,
        manifest=manifest,
        validation_authority=validation_authority,
        base_selection_authority_v2=base,
        committed_at_ms=max(
            committed_at_ms + 1,
            base.loaded_source.commit.committed_at_ms + 1,
        ),
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_DATASET",
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V3_SPEC_SHA256",
    "MassiveAdaptiveRLPolicySelectionAuthorityV3",
    "MassiveAdaptiveRLPolicySelectionV3Error",
    "authorize_massive_adaptive_rl_policy_selection_authority_v3",
    "build_massive_adaptive_rl_policy_selection_authority_v3",
    "load_massive_adaptive_rl_policy_selection_authority_v3",
    "materialize_massive_adaptive_rl_policy_selection_authority_v3",
    "parse_massive_adaptive_rl_policy_selection_authority_v3",
    "policy_selection_authority_relative_path_v3",
    "policy_selection_v2_witness_artifact_id_v3",
    "policy_selection_v2_witness_relative_path_v3",
    "run_or_resume_massive_adaptive_rl_policy_selection_authority_v3",
]
