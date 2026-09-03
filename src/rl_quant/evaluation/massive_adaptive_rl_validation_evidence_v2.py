"""V2 source-generation envelopes for adaptive-RL validation evidence.

The existing V1 evaluators remain exact computational witnesses.  They do not
carry the validation-complete source generation because that generation did
not exist when their schemas were defined.  This module wraps each persisted
V1 inner-validation outcome and each persisted V1 fold aggregate with the
exact V2 source, registry, and all-fold input barrier that had to precede it.

Generic reload is integrity-only.  Runtime authorization requires the exact
V1 evidence plus the exact replayed V2 authorities; a V1-only outcome cannot
be promoted through this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
import json
from pathlib import Path
from typing import TypeAlias, cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_cost_ladder_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256,
    MassiveAdaptiveRLCostLadderAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_validation_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveRLFixedControlValidationAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveRLFoldValidationAuthorityV1,
    fold_validation_authority_relative_path_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V2_SPEC_SHA256,
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    validate_massive_adaptive_rl_validation_outcome_barrier_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_trace_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256,
    MassiveAdaptiveRLPolicyTraceAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    validation_cost_ladder_relative_path_v1,
    validation_fixed_control_relative_path_v1,
    validation_primary_trace_relative_path_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v2 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SPEC_SHA256,
    MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    MassiveAdaptiveRLValidationSourcesAuthorityV2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MassiveAdaptiveRLExperimentManifestV4,
)


MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-validation-outcome-authority-v2"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_DATASET = (
    "massive-adaptive-rl-validation-outcome-authority-v2"
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fold-validation-authority-v2"
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_DATASET = (
    "massive-adaptive-rl-fold-validation-authority-v2"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_SCHEMA,
            "encoding": "canonical-json-v2-source-generation-outcome-envelope",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SCHEMA,
        "encoding": "canonical-json-v2-source-generation-fold-envelope",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V2_SPEC_SHA256 = semantic_sha256(
    {
        "base_outcome": "exact-persisted-runtime-replayed-inner-validation-v1",
        "base_outcome_schemas": (
            MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA,
            MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SCHEMA,
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SCHEMA,
        ),
        "base_outcome_implementation_source_sha256s": (
            MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256,
            MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256,
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
        ),
        "base_fc06_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SPEC_SHA256
        ),
        "validation_inputs_v2_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_INPUTS_V2_SOURCE_SHA256
        ),
        "validation_sources_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SOURCES_AUTHORITY_V2_SPEC_SHA256
        ),
        "validation_registry_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_ENVIRONMENT_REGISTRY_V2_SPEC_SHA256
        ),
        "four_fold_barrier_v2_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_V2_SPEC_SHA256
        ),
        "four_fold_barrier_v2_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_INPUTS_AUTHORITY_V2_SOURCE_SHA256
        ),
        "ordering": "v2-barrier-before-v1-outcome-before-v2-envelope",
        "checkpoint_population": "ppo-checkpoint-must-be-barrier-member",
        "publication": "manifest-fold-role-subject-derived-create-only",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V2_SPEC_SHA256 = semantic_sha256(
    {
        "base_fold": "exact-persisted-runtime-replayed-fold-validation-v1",
        "base_fold_schema": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SCHEMA,
        "base_fold_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SPEC_SHA256
        ),
        "base_fold_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V1_SOURCE_SHA256
        ),
        "inputs": "exact-v2-source-registry-and-four-fold-barrier",
        "outcomes": "exact-v2-primary-ladder-and-fc06-envelopes",
        "candidate_population": "exact-preregistered-checkpoint-inventory",
        "ordering": "all-v2-outcome-envelopes-and-v1-fold-before-v2-fold",
        "publication": "manifest-and-fold-derived-create-only",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)

_PRIMARY = "ppo-primary"
_LADDER = "ppo-cost-ladder"
_FC06 = "fc06-primary"
_OUTCOME_KINDS = (_PRIMARY, _LADDER, _FC06)


class MassiveAdaptiveRLValidationEvidenceV2Error(ValueError):
    """V2 validation evidence is absent, mixed, late, or inconsistent."""


ValidationOutcomeAuthorityV1: TypeAlias = (
    MassiveAdaptiveRLPolicyTraceAuthorityV1
    | MassiveAdaptiveRLCostLadderAuthorityV1
    | MassiveAdaptiveRLFixedControlValidationAuthorityV1
)


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
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
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation evidence V2 source transaction is incomplete"
        )
    return all(present)


@dataclass(frozen=True, slots=True)
class _OutcomeFactsV2:
    outcome_kind: str
    fold_index: int
    subject_receipt_sha256: str
    checkpoint_authority_receipt_sha256: str | None
    base_outcome_receipt_sha256: str
    base_outcome_schema: str
    base_outcome_implementation_source_sha256: str
    base_outcome_source_receipt_sha256: str
    base_outcome_commit_receipt_sha256: str
    base_outcome_committed_at_ms: int
    base_validation_sources_v1_receipt_sha256: str
    base_validation_registry_v1_receipt_sha256: str
    base_validation_registry_v1_source_receipt_sha256: str
    base_validation_registry_v1_commit_receipt_sha256: str
    base_validation_registry_v1_committed_at_ms: int
    base_four_fold_validation_inputs_v1_receipt_sha256: str
    base_four_fold_validation_inputs_v1_source_receipt_sha256: str
    base_four_fold_validation_inputs_v1_commit_receipt_sha256: str
    base_four_fold_validation_inputs_v1_committed_at_ms: int
    validation_context_receipt_sha256: str
    source_data_qualified: bool


def _outcome_facts_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    outcome: ValidationOutcomeAuthorityV1,
) -> _OutcomeFactsV2:
    outcome.validate()
    loaded = outcome.loaded_source
    loaded.validate()
    if type(outcome) is MassiveAdaptiveRLPolicyTraceAuthorityV1:
        if (
            outcome.evaluation_role != "inner_validation"
            or not outcome.development_policy_evaluation_authorized
        ):
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "V2 primary evidence requires an authorized inner-validation trace"
            )
        kind = _PRIMARY
        subject = outcome.checkpoint_authority_receipt_sha256
        checkpoint = subject
        expected_path = validation_primary_trace_relative_path_v1(
            manifest=manifest,
            fold_index=outcome.fold_index,
            checkpoint_authority_receipt_sha256=subject,
        )
    elif type(outcome) is MassiveAdaptiveRLCostLadderAuthorityV1:
        if (
            outcome.evaluation_role != "inner_validation"
            or not outcome.development_policy_selection_authorized
        ):
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "V2 ladder evidence requires an authorized inner-validation ladder"
            )
        kind = _LADDER
        subject = outcome.checkpoint_authority_receipt_sha256
        checkpoint = subject
        expected_path = validation_cost_ladder_relative_path_v1(
            manifest=manifest,
            fold_index=outcome.fold_index,
            checkpoint_authority_receipt_sha256=subject,
        )
    elif type(outcome) is MassiveAdaptiveRLFixedControlValidationAuthorityV1:
        if not outcome.development_stage_authorized:
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "V2 FC06 evidence requires an authorized validation trace"
            )
        kind = _FC06
        subject = outcome.fixed_control_selection_authority_receipt_sha256
        checkpoint = None
        expected_path = validation_fixed_control_relative_path_v1(
            manifest=manifest,
            fold_index=outcome.fold_index,
        )
    else:
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation outcome V2 requires an exact V1 evidence type"
        )
    if loaded.payload_relative_path != expected_path:
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation outcome V1 path is not canonical"
        )
    source_receipt = loaded.receipt.receipt_sha256
    commit_receipt = loaded.commit.receipt_sha256
    for value in (
        subject,
        outcome.semantic_receipt_sha256,
        outcome.implementation_source_sha256,
        source_receipt,
        commit_receipt,
        outcome.validation_sources_authority_receipt_sha256,
        outcome.validation_environment_registry_receipt_sha256,
        outcome.validation_environment_registry_source_receipt_sha256,
        outcome.validation_environment_registry_commit_receipt_sha256,
        outcome.four_fold_validation_inputs_authority_receipt_sha256,
        outcome.four_fold_validation_inputs_source_receipt_sha256,
        outcome.four_fold_validation_inputs_commit_receipt_sha256,
        outcome.validation_context_receipt_sha256,
    ):
        _digest("validation outcome V1", value)
    return _OutcomeFactsV2(
        outcome_kind=kind,
        fold_index=outcome.fold_index,
        subject_receipt_sha256=subject,
        checkpoint_authority_receipt_sha256=checkpoint,
        base_outcome_receipt_sha256=outcome.semantic_receipt_sha256,
        base_outcome_schema=outcome.schema,
        base_outcome_implementation_source_sha256=(
            outcome.implementation_source_sha256
        ),
        base_outcome_source_receipt_sha256=source_receipt,
        base_outcome_commit_receipt_sha256=commit_receipt,
        base_outcome_committed_at_ms=loaded.commit.committed_at_ms,
        base_validation_sources_v1_receipt_sha256=cast(
            str, outcome.validation_sources_authority_receipt_sha256
        ),
        base_validation_registry_v1_receipt_sha256=cast(
            str, outcome.validation_environment_registry_receipt_sha256
        ),
        base_validation_registry_v1_source_receipt_sha256=cast(
            str, outcome.validation_environment_registry_source_receipt_sha256
        ),
        base_validation_registry_v1_commit_receipt_sha256=cast(
            str, outcome.validation_environment_registry_commit_receipt_sha256
        ),
        base_validation_registry_v1_committed_at_ms=cast(
            int, outcome.validation_environment_registry_committed_at_ms
        ),
        base_four_fold_validation_inputs_v1_receipt_sha256=cast(
            str, outcome.four_fold_validation_inputs_authority_receipt_sha256
        ),
        base_four_fold_validation_inputs_v1_source_receipt_sha256=cast(
            str, outcome.four_fold_validation_inputs_source_receipt_sha256
        ),
        base_four_fold_validation_inputs_v1_commit_receipt_sha256=cast(
            str, outcome.four_fold_validation_inputs_commit_receipt_sha256
        ),
        base_four_fold_validation_inputs_v1_committed_at_ms=cast(
            int, outcome.four_fold_validation_inputs_committed_at_ms
        ),
        validation_context_receipt_sha256=outcome.validation_context_receipt_sha256,
        source_data_qualified=outcome.source_data_qualified,
    )


def validation_outcome_authority_relative_path_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    outcome_kind: str,
    subject_receipt_sha256: str,
) -> str:
    manifest.validate()
    if (
        isinstance(fold_index, bool)
        or fold_index not in range(4)
        or outcome_kind not in _OUTCOME_KINDS
    ):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation outcome V2 canonical identity differs"
        )
    subject = _digest("validation outcome V2 subject", subject_receipt_sha256)
    return (
        "massive-adaptive/rl-validation-outcome-authority-v2/"
        f"v4-{manifest.semantic_receipt_sha256}-fold-{fold_index}-"
        f"{outcome_kind}-{subject}.json"
    )


def fold_validation_authority_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "fold-validation V2 fold differs"
        )
    return (
        "massive-adaptive/rl-fold-validation-authority-v2/"
        f"v4-{manifest.semantic_receipt_sha256}-fold-{fold_index}.json"
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationOutcomeAuthorityV2:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_index: int
    outcome_kind: str
    subject_receipt_sha256: str
    checkpoint_authority_receipt_sha256: str | None
    base_outcome_receipt_sha256: str
    base_outcome_schema: str
    base_outcome_implementation_source_sha256: str
    base_outcome_source_receipt_sha256: str
    base_outcome_commit_receipt_sha256: str
    base_outcome_committed_at_ms: int
    validation_sources_v2_receipt_sha256: str
    validation_sources_v2_source_receipt_sha256: str
    validation_sources_v2_commit_receipt_sha256: str
    validation_registry_v2_receipt_sha256: str
    validation_registry_v2_source_receipt_sha256: str
    validation_registry_v2_commit_receipt_sha256: str
    validation_registry_v2_committed_at_ms: int
    four_fold_validation_inputs_v2_receipt_sha256: str
    four_fold_validation_inputs_v2_source_receipt_sha256: str
    four_fold_validation_inputs_v2_commit_receipt_sha256: str
    four_fold_validation_inputs_v2_committed_at_ms: int
    runtime_sources_v2_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    base_validation_sources_v1_receipt_sha256: str
    base_validation_registry_v1_receipt_sha256: str
    base_validation_registry_v1_source_receipt_sha256: str
    base_validation_registry_v1_commit_receipt_sha256: str
    base_validation_registry_v1_committed_at_ms: int
    base_four_fold_validation_inputs_v1_receipt_sha256: str
    base_four_fold_validation_inputs_v1_source_receipt_sha256: str
    base_four_fold_validation_inputs_v1_commit_receipt_sha256: str
    base_four_fold_validation_inputs_v1_committed_at_ms: int
    validation_context_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_evidence_replayed: bool = False
    development_validation_evidence_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV4 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_base_outcome: ValidationOutcomeAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_validation_sources_v2: (
        MassiveAdaptiveRLValidationSourcesAuthorityV2 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_validation_registry_v2: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV2 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_four_fold_barrier_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2 | None
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
                "runtime_evidence_replayed",
                "development_validation_evidence_authorized",
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
            and self.runtime_evidence_replayed
            and self.development_validation_evidence_authorized
            and self.source_data_qualified
        )

    @property
    def base_outcome(self) -> ValidationOutcomeAuthorityV1:
        self.validate()
        if self._runtime_base_outcome is None:
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "validation outcome V2 has no runtime V1 evidence"
            )
        return self._runtime_base_outcome

    def validate(self) -> None:
        runtime_values = (
            self._runtime_manifest,
            self._runtime_base_outcome,
            self._runtime_validation_sources_v2,
            self._runtime_validation_registry_v2,
            self._runtime_four_fold_barrier_v2,
        )
        runtime = all(value is not None for value in runtime_values)
        if any(value is not None for value in runtime_values) != runtime:
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "validation outcome V2 runtime is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        expected_authorized = bool(runtime and self.source_data_qualified)
        expected_base_identity = {
            _PRIMARY: (
                MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA,
                MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256,
            ),
            _LADDER: (
                MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SCHEMA,
                MASSIVE_ADAPTIVE_RL_COST_LADDER_AUTHORITY_V1_SOURCE_SHA256,
            ),
            _FC06: (
                MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SCHEMA,
                MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_VALIDATION_AUTHORITY_V1_SOURCE_SHA256,
            ),
        }.get(self.outcome_kind)
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or self.outcome_kind not in _OUTCOME_KINDS
            or expected_base_identity
            != (
                self.base_outcome_schema,
                self.base_outcome_implementation_source_sha256,
            )
            or (self.outcome_kind == _FC06)
            != (self.checkpoint_authority_receipt_sha256 is None)
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_evidence_replayed != runtime
            or self.development_validation_evidence_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "validation outcome authority V2 differs"
            )
        if runtime:
            assert self._runtime_manifest is not None
            assert self._runtime_base_outcome is not None
            assert self._runtime_validation_sources_v2 is not None
            assert self._runtime_validation_registry_v2 is not None
            assert self._runtime_four_fold_barrier_v2 is not None
            expected = build_massive_adaptive_rl_validation_outcome_authority_v2(
                manifest=self._runtime_manifest,
                base_outcome=self._runtime_base_outcome,
                validation_sources_v2=self._runtime_validation_sources_v2,
                validation_registry_v2=self._runtime_validation_registry_v2,
                four_fold_validation_inputs_v2=self._runtime_four_fold_barrier_v2,
            )
            if self.semantic_unsigned() != expected.semantic_unsigned():
                raise MassiveAdaptiveRLValidationEvidenceV2Error(
                    "validation outcome V2 runtime evidence differs"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= self.base_outcome_committed_at_ms
        ):
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "validation outcome V2 source transaction differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256") and value is not None:
                _digest("validation outcome V2", value)
        for committed_at_ms in (
            self.base_outcome_committed_at_ms,
            self.validation_registry_v2_committed_at_ms,
            self.four_fold_validation_inputs_v2_committed_at_ms,
            self.base_validation_registry_v1_committed_at_ms,
            self.base_four_fold_validation_inputs_v1_committed_at_ms,
        ):
            if (
                isinstance(committed_at_ms, bool)
                or not isinstance(committed_at_ms, int)
                or committed_at_ms < 0
            ):
                raise MassiveAdaptiveRLValidationEvidenceV2Error(
                    "validation outcome V2 commit time differs"
                )
        if not (
            self.base_validation_registry_v1_committed_at_ms
            < self.base_four_fold_validation_inputs_v1_committed_at_ms
            < self.four_fold_validation_inputs_v2_committed_at_ms
            < self.base_outcome_committed_at_ms
            and self.validation_registry_v2_committed_at_ms
            < self.four_fold_validation_inputs_v2_committed_at_ms
        ):
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "validation outcome V2 chronology differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_validation_outcome_authority_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    base_outcome: ValidationOutcomeAuthorityV1,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_registry_v2: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    four_fold_validation_inputs_v2: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV2:
    manifest.validate()
    facts = _outcome_facts_v2(manifest=manifest, outcome=base_outcome)
    validation_sources_v2.validate()
    validation_registry_v2.validate()
    four_fold_validation_inputs_v2.validate()
    if (
        not validation_sources_v2.development_stage_authorized
        or not validation_registry_v2.development_stage_authorized
        or not four_fold_validation_inputs_v2.development_stage_authorized
        or facts.fold_index != validation_sources_v2.fold_index
        or facts.fold_index != validation_registry_v2.fold_index
        or facts.base_validation_sources_v1_receipt_sha256
        != validation_sources_v2.base_validation_sources_v1_receipt_sha256
        or facts.base_validation_registry_v1_receipt_sha256
        != validation_registry_v2.base_validation_registry_v1_receipt_sha256
        or facts.base_validation_registry_v1_source_receipt_sha256
        != validation_registry_v2.base_validation_registry_v1_source_receipt_sha256
        or facts.base_validation_registry_v1_commit_receipt_sha256
        != validation_registry_v2.base_validation_registry_v1_commit_receipt_sha256
        or facts.base_validation_registry_v1_committed_at_ms
        != validation_registry_v2.base_validation_registry_v1_committed_at_ms
        or facts.base_four_fold_validation_inputs_v1_receipt_sha256
        != four_fold_validation_inputs_v2.base_four_fold_validation_inputs_v1_receipt_sha256
        or facts.base_four_fold_validation_inputs_v1_source_receipt_sha256
        != four_fold_validation_inputs_v2.base_four_fold_validation_inputs_v1_source_receipt_sha256
        or facts.base_four_fold_validation_inputs_v1_commit_receipt_sha256
        != four_fold_validation_inputs_v2.base_four_fold_validation_inputs_v1_commit_receipt_sha256
        or facts.base_four_fold_validation_inputs_v1_committed_at_ms
        != four_fold_validation_inputs_v2.base_four_fold_validation_inputs_v1_committed_at_ms
        or facts.validation_context_receipt_sha256
        != validation_registry_v2.validation_context_receipt_sha256
        or validation_registry_v2.validation_sources_v2_receipt_sha256
        != validation_sources_v2.semantic_receipt_sha256
        or validation_sources_v2.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or four_fold_validation_inputs_v2.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation outcome V2 source generation differs"
        )
    validate_massive_adaptive_rl_validation_outcome_barrier_v2(
        authority=four_fold_validation_inputs_v2,
        validation_environment_registry=validation_registry_v2,
        fold_index=facts.fold_index,
        outcome_committed_at_ms=facts.base_outcome_committed_at_ms,
        checkpoint_authority_receipt_sha256=(facts.checkpoint_authority_receipt_sha256),
    )
    source_receipt = validation_sources_v2.source_receipt_sha256
    source_commit = validation_sources_v2.source_transaction_receipt_sha256
    registry_source = validation_registry_v2.source_receipt_sha256
    registry_commit = validation_registry_v2.source_transaction_receipt_sha256
    registry_time = validation_registry_v2.source_transaction_committed_at_ms
    barrier_source = four_fold_validation_inputs_v2.source_receipt_sha256
    barrier_commit = four_fold_validation_inputs_v2.source_transaction_receipt_sha256
    barrier_time = four_fold_validation_inputs_v2.source_transaction_committed_at_ms
    if any(
        value is None
        for value in (
            source_receipt,
            source_commit,
            registry_source,
            registry_commit,
            registry_time,
            barrier_source,
            barrier_commit,
            barrier_time,
        )
    ):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation outcome V2 inputs are not persisted"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        **{
            descriptor.name: getattr(facts, descriptor.name)
            for descriptor in fields(facts)
        },
        "validation_sources_v2_receipt_sha256": (
            validation_sources_v2.semantic_receipt_sha256
        ),
        "validation_sources_v2_source_receipt_sha256": cast(str, source_receipt),
        "validation_sources_v2_commit_receipt_sha256": cast(str, source_commit),
        "validation_registry_v2_receipt_sha256": (
            validation_registry_v2.semantic_receipt_sha256
        ),
        "validation_registry_v2_source_receipt_sha256": cast(str, registry_source),
        "validation_registry_v2_commit_receipt_sha256": cast(str, registry_commit),
        "validation_registry_v2_committed_at_ms": cast(int, registry_time),
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
        "runtime_sources_v2_receipt_sha256": (
            four_fold_validation_inputs_v2.runtime_sources_v2_receipt_sha256
        ),
        "training_source_projection_sha256": (
            four_fold_validation_inputs_v2.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            four_fold_validation_inputs_v2.validation_source_projection_sha256
        ),
        "source_data_qualified": bool(
            facts.source_data_qualified
            and validation_sources_v2.source_data_qualified
            and validation_registry_v2.source_data_qualified
            and four_fold_validation_inputs_v2.source_data_qualified
        ),
    }
    provisional = MassiveAdaptiveRLValidationOutcomeAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_evidence_replayed=True,
        development_validation_evidence_authorized=bool(body["source_data_qualified"]),
        _runtime_manifest=manifest,
        _runtime_base_outcome=base_outcome,
        _runtime_validation_sources_v2=validation_sources_v2,
        _runtime_validation_registry_v2=validation_registry_v2,
        _runtime_four_fold_barrier_v2=four_fold_validation_inputs_v2,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    _validate_outcome_runtime_without_recursive_build(result, manifest=manifest)
    return result


def _validate_outcome_runtime_without_recursive_build(
    authority: MassiveAdaptiveRLValidationOutcomeAuthorityV2,
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
) -> None:
    """Validate a freshly built authority without re-entering its builder."""

    runtime = authority._runtime_base_outcome is not None
    if not runtime:
        authority.validate()
        return
    saved = replace(
        authority,
        _runtime_manifest=None,
        _runtime_base_outcome=None,
        _runtime_validation_sources_v2=None,
        _runtime_validation_registry_v2=None,
        _runtime_four_fold_barrier_v2=None,
        runtime_evidence_replayed=False,
        development_validation_evidence_authorized=False,
    )
    saved.validate()
    if authority.manifest_v4_receipt_sha256 != manifest.semantic_receipt_sha256:
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation outcome V2 Manifest differs"
        )


def _parse_body(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation evidence V2 is not canonical JSON"
        )
    return dict(cast(Mapping[str, object], value))


def parse_massive_adaptive_rl_validation_outcome_authority_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV2:
    body = _parse_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLValidationOutcomeAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_validation_outcome_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    outcome_kind: str,
    subject_receipt_sha256: str,
    verified_at_ms: int,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV2:
    return parse_massive_adaptive_rl_validation_outcome_authority_v2(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=validation_outcome_authority_relative_path_v2(
                manifest=manifest,
                fold_index=fold_index,
                outcome_kind=outcome_kind,
                subject_receipt_sha256=subject_receipt_sha256,
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_validation_outcome_authority_v2(
    *,
    authority: MassiveAdaptiveRLValidationOutcomeAuthorityV2,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    base_outcome: ValidationOutcomeAuthorityV1,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_registry_v2: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    four_fold_validation_inputs_v2: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV2:
    authority.validate()
    expected = build_massive_adaptive_rl_validation_outcome_authority_v2(
        manifest=manifest,
        base_outcome=base_outcome,
        validation_sources_v2=validation_sources_v2,
        validation_registry_v2=validation_registry_v2,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
    )
    expected_relative = validation_outcome_authority_relative_path_v2(
        manifest=manifest,
        fold_index=expected.fold_index,
        outcome_kind=expected.outcome_kind,
        subject_receipt_sha256=expected.subject_receipt_sha256,
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path != expected_relative
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation outcome authority V2 does not replay"
        )
    result = replace(
        authority,
        runtime_evidence_replayed=True,
        development_validation_evidence_authorized=authority.source_data_qualified,
        _runtime_manifest=manifest,
        _runtime_base_outcome=base_outcome,
        _runtime_validation_sources_v2=validation_sources_v2,
        _runtime_validation_registry_v2=validation_registry_v2,
        _runtime_four_fold_barrier_v2=four_fold_validation_inputs_v2,
    )
    _validate_outcome_runtime_without_recursive_build(result, manifest=manifest)
    return result


def materialize_massive_adaptive_rl_validation_outcome_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    base_outcome: ValidationOutcomeAuthorityV1,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_registry_v2: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    four_fold_validation_inputs_v2: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    committed_at_ms: int,
) -> MassiveAdaptiveRLValidationOutcomeAuthorityV2:
    authority = build_massive_adaptive_rl_validation_outcome_authority_v2(
        manifest=manifest,
        base_outcome=base_outcome,
        validation_sources_v2=validation_sources_v2,
        validation_registry_v2=validation_registry_v2,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
    )
    relative = validation_outcome_authority_relative_path_v2(
        manifest=manifest,
        fold_index=authority.fold_index,
        outcome_kind=authority.outcome_kind,
        subject_receipt_sha256=authority.subject_receipt_sha256,
    )
    if _transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation outcome authority V2 already exists"
        )
    if (
        isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms <= authority.base_outcome_committed_at_ms
    ):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "validation outcome V2 must follow its V1 outcome"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-VALIDATION-OUTCOME-V2-{authority.outcome_kind}-"
            f"FOLD{authority.fold_index}"
        ),
    )
    parsed = parse_massive_adaptive_rl_validation_outcome_authority_v2(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=committed_at_ms,
        ),
    )
    return authorize_massive_adaptive_rl_validation_outcome_authority_v2(
        authority=parsed,
        manifest=manifest,
        base_outcome=base_outcome,
        validation_sources_v2=validation_sources_v2,
        validation_registry_v2=validation_registry_v2,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFoldValidationAuthorityV2:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_index: int
    fold_fit_authority_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    runtime_sources_v2_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    validation_sources_v2_receipt_sha256: str
    validation_sources_v2_source_receipt_sha256: str
    validation_sources_v2_commit_receipt_sha256: str
    validation_registry_v2_receipt_sha256: str
    validation_registry_v2_source_receipt_sha256: str
    validation_registry_v2_commit_receipt_sha256: str
    four_fold_validation_inputs_v2_receipt_sha256: str
    four_fold_validation_inputs_v2_source_receipt_sha256: str
    four_fold_validation_inputs_v2_commit_receipt_sha256: str
    four_fold_validation_inputs_v2_committed_at_ms: int
    expected_checkpoint_authority_receipts: tuple[str, ...]
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
    base_fold_validation_v1_source_receipt_sha256: str
    base_fold_validation_v1_commit_receipt_sha256: str
    base_fold_validation_v1_committed_at_ms: int
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_validation_replayed: bool = False
    development_validation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV4 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_base_fold: MassiveAdaptiveRLFoldValidationAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_validation_sources_v2: (
        MassiveAdaptiveRLValidationSourcesAuthorityV2 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_validation_registry_v2: (
        MassiveAdaptiveRLValidationEnvironmentRegistryV2 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_four_fold_barrier_v2: (
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_primary_outcomes_v2: tuple[
        MassiveAdaptiveRLValidationOutcomeAuthorityV2, ...
    ] = field(default=(), compare=False, repr=False)
    _runtime_ladder_outcomes_v2: tuple[
        MassiveAdaptiveRLValidationOutcomeAuthorityV2, ...
    ] = field(default=(), compare=False, repr=False)
    _runtime_fixed_outcome_v2: MassiveAdaptiveRLValidationOutcomeAuthorityV2 | None = (
        field(default=None, compare=False, repr=False)
    )
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
                "runtime_validation_replayed",
                "development_validation_authorized",
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
            and self.runtime_validation_replayed
            and self.development_validation_authorized
            and self.source_data_qualified
        )

    @property
    def base_fold_validation_v1(self) -> MassiveAdaptiveRLFoldValidationAuthorityV1:
        self.validate()
        if self._runtime_base_fold is None:
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "fold-validation V2 has no runtime V1 witness"
            )
        return self._runtime_base_fold

    def validate(self) -> None:
        scalar_runtime = (
            self._runtime_manifest,
            self._runtime_base_fold,
            self._runtime_validation_sources_v2,
            self._runtime_validation_registry_v2,
            self._runtime_four_fold_barrier_v2,
            self._runtime_fixed_outcome_v2,
        )
        runtime = all(value is not None for value in scalar_runtime) and bool(
            self._runtime_primary_outcomes_v2 and self._runtime_ladder_outcomes_v2
        )
        any_runtime = any(value is not None for value in scalar_runtime) or bool(
            self._runtime_primary_outcomes_v2 or self._runtime_ladder_outcomes_v2
        )
        if any_runtime != runtime:
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "fold-validation V2 runtime is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        expected_authorized = bool(runtime and self.source_data_qualified)
        inventories: tuple[Sequence[object], ...] = (
            self.expected_checkpoint_authority_receipts,
            self.primary_outcome_v2_receipts,
            self.primary_outcome_v2_source_receipts,
            self.primary_outcome_v2_commit_receipts,
            self.ladder_outcome_v2_receipts,
            self.ladder_outcome_v2_source_receipts,
            self.ladder_outcome_v2_commit_receipts,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or len(self.expected_checkpoint_authority_receipts) != self.fold_index + 1
            or any(
                len(rows) != len(self.expected_checkpoint_authority_receipts)
                for rows in inventories[1:]
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_validation_replayed != runtime
            or self.development_validation_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or self.four_fold_validation_inputs_v2_committed_at_ms
            >= self.base_fold_validation_v1_committed_at_ms
        ):
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "fold-validation authority V2 differs"
            )
        if runtime:
            assert self._runtime_manifest is not None
            assert self._runtime_base_fold is not None
            assert self._runtime_validation_sources_v2 is not None
            assert self._runtime_validation_registry_v2 is not None
            assert self._runtime_four_fold_barrier_v2 is not None
            assert self._runtime_fixed_outcome_v2 is not None
            expected = build_massive_adaptive_rl_fold_validation_authority_v2(
                manifest=self._runtime_manifest,
                base_fold_validation_v1=self._runtime_base_fold,
                validation_sources_v2=self._runtime_validation_sources_v2,
                validation_registry_v2=self._runtime_validation_registry_v2,
                four_fold_validation_inputs_v2=self._runtime_four_fold_barrier_v2,
                primary_outcomes_v2=self._runtime_primary_outcomes_v2,
                ladder_outcomes_v2=self._runtime_ladder_outcomes_v2,
                fixed_control_outcome_v2=self._runtime_fixed_outcome_v2,
            )
            if self.semantic_unsigned() != expected.semantic_unsigned():
                raise MassiveAdaptiveRLValidationEvidenceV2Error(
                    "fold-validation V2 runtime evidence differs"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= self.base_fold_validation_v1_committed_at_ms
        ):
            raise MassiveAdaptiveRLValidationEvidenceV2Error(
                "fold-validation V2 source transaction differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest("fold-validation V2", value)
        for inventory in inventories:
            for value in inventory:
                _digest("fold-validation V2 inventory", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _build_fold_validation_authority_v2_unchecked(
    *,
    base_fold_validation_v1: MassiveAdaptiveRLFoldValidationAuthorityV1,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_registry_v2: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    four_fold_validation_inputs_v2: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    primary_outcomes_v2: Sequence[MassiveAdaptiveRLValidationOutcomeAuthorityV2],
    ladder_outcomes_v2: Sequence[MassiveAdaptiveRLValidationOutcomeAuthorityV2],
    fixed_control_outcome_v2: MassiveAdaptiveRLValidationOutcomeAuthorityV2,
) -> MassiveAdaptiveRLFoldValidationAuthorityV2:
    primary = tuple(primary_outcomes_v2)
    ladders = tuple(ladder_outcomes_v2)
    fixed = fixed_control_outcome_v2
    base_loaded = base_fold_validation_v1.loaded_source
    source_receipts = tuple(cast(str, row.source_receipt_sha256) for row in primary)
    source_commits = tuple(
        cast(str, row.source_transaction_receipt_sha256) for row in primary
    )
    ladder_sources = tuple(cast(str, row.source_receipt_sha256) for row in ladders)
    ladder_commits = tuple(
        cast(str, row.source_transaction_receipt_sha256) for row in ladders
    )
    barrier_source = cast(str, four_fold_validation_inputs_v2.source_receipt_sha256)
    barrier_commit = cast(
        str, four_fold_validation_inputs_v2.source_transaction_receipt_sha256
    )
    barrier_time = cast(
        int, four_fold_validation_inputs_v2.source_transaction_committed_at_ms
    )
    body = {
        "experiment_id": base_fold_validation_v1.experiment_id,
        "manifest_v4_receipt_sha256": (
            base_fold_validation_v1.manifest_v4_receipt_sha256
        ),
        "training_manifest_v3_receipt_sha256": (
            base_fold_validation_v1.training_manifest_v3_receipt_sha256
        ),
        "fold_index": base_fold_validation_v1.fold_index,
        "fold_fit_authority_receipt_sha256": (
            base_fold_validation_v1.fold_fit_authority_receipt_sha256
        ),
        "four_fold_fit_authority_receipt_sha256": (
            base_fold_validation_v1.four_fold_fit_authority_receipt_sha256
        ),
        "runtime_sources_v2_receipt_sha256": (
            four_fold_validation_inputs_v2.runtime_sources_v2_receipt_sha256
        ),
        "training_source_projection_sha256": (
            four_fold_validation_inputs_v2.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            four_fold_validation_inputs_v2.validation_source_projection_sha256
        ),
        "validation_sources_v2_receipt_sha256": (
            validation_sources_v2.semantic_receipt_sha256
        ),
        "validation_sources_v2_source_receipt_sha256": cast(
            str, validation_sources_v2.source_receipt_sha256
        ),
        "validation_sources_v2_commit_receipt_sha256": cast(
            str, validation_sources_v2.source_transaction_receipt_sha256
        ),
        "validation_registry_v2_receipt_sha256": (
            validation_registry_v2.semantic_receipt_sha256
        ),
        "validation_registry_v2_source_receipt_sha256": cast(
            str, validation_registry_v2.source_receipt_sha256
        ),
        "validation_registry_v2_commit_receipt_sha256": cast(
            str, validation_registry_v2.source_transaction_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_receipt_sha256": (
            four_fold_validation_inputs_v2.semantic_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_source_receipt_sha256": barrier_source,
        "four_fold_validation_inputs_v2_commit_receipt_sha256": barrier_commit,
        "four_fold_validation_inputs_v2_committed_at_ms": barrier_time,
        "expected_checkpoint_authority_receipts": (
            base_fold_validation_v1.expected_checkpoint_authority_receipts
        ),
        "primary_outcome_v2_receipts": tuple(
            row.semantic_receipt_sha256 for row in primary
        ),
        "primary_outcome_v2_source_receipts": source_receipts,
        "primary_outcome_v2_commit_receipts": source_commits,
        "ladder_outcome_v2_receipts": tuple(
            row.semantic_receipt_sha256 for row in ladders
        ),
        "ladder_outcome_v2_source_receipts": ladder_sources,
        "ladder_outcome_v2_commit_receipts": ladder_commits,
        "fixed_control_outcome_v2_receipt_sha256": fixed.semantic_receipt_sha256,
        "fixed_control_outcome_v2_source_receipt_sha256": cast(
            str, fixed.source_receipt_sha256
        ),
        "fixed_control_outcome_v2_commit_receipt_sha256": cast(
            str, fixed.source_transaction_receipt_sha256
        ),
        "base_fold_validation_v1_receipt_sha256": (
            base_fold_validation_v1.semantic_receipt_sha256
        ),
        "base_fold_validation_v1_source_receipt_sha256": (
            base_loaded.receipt.receipt_sha256
        ),
        "base_fold_validation_v1_commit_receipt_sha256": (
            base_loaded.commit.receipt_sha256
        ),
        "base_fold_validation_v1_committed_at_ms": (base_loaded.commit.committed_at_ms),
        "source_data_qualified": bool(
            base_fold_validation_v1.source_data_qualified
            and validation_sources_v2.source_data_qualified
            and validation_registry_v2.source_data_qualified
            and four_fold_validation_inputs_v2.source_data_qualified
            and all(row.source_data_qualified for row in (*primary, *ladders, fixed))
        ),
    }
    provisional = MassiveAdaptiveRLFoldValidationAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_validation_replayed=True,
        development_validation_authorized=bool(body["source_data_qualified"]),
        _runtime_base_fold=base_fold_validation_v1,
        _runtime_validation_sources_v2=validation_sources_v2,
        _runtime_validation_registry_v2=validation_registry_v2,
        _runtime_four_fold_barrier_v2=four_fold_validation_inputs_v2,
        _runtime_primary_outcomes_v2=primary,
        _runtime_ladder_outcomes_v2=ladders,
        _runtime_fixed_outcome_v2=fixed,
    )
    return replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )


def build_massive_adaptive_rl_fold_validation_authority_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    base_fold_validation_v1: MassiveAdaptiveRLFoldValidationAuthorityV1,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_registry_v2: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    four_fold_validation_inputs_v2: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    primary_outcomes_v2: Sequence[MassiveAdaptiveRLValidationOutcomeAuthorityV2],
    ladder_outcomes_v2: Sequence[MassiveAdaptiveRLValidationOutcomeAuthorityV2],
    fixed_control_outcome_v2: MassiveAdaptiveRLValidationOutcomeAuthorityV2,
) -> MassiveAdaptiveRLFoldValidationAuthorityV2:
    manifest.validate()
    base_fold_validation_v1.validate()
    validation_sources_v2.validate()
    validation_registry_v2.validate()
    four_fold_validation_inputs_v2.validate()
    primary = tuple(primary_outcomes_v2)
    ladders = tuple(ladder_outcomes_v2)
    fixed = fixed_control_outcome_v2
    for row in (*primary, *ladders, fixed):
        row.validate()
    fold_index = base_fold_validation_v1.fold_index
    barrier_index = four_fold_validation_inputs_v2.fold_indices.index(fold_index)
    if (
        not base_fold_validation_v1.development_stage_authorized
        or not validation_sources_v2.development_stage_authorized
        or not validation_registry_v2.development_stage_authorized
        or not four_fold_validation_inputs_v2.development_stage_authorized
        or not all(
            row.development_stage_authorized for row in (*primary, *ladders, fixed)
        )
        or base_fold_validation_v1.loaded_source.payload_relative_path
        != fold_validation_authority_relative_path_v1(
            manifest=manifest, fold_index=fold_index
        )
        or base_fold_validation_v1.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
        or validation_sources_v2.fold_index != fold_index
        or validation_registry_v2.fold_index != fold_index
        or base_fold_validation_v1.validation_sources_authority_receipt_sha256
        != validation_sources_v2.base_validation_sources_v1_receipt_sha256
        or base_fold_validation_v1.validation_environment_registry_receipt_sha256
        != validation_registry_v2.base_validation_registry_v1_receipt_sha256
        or base_fold_validation_v1.four_fold_validation_inputs_authority_receipt_sha256
        != four_fold_validation_inputs_v2.base_four_fold_validation_inputs_v1_receipt_sha256
        or four_fold_validation_inputs_v2.validation_sources_v2_receipts[barrier_index]
        != validation_sources_v2.semantic_receipt_sha256
        or four_fold_validation_inputs_v2.validation_environment_registry_v2_receipts[
            barrier_index
        ]
        != validation_registry_v2.semantic_receipt_sha256
        or tuple(row.outcome_kind for row in primary) != (_PRIMARY,) * len(primary)
        or tuple(row.outcome_kind for row in ladders) != (_LADDER,) * len(ladders)
        or fixed.outcome_kind != _FC06
        or tuple(row.checkpoint_authority_receipt_sha256 for row in primary)
        != base_fold_validation_v1.expected_checkpoint_authority_receipts
        or tuple(row.checkpoint_authority_receipt_sha256 for row in ladders)
        != base_fold_validation_v1.expected_checkpoint_authority_receipts
        or tuple(row.base_outcome_receipt_sha256 for row in primary)
        != base_fold_validation_v1.primary_trace_authority_receipts
        or tuple(row.base_outcome_receipt_sha256 for row in ladders)
        != base_fold_validation_v1.cost_ladder_authority_receipts
        or fixed.base_outcome_receipt_sha256
        != base_fold_validation_v1.fixed_control_validation_authority_receipt_sha256
        or any(
            row.four_fold_validation_inputs_v2_receipt_sha256
            != four_fold_validation_inputs_v2.semantic_receipt_sha256
            or row.validation_registry_v2_receipt_sha256
            != validation_registry_v2.semantic_receipt_sha256
            for row in (*primary, *ladders, fixed)
        )
    ):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "fold-validation V2 evidence generation differs"
        )
    result = _build_fold_validation_authority_v2_unchecked(
        base_fold_validation_v1=base_fold_validation_v1,
        validation_sources_v2=validation_sources_v2,
        validation_registry_v2=validation_registry_v2,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        primary_outcomes_v2=primary,
        ladder_outcomes_v2=ladders,
        fixed_control_outcome_v2=fixed,
    )
    result = replace(result, _runtime_manifest=manifest)
    saved = replace(
        result,
        _runtime_manifest=None,
        _runtime_base_fold=None,
        _runtime_validation_sources_v2=None,
        _runtime_validation_registry_v2=None,
        _runtime_four_fold_barrier_v2=None,
        _runtime_primary_outcomes_v2=(),
        _runtime_ladder_outcomes_v2=(),
        _runtime_fixed_outcome_v2=None,
        runtime_validation_replayed=False,
        development_validation_authorized=False,
    )
    saved.validate()
    return result


def parse_massive_adaptive_rl_fold_validation_authority_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFoldValidationAuthorityV2:
    body = _parse_body(root=root, loaded_source=loaded_source)
    for name in (
        "expected_checkpoint_authority_receipts",
        "primary_outcome_v2_receipts",
        "primary_outcome_v2_source_receipts",
        "primary_outcome_v2_commit_receipts",
        "ladder_outcome_v2_receipts",
        "ladder_outcome_v2_source_receipts",
        "ladder_outcome_v2_commit_receipts",
    ):
        body[name] = tuple(cast(Sequence[str], body[name]))
    result = MassiveAdaptiveRLFoldValidationAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_fold_validation_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    verified_at_ms: int,
) -> MassiveAdaptiveRLFoldValidationAuthorityV2:
    return parse_massive_adaptive_rl_fold_validation_authority_v2(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=fold_validation_authority_relative_path_v2(
                manifest=manifest, fold_index=fold_index
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_fold_validation_authority_v2(
    *,
    authority: MassiveAdaptiveRLFoldValidationAuthorityV2,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    base_fold_validation_v1: MassiveAdaptiveRLFoldValidationAuthorityV1,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_registry_v2: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    four_fold_validation_inputs_v2: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    primary_outcomes_v2: Sequence[MassiveAdaptiveRLValidationOutcomeAuthorityV2],
    ladder_outcomes_v2: Sequence[MassiveAdaptiveRLValidationOutcomeAuthorityV2],
    fixed_control_outcome_v2: MassiveAdaptiveRLValidationOutcomeAuthorityV2,
) -> MassiveAdaptiveRLFoldValidationAuthorityV2:
    authority.validate()
    expected = build_massive_adaptive_rl_fold_validation_authority_v2(
        manifest=manifest,
        base_fold_validation_v1=base_fold_validation_v1,
        validation_sources_v2=validation_sources_v2,
        validation_registry_v2=validation_registry_v2,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        primary_outcomes_v2=primary_outcomes_v2,
        ladder_outcomes_v2=ladder_outcomes_v2,
        fixed_control_outcome_v2=fixed_control_outcome_v2,
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != fold_validation_authority_relative_path_v2(
            manifest=manifest, fold_index=authority.fold_index
        )
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "fold-validation authority V2 does not replay"
        )
    result = replace(
        authority,
        runtime_validation_replayed=True,
        development_validation_authorized=authority.source_data_qualified,
        _runtime_manifest=manifest,
        _runtime_base_fold=base_fold_validation_v1,
        _runtime_validation_sources_v2=validation_sources_v2,
        _runtime_validation_registry_v2=validation_registry_v2,
        _runtime_four_fold_barrier_v2=four_fold_validation_inputs_v2,
        _runtime_primary_outcomes_v2=tuple(primary_outcomes_v2),
        _runtime_ladder_outcomes_v2=tuple(ladder_outcomes_v2),
        _runtime_fixed_outcome_v2=fixed_control_outcome_v2,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_fold_validation_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    base_fold_validation_v1: MassiveAdaptiveRLFoldValidationAuthorityV1,
    validation_sources_v2: MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_registry_v2: MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    four_fold_validation_inputs_v2: MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
    primary_outcomes_v2: Sequence[MassiveAdaptiveRLValidationOutcomeAuthorityV2],
    ladder_outcomes_v2: Sequence[MassiveAdaptiveRLValidationOutcomeAuthorityV2],
    fixed_control_outcome_v2: MassiveAdaptiveRLValidationOutcomeAuthorityV2,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFoldValidationAuthorityV2:
    authority = build_massive_adaptive_rl_fold_validation_authority_v2(
        manifest=manifest,
        base_fold_validation_v1=base_fold_validation_v1,
        validation_sources_v2=validation_sources_v2,
        validation_registry_v2=validation_registry_v2,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        primary_outcomes_v2=primary_outcomes_v2,
        ladder_outcomes_v2=ladder_outcomes_v2,
        fixed_control_outcome_v2=fixed_control_outcome_v2,
    )
    relative = fold_validation_authority_relative_path_v2(
        manifest=manifest, fold_index=authority.fold_index
    )
    if _transaction_exists(root=root, relative=relative):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "fold-validation authority V2 already exists"
        )
    child_times = tuple(
        cast(int, row.source_transaction_committed_at_ms)
        for row in (
            *tuple(primary_outcomes_v2),
            *tuple(ladder_outcomes_v2),
            fixed_control_outcome_v2,
        )
    )
    prerequisite = max(authority.base_fold_validation_v1_committed_at_ms, *child_times)
    if (
        isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms <= prerequisite
    ):
        raise MassiveAdaptiveRLValidationEvidenceV2Error(
            "fold-validation V2 must follow all evidence"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            f"ADAPTIVE-RL-FOLD-VALIDATION-V2-{manifest.experiment_id}-"
            f"FOLD{authority.fold_index}"
        ),
    )
    parsed = parse_massive_adaptive_rl_fold_validation_authority_v2(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=committed_at_ms,
        ),
    )
    return authorize_massive_adaptive_rl_fold_validation_authority_v2(
        authority=parsed,
        manifest=manifest,
        base_fold_validation_v1=base_fold_validation_v1,
        validation_sources_v2=validation_sources_v2,
        validation_registry_v2=validation_registry_v2,
        four_fold_validation_inputs_v2=four_fold_validation_inputs_v2,
        primary_outcomes_v2=primary_outcomes_v2,
        ladder_outcomes_v2=ladder_outcomes_v2,
        fixed_control_outcome_v2=fixed_control_outcome_v2,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_DATASET",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_EVIDENCE_V2_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_DATASET",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_AUTHORITY_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_OUTCOME_V2_SPEC_SHA256",
    "MassiveAdaptiveRLFoldValidationAuthorityV2",
    "MassiveAdaptiveRLValidationEvidenceV2Error",
    "MassiveAdaptiveRLValidationOutcomeAuthorityV2",
    "authorize_massive_adaptive_rl_fold_validation_authority_v2",
    "authorize_massive_adaptive_rl_validation_outcome_authority_v2",
    "build_massive_adaptive_rl_fold_validation_authority_v2",
    "build_massive_adaptive_rl_validation_outcome_authority_v2",
    "fold_validation_authority_relative_path_v2",
    "load_massive_adaptive_rl_fold_validation_authority_v2",
    "load_massive_adaptive_rl_validation_outcome_authority_v2",
    "materialize_massive_adaptive_rl_fold_validation_authority_v2",
    "materialize_massive_adaptive_rl_validation_outcome_authority_v2",
    "parse_massive_adaptive_rl_fold_validation_authority_v2",
    "parse_massive_adaptive_rl_validation_outcome_authority_v2",
    "validation_outcome_authority_relative_path_v2",
]
