"""Completion authority for one attested validation fold execution.

The authority does not recompute selection.  It binds the exact Selection V3
authority to the persisted numerical environment and to every canonical source
transaction that preceded it.  Each stage records its ordinal and predecessor
commit receipt, so chronological order does not depend on invented one-
millisecond timestamps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_executor_v1 import (
    _CanonicalValidationStageV1,
    _canonical_validation_stages_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v3 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV3,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MassiveAdaptiveRLExperimentManifestV4,
)
from rl_quant.workflows.massive_adaptive_rl_validation_execution_environment_v1 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
    MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1,
)


MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fold-validation-execution-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-fold-validation-execution-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SCHEMA,
            "encoding": "canonical-json-attested-fold-execution",
            "generic_reload": "integrity-only",
        }
    )
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-selection-v3-and-validation-execution-environment-v1",
        "stage_inventory": "complete-canonical-fold-stage-source-transactions",
        "ordering": "zero-based-stage-ordinal-plus-predecessor-commit-chain",
        "timestamps": "observed-source-commit-times-strictly-monotonic",
        "portability": "same-sealed-inode-filesystem-only-v1",
        "validation_execution_environment_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256
        ),
        "publication": "manifest-and-fold-derived-create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "policy_freezing": "runtime-replay-required",
        "profitability_reporting": False,
        "outer_evaluation": False,
        "lockbox_access": False,
    }
)


class MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(ValueError):
    """The attested fold execution is incomplete, mixed, or noncanonical."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution stage path differs"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution stage path differs"
        )
    return path.as_posix()


def _transaction_exists(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority source transaction is incomplete"
        )
    return all(present)


def fold_validation_execution_authority_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority index differs"
        )
    return (
        "massive-adaptive/rl-fold-validation-execution-authority-v1/"
        f"v4-{manifest.semantic_receipt_sha256}-fold-{fold_index}.json"
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLValidationExecutionStageV1:
    stage_ordinal: int
    stage_name: str
    relative_payload_path: str
    authority_receipt_sha256: str
    source_receipt_sha256: str
    commit_receipt_sha256: str
    committed_at_ms: int
    observed_published_at_ms: int
    previous_stage_commit_receipt_sha256: str

    def validate(self) -> None:
        if (
            isinstance(self.stage_ordinal, bool)
            or not isinstance(self.stage_ordinal, int)
            or self.stage_ordinal < 0
            or not isinstance(self.stage_name, str)
            or not self.stage_name
            or self.stage_name != self.stage_name.strip()
            or isinstance(self.committed_at_ms, bool)
            or not isinstance(self.committed_at_ms, int)
            or self.committed_at_ms < 0
            or isinstance(self.observed_published_at_ms, bool)
            or not isinstance(self.observed_published_at_ms, int)
            or self.observed_published_at_ms < 0
        ):
            raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
                "fold execution stage metadata differs"
            )
        _safe_relative_path(self.relative_payload_path)
        for name in (
            "authority_receipt_sha256",
            "source_receipt_sha256",
            "commit_receipt_sha256",
            "previous_stage_commit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
    experiment_id: str
    manifest_v4_receipt_sha256: str
    training_manifest_v3_receipt_sha256: str
    fold_index: int
    runtime_sources_v2_receipt_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    four_fold_validation_inputs_v2_receipt_sha256: str
    checkpoint_authority_receipts: tuple[str, ...]
    fixed_control_selection_authority_receipt_sha256: str
    validation_execution_environment_receipt_sha256: str
    validation_execution_environment_source_receipt_sha256: str
    validation_execution_environment_commit_receipt_sha256: str
    validation_execution_environment_committed_at_ms: int
    validation_execution_environment_observed_published_at_ms: int
    scientific_execution_fingerprint_sha256: str
    policy_selection_v3_receipt_sha256: str
    policy_selection_v3_source_receipt_sha256: str
    policy_selection_v3_commit_receipt_sha256: str
    policy_selection_v3_committed_at_ms: int
    stages: tuple[MassiveAdaptiveRLValidationExecutionStageV1, ...]
    stage_inventory_sha256: str
    execution_started_at_ms: int
    execution_completed_at_ms: int
    execution_observed_started_at_ms: int
    execution_observed_completed_at_ms: int
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_execution_replayed: bool = False
    development_policy_selection_authorized: bool = False
    policy_freezing_authorized: bool = False
    outer_diagnostic_preparation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SCHEMA
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV4 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_environment: (
        MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_selection_v3: MassiveAdaptiveRLPolicySelectionAuthorityV3 | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: (
                tuple(asdict(stage) for stage in self.stages)
                if descriptor.name == "stages"
                else getattr(self, descriptor.name)
            )
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name
            not in {
                "semantic_receipt_sha256",
                "runtime_execution_replayed",
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
            and self.runtime_execution_replayed
            and self.development_policy_selection_authorized
            and self.policy_freezing_authorized
            and self.outer_diagnostic_preparation_authorized
            and self.source_data_qualified
        )

    @property
    def positive_profitability_authorization_eligible(self) -> bool:
        return bool(
            self.development_stage_authorized
            and self._runtime_selection_v3 is not None
            and self._runtime_selection_v3.positive_profitability_authorization_eligible
        )

    @property
    def policy_selection_v3(self) -> MassiveAdaptiveRLPolicySelectionAuthorityV3:
        if self._runtime_selection_v3 is None:
            raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
                "fold execution authority has no runtime Selection V3"
            )
        return self._runtime_selection_v3

    def validate(self) -> None:
        runtime_parts = (
            self._runtime_manifest,
            self._runtime_environment,
            self._runtime_selection_v3,
        )
        runtime_present = any(value is not None for value in runtime_parts)
        if runtime_present != all(value is not None for value in runtime_parts):
            raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
                "fold execution authority runtime is partial"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
        for stage in self.stages:
            stage.validate()
        if runtime_present:
            runtime_manifest = cast(
                MassiveAdaptiveRLExperimentManifestV4, self._runtime_manifest
            )
            runtime_environment = cast(
                MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1,
                self._runtime_environment,
            )
            runtime_selection_v3 = cast(
                MassiveAdaptiveRLPolicySelectionAuthorityV3,
                self._runtime_selection_v3,
            )
            runtime_manifest.validate()
            runtime_environment.validate()
            runtime_selection_v3.validate()
            expected_stages = _canonical_validation_stages_v1(
                manifest=runtime_manifest,
                fold_index=self.fold_index,
                checkpoint_authority_receipts=self.checkpoint_authority_receipts,
                fixed_control_selection_authority_receipt_sha256=(
                    self.fixed_control_selection_authority_receipt_sha256
                ),
            )
            if (
                runtime_manifest.semantic_receipt_sha256
                != self.manifest_v4_receipt_sha256
                or runtime_environment.semantic_receipt_sha256
                != self.validation_execution_environment_receipt_sha256
                or runtime_selection_v3.semantic_receipt_sha256
                != self.policy_selection_v3_receipt_sha256
                or tuple(stage.name for stage in expected_stages)
                != tuple(stage.stage_name for stage in self.stages)
                or tuple(stage.relative_path for stage in expected_stages)
                != tuple(stage.relative_payload_path for stage in self.stages)
                or not runtime_environment.development_stage_authorized
                or not runtime_selection_v3.development_stage_authorized
            ):
                raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
                    "fold execution authority runtime differs"
                )
        ordinals = tuple(stage.stage_ordinal for stage in self.stages)
        names = tuple(stage.stage_name for stage in self.stages)
        paths = tuple(stage.relative_payload_path for stage in self.stages)
        times = tuple(stage.committed_at_ms for stage in self.stages)
        observed_times = tuple(
            stage.observed_published_at_ms for stage in self.stages
        )
        timestamps = (
            self.validation_execution_environment_committed_at_ms,
            self.validation_execution_environment_observed_published_at_ms,
            self.policy_selection_v3_committed_at_ms,
            self.execution_started_at_ms,
            self.execution_completed_at_ms,
            self.execution_observed_started_at_ms,
            self.execution_observed_completed_at_ms,
        )
        boolean_fields = (
            self.source_data_qualified,
            self.runtime_execution_replayed,
            self.development_policy_selection_authorized,
            self.policy_freezing_authorized,
            self.outer_diagnostic_preparation_authorized,
            self.profitability_reporting_authorized,
            self.outer_evaluation_authorized,
            self.lockbox_access_authorized,
        )
        expected_count = (self.fold_index + 1) * 4 + 6
        expected_flags = bool(
            runtime_present
            and self.source_transaction_verified
            and self.source_data_qualified
            and self._runtime_environment is not None
            and self._runtime_environment.development_stage_authorized
            and self._runtime_selection_v3 is not None
            and self._runtime_selection_v3.development_stage_authorized
        )
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in timestamps
            )
            or any(type(value) is not bool for value in boolean_fields)
            or len(self.checkpoint_authority_receipts) != self.fold_index + 1
            or len(set(self.checkpoint_authority_receipts))
            != len(self.checkpoint_authority_receipts)
            or len(self.stages) != expected_count
            or ordinals != tuple(range(expected_count))
            or len(set(names)) != 10
            or len(set(paths)) != len(paths)
            or any(right <= left for left, right in zip(times, times[1:]))
            or any(
                right < left
                for left, right in zip(observed_times, observed_times[1:])
            )
            or self.stages[0].previous_stage_commit_receipt_sha256
            != self.validation_execution_environment_commit_receipt_sha256
            or any(
                current.previous_stage_commit_receipt_sha256
                != previous.commit_receipt_sha256
                for previous, current in zip(self.stages, self.stages[1:])
            )
            or self.stages[-1].authority_receipt_sha256
            != self.policy_selection_v3_receipt_sha256
            or self.stages[-1].source_receipt_sha256
            != self.policy_selection_v3_source_receipt_sha256
            or self.stages[-1].commit_receipt_sha256
            != self.policy_selection_v3_commit_receipt_sha256
            or self.stages[-1].committed_at_ms
            != self.policy_selection_v3_committed_at_ms
            or self.validation_execution_environment_committed_at_ms >= times[0]
            or self.validation_execution_environment_observed_published_at_ms
            > observed_times[0]
            or self.execution_started_at_ms != times[0]
            or self.execution_completed_at_ms != times[-1]
            or self.execution_observed_started_at_ms != observed_times[0]
            or self.execution_observed_completed_at_ms != observed_times[-1]
            or self.stage_inventory_sha256
            != semantic_sha256(tuple(asdict(stage) for stage in self.stages))
            or self.runtime_execution_replayed != runtime_present
            or self.development_policy_selection_authorized != expected_flags
            or self.policy_freezing_authorized != expected_flags
            or self.outer_diagnostic_preparation_authorized != expected_flags
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
                "fold validation execution authority differs"
            )
        for name in (
            "manifest_v4_receipt_sha256",
            "training_manifest_v3_receipt_sha256",
            "runtime_sources_v2_receipt_sha256",
            "four_fold_fit_authority_receipt_sha256",
            "four_fold_validation_inputs_v2_receipt_sha256",
            "fixed_control_selection_authority_receipt_sha256",
            "validation_execution_environment_receipt_sha256",
            "validation_execution_environment_source_receipt_sha256",
            "validation_execution_environment_commit_receipt_sha256",
            "scientific_execution_fingerprint_sha256",
            "policy_selection_v3_receipt_sha256",
            "policy_selection_v3_source_receipt_sha256",
            "policy_selection_v3_commit_receipt_sha256",
            "stage_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for receipt in self.checkpoint_authority_receipts:
            _digest("checkpoint authority", receipt)
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= self.execution_completed_at_ms
        ):
            raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
                "fold execution authority source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _stage_records(
    *,
    root: str | Path,
    stages: Sequence[_CanonicalValidationStageV1],
    environment_commit_receipt_sha256: str,
    verified_at_ms: int,
) -> tuple[MassiveAdaptiveRLValidationExecutionStageV1, ...]:
    previous = environment_commit_receipt_sha256
    records: list[MassiveAdaptiveRLValidationExecutionStageV1] = []
    for ordinal, stage in enumerate(stages):
        loaded = load_massive_source_bundle(
            root=root,
            relative_payload_path=stage.relative_path,
            verified_at_ms=verified_at_ms,
        )
        record = MassiveAdaptiveRLValidationExecutionStageV1(
            stage_ordinal=ordinal,
            stage_name=stage.name,
            relative_payload_path=stage.relative_path,
            authority_receipt_sha256=(loaded.receipt.entitlement_receipt_sha256),
            source_receipt_sha256=loaded.receipt.receipt_sha256,
            commit_receipt_sha256=loaded.commit.receipt_sha256,
            committed_at_ms=loaded.commit.committed_at_ms,
            observed_published_at_ms=loaded.payload_ctime_ns // 1_000_000,
            previous_stage_commit_receipt_sha256=previous,
        )
        record.validate()
        records.append(record)
        previous = loaded.commit.receipt_sha256
    return tuple(records)


def build_massive_adaptive_rl_fold_validation_execution_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    checkpoint_authority_receipts: tuple[str, ...],
    fixed_control_selection_authority_receipt_sha256: str,
    validation_execution_environment: (
        MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1
    ),
    policy_selection_v3: MassiveAdaptiveRLPolicySelectionAuthorityV3,
    verified_at_ms: int,
) -> MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
    manifest.validate()
    validation_execution_environment.validate()
    policy_selection_v3.validate()
    environment_source = validation_execution_environment.source_receipt_sha256
    environment_commit = (
        validation_execution_environment.source_transaction_receipt_sha256
    )
    environment_time = (
        validation_execution_environment.source_transaction_committed_at_ms
    )
    environment_observed_time = (
        validation_execution_environment.source_transaction_observed_published_at_ms
    )
    selection_source = policy_selection_v3.source_receipt_sha256
    selection_commit = policy_selection_v3.source_transaction_receipt_sha256
    selection_time = policy_selection_v3.source_transaction_committed_at_ms
    if (
        not validation_execution_environment.development_stage_authorized
        or not policy_selection_v3.development_stage_authorized
        or environment_source is None
        or environment_commit is None
        or environment_time is None
        or environment_observed_time is None
        or selection_source is None
        or selection_commit is None
        or selection_time is None
        or manifest.experiment_id != validation_execution_environment.experiment_id
        or manifest.experiment_id != policy_selection_v3.experiment_id
        or manifest.semantic_receipt_sha256
        != validation_execution_environment.manifest_v4_receipt_sha256
        or manifest.semantic_receipt_sha256
        != policy_selection_v3.manifest_v4_receipt_sha256
        or fold_index != policy_selection_v3.fold_index
        or policy_selection_v3.runtime_sources_v2_receipt_sha256
        != validation_execution_environment.runtime_sources_v2_receipt_sha256
        or policy_selection_v3.source_bundle_v2_receipt_sha256
        != validation_execution_environment.source_bundle_v2_receipt_sha256
        or policy_selection_v3.runtime_source_graph_v2_receipt_sha256
        != validation_execution_environment.runtime_source_graph_v2_receipt_sha256
        or policy_selection_v3.runtime_source_graph_v2_witness_receipt_sha256
        != validation_execution_environment.runtime_source_graph_v2_witness_receipt_sha256
        or policy_selection_v3.replay_dependency_index_v2_receipt_sha256
        != validation_execution_environment.replay_dependency_index_v2_receipt_sha256
        or policy_selection_v3.four_fold_validation_inputs_v2_receipt_sha256
        != validation_execution_environment.four_fold_validation_inputs_v2_receipt_sha256
        or policy_selection_v3.expected_candidate_checkpoint_authority_receipts
        != checkpoint_authority_receipts
    ):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority inputs differ"
        )
    stages = _canonical_validation_stages_v1(
        manifest=manifest,
        fold_index=fold_index,
        checkpoint_authority_receipts=checkpoint_authority_receipts,
        fixed_control_selection_authority_receipt_sha256=(
            fixed_control_selection_authority_receipt_sha256
        ),
    )
    records = _stage_records(
        root=root,
        stages=stages,
        environment_commit_receipt_sha256=environment_commit,
        verified_at_ms=verified_at_ms,
    )
    source_data_qualified = bool(
        validation_execution_environment.source_data_qualified
        and policy_selection_v3.source_data_qualified
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_index": fold_index,
        "runtime_sources_v2_receipt_sha256": (
            validation_execution_environment.runtime_sources_v2_receipt_sha256
        ),
        "four_fold_fit_authority_receipt_sha256": (
            validation_execution_environment.four_fold_fit_authority_receipt_sha256
        ),
        "four_fold_validation_inputs_v2_receipt_sha256": (
            validation_execution_environment.four_fold_validation_inputs_v2_receipt_sha256
        ),
        "checkpoint_authority_receipts": checkpoint_authority_receipts,
        "fixed_control_selection_authority_receipt_sha256": (
            fixed_control_selection_authority_receipt_sha256
        ),
        "validation_execution_environment_receipt_sha256": (
            validation_execution_environment.semantic_receipt_sha256
        ),
        "validation_execution_environment_source_receipt_sha256": (environment_source),
        "validation_execution_environment_commit_receipt_sha256": (environment_commit),
        "validation_execution_environment_committed_at_ms": environment_time,
        "validation_execution_environment_observed_published_at_ms": (
            environment_observed_time
        ),
        "scientific_execution_fingerprint_sha256": (
            validation_execution_environment.scientific_execution_fingerprint_sha256
        ),
        "policy_selection_v3_receipt_sha256": (
            policy_selection_v3.semantic_receipt_sha256
        ),
        "policy_selection_v3_source_receipt_sha256": selection_source,
        "policy_selection_v3_commit_receipt_sha256": selection_commit,
        "policy_selection_v3_committed_at_ms": selection_time,
        "stages": records,
        "stage_inventory_sha256": semantic_sha256(
            tuple(asdict(stage) for stage in records)
        ),
        "execution_started_at_ms": records[0].committed_at_ms,
        "execution_completed_at_ms": records[-1].committed_at_ms,
        "execution_observed_started_at_ms": records[0].observed_published_at_ms,
        "execution_observed_completed_at_ms": records[-1].observed_published_at_ms,
        "source_data_qualified": source_data_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFoldValidationExecutionAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_execution_replayed=True,
        development_policy_selection_authorized=False,
        policy_freezing_authorized=False,
        outer_diagnostic_preparation_authorized=False,
        _runtime_manifest=manifest,
        _runtime_environment=validation_execution_environment,
        _runtime_selection_v3=policy_selection_v3,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _payload(
    authority: MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
) -> dict[str, object]:
    authority.validate()
    return {
        **authority.semantic_unsigned(),
        "semantic_receipt_sha256": authority.semantic_receipt_sha256,
    }


def _parse_stage(value: object) -> MassiveAdaptiveRLValidationExecutionStageV1:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution stage payload differs"
        )
    try:
        result = MassiveAdaptiveRLValidationExecutionStageV1(**dict(value))
    except TypeError as error:
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution stage fields differ"
        ) from error
    result.validate()
    return result


def _parse(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    checkpoints = body.get("checkpoint_authority_receipts")
    stages = body.get("stages")
    if (
        not isinstance(checkpoints, list)
        or not all(isinstance(item, str) for item in checkpoints)
        or not isinstance(stages, list)
    ):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority inventories differ"
        )
    body["checkpoint_authority_receipts"] = tuple(checkpoints)
    body["stages"] = tuple(_parse_stage(item) for item in stages)
    try:
        result = MassiveAdaptiveRLFoldValidationExecutionAuthorityV1(
            **body,  # type: ignore[arg-type]
            runtime_execution_replayed=False,
            development_policy_selection_authorized=False,
            policy_freezing_authorized=False,
            outer_diagnostic_preparation_authorized=False,
            _loaded_source=loaded_source,
        )
    except TypeError as error:
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority fields differ"
        ) from error
    result.validate()
    if raw != canonical_json_file_bytes(_payload(result)):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority payload did not round trip"
        )
    return result


def load_massive_adaptive_rl_fold_validation_execution_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    verified_at_ms: int,
) -> MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=fold_validation_execution_authority_relative_path_v1(
            manifest=manifest, fold_index=fold_index
        ),
        verified_at_ms=verified_at_ms,
    )
    return _parse(root=root, loaded_source=loaded)


def materialize_massive_adaptive_rl_fold_validation_execution_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    authority: MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
    manifest.validate()
    authority.validate()
    if (
        not authority.runtime_execution_replayed
        or authority.source_transaction_verified
        or not authority.source_data_qualified
        or isinstance(committed_at_ms, bool)
        or not isinstance(committed_at_ms, int)
        or committed_at_ms <= authority.execution_completed_at_ms
        or authority.manifest_v4_receipt_sha256
        != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority materialization is not authorized"
        )
    relative = fold_validation_execution_authority_relative_path_v1(
        manifest=manifest, fold_index=authority.fold_index
    )
    Path(root).mkdir(parents=True, exist_ok=True)
    if Path(root).is_symlink():
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority root is a symlink"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(authority))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=(
            "ADAPTIVE-RL-FOLD-VALIDATION-EXECUTION-AUTHORITY-V1-"
            f"{manifest.semantic_receipt_sha256}-{authority.fold_index}"
        ),
    )
    return load_massive_adaptive_rl_fold_validation_execution_authority_v1(
        root=root,
        manifest=manifest,
        fold_index=authority.fold_index,
        verified_at_ms=committed_at_ms,
    )


def authorize_massive_adaptive_rl_fold_validation_execution_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    checkpoint_authority_receipts: tuple[str, ...],
    fixed_control_selection_authority_receipt_sha256: str,
    validation_execution_environment: (
        MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1
    ),
    policy_selection_v3: MassiveAdaptiveRLPolicySelectionAuthorityV3,
    verified_at_ms: int,
) -> MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
    authority.validate()
    if not authority.source_transaction_verified:
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution replay requires persisted integrity"
        )
    active = build_massive_adaptive_rl_fold_validation_execution_authority_v1(
        root=root,
        manifest=manifest,
        fold_index=authority.fold_index,
        checkpoint_authority_receipts=checkpoint_authority_receipts,
        fixed_control_selection_authority_receipt_sha256=(
            fixed_control_selection_authority_receipt_sha256
        ),
        validation_execution_environment=validation_execution_environment,
        policy_selection_v3=policy_selection_v3,
        verified_at_ms=verified_at_ms,
    )
    if canonical_json_file_bytes(_payload(active)) != canonical_json_file_bytes(
        _payload(authority)
    ):
        raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
            "fold execution authority did not replay"
        )
    result = replace(
        authority,
        runtime_execution_replayed=True,
        development_policy_selection_authorized=authority.source_data_qualified,
        policy_freezing_authorized=authority.source_data_qualified,
        outer_diagnostic_preparation_authorized=authority.source_data_qualified,
        _runtime_manifest=manifest,
        _runtime_environment=validation_execution_environment,
        _runtime_selection_v3=policy_selection_v3,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_rl_fold_validation_execution_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    fold_index: int,
    checkpoint_authority_receipts: tuple[str, ...],
    fixed_control_selection_authority_receipt_sha256: str,
    validation_execution_environment: (
        MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1
    ),
    policy_selection_v3: MassiveAdaptiveRLPolicySelectionAuthorityV3,
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
    relative = fold_validation_execution_authority_relative_path_v1(
        manifest=manifest, fold_index=fold_index
    )
    if _transaction_exists(root=root, relative=relative):
        generic = load_massive_adaptive_rl_fold_validation_execution_authority_v1(
            root=root,
            manifest=manifest,
            fold_index=fold_index,
            verified_at_ms=committed_at_ms,
        )
    else:
        if not allow_materialize:
            raise MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error(
                "canonical fold execution authority is absent"
            )
        active = build_massive_adaptive_rl_fold_validation_execution_authority_v1(
            root=root,
            manifest=manifest,
            fold_index=fold_index,
            checkpoint_authority_receipts=checkpoint_authority_receipts,
            fixed_control_selection_authority_receipt_sha256=(
                fixed_control_selection_authority_receipt_sha256
            ),
            validation_execution_environment=validation_execution_environment,
            policy_selection_v3=policy_selection_v3,
            verified_at_ms=committed_at_ms,
        )
        generic = (
            materialize_massive_adaptive_rl_fold_validation_execution_authority_v1(
                root=root,
                manifest=manifest,
                authority=active,
                committed_at_ms=committed_at_ms,
            )
        )
    return authorize_massive_adaptive_rl_fold_validation_execution_authority_v1(
        root=root,
        authority=generic,
        manifest=manifest,
        checkpoint_authority_receipts=checkpoint_authority_receipts,
        fixed_control_selection_authority_receipt_sha256=(
            fixed_control_selection_authority_receipt_sha256
        ),
        validation_execution_environment=validation_execution_environment,
        policy_selection_v3=policy_selection_v3,
        verified_at_ms=committed_at_ms,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFoldValidationExecutionAuthorityV1",
    "MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error",
    "MassiveAdaptiveRLValidationExecutionStageV1",
    "authorize_massive_adaptive_rl_fold_validation_execution_authority_v1",
    "build_massive_adaptive_rl_fold_validation_execution_authority_v1",
    "fold_validation_execution_authority_relative_path_v1",
    "load_massive_adaptive_rl_fold_validation_execution_authority_v1",
    "materialize_massive_adaptive_rl_fold_validation_execution_authority_v1",
    "run_or_resume_massive_adaptive_rl_fold_validation_execution_authority_v1",
]
