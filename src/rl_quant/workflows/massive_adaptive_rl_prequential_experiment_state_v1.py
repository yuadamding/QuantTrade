"""Append-only Manifest-V5 prequential experiment state.

The component authorities remain the source of scientific and economic truth.
This ledger records their one legal global order.  A state is authorizing only
after its exact stage artifact and its own create-only source transaction have
both been replayed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from io import BytesIO
import json
from pathlib import Path
import time
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
    MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)
from rl_quant.workflows.massive_adaptive_rl_full_cold_replay_v1 import (
    MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SCHEMA,
)


MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_DATASET = (
    "massive-adaptive-rl-prequential-experiment-state-v1"
)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SCHEMA,
            "encoding": "canonical-json-create-only-immediate-predecessor-chain",
            "generic_reload": "nonauthorizing",
        }
    )
)


class MassiveAdaptiveRLPrequentialStageV1(str, Enum):
    TRAINED = "trained"
    EXECUTION_IMPLEMENTATION_REGISTERED = "execution-implementation-registered"
    INITIAL_VALIDATION_INPUTS_COMMITTED = "initial-validation-inputs-committed"
    POLICY_0_FROZEN = "policy-0-frozen"
    POLICY_1_FROZEN = "policy-1-frozen"
    OUTER_0_SEALED = "outer-0-sealed"
    VALIDATION_2_RELEASED = "validation-2-released"
    POLICY_2_FROZEN = "policy-2-frozen"
    OUTER_1_SEALED = "outer-1-sealed"
    VALIDATION_3_RELEASED = "validation-3-released"
    POLICY_3_FROZEN = "policy-3-frozen"
    OUTER_2_SEALED = "outer-2-sealed"
    OUTER_3_SEALED = "outer-3-sealed"
    PROFITABILITY_REPORT_PUBLISHED = "profitability-report-published"
    FULL_COLD_REPLAY_VERIFIED = "full-cold-replay-verified"


MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1 = tuple(
    MassiveAdaptiveRLPrequentialStageV1(value)
    for value in MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1
)

_STAGE_ARTIFACT_SCHEMAS = {
    MassiveAdaptiveRLPrequentialStageV1.TRAINED: (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_FIT_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.EXECUTION_IMPLEMENTATION_REGISTERED: (
        MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.INITIAL_VALIDATION_INPUTS_COMMITTED: (
        MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.POLICY_0_FROZEN: (
        MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.POLICY_1_FROZEN: (
        MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.OUTER_0_SEALED: (
        MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.VALIDATION_2_RELEASED: (
        MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.POLICY_2_FROZEN: (
        MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.OUTER_1_SEALED: (
        MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.VALIDATION_3_RELEASED: (
        MASSIVE_ADAPTIVE_RL_VALIDATION_RELEASE_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN: (
        MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.OUTER_2_SEALED: (
        MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.OUTER_3_SEALED: (
        MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED: (
        MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA
    ),
    MassiveAdaptiveRLPrequentialStageV1.FULL_COLD_REPLAY_VERIFIED: (
        MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SCHEMA
    ),
}


class MassiveAdaptiveRLPrequentialExperimentStateV1Error(ValueError):
    """The V5 state chain, artifact lineage, or replay differs."""


class MassiveAdaptiveRLPrequentialExperimentStateV1StaleError(
    MassiveAdaptiveRLPrequentialExperimentStateV1Error
):
    """A state writer did not descend from the persisted chain head."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: object) -> str:
    return _digest(name, value)


def _timestamp(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            f"{name} is absent or invalid"
        )
    return value


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential experiment ID is not path safe"
        )
    return value


def _state_relative_path(
    *, experiment_id: str, stage: MassiveAdaptiveRLPrequentialStageV1
) -> str:
    index = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1.index(stage)
    return (
        f"adaptive-rl/{_identifier(experiment_id)}/prequential-experiment-state-v1/"
        f"{index:03d}-{stage.value}.json"
    )


def massive_adaptive_rl_prequential_state_relative_path_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    stage: MassiveAdaptiveRLPrequentialStageV1,
) -> str:
    manifest.validate()
    if type(stage) is not MassiveAdaptiveRLPrequentialStageV1:
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential stage generation differs"
        )
    return _state_relative_path(experiment_id=manifest.experiment_id, stage=stage)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPrequentialExperimentStateV1:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    manifest_v5_registration_receipt_sha256: str
    execution_implementation_registration_receipt_sha256: str
    sequence_index: int
    stage: MassiveAdaptiveRLPrequentialStageV1
    immediate_predecessor_state_receipt_sha256: str | None
    immediate_predecessor_state_committed_at_ms: int | None
    previous_stage_artifact_committed_at_ms: int | None
    stage_artifact_schema: str
    stage_artifact_semantic_receipt_sha256: str
    stage_artifact_source_receipt_sha256: str
    stage_artifact_commit_receipt_sha256: str
    stage_artifact_committed_at_ms: int
    policy_schedule_disposition: str | None
    policy_schedule_qualified: bool | None
    profitability_gates_passed: bool | None
    source_data_qualified: bool
    blocker_code: str | None
    semantic_receipt_sha256: str
    runtime_state_replayed: bool = False
    prequential_execution_authorized: bool = False
    development_profitability_reporting_authorized: bool = False
    full_cold_replay_verified: bool = False
    positive_profitability_authorization_eligible: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SCHEMA
    _runtime_stage_artifact: object | None = field(
        default=None, compare=False, repr=False
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
                "runtime_state_replayed",
                "prequential_execution_authorized",
                "development_profitability_reporting_authorized",
                "full_cold_replay_verified",
                "positive_profitability_authorization_eligible",
            }
        }

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

    def validate(self) -> None:
        runtime = self._runtime_stage_artifact is not None
        expected_index = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1.index(
            self.stage
        )
        has_predecessor = self.sequence_index > 0
        report_or_later = (
            self.sequence_index
            >= MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1.index(
                MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED
            )
        )
        policy_or_later = (
            self.sequence_index
            >= MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1.index(
                MassiveAdaptiveRLPrequentialStageV1.POLICY_0_FROZEN
            )
        )
        cold = (
            self.stage is MassiveAdaptiveRLPrequentialStageV1.FULL_COLD_REPLAY_VERIFIED
        )
        expected_positive = bool(
            runtime
            and cold
            and self.source_data_qualified
            and self.policy_schedule_qualified
            and self.profitability_gates_passed
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SCHEMA
            or _identifier(self.experiment_id) != self.experiment_id
            or isinstance(self.sequence_index, bool)
            or self.sequence_index != expected_index
            or (self.immediate_predecessor_state_receipt_sha256 is not None)
            != has_predecessor
            or (self.immediate_predecessor_state_committed_at_ms is not None)
            != has_predecessor
            or (self.previous_stage_artifact_committed_at_ms is not None)
            != has_predecessor
            or self.stage_artifact_schema != _STAGE_ARTIFACT_SCHEMAS[self.stage]
            or not isinstance(self.source_data_qualified, bool)
            or self.blocker_code is not None
            or (self.policy_schedule_disposition is not None) != policy_or_later
            or (self.policy_schedule_disposition is not None)
            != (self.policy_schedule_qualified is not None)
            or self.policy_schedule_disposition is not None
            and self.policy_schedule_disposition
            not in {"policy-prefix-qualified", "policy-prefix-diagnostic-only"}
            or self.policy_schedule_qualified
            != (
                None
                if self.policy_schedule_disposition is None
                else self.policy_schedule_disposition == "policy-prefix-qualified"
            )
            or report_or_later != (self.profitability_gates_passed is not None)
            or not isinstance(self.profitability_gates_passed, (bool, type(None)))
            or self.runtime_state_replayed != runtime
            or self.prequential_execution_authorized
            != bool(runtime and self.source_data_qualified)
            or self.development_profitability_reporting_authorized
            != bool(runtime and report_or_later and self.source_data_qualified)
            or self.full_cold_replay_verified != bool(runtime and cold)
            or self.positive_profitability_authorization_eligible != expected_positive
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "prequential experiment state differs"
            )
        _timestamp("stage artifact timestamp", self.stage_artifact_committed_at_ms)
        for value in (
            self.manifest_v5_receipt_sha256,
            self.manifest_v5_registration_receipt_sha256,
            self.execution_implementation_registration_receipt_sha256,
            self.stage_artifact_semantic_receipt_sha256,
            self.stage_artifact_source_receipt_sha256,
            self.stage_artifact_commit_receipt_sha256,
            self.semantic_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
        ):
            _digest("prequential state receipt", value)
        if self.immediate_predecessor_state_receipt_sha256 is not None:
            _digest(
                "predecessor state receipt",
                self.immediate_predecessor_state_receipt_sha256,
            )
            _timestamp(
                "predecessor state timestamp",
                self.immediate_predecessor_state_committed_at_ms,
            )
            _timestamp(
                "previous stage artifact timestamp",
                self.previous_stage_artifact_committed_at_ms,
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
            expected_relative = _state_relative_path(
                experiment_id=self.experiment_id, stage=self.stage
            )
            if (
                self._loaded_source.payload_relative_path != expected_relative
                or self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.semantic_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= max(
                    self.stage_artifact_committed_at_ms,
                    self.immediate_predecessor_state_committed_at_ms or -1,
                )
            ):
                raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                    "prequential state source transaction differs"
                )
        if runtime:
            facts = _stage_artifact_facts(
                manifest_receipt=self.manifest_v5_receipt_sha256,
                execution_registration_receipt=(
                    self.execution_implementation_registration_receipt_sha256
                ),
                artifact=self._runtime_stage_artifact,
            )
            if (
                facts.stage is not self.stage
                or facts.schema != self.stage_artifact_schema
                or facts.semantic_receipt != self.stage_artifact_semantic_receipt_sha256
                or facts.source_receipt != self.stage_artifact_source_receipt_sha256
                or facts.commit_receipt != self.stage_artifact_commit_receipt_sha256
                or facts.committed_at_ms != self.stage_artifact_committed_at_ms
                or facts.source_data_qualified != self.source_data_qualified
            ):
                raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                    "prequential runtime stage artifact differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class _StageArtifactFactsV1:
    stage: MassiveAdaptiveRLPrequentialStageV1
    schema: str
    semantic_receipt: str
    source_receipt: str
    commit_receipt: str
    committed_at_ms: int
    source_data_qualified: bool
    policy_schedule_disposition: str | None
    policy_schedule_qualified: bool | None
    profitability_gates_passed: bool | None


def _source_transaction_facts(artifact: object) -> tuple[str, str, int]:
    source = getattr(artifact, "source_receipt_sha256", None)
    commit = getattr(artifact, "source_transaction_receipt_sha256", None)
    committed_at_ms = getattr(artifact, "source_transaction_committed_at_ms", None)
    loaded = getattr(artifact, "_loaded_source", None)
    if source is None and loaded is not None:
        source = getattr(getattr(loaded, "receipt", None), "receipt_sha256", None)
    if commit is None and loaded is not None:
        commit = getattr(getattr(loaded, "commit", None), "receipt_sha256", None)
    if committed_at_ms is None and loaded is not None:
        committed_at_ms = getattr(
            getattr(loaded, "commit", None), "committed_at_ms", None
        )
    return (
        _required_digest("stage artifact source receipt", source),
        _required_digest("stage artifact commit receipt", commit),
        _timestamp("stage artifact commit timestamp", committed_at_ms),
    )


def _stage_artifact_facts(
    *, manifest_receipt: str, execution_registration_receipt: str, artifact: object
) -> _StageArtifactFactsV1:
    from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
    )
    from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v2 import (
        MassiveAdaptiveRLProfitabilityReportAuthorityV2,
    )
    from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
        MassiveAdaptiveRLValidationReleaseAuthorityV1,
    )
    from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
    )
    from rl_quant.workflows.massive_adaptive_rl_full_cold_replay_v1 import (
        MassiveAdaptiveRLFullColdReplayAuthorityV1,
    )
    from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
        MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    )

    if type(artifact) is MassiveAdaptiveRLFourFoldFitAuthorityV1:
        stage = MassiveAdaptiveRLPrequentialStageV1.TRAINED
        runtime_authorized = bool(getattr(artifact, "development_stage_authorized"))
        disposition = None
        schedule_qualified = None
        gates = None
    elif (
        type(artifact)
        is MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ):
        stage = MassiveAdaptiveRLPrequentialStageV1.EXECUTION_IMPLEMENTATION_REGISTERED
        runtime_authorized = bool(getattr(artifact, "development_execution_registered"))
        disposition = None
        schedule_qualified = None
        gates = None
    elif type(artifact) is MassiveAdaptiveRLValidationReleaseAuthorityV1:
        runtime_authorized = bool(getattr(artifact, "development_stage_authorized"))
        release_kind = getattr(artifact, "release_kind", None)
        stage_by_release = {
            "initial-folds-0-1": MassiveAdaptiveRLPrequentialStageV1.INITIAL_VALIDATION_INPUTS_COMMITTED,
            "post-outer-0-fold-2": MassiveAdaptiveRLPrequentialStageV1.VALIDATION_2_RELEASED,
            "post-outer-1-fold-3": MassiveAdaptiveRLPrequentialStageV1.VALIDATION_3_RELEASED,
        }
        try:
            stage = stage_by_release[cast(str, release_kind)]
        except KeyError as error:
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "validation-release stage differs"
            ) from error
        disposition = None
        schedule_qualified = None
        gates = None
    elif type(artifact) is MassiveAdaptiveRLWalkForwardPolicyScheduleV1:
        runtime_authorized = bool(getattr(artifact, "development_stage_authorized"))
        stage_by_folds = {
            (0,): MassiveAdaptiveRLPrequentialStageV1.POLICY_0_FROZEN,
            (0, 1): MassiveAdaptiveRLPrequentialStageV1.POLICY_1_FROZEN,
            (0, 1, 2): MassiveAdaptiveRLPrequentialStageV1.POLICY_2_FROZEN,
            (0, 1, 2, 3): MassiveAdaptiveRLPrequentialStageV1.POLICY_3_FROZEN,
        }
        try:
            stage = stage_by_folds[tuple(getattr(artifact, "fold_indices", ()))]
        except KeyError as error:
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "policy-schedule state prefix differs"
            ) from error
        disposition = cast(str, getattr(artifact, "policy_schedule_disposition"))
        schedule_qualified = disposition == "policy-prefix-qualified"
        gates = None
    elif type(artifact) is MassiveAdaptiveRLOuterFoldSealAuthorityV1:
        runtime_authorized = bool(getattr(artifact, "development_outer_fold_sealed"))
        fold_index = getattr(artifact, "fold_index", None)
        stage_by_fold = {
            0: MassiveAdaptiveRLPrequentialStageV1.OUTER_0_SEALED,
            1: MassiveAdaptiveRLPrequentialStageV1.OUTER_1_SEALED,
            2: MassiveAdaptiveRLPrequentialStageV1.OUTER_2_SEALED,
            3: MassiveAdaptiveRLPrequentialStageV1.OUTER_3_SEALED,
        }
        try:
            stage = stage_by_fold[cast(int, fold_index)]
        except KeyError as error:
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "outer-seal state fold differs"
            ) from error
        disposition = None
        schedule_qualified = None
        gates = None
    elif type(artifact) is MassiveAdaptiveRLProfitabilityReportAuthorityV2:
        runtime_authorized = bool(
            getattr(artifact, "development_profitability_reporting_authorized")
        )
        stage = MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED
        disposition = cast(str, getattr(artifact, "policy_schedule_disposition"))
        schedule_qualified = bool(getattr(artifact, "policy_schedule_qualified"))
        gates = bool(getattr(artifact, "profitability_gates_passed"))
    elif type(artifact) is MassiveAdaptiveRLFullColdReplayAuthorityV1:
        runtime_authorized = bool(
            getattr(artifact, "development_full_cold_replay_verified")
        )
        stage = MassiveAdaptiveRLPrequentialStageV1.FULL_COLD_REPLAY_VERIFIED
        disposition = cast(str, getattr(artifact, "policy_schedule_disposition"))
        schedule_qualified = bool(getattr(artifact, "policy_schedule_qualified"))
        gates = bool(getattr(artifact, "profitability_gates_passed"))
    else:
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential stage artifact generation differs"
        )
    getattr(artifact, "validate")()
    if not runtime_authorized:
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential stage artifact has not been exactly replay-authorized"
        )
    artifact_manifest = getattr(artifact, "manifest_v5_receipt_sha256", None)
    if stage is MassiveAdaptiveRLPrequentialStageV1.TRAINED:
        artifact_manifest = manifest_receipt
    artifact_execution = getattr(
        artifact, "execution_implementation_registration_receipt_sha256", None
    )
    if stage is MassiveAdaptiveRLPrequentialStageV1.TRAINED:
        artifact_execution = execution_registration_receipt
    elif (
        stage is MassiveAdaptiveRLPrequentialStageV1.EXECUTION_IMPLEMENTATION_REGISTERED
    ):
        artifact_execution = getattr(artifact, "semantic_receipt_sha256", None)
    elif type(artifact) is MassiveAdaptiveRLValidationReleaseAuthorityV1:
        artifact_execution = getattr(
            artifact,
            "execution_implementation_registration_authority_receipt_sha256",
            None,
        )
    if (
        artifact_manifest != manifest_receipt
        or artifact_execution != execution_registration_receipt
    ):
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential stage artifact protocol lineage differs"
        )
    source, commit, committed_at_ms = _source_transaction_facts(artifact)
    source_qualified = getattr(artifact, "source_data_qualified", None)
    if type(source_qualified) is not bool or not source_qualified:
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential stage artifact is not source-qualified"
        )
    semantic = _required_digest(
        "stage artifact semantic receipt",
        getattr(artifact, "semantic_receipt_sha256", None),
    )
    schema = str(getattr(artifact, "schema", ""))
    if schema != _STAGE_ARTIFACT_SCHEMAS[stage]:
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential stage artifact schema differs"
        )
    return _StageArtifactFactsV1(
        stage=stage,
        schema=schema,
        semantic_receipt=semantic,
        source_receipt=source,
        commit_receipt=commit,
        committed_at_ms=committed_at_ms,
        source_data_qualified=source_qualified,
        policy_schedule_disposition=disposition,
        policy_schedule_qualified=schedule_qualified,
        profitability_gates_passed=gates,
    )


def _transaction_state(*, root: str | Path, relative: str) -> tuple[bool, bool]:
    payload = Path(root) / relative
    present = tuple(
        path.exists() or path.is_symlink()
        for path in (
            payload,
            payload.with_name(payload.name + ".receipt.json"),
            payload.with_name(payload.name + ".commit.json"),
        )
    )
    return all(present), any(present) and not all(present)


def _parse_state(
    *, root: str | Path, loaded: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLPrequentialExperimentStateV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential state payload is not canonical JSON"
        )
    body = dict(value)
    body["stage"] = MassiveAdaptiveRLPrequentialStageV1(str(body["stage"]))
    result = MassiveAdaptiveRLPrequentialExperimentStateV1(
        **body,
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_prequential_experiment_states_v1(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV5
) -> tuple[MassiveAdaptiveRLPrequentialExperimentStateV1, ...]:
    """Load the one contiguous V5 state prefix; reject gaps and branches."""

    manifest.validate()
    directory = (
        Path(root)
        / "adaptive-rl"
        / manifest.experiment_id
        / "prequential-experiment-state-v1"
    )
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential state directory is invalid"
        )
    allowed_names: set[str] = set()
    for stage in MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1:
        name = Path(
            _state_relative_path(experiment_id=manifest.experiment_id, stage=stage)
        ).name
        allowed_names.update({name, name + ".receipt.json", name + ".commit.json"})
    observed = tuple(directory.iterdir())
    if any(
        path.is_symlink() or not path.is_file() or path.name not in allowed_names
        for path in observed
    ):
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential state inventory contains an unexpected descendant"
        )
    states: list[MassiveAdaptiveRLPrequentialExperimentStateV1] = []
    gap = False
    for stage in MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1:
        relative = _state_relative_path(
            experiment_id=manifest.experiment_id, stage=stage
        )
        complete, partial = _transaction_state(root=root, relative=relative)
        if partial:
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "prequential state transaction is incomplete"
            )
        if not complete:
            gap = True
            continue
        if gap:
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "prequential state chain contains a gap or branch"
            )
        loaded = load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=time.time_ns() // 1_000_000,
        )
        states.append(_parse_state(root=root, loaded=loaded))
    result = tuple(states)
    for index, state in enumerate(result):
        if (
            state.sequence_index != index
            or state.stage is not MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1[index]
            or state.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "prequential state sequence differs"
            )
        if index:
            previous = result[index - 1]
            if (
                state.immediate_predecessor_state_receipt_sha256
                != previous.semantic_receipt_sha256
                or state.immediate_predecessor_state_committed_at_ms
                != previous.source_transaction_committed_at_ms
                or state.previous_stage_artifact_committed_at_ms
                != previous.stage_artifact_committed_at_ms
                or cast(int, state.source_transaction_committed_at_ms)
                <= cast(int, previous.source_transaction_committed_at_ms)
                or state.manifest_v5_registration_receipt_sha256
                != previous.manifest_v5_registration_receipt_sha256
                or state.execution_implementation_registration_receipt_sha256
                != previous.execution_implementation_registration_receipt_sha256
                or previous.policy_schedule_disposition
                == "policy-prefix-diagnostic-only"
                and state.policy_schedule_disposition == "policy-prefix-qualified"
            ):
                raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                    "prequential immediate-predecessor chain differs"
                )
    return result


def _state_body(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    previous: MassiveAdaptiveRLPrequentialExperimentStateV1 | None,
    facts: _StageArtifactFactsV1,
) -> dict[str, object]:
    expected_index = 0 if previous is None else previous.sequence_index + 1
    if (
        expected_index >= len(MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1)
        or facts.stage
        is not MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1[expected_index]
    ):
        raise MassiveAdaptiveRLPrequentialExperimentStateV1StaleError(
            "prequential state transition is not the exact next stage"
        )
    disposition = facts.policy_schedule_disposition
    qualified = facts.policy_schedule_qualified
    gates = facts.profitability_gates_passed
    if previous is not None:
        if disposition is None:
            disposition = previous.policy_schedule_disposition
            qualified = previous.policy_schedule_qualified
        if gates is None:
            gates = previous.profitability_gates_passed
        if (
            previous.policy_schedule_disposition == "policy-prefix-diagnostic-only"
            and disposition == "policy-prefix-qualified"
        ):
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "diagnostic policy schedule cannot become qualified"
            )
    return {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "manifest_v5_registration_receipt_sha256": (
            manifest_registration.semantic_receipt_sha256
        ),
        "execution_implementation_registration_receipt_sha256": (
            execution_registration.semantic_receipt_sha256
        ),
        "sequence_index": expected_index,
        "stage": facts.stage,
        "immediate_predecessor_state_receipt_sha256": (
            None if previous is None else previous.semantic_receipt_sha256
        ),
        "immediate_predecessor_state_committed_at_ms": (
            None if previous is None else previous.source_transaction_committed_at_ms
        ),
        "previous_stage_artifact_committed_at_ms": (
            None if previous is None else previous.stage_artifact_committed_at_ms
        ),
        "stage_artifact_schema": facts.schema,
        "stage_artifact_semantic_receipt_sha256": facts.semantic_receipt,
        "stage_artifact_source_receipt_sha256": facts.source_receipt,
        "stage_artifact_commit_receipt_sha256": facts.commit_receipt,
        "stage_artifact_committed_at_ms": facts.committed_at_ms,
        "policy_schedule_disposition": disposition,
        "policy_schedule_qualified": qualified,
        "profitability_gates_passed": gates,
        "source_data_qualified": bool(
            facts.source_data_qualified
            and (previous is None or previous.source_data_qualified)
        ),
        "blocker_code": None,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SCHEMA,
    }


def run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    stage_artifact: object,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLPrequentialExperimentStateV1:
    """Append or replay exactly one artifact-derived state transition."""

    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(execution_registration)
        is not MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
        or type(allow_materialize) is not bool
    ):
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential state requires exact Manifest-V5 roots"
        )
    manifest.validate()
    manifest_registration.validate()
    execution_registration.validate()
    if (
        not manifest_registration.development_protocol_registered
        or not execution_registration.development_execution_registered
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or execution_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or execution_registration.manifest_v5_registration_authority_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "prequential state registration lineage differs"
        )
    from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
    )

    if (
        type(stage_artifact) is MassiveAdaptiveRLFourFoldFitAuthorityV1
        and stage_artifact.semantic_receipt_sha256
        != execution_registration.four_fold_fit_authority_receipt_sha256
    ):
        raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
            "trained state does not match the registered four-fold fit"
        )
    facts = _stage_artifact_facts(
        manifest_receipt=manifest.semantic_receipt_sha256,
        execution_registration_receipt=execution_registration.semantic_receipt_sha256,
        artifact=stage_artifact,
    )
    relative = _state_relative_path(
        experiment_id=manifest.experiment_id, stage=facts.stage
    )
    with massive_adaptive_rl_experiment_materialization_lock_v1(
        artifact_root=root, experiment_id=manifest.experiment_id
    ):
        states = load_massive_adaptive_rl_prequential_experiment_states_v1(
            root=root, manifest=manifest
        )
        index = MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1.index(facts.stage)
        if index > len(states):
            raise MassiveAdaptiveRLPrequentialExperimentStateV1StaleError(
                "prequential state cannot skip a stage"
            )
        previous = None if index == 0 else states[index - 1]
        body = _state_body(
            manifest=manifest,
            manifest_registration=manifest_registration,
            execution_registration=execution_registration,
            previous=previous,
            facts=facts,
        )
        provisional = MassiveAdaptiveRLPrequentialExperimentStateV1(
            **body,  # type: ignore[arg-type]
            semantic_receipt_sha256="0" * 64,
        )
        expected = replace(
            provisional,
            semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
        )
        expected.validate()
        complete, partial = _transaction_state(root=root, relative=relative)
        if partial:
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "prequential state transaction is incomplete"
            )
        if not complete:
            if index != len(states):
                raise MassiveAdaptiveRLPrequentialExperimentStateV1StaleError(
                    "prequential state history has a conflicting head"
                )
            if not allow_materialize:
                raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                    "prequential state transition is absent"
                )
            predecessor_state_time = (
                -1
                if previous is None
                else _timestamp(
                    "predecessor state timestamp",
                    previous.source_transaction_committed_at_ms,
                )
            )
            committed_at_ms = (
                max(
                    time.time_ns() // 1_000_000,
                    expected.stage_artifact_committed_at_ms,
                    predecessor_state_time,
                )
                + 1
            )
            capability = issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1(
                root=root, authority=manifest_registration
            )
            with massive_adaptive_rl_manifest_v5_writer_scope_v1(
                root=root, capability=capability
            ):
                publish_massive_source_object(
                    stream=BytesIO(
                        canonical_json_file_bytes(expected.semantic_unsigned())
                    ),
                    root=root,
                    relative_payload_path=relative,
                    dataset_id=(
                        MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_DATASET
                    ),
                    source_object_key=relative,
                    requested_at_ms=committed_at_ms,
                    downloaded_at_ms=committed_at_ms,
                    schema_sha256=(
                        MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SCHEMA_SHA256
                    ),
                    entitlement_receipt_sha256=expected.semantic_receipt_sha256,
                    committed_at_ms=committed_at_ms,
                    request_id=(
                        "ADAPTIVE-RL-PREQUENTIAL-STATE-V1-"
                        f"{manifest.experiment_id}-{index:03d}"
                    ),
                )
        parsed = _parse_state(
            root=root,
            loaded=load_massive_source_bundle(
                root=root,
                relative_payload_path=relative,
                verified_at_ms=time.time_ns() // 1_000_000,
            ),
        )
        if parsed.semantic_unsigned() != expected.semantic_unsigned():
            raise MassiveAdaptiveRLPrequentialExperimentStateV1Error(
                "prequential persisted state does not replay"
            )
        result = replace(
            parsed,
            runtime_state_replayed=True,
            prequential_execution_authorized=parsed.source_data_qualified,
            development_profitability_reporting_authorized=bool(
                parsed.source_data_qualified
                and parsed.sequence_index
                >= MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1.index(
                    MassiveAdaptiveRLPrequentialStageV1.PROFITABILITY_REPORT_PUBLISHED
                )
            ),
            full_cold_replay_verified=bool(
                parsed.source_data_qualified
                and parsed.stage
                is MassiveAdaptiveRLPrequentialStageV1.FULL_COLD_REPLAY_VERIFIED
            ),
            positive_profitability_authorization_eligible=bool(
                parsed.source_data_qualified
                and parsed.stage
                is MassiveAdaptiveRLPrequentialStageV1.FULL_COLD_REPLAY_VERIFIED
                and parsed.policy_schedule_qualified
                and parsed.profitability_gates_passed
            ),
            _runtime_stage_artifact=stage_artifact,
        )
        result.validate()
        return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_EXPERIMENT_STATE_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_ORDER_V1",
    "MassiveAdaptiveRLPrequentialExperimentStateV1",
    "MassiveAdaptiveRLPrequentialExperimentStateV1Error",
    "MassiveAdaptiveRLPrequentialExperimentStateV1StaleError",
    "MassiveAdaptiveRLPrequentialStageV1",
    "load_massive_adaptive_rl_prequential_experiment_states_v1",
    "massive_adaptive_rl_prequential_state_relative_path_v1",
    "run_or_resume_massive_adaptive_rl_prequential_experiment_state_v1",
]
