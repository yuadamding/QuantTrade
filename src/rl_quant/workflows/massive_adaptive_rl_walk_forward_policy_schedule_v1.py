"""Create-only prequential policy-schedule prefixes for Manifest V5."""

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
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_frozen_rl_policy_v2 import (
    MassiveAdaptiveFrozenRLPolicyV2,
)
from rl_quant.training.massive_adaptive_rl_frozen_fc06_v2 import (
    MassiveAdaptiveRLFrozenFC06V2,
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
    MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)

if TYPE_CHECKING:
    from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
    )

MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_DATASET = (
    "massive-adaptive-rl-walk-forward-policy-schedule-v1"
)
MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA,
            "encoding": "canonical-json-exact-v5-policy-prefix",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1 = "policy-prefix-qualified"
MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_DIAGNOSTIC_V1 = "policy-prefix-diagnostic-only"


class MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(ValueError):
    """A policy prefix, its freeze lineage, or its causal seals differ."""


class MassiveAdaptiveRLWalkForwardPolicyScheduleV1LeaseUnavailable(RuntimeError):
    """Another process owns this schedule-prefix transaction."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(f"{name} is absent")
    return _digest(name, value)


def _timestamp(value: int | None, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
            f"{name} is absent or invalid"
        )
    return value


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def walk_forward_policy_schedule_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5, through_fold_index: int
) -> str:
    manifest.validate()
    if isinstance(through_fold_index, bool) or through_fold_index not in range(4):
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
            "policy-schedule prefix must end at fold 0, 1, 2, or 3"
        )
    return (
        f"adaptive-rl/{manifest.experiment_id}/walk-forward-policy-schedule-v1/"
        f"prefix-through-fold-{through_fold_index}.json"
    )


def _transaction_state(root: str | Path, relative: str) -> tuple[bool, bool]:
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


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLWalkForwardPolicyScheduleV1:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    manifest_v5_registration_receipt_sha256: str
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    split_plan_receipt_sha256: str
    outer_fold_receipts: tuple[str, ...]
    outer_session_date_inventories: tuple[tuple[str, ...], ...]
    fold_indices: tuple[int, ...]
    frozen_ppo_policy_receipts: tuple[str, ...]
    frozen_ppo_source_receipts: tuple[str, ...]
    frozen_ppo_commit_receipts: tuple[str, ...]
    frozen_ppo_committed_at_ms: tuple[int, ...]
    frozen_fc06_control_receipts: tuple[str, ...]
    frozen_fc06_source_receipts: tuple[str, ...]
    frozen_fc06_commit_receipts: tuple[str, ...]
    frozen_fc06_committed_at_ms: tuple[int, ...]
    validation_release_authority_receipts: tuple[str, ...]
    selected_candidate_validation_eligible: tuple[bool, ...]
    predecessor_outer_fold_indices: tuple[int, ...]
    predecessor_outer_fold_seal_receipts: tuple[str, ...]
    predecessor_outer_fold_seal_source_receipts: tuple[str, ...]
    predecessor_outer_fold_seal_commit_receipts: tuple[str, ...]
    predecessor_outer_fold_seal_committed_at_ms: tuple[int, ...]
    policy_schedule_disposition: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_schedule_replayed: bool = False
    development_outer_schedule_authorized: bool = False
    profitability_reporting_authorized: bool = False
    positive_profitability_authorization_eligible: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA
    _runtime_frozen_ppo_policies: tuple[MassiveAdaptiveFrozenRLPolicyV2, ...] = field(
        default=(), compare=False, repr=False
    )
    _runtime_frozen_fc06_controls: tuple[MassiveAdaptiveRLFrozenFC06V2, ...] = field(
        default=(), compare=False, repr=False
    )
    _runtime_predecessor_seals: tuple[
        MassiveAdaptiveRLOuterFoldSealAuthorityV1, ...
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
                "runtime_schedule_replayed",
                "development_outer_schedule_authorized",
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

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self._loaded_source is not None
            and self.runtime_schedule_replayed
            and self.development_outer_schedule_authorized
            and self.source_data_qualified
        )

    @property
    def complete_schedule(self) -> bool:
        return self.fold_indices == (0, 1, 2, 3)

    @property
    def all_policies_validation_eligible(self) -> bool:
        return bool(
            self.complete_schedule and all(self.selected_candidate_validation_eligible)
        )

    def authorizes_outer_fold(self, fold_index: int) -> bool:
        self.validate()
        required_length = (2, 3, 4, 4)[fold_index] if fold_index in range(4) else 99
        return bool(
            self.development_stage_authorized
            and len(self.fold_indices) >= required_length
        )

    def frozen_policy(self, fold_index: int) -> MassiveAdaptiveFrozenRLPolicyV2:
        self.validate()
        if not self.development_stage_authorized or fold_index not in self.fold_indices:
            raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                "frozen PPO policy is not available from this schedule prefix"
            )
        return self._runtime_frozen_ppo_policies[self.fold_indices.index(fold_index)]

    def frozen_control(self, fold_index: int) -> MassiveAdaptiveRLFrozenFC06V2:
        self.validate()
        if not self.development_stage_authorized or fold_index not in self.fold_indices:
            raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                "frozen FC06 control is not available from this schedule prefix"
            )
        return self._runtime_frozen_fc06_controls[self.fold_indices.index(fold_index)]

    def validate(self) -> None:
        count = len(self.fold_indices)
        frozen_ppo_times = tuple(
            _timestamp(value, "frozen PPO time")
            for value in self.frozen_ppo_committed_at_ms
        )
        frozen_fc06_times = tuple(
            _timestamp(value, "frozen FC06 time")
            for value in self.frozen_fc06_committed_at_ms
        )
        predecessor_seal_times = tuple(
            _timestamp(value, "outer seal time")
            for value in self.predecessor_outer_fold_seal_committed_at_ms
        )
        inventories = (
            self.frozen_ppo_policy_receipts,
            self.frozen_ppo_source_receipts,
            self.frozen_ppo_commit_receipts,
            self.frozen_ppo_committed_at_ms,
            self.frozen_fc06_control_receipts,
            self.frozen_fc06_source_receipts,
            self.frozen_fc06_commit_receipts,
            self.frozen_fc06_committed_at_ms,
            self.validation_release_authority_receipts,
            self.selected_candidate_validation_eligible,
        )
        runtime = bool(self._runtime_frozen_ppo_policies)
        expected_disposition = (
            MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1
            if all(self.selected_candidate_validation_eligible)
            else MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_DIAGNOSTIC_V1
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA
            or not self.experiment_id
            or len(self.outer_fold_receipts) != 4
            or len(self.outer_session_date_inventories) != 4
            or any(
                len(row) != 126 or row != tuple(sorted(set(row)))
                for row in self.outer_session_date_inventories
            )
            or self.fold_indices not in ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
            or any(len(values) != count for values in inventories)
            or any(
                not isinstance(value, bool)
                for value in self.selected_candidate_validation_eligible
            )
            or self.predecessor_outer_fold_indices != tuple(range(max(0, count - 2)))
            or len(self.predecessor_outer_fold_seal_receipts) != max(0, count - 2)
            or len(self.predecessor_outer_fold_seal_source_receipts)
            != max(0, count - 2)
            or len(self.predecessor_outer_fold_seal_commit_receipts)
            != max(0, count - 2)
            or len(self.predecessor_outer_fold_seal_committed_at_ms)
            != max(0, count - 2)
            or any(
                seal_time
                >= min(frozen_ppo_times[index + 2], frozen_fc06_times[index + 2])
                for index, seal_time in enumerate(predecessor_seal_times)
            )
            or self.policy_schedule_disposition != expected_disposition
            or not isinstance(self.source_data_qualified, bool)
            or bool(self._runtime_frozen_fc06_controls) != runtime
            or len(self._runtime_frozen_ppo_policies) not in (0, count)
            or len(self._runtime_frozen_fc06_controls) not in (0, count)
            or len(self._runtime_predecessor_seals)
            != (max(0, count - 2) if runtime else 0)
            or self.runtime_schedule_replayed != runtime
            or self.development_outer_schedule_authorized
            != bool(runtime and self.source_data_qualified)
            or self.profitability_reporting_authorized
            or self.positive_profitability_authorization_eligible
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                "walk-forward policy schedule differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        for values in (
            self.frozen_ppo_policy_receipts,
            self.frozen_ppo_source_receipts,
            self.frozen_ppo_commit_receipts,
            self.frozen_fc06_control_receipts,
            self.frozen_fc06_source_receipts,
            self.frozen_fc06_commit_receipts,
            self.validation_release_authority_receipts,
            self.predecessor_outer_fold_seal_receipts,
            self.predecessor_outer_fold_seal_source_receipts,
            self.predecessor_outer_fold_seal_commit_receipts,
            self.outer_fold_receipts,
        ):
            for value in values:
                _digest("policy schedule inventory", value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            latest_freeze = max(
                *frozen_ppo_times,
                *frozen_fc06_times,
            )
            latest_predecessor_seal = max(
                predecessor_seal_times,
                default=-1,
            )
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.semantic_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= max(latest_freeze, latest_predecessor_seal)
            ):
                raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                    "policy-schedule source transaction differs"
                )
        if runtime:
            from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
                MassiveAdaptiveRLOuterFoldSealAuthorityV1,
            )

            for ppo, fc06, index in zip(
                self._runtime_frozen_ppo_policies,
                self._runtime_frozen_fc06_controls,
                self.fold_indices,
                strict=True,
            ):
                ppo.validate()
                fc06.validate()
                if (
                    not ppo.development_stage_authorized
                    or not fc06.development_stage_authorized
                    or ppo.fold_index != index
                    or fc06.fold_index != index
                    or ppo.policy_selection_authority_receipt_sha256
                    != fc06.policy_selection_authority_receipt_sha256
                    or ppo.manifest_v5_receipt_sha256 != self.manifest_v5_receipt_sha256
                    or fc06.manifest_v5_receipt_sha256
                    != self.manifest_v5_receipt_sha256
                ):
                    raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                        "runtime frozen policy pair differs"
                    )
            for seal, index, receipt, source, commit, committed_at_ms in zip(
                self._runtime_predecessor_seals,
                self.predecessor_outer_fold_indices,
                self.predecessor_outer_fold_seal_receipts,
                self.predecessor_outer_fold_seal_source_receipts,
                self.predecessor_outer_fold_seal_commit_receipts,
                self.predecessor_outer_fold_seal_committed_at_ms,
                strict=True,
            ):
                if type(seal) is not MassiveAdaptiveRLOuterFoldSealAuthorityV1:
                    raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                        "predecessor outer seal type differs"
                    )
                seal.validate()
                delayed_release = self._runtime_frozen_ppo_policies[
                    index + 2
                ].policy_selection_authority.fold_validation_authority.release_authority
                if (
                    seal.fold_index != index
                    or seal.semantic_receipt_sha256 != receipt
                    or seal.source_receipt_sha256 != source
                    or seal.source_transaction_receipt_sha256 != commit
                    or seal.source_transaction_committed_at_ms != committed_at_ms
                    or seal.manifest_v5_receipt_sha256
                    != self.manifest_v5_receipt_sha256
                    or seal.execution_implementation_registration_receipt_sha256
                    != self.execution_implementation_registration_receipt_sha256
                    or seal.scientific_execution_fingerprint_sha256
                    != self.scientific_execution_fingerprint_sha256
                    or not seal.development_outer_fold_sealed
                    or delayed_release.predecessor_outer_fold_index != index
                    or delayed_release.predecessor_outer_fold_seal_receipt_sha256
                    != receipt
                    or delayed_release.predecessor_outer_fold_seal_source_receipt_sha256
                    != source
                    or delayed_release.predecessor_outer_fold_seal_commit_receipt_sha256
                    != commit
                    or delayed_release.predecessor_outer_fold_seal_committed_at_ms
                    != committed_at_ms
                ):
                    raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                        "predecessor outer seal differs"
                    )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_walk_forward_policy_schedule_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    frozen_ppo_policies: Sequence[MassiveAdaptiveFrozenRLPolicyV2],
    frozen_fc06_controls: Sequence[MassiveAdaptiveRLFrozenFC06V2],
    predecessor_outer_fold_seals: Sequence[
        MassiveAdaptiveRLOuterFoldSealAuthorityV1
    ] = (),
) -> MassiveAdaptiveRLWalkForwardPolicyScheduleV1:
    from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
        MassiveAdaptiveRLOuterFoldSealAuthorityV1,
    )

    ppo = tuple(frozen_ppo_policies)
    fc06 = tuple(frozen_fc06_controls)
    seals = tuple(predecessor_outer_fold_seals)
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(execution_registration)
        is not MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
        or any(type(row) is not MassiveAdaptiveFrozenRLPolicyV2 for row in ppo)
        or any(type(row) is not MassiveAdaptiveRLFrozenFC06V2 for row in fc06)
        or any(
            type(row) is not MassiveAdaptiveRLOuterFoldSealAuthorityV1 for row in seals
        )
    ):
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
            "walk-forward policy schedule requires exact V5 authority generations"
        )
    for authority in (
        manifest,
        manifest_registration,
        execution_registration,
        *ppo,
        *fc06,
        *seals,
    ):
        authority.validate()
    indices = tuple(row.fold_index for row in ppo)
    if not ppo:
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
            "walk-forward policy schedule requires frozen PPO policies"
        )
    initial_inputs = ppo[
        0
    ].policy_selection_authority.fold_validation_authority.release_authority.initial_validation_inputs
    plan = initial_inputs.prequential_validation_plan
    split_plan = initial_inputs.runtime_sources_v2.base_runtime_sources_v1.split_plan
    delayed_releases = tuple(
        policy.policy_selection_authority.fold_validation_authority.release_authority
        for policy in ppo[2:]
    )
    if (
        indices not in ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
        or tuple(row.fold_index for row in fc06) != indices
        or len(seals) != max(0, len(indices) - 2)
        or not manifest_registration.development_protocol_registered
        or not execution_registration.development_execution_registered
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or execution_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or plan.split_plan_receipt_sha256 != split_plan.semantic_receipt_sha256
        or any(
            row.execution_implementation_registration_receipt_sha256
            != execution_registration.semantic_receipt_sha256
            or row.scientific_execution_fingerprint_sha256
            != execution_registration.scientific_execution_fingerprint_sha256
            for row in (*ppo, *fc06)
        )
        or any(
            seal.fold_index != index
            or seal.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
            or seal.execution_implementation_registration_receipt_sha256
            != execution_registration.semantic_receipt_sha256
            or not seal.development_outer_fold_sealed
            for index, seal in enumerate(seals)
        )
        or any(
            _timestamp(seal.source_transaction_committed_at_ms, "outer seal time")
            >= min(
                _timestamp(
                    ppo[index + 2].source_transaction_committed_at_ms,
                    "delayed frozen PPO time",
                ),
                _timestamp(
                    fc06[index + 2].source_transaction_committed_at_ms,
                    "delayed frozen FC06 time",
                ),
            )
            for index, seal in enumerate(seals)
        )
        or any(
            release.predecessor_outer_fold_index != index
            or release.predecessor_outer_fold_seal_receipt_sha256
            != seal.semantic_receipt_sha256
            or release.predecessor_outer_fold_seal_source_receipt_sha256
            != seal.source_receipt_sha256
            or release.predecessor_outer_fold_seal_commit_receipt_sha256
            != seal.source_transaction_receipt_sha256
            or release.predecessor_outer_fold_seal_committed_at_ms
            != seal.source_transaction_committed_at_ms
            for index, (seal, release) in enumerate(
                zip(seals, delayed_releases, strict=True)
            )
        )
    ):
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
            "walk-forward policy schedule roots differ"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "manifest_v5_registration_receipt_sha256": manifest_registration.semantic_receipt_sha256,
        "execution_implementation_registration_receipt_sha256": execution_registration.semantic_receipt_sha256,
        "scientific_execution_fingerprint_sha256": execution_registration.scientific_execution_fingerprint_sha256,
        "split_plan_receipt_sha256": plan.split_plan_receipt_sha256,
        "outer_fold_receipts": tuple(
            row.receipt_sha256 for row in split_plan.outer_folds
        ),
        "outer_session_date_inventories": plan.outer_session_date_inventories,
        "fold_indices": indices,
        "frozen_ppo_policy_receipts": tuple(row.semantic_receipt_sha256 for row in ppo),
        "frozen_ppo_source_receipts": tuple(
            _required(row.source_receipt_sha256, "frozen PPO source") for row in ppo
        ),
        "frozen_ppo_commit_receipts": tuple(
            _required(row.source_transaction_receipt_sha256, "frozen PPO commit")
            for row in ppo
        ),
        "frozen_ppo_committed_at_ms": tuple(
            _timestamp(row.source_transaction_committed_at_ms, "frozen PPO time")
            for row in ppo
        ),
        "frozen_fc06_control_receipts": tuple(
            row.semantic_receipt_sha256 for row in fc06
        ),
        "frozen_fc06_source_receipts": tuple(
            _required(row.source_receipt_sha256, "frozen FC06 source") for row in fc06
        ),
        "frozen_fc06_commit_receipts": tuple(
            _required(row.source_transaction_receipt_sha256, "frozen FC06 commit")
            for row in fc06
        ),
        "frozen_fc06_committed_at_ms": tuple(
            _timestamp(row.source_transaction_committed_at_ms, "frozen FC06 time")
            for row in fc06
        ),
        "validation_release_authority_receipts": tuple(
            row.validation_release_authority_receipt_sha256 for row in ppo
        ),
        "selected_candidate_validation_eligible": tuple(
            row.selected_candidate_validation_eligible for row in ppo
        ),
        "predecessor_outer_fold_indices": tuple(range(len(seals))),
        "predecessor_outer_fold_seal_receipts": tuple(
            cast(str, getattr(row, "semantic_receipt_sha256")) for row in seals
        ),
        "predecessor_outer_fold_seal_source_receipts": tuple(
            _required(row.source_receipt_sha256, "outer seal source") for row in seals
        ),
        "predecessor_outer_fold_seal_commit_receipts": tuple(
            _required(row.source_transaction_receipt_sha256, "outer seal commit")
            for row in seals
        ),
        "predecessor_outer_fold_seal_committed_at_ms": tuple(
            _timestamp(row.source_transaction_committed_at_ms, "outer seal time")
            for row in seals
        ),
        "policy_schedule_disposition": (
            MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1
            if all(row.selected_candidate_validation_eligible for row in ppo)
            else MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_DIAGNOSTIC_V1
        ),
        "source_data_qualified": bool(
            execution_registration.source_data_qualified
            and all(row.source_data_qualified for row in ppo)
            and all(row.source_data_qualified for row in fc06)
            and all(getattr(row, "source_data_qualified", False) for row in seals)
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SCHEMA,
    }
    provisional = MassiveAdaptiveRLWalkForwardPolicyScheduleV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_schedule_replayed=True,
        development_outer_schedule_authorized=bool(body["source_data_qualified"]),
        _runtime_frozen_ppo_policies=ppo,
        _runtime_frozen_fc06_controls=fc06,
        _runtime_predecessor_seals=seals,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse(
    *, root: str | Path, loaded: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLWalkForwardPolicyScheduleV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
            "policy-schedule source is not canonical JSON"
        )
    body = dict(value)
    for name in (
        "fold_indices",
        "frozen_ppo_policy_receipts",
        "frozen_ppo_source_receipts",
        "frozen_ppo_commit_receipts",
        "frozen_ppo_committed_at_ms",
        "frozen_fc06_control_receipts",
        "frozen_fc06_source_receipts",
        "frozen_fc06_commit_receipts",
        "frozen_fc06_committed_at_ms",
        "validation_release_authority_receipts",
        "selected_candidate_validation_eligible",
        "predecessor_outer_fold_indices",
        "predecessor_outer_fold_seal_receipts",
        "predecessor_outer_fold_seal_source_receipts",
        "predecessor_outer_fold_seal_commit_receipts",
        "predecessor_outer_fold_seal_committed_at_ms",
        "outer_fold_receipts",
    ):
        body[name] = tuple(cast(Sequence[object], body[name]))
    body["outer_session_date_inventories"] = tuple(
        tuple(str(item) for item in cast(Sequence[object], row))
        for row in cast(Sequence[object], body["outer_session_date_inventories"])
    )
    result = MassiveAdaptiveRLWalkForwardPolicyScheduleV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    frozen_ppo_policies: Sequence[MassiveAdaptiveFrozenRLPolicyV2],
    frozen_fc06_controls: Sequence[MassiveAdaptiveRLFrozenFC06V2],
    predecessor_outer_fold_seals: Sequence[
        MassiveAdaptiveRLOuterFoldSealAuthorityV1
    ] = (),
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLWalkForwardPolicyScheduleV1:
    expected = build_massive_adaptive_rl_walk_forward_policy_schedule_v1(
        manifest=manifest,
        manifest_registration=manifest_registration,
        execution_registration=execution_registration,
        frozen_ppo_policies=frozen_ppo_policies,
        frozen_fc06_controls=frozen_fc06_controls,
        predecessor_outer_fold_seals=predecessor_outer_fold_seals,
    )
    relative = walk_forward_policy_schedule_relative_path_v1(
        manifest=manifest, through_fold_index=expected.fold_indices[-1]
    )
    try:
        with massive_adaptive_rl_experiment_materialization_lock_v1(
            artifact_root=root, experiment_id=manifest.experiment_id
        ):
            complete, partial = _transaction_state(root, relative)
            if partial:
                raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                    "policy-schedule transaction is incomplete"
                )
            if not complete:
                if not allow_materialize:
                    raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                        "policy-schedule prefix is absent"
                    )
                committed_at_ms = (
                    max(
                        _now_ms(),
                        *expected.frozen_ppo_committed_at_ms,
                        *expected.frozen_fc06_committed_at_ms,
                        *(
                            _timestamp(
                                seal.source_transaction_committed_at_ms,
                                "outer seal time",
                            )
                            for seal in predecessor_outer_fold_seals
                        ),
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
                        dataset_id=MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_DATASET,
                        source_object_key=relative,
                        requested_at_ms=committed_at_ms,
                        downloaded_at_ms=committed_at_ms,
                        schema_sha256=MASSIVE_ADAPTIVE_RL_WALK_FORWARD_POLICY_SCHEDULE_V1_SOURCE_SCHEMA_SHA256,
                        entitlement_receipt_sha256=expected.semantic_receipt_sha256,
                        committed_at_ms=committed_at_ms,
                        request_id=f"ADAPTIVE-RL-WALK-FORWARD-SCHEDULE-{manifest.experiment_id}-{expected.fold_indices[-1]}",
                    )
            loaded = load_massive_source_bundle(
                root=root, relative_payload_path=relative, verified_at_ms=_now_ms()
            )
            parsed = _parse(root=root, loaded=loaded)
            if parsed.semantic_unsigned() != expected.semantic_unsigned():
                raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
                    "policy-schedule prefix does not replay"
                )
            result = replace(
                parsed,
                runtime_schedule_replayed=True,
                development_outer_schedule_authorized=parsed.source_data_qualified,
                _runtime_frozen_ppo_policies=tuple(frozen_ppo_policies),
                _runtime_frozen_fc06_controls=tuple(frozen_fc06_controls),
                _runtime_predecessor_seals=tuple(predecessor_outer_fold_seals),
            )
            result.validate()
            return result
    except MassiveAdaptiveRLExperimentLockV1Unavailable as error:
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1LeaseUnavailable(
            "policy-schedule publication is already owned"
        ) from error
    except MassiveAdaptiveRLExperimentLockV1Error as error:
        raise MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error(
            "policy-schedule publication lock is invalid"
        ) from error


__all__ = [
    "MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_DIAGNOSTIC_V1",
    "MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1",
    "MassiveAdaptiveRLWalkForwardPolicyScheduleV1",
    "MassiveAdaptiveRLWalkForwardPolicyScheduleV1Error",
    "build_massive_adaptive_rl_walk_forward_policy_schedule_v1",
    "run_or_resume_massive_adaptive_rl_walk_forward_policy_schedule_v1",
    "walk_forward_policy_schedule_relative_path_v1",
]
