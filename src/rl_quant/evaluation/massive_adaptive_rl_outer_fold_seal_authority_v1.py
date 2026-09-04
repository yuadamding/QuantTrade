"""Create-only authenticated seal for one Manifest-V5 outer fold."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
import json
import math
from pathlib import Path
import time
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v2 import (
    MassiveAdaptiveOuterAccessCommitmentV2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_rollout_authority_v2 import (
    MassiveAdaptiveRLOuterRolloutAuthorityV2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
    MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-outer-fold-seal-authority-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA,
        "payload": "complete-shared-economics-fold-seal",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(ValueError):
    """The outer access, rollout, schedule, or seal chronology differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: str | None) -> str:
    if value is None:
        raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(f"{name} is absent")
    return _digest(name, value)


def _required_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
            f"{name} is absent or invalid"
        )
    return value


def outer_fold_seal_authority_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
            "outer-fold seal index differs"
        )
    return (
        f"adaptive-rl/{manifest.experiment_id}/outer-fold-seal-authority-v1/"
        f"fold-{fold_index}.json"
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterFoldSealAuthorityV1:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    fold_index: int
    policy_schedule_receipt_sha256: str
    policy_schedule_source_receipt_sha256: str
    policy_schedule_commit_receipt_sha256: str
    outer_access_commitment_receipt_sha256: str
    outer_access_source_receipt_sha256: str
    outer_access_commit_receipt_sha256: str
    outer_access_committed_at_ms: int
    outer_rollout_authority_receipt_sha256: str
    outer_rollout_source_receipt_sha256: str
    outer_rollout_commit_receipt_sha256: str
    outer_rollout_committed_at_ms: int
    outer_rollout_receipt_sha256: str
    frozen_ppo_policy_receipt_sha256: str
    frozen_fc06_control_receipt_sha256: str
    decision_session_dates: tuple[str, ...]
    outer_decision_inventory_sha256: str
    ppo_action_inventory_sha256: str
    ppo_primary_transition_inventory_sha256: str
    ppo_low_cost_transition_inventory_sha256: str
    ppo_high_cost_transition_inventory_sha256: str
    fixed_control_transition_inventory_sha256: str
    fixed_control_low_cost_transition_inventory_sha256: str
    fixed_control_high_cost_transition_inventory_sha256: str
    decision_target_inventory_sha256: str
    fixed_control_decision_target_inventory_sha256: str
    primary_terminal_liquidation_adjusted_return: float
    low_cost_terminal_liquidation_adjusted_return: float
    high_cost_terminal_liquidation_adjusted_return: float
    fixed_control_terminal_liquidation_adjusted_return: float
    fixed_control_low_cost_terminal_liquidation_adjusted_return: float
    fixed_control_high_cost_terminal_liquidation_adjusted_return: float
    ppo_cost_ladder_monotone: bool
    fixed_control_cost_ladder_monotone: bool
    maximum_drawdown: float
    policy_validation_eligible: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_seal_replayed: bool = False
    development_outer_fold_sealed: bool = False
    validation_release_authorized: bool = False
    profitability_reporting_authorized: bool = False
    positive_profitability_authorization_eligible: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA
    _runtime_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_access: MassiveAdaptiveOuterAccessCommitmentV2 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_rollout: MassiveAdaptiveRLOuterRolloutAuthorityV2 | None = field(
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
                "runtime_seal_replayed",
                "development_outer_fold_sealed",
                "validation_release_authorized",
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
    def rollout_authority(self) -> MassiveAdaptiveRLOuterRolloutAuthorityV2:
        self.validate()
        if self._runtime_rollout is None or not self.development_outer_fold_sealed:
            raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
                "sealed rollout has not been exactly replayed"
            )
        return self._runtime_rollout

    @property
    def policy_schedule(self) -> MassiveAdaptiveRLWalkForwardPolicyScheduleV1:
        self.validate()
        if self._runtime_schedule is None or not self.development_outer_fold_sealed:
            raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
                "sealed policy schedule has not been exactly replayed"
            )
        return self._runtime_schedule

    def validate(self) -> None:
        runtime_values = (
            self._runtime_schedule,
            self._runtime_access,
            self._runtime_rollout,
        )
        runtime = all(value is not None for value in runtime_values)
        economic_values = (
            self.primary_terminal_liquidation_adjusted_return,
            self.low_cost_terminal_liquidation_adjusted_return,
            self.high_cost_terminal_liquidation_adjusted_return,
            self.fixed_control_terminal_liquidation_adjusted_return,
            self.fixed_control_low_cost_terminal_liquidation_adjusted_return,
            self.fixed_control_high_cost_terminal_liquidation_adjusted_return,
            self.maximum_drawdown,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or len(self.decision_session_dates) != 126
            or self.decision_session_dates
            != tuple(sorted(set(self.decision_session_dates)))
            or self.outer_decision_inventory_sha256
            != semantic_sha256(self.decision_session_dates)
            or any(not math.isfinite(value) for value in economic_values)
            or min(economic_values[:-1]) <= -1.0
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or not isinstance(self.ppo_cost_ladder_monotone, bool)
            or self.ppo_cost_ladder_monotone
            != (
                self.low_cost_terminal_liquidation_adjusted_return
                >= self.primary_terminal_liquidation_adjusted_return
                >= self.high_cost_terminal_liquidation_adjusted_return
            )
            or not isinstance(self.fixed_control_cost_ladder_monotone, bool)
            or self.fixed_control_cost_ladder_monotone
            != (
                self.fixed_control_low_cost_terminal_liquidation_adjusted_return
                >= self.fixed_control_terminal_liquidation_adjusted_return
                >= self.fixed_control_high_cost_terminal_liquidation_adjusted_return
            )
            or not isinstance(self.policy_validation_eligible, bool)
            or not isinstance(self.source_data_qualified, bool)
            or any(value is not None for value in runtime_values) != runtime
            or self.runtime_seal_replayed != runtime
            or self.development_outer_fold_sealed
            != bool(runtime and self.source_data_qualified)
            or self.validation_release_authorized
            != bool(
                runtime and self.source_data_qualified and self.fold_index in (0, 1)
            )
            or self.profitability_reporting_authorized
            or self.positive_profitability_authorization_eligible
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
                "outer-fold seal differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.semantic_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= self.outer_rollout_committed_at_ms
            ):
                raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
                    "outer-fold seal source transaction differs"
                )
        if runtime:
            assert self._runtime_schedule is not None
            assert self._runtime_access is not None
            assert self._runtime_rollout is not None
            self._runtime_schedule.validate()
            self._runtime_access.validate()
            self._runtime_rollout.validate()
            if (
                not self._runtime_schedule.authorizes_outer_fold(self.fold_index)
                or self._runtime_schedule.experiment_id != self.experiment_id
                or self._runtime_schedule.manifest_v5_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_schedule.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_schedule.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_schedule.semantic_receipt_sha256
                != self.policy_schedule_receipt_sha256
                or self._runtime_schedule.source_receipt_sha256
                != self.policy_schedule_source_receipt_sha256
                or self._runtime_schedule.source_transaction_receipt_sha256
                != self.policy_schedule_commit_receipt_sha256
                or not self._runtime_access.outer_input_access_authorized
                or self._runtime_access.semantic_receipt_sha256
                != self.outer_access_commitment_receipt_sha256
                or self._runtime_access.source_receipt_sha256
                != self.outer_access_source_receipt_sha256
                or self._runtime_access.source_transaction_receipt_sha256
                != self.outer_access_commit_receipt_sha256
                or self._runtime_access.source_transaction_committed_at_ms
                != self.outer_access_committed_at_ms
                or self._runtime_access.policy_schedule_receipt_sha256
                != self.policy_schedule_receipt_sha256
                or not self._runtime_rollout.outer_evaluation_authorized
                or self._runtime_rollout.semantic_receipt_sha256
                != self.outer_rollout_authority_receipt_sha256
                or self._runtime_rollout.source_receipt_sha256
                != self.outer_rollout_source_receipt_sha256
                or self._runtime_rollout.source_transaction_receipt_sha256
                != self.outer_rollout_commit_receipt_sha256
                or self._runtime_rollout.source_transaction_committed_at_ms
                != self.outer_rollout_committed_at_ms
                or self._runtime_rollout.outer_rollout_receipt_sha256
                != self.outer_rollout_receipt_sha256
                or self._runtime_rollout.fold_index != self.fold_index
                or self._runtime_rollout.decision_session_dates
                != self.decision_session_dates
                or self._runtime_rollout.ppo_cost_ladder_monotone
                != self.ppo_cost_ladder_monotone
                or self._runtime_rollout.fixed_control_cost_ladder_monotone
                != self.fixed_control_cost_ladder_monotone
            ):
                raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
                    "outer-fold seal runtime lineage differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_outer_fold_seal_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    policy_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2,
    outer_rollout: MassiveAdaptiveRLOuterRolloutAuthorityV2,
) -> MassiveAdaptiveRLOuterFoldSealAuthorityV1:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(policy_schedule) is not MassiveAdaptiveRLWalkForwardPolicyScheduleV1
        or type(outer_access) is not MassiveAdaptiveOuterAccessCommitmentV2
        or type(outer_rollout) is not MassiveAdaptiveRLOuterRolloutAuthorityV2
    ):
        raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
            "outer-fold seal requires exact Manifest-V5 authority generations"
        )
    for authority in (manifest, policy_schedule, outer_access, outer_rollout):
        authority.validate()
    computation = outer_rollout.rollout
    fold_index = outer_access.fold_index
    if (
        outer_rollout.fold_index != fold_index
        or manifest.experiment_id != policy_schedule.experiment_id
        or manifest.experiment_id != outer_access.experiment_id
        or manifest.experiment_id != outer_rollout.experiment_id
        or manifest.semantic_receipt_sha256
        != policy_schedule.manifest_v5_receipt_sha256
        or manifest.semantic_receipt_sha256 != outer_access.manifest_v5_receipt_sha256
        or manifest.semantic_receipt_sha256 != outer_rollout.manifest_v5_receipt_sha256
        or manifest.scientific_protocol_projection_sha256
        != policy_schedule.scientific_protocol_projection_sha256
        or manifest.scientific_protocol_projection_sha256
        != outer_access.scientific_protocol_projection_sha256
        or manifest.scientific_protocol_projection_sha256
        != outer_rollout.scientific_protocol_projection_sha256
        or policy_schedule.execution_implementation_registration_receipt_sha256
        != outer_access.execution_implementation_registration_receipt_sha256
        or policy_schedule.execution_implementation_registration_receipt_sha256
        != outer_rollout.execution_implementation_registration_receipt_sha256
        or policy_schedule.scientific_execution_fingerprint_sha256
        != outer_access.scientific_execution_fingerprint_sha256
        or policy_schedule.scientific_execution_fingerprint_sha256
        != outer_rollout.scientific_execution_fingerprint_sha256
        or policy_schedule.semantic_receipt_sha256
        != outer_access.policy_schedule_receipt_sha256
        or policy_schedule.source_receipt_sha256
        != outer_access.policy_schedule_source_receipt_sha256
        or policy_schedule.source_transaction_receipt_sha256
        != outer_access.policy_schedule_commit_receipt_sha256
        or outer_rollout.outer_access_commitment_receipt_sha256
        != outer_access.semantic_receipt_sha256
        or outer_rollout.outer_access_source_receipt_sha256
        != outer_access.source_receipt_sha256
        or outer_rollout.outer_access_commit_receipt_sha256
        != outer_access.source_transaction_receipt_sha256
        or outer_rollout.outer_access_committed_at_ms
        != outer_access.source_transaction_committed_at_ms
        or outer_access.outer_decision_session_dates
        != outer_rollout.decision_session_dates
    ):
        raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
            "outer-fold seal roots differ"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "execution_implementation_registration_receipt_sha256": (
            outer_access.execution_implementation_registration_receipt_sha256
        ),
        "scientific_execution_fingerprint_sha256": (
            outer_access.scientific_execution_fingerprint_sha256
        ),
        "fold_index": fold_index,
        "policy_schedule_receipt_sha256": policy_schedule.semantic_receipt_sha256,
        "policy_schedule_source_receipt_sha256": _required_digest(
            "policy schedule source", policy_schedule.source_receipt_sha256
        ),
        "policy_schedule_commit_receipt_sha256": _required_digest(
            "policy schedule commit", policy_schedule.source_transaction_receipt_sha256
        ),
        "outer_access_commitment_receipt_sha256": outer_access.semantic_receipt_sha256,
        "outer_access_source_receipt_sha256": _required_digest(
            "outer access source", outer_access.source_receipt_sha256
        ),
        "outer_access_commit_receipt_sha256": _required_digest(
            "outer access commit", outer_access.source_transaction_receipt_sha256
        ),
        "outer_access_committed_at_ms": _required_time(
            "outer access time", outer_access.source_transaction_committed_at_ms
        ),
        "outer_rollout_authority_receipt_sha256": outer_rollout.semantic_receipt_sha256,
        "outer_rollout_source_receipt_sha256": _required_digest(
            "outer rollout source", outer_rollout.source_receipt_sha256
        ),
        "outer_rollout_commit_receipt_sha256": _required_digest(
            "outer rollout commit", outer_rollout.source_transaction_receipt_sha256
        ),
        "outer_rollout_committed_at_ms": _required_time(
            "outer rollout time", outer_rollout.source_transaction_committed_at_ms
        ),
        "outer_rollout_receipt_sha256": outer_rollout.outer_rollout_receipt_sha256,
        "frozen_ppo_policy_receipt_sha256": outer_access.frozen_ppo_policy_receipt_sha256,
        "frozen_fc06_control_receipt_sha256": outer_access.frozen_fc06_control_receipt_sha256,
        "decision_session_dates": outer_rollout.decision_session_dates,
        "outer_decision_inventory_sha256": semantic_sha256(
            outer_rollout.decision_session_dates
        ),
        "ppo_action_inventory_sha256": outer_rollout.ppo_action_inventory_sha256,
        "ppo_primary_transition_inventory_sha256": (
            outer_rollout.ppo_primary_transition_inventory_sha256
        ),
        "ppo_low_cost_transition_inventory_sha256": (
            outer_rollout.ppo_low_cost_transition_inventory_sha256
        ),
        "ppo_high_cost_transition_inventory_sha256": (
            outer_rollout.ppo_high_cost_transition_inventory_sha256
        ),
        "fixed_control_transition_inventory_sha256": (
            outer_rollout.fixed_control_transition_inventory_sha256
        ),
        "fixed_control_low_cost_transition_inventory_sha256": (
            outer_rollout.fixed_control_low_cost_transition_inventory_sha256
        ),
        "fixed_control_high_cost_transition_inventory_sha256": (
            outer_rollout.fixed_control_high_cost_transition_inventory_sha256
        ),
        "decision_target_inventory_sha256": outer_rollout.decision_target_inventory_sha256,
        "fixed_control_decision_target_inventory_sha256": (
            outer_rollout.fixed_control_decision_target_inventory_sha256
        ),
        "primary_terminal_liquidation_adjusted_return": (
            computation.primary_terminal_liquidation_adjusted_return
        ),
        "low_cost_terminal_liquidation_adjusted_return": (
            computation.low_cost_terminal_liquidation_adjusted_return
        ),
        "high_cost_terminal_liquidation_adjusted_return": (
            computation.high_cost_terminal_liquidation_adjusted_return
        ),
        "fixed_control_terminal_liquidation_adjusted_return": (
            computation.fixed_control_terminal_liquidation_adjusted_return
        ),
        "fixed_control_low_cost_terminal_liquidation_adjusted_return": (
            computation.fixed_control_low_cost_terminal_liquidation_adjusted_return
        ),
        "fixed_control_high_cost_terminal_liquidation_adjusted_return": (
            computation.fixed_control_high_cost_terminal_liquidation_adjusted_return
        ),
        "ppo_cost_ladder_monotone": computation.ppo_cost_ladder_monotone,
        "fixed_control_cost_ladder_monotone": (
            computation.fixed_control_cost_ladder_monotone
        ),
        "maximum_drawdown": computation.maximum_drawdown,
        "policy_validation_eligible": outer_access.policy_validation_eligible,
        "source_data_qualified": bool(
            policy_schedule.source_data_qualified
            and outer_access.source_data_qualified
            and outer_rollout.source_data_qualified
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SCHEMA,
    }
    provisional = MassiveAdaptiveRLOuterFoldSealAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_seal_replayed=True,
        development_outer_fold_sealed=bool(body["source_data_qualified"]),
        validation_release_authorized=bool(
            body["source_data_qualified"] and fold_index in (0, 1)
        ),
        _runtime_schedule=policy_schedule,
        _runtime_access=outer_access,
        _runtime_rollout=outer_rollout,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse(
    *, root: str | Path, loaded: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLOuterFoldSealAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
            "outer-fold seal payload is not canonical JSON"
        )
    body = dict(value)
    body["decision_session_dates"] = tuple(
        str(item) for item in cast(Sequence[object], body["decision_session_dates"])
    )
    result = MassiveAdaptiveRLOuterFoldSealAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_rl_outer_fold_seal_authority_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    policy_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2,
    outer_rollout: MassiveAdaptiveRLOuterRolloutAuthorityV2,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLOuterFoldSealAuthorityV1:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(policy_schedule) is not MassiveAdaptiveRLWalkForwardPolicyScheduleV1
        or type(outer_access) is not MassiveAdaptiveOuterAccessCommitmentV2
        or type(outer_rollout) is not MassiveAdaptiveRLOuterRolloutAuthorityV2
    ):
        raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
            "outer-fold seal requires exact Manifest-V5 authorities"
        )
    manifest_registration.validate()
    if (
        not manifest_registration.development_protocol_registered
        or manifest_registration.experiment_id != manifest.experiment_id
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or policy_schedule.manifest_v5_registration_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
        or outer_access.manifest_v5_registration_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
            "outer-fold seal registration lineage differs"
        )
    expected = build_massive_adaptive_rl_outer_fold_seal_authority_v1(
        manifest=manifest,
        policy_schedule=policy_schedule,
        outer_access=outer_access,
        outer_rollout=outer_rollout,
    )
    relative = outer_fold_seal_authority_relative_path_v1(
        manifest=manifest, fold_index=expected.fold_index
    )
    with massive_adaptive_rl_experiment_materialization_lock_v1(
        artifact_root=root, experiment_id=manifest.experiment_id
    ):
        payload = Path(root) / relative
        transaction_paths = (
            payload,
            payload.with_name(payload.name + ".receipt.json"),
            payload.with_name(payload.name + ".commit.json"),
        )
        present = tuple(
            path.exists() or path.is_symlink() for path in transaction_paths
        )
        if any(present) and not all(present):
            raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
                "outer-fold seal transaction is incomplete"
            )
        if not all(present):
            if not allow_materialize:
                raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
                    "outer-fold seal is absent"
                )
            committed_at_ms = (
                max(time.time_ns() // 1_000_000, expected.outer_rollout_committed_at_ms)
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
                    dataset_id=MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_DATASET,
                    source_object_key=relative,
                    requested_at_ms=committed_at_ms,
                    downloaded_at_ms=committed_at_ms,
                    schema_sha256=MASSIVE_ADAPTIVE_RL_OUTER_FOLD_SEAL_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
                    entitlement_receipt_sha256=expected.semantic_receipt_sha256,
                    committed_at_ms=committed_at_ms,
                    request_id=(
                        f"ADAPTIVE-RL-OUTER-FOLD-SEAL-{manifest.experiment_id}-"
                        f"FOLD{expected.fold_index}"
                    ),
                )
        parsed = _parse(
            root=root,
            loaded=load_massive_source_bundle(
                root=root,
                relative_payload_path=relative,
                verified_at_ms=time.time_ns() // 1_000_000,
            ),
        )
        if parsed.semantic_unsigned() != expected.semantic_unsigned():
            raise MassiveAdaptiveRLOuterFoldSealAuthorityV1Error(
                "outer-fold seal does not replay"
            )
        result = replace(
            parsed,
            runtime_seal_replayed=True,
            development_outer_fold_sealed=parsed.source_data_qualified,
            validation_release_authorized=bool(
                parsed.source_data_qualified and parsed.fold_index in (0, 1)
            ),
            _runtime_schedule=policy_schedule,
            _runtime_access=outer_access,
            _runtime_rollout=outer_rollout,
        )
        result.validate()
        return result


__all__ = [
    "MassiveAdaptiveRLOuterFoldSealAuthorityV1",
    "MassiveAdaptiveRLOuterFoldSealAuthorityV1Error",
    "build_massive_adaptive_rl_outer_fold_seal_authority_v1",
    "outer_fold_seal_authority_relative_path_v1",
    "run_or_resume_massive_adaptive_rl_outer_fold_seal_authority_v1",
]
