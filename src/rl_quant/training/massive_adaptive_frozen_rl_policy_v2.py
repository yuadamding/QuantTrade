"""Create-only V5 frozen PPO policy derived from Selection V4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
import hashlib
from io import BytesIO
from pathlib import Path
import time
from typing import cast

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptiveRLCheckpointV1
from rl_quant.training.massive_adaptive_rl_policy_selection_v4 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV4,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    MassiveAdaptiveRLCheckpointAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SCHEMA,
    MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)


MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_DATASET = "massive-adaptive-frozen-rl-policy-v2"
MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SCHEMA,
        "payload": "weights-only-inference-state-and-v5-lineage",
        "selection": "exact-policy-selection-v4",
        "optimizer_payload": False,
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_RL_IDENTITY_NORMALIZATION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "kind": "feature-defined-scaling-with-no-fitted-normalizer-v1",
        "mutable_outer_preprocessing": False,
    }
)


class MassiveAdaptiveFrozenRLPolicyV2Error(ValueError):
    """The selected checkpoint, frozen state, or V5 lineage differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: str | None) -> str:
    if value is None:
        raise MassiveAdaptiveFrozenRLPolicyV2Error(f"{name} is absent")
    return _digest(name, value)


def _required_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveFrozenRLPolicyV2Error(f"{name} is absent or invalid")
    return value


def _wall_clock_after(value: int) -> int:
    now = time.time_ns() // 1_000_000
    if isinstance(now, bool) or not isinstance(now, int) or now < 0:
        raise MassiveAdaptiveFrozenRLPolicyV2Error("frozen-policy clock differs")
    return max(now, value + 1)


def _tensor_receipt(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_receipt(value: Mapping[str, torch.Tensor]) -> str:
    return semantic_sha256(
        tuple((name, _tensor_receipt(tensor)) for name, tensor in sorted(value.items()))
    )


def _partition_state(
    value: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    actor = {
        name: tensor
        for name, tensor in value.items()
        if name == "actor_log_std" or name.startswith(("actor.", "actor_mean."))
    }
    critic = {
        name: tensor
        for name, tensor in value.items()
        if name.startswith(("critic.", "value_head."))
    }
    if not actor or not critic or set(actor) | set(critic) != set(value):
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            "checkpoint model state cannot be partitioned into actor and critic"
        )
    return actor, critic


def frozen_rl_policy_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveFrozenRLPolicyV2Error("frozen-policy fold differs")
    return f"adaptive-rl/{manifest.experiment_id}/frozen-policy-v2/fold-{fold_index}.pt"


def _transaction_state(*, root: str | Path, relative: str) -> tuple[bool, bool]:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    return all(present), any(present) and not all(present)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveFrozenRLPolicyV2:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    validation_release_authority_receipt_sha256: str
    fold_validation_authority_receipt_sha256: str
    policy_selection_authority_receipt_sha256: str
    policy_selection_source_receipt_sha256: str
    policy_selection_commit_receipt_sha256: str
    policy_selection_committed_at_ms: int
    fold_index: int
    selected_checkpoint_authority_receipt_sha256: str
    selected_checkpoint_authority_source_receipt_sha256: str
    selected_checkpoint_authority_commit_receipt_sha256: str
    selected_checkpoint_authority_committed_at_ms: int
    selected_checkpoint_receipt_sha256: str
    selected_model_state_receipt_sha256: str
    selected_update_index: int
    selected_candidate_validation_eligible: bool
    validation_eligibility_failures: tuple[str, ...]
    training_forecast_authority_receipt_sha256: str
    actor_state_receipt_sha256: str
    critic_state_receipt_sha256: str
    frozen_model_state_receipt_sha256: str
    actor_state_keys: tuple[str, ...]
    critic_state_keys: tuple[str, ...]
    actor_optimizer_state_provenance_receipt_sha256: str
    critic_optimizer_state_provenance_receipt_sha256: str
    normalization_specification_sha256: str
    normalization_state_receipt_sha256: str
    ppo_config_receipt_sha256: str
    observation_specification_sha256: str
    action_specification_sha256: str
    reward_specification_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_policy_replayed: bool = False
    development_outer_policy_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SCHEMA
    _runtime_selection: MassiveAdaptiveRLPolicySelectionAuthorityV4 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_model_state: dict[str, torch.Tensor] | None = field(
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
                "runtime_policy_replayed",
                "development_outer_policy_authorized",
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
            and self.runtime_policy_replayed
            and self.development_outer_policy_authorized
            and self.source_data_qualified
        )

    @property
    def positive_profitability_authorization_eligible(self) -> bool:
        return bool(
            self.development_stage_authorized
            and self.selected_candidate_validation_eligible
        )

    @property
    def runtime_model_state(self) -> dict[str, torch.Tensor]:
        self.validate()
        if self._runtime_model_state is None or not self.development_stage_authorized:
            raise MassiveAdaptiveFrozenRLPolicyV2Error(
                "frozen PPO state has not been exactly replayed"
            )
        return {
            name: value.clone() for name, value in self._runtime_model_state.items()
        }

    @property
    def policy_selection_authority(self) -> MassiveAdaptiveRLPolicySelectionAuthorityV4:
        """Return the exact replayed Selection V4 bound by this frozen policy."""

        self.validate()
        if self._runtime_selection is None or not self.development_stage_authorized:
            raise MassiveAdaptiveFrozenRLPolicyV2Error(
                "frozen PPO selection has not been exactly replayed"
            )
        return self._runtime_selection

    def validate(self) -> None:
        runtime_present = (
            self._runtime_selection is not None
            and self._runtime_model_state is not None
        )
        any_runtime = (
            self._runtime_selection is not None or self._runtime_model_state is not None
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or isinstance(self.selected_update_index, bool)
            or not isinstance(self.selected_update_index, int)
            or self.selected_update_index < 0
            or isinstance(self.policy_selection_committed_at_ms, bool)
            or not isinstance(self.policy_selection_committed_at_ms, int)
            or self.policy_selection_committed_at_ms < 0
            or isinstance(self.selected_checkpoint_authority_committed_at_ms, bool)
            or not isinstance(self.selected_checkpoint_authority_committed_at_ms, int)
            or self.selected_checkpoint_authority_committed_at_ms < 0
            or self.selected_checkpoint_authority_committed_at_ms
            >= self.policy_selection_committed_at_ms
            or any(
                not isinstance(value, str) or not value
                for value in self.validation_eligibility_failures
            )
            or self.validation_eligibility_failures
            != tuple(sorted(set(self.validation_eligibility_failures)))
            or not isinstance(self.selected_candidate_validation_eligible, bool)
            or self.selected_candidate_validation_eligible
            != (not self.validation_eligibility_failures)
            or not self.actor_state_keys
            or not self.critic_state_keys
            or any(
                not isinstance(value, str) or not value
                for value in (*self.actor_state_keys, *self.critic_state_keys)
            )
            or len(set(self.actor_state_keys)) != len(self.actor_state_keys)
            or len(set(self.critic_state_keys)) != len(self.critic_state_keys)
            or set(self.actor_state_keys) & set(self.critic_state_keys)
            or self.actor_state_keys != tuple(sorted(self.actor_state_keys))
            or self.critic_state_keys != tuple(sorted(self.critic_state_keys))
            or self.normalization_specification_sha256
            != MASSIVE_ADAPTIVE_RL_IDENTITY_NORMALIZATION_V1_SPEC_SHA256
            or self.normalization_state_receipt_sha256
            != semantic_sha256(
                (
                    self.normalization_specification_sha256,
                    self.observation_specification_sha256,
                )
            )
            or not isinstance(self.source_data_qualified, bool)
            or any_runtime != runtime_present
            or self.runtime_policy_replayed != runtime_present
            or self.development_outer_policy_authorized
            != bool(runtime_present and self.source_data_qualified)
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveFrozenRLPolicyV2Error("frozen PPO policy V2 differs")
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.policy_selection_authority_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= self.policy_selection_committed_at_ms
            ):
                raise MassiveAdaptiveFrozenRLPolicyV2Error(
                    "frozen PPO source transaction differs"
                )
        if runtime_present:
            assert self._runtime_selection is not None
            assert self._runtime_model_state is not None
            self._runtime_selection.validate()
            state = self._runtime_model_state
            actor, critic = _partition_state(state)
            checkpoint_authority = self._runtime_selection.selected_checkpoint_authority
            checkpoint = checkpoint_authority.runtime_checkpoint
            if (
                not self._runtime_selection.development_stage_authorized
                or checkpoint is None
                or self._runtime_selection.semantic_receipt_sha256
                != self.policy_selection_authority_receipt_sha256
                or self._runtime_selection.source_receipt_sha256
                != self.policy_selection_source_receipt_sha256
                or self._runtime_selection.source_transaction_receipt_sha256
                != self.policy_selection_commit_receipt_sha256
                or self._runtime_selection.source_transaction_committed_at_ms
                != self.policy_selection_committed_at_ms
                or self._runtime_selection.manifest_v5_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_selection.scientific_protocol_projection_sha256
                != self.scientific_protocol_projection_sha256
                or self._runtime_selection.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_selection.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_selection.validation_release_authority_receipt_sha256
                != self.validation_release_authority_receipt_sha256
                or self._runtime_selection.fold_validation_authority_receipt_sha256
                != self.fold_validation_authority_receipt_sha256
                or self._runtime_selection.fold_index != self.fold_index
                or self._runtime_selection.selected_checkpoint_authority_receipt_sha256
                != self.selected_checkpoint_authority_receipt_sha256
                or self._runtime_selection.selected_checkpoint_receipt_sha256
                != self.selected_checkpoint_receipt_sha256
                or self._runtime_selection.selected_model_state_receipt_sha256
                != self.selected_model_state_receipt_sha256
                or self._runtime_selection.selected_update_index
                != self.selected_update_index
                or self._runtime_selection.selected_candidate_validation_eligible
                != self.selected_candidate_validation_eligible
                or self._runtime_selection.validation_eligibility_failures
                != self.validation_eligibility_failures
                or self._runtime_selection.training_forecast_authority_receipt_sha256
                != self.training_forecast_authority_receipt_sha256
                or checkpoint_authority.checkpoint_source_receipt_sha256
                != self.selected_checkpoint_authority_source_receipt_sha256
                or checkpoint_authority.loaded_source.commit.receipt_sha256
                != self.selected_checkpoint_authority_commit_receipt_sha256
                or checkpoint_authority.loaded_source.commit.committed_at_ms
                != self.selected_checkpoint_authority_committed_at_ms
                or self.frozen_model_state_receipt_sha256 != _state_receipt(state)
                or self.actor_state_receipt_sha256 != _state_receipt(actor)
                or self.critic_state_receipt_sha256 != _state_receipt(critic)
                or self.actor_state_keys != tuple(sorted(actor))
                or self.critic_state_keys != tuple(sorted(critic))
                or self.actor_optimizer_state_provenance_receipt_sha256
                != checkpoint.actor_optimizer_state_receipt_sha256
                or self.critic_optimizer_state_provenance_receipt_sha256
                != checkpoint.critic_optimizer_state_receipt_sha256
                or self.ppo_config_receipt_sha256
                != checkpoint.ppo_config_receipt_sha256
                or self.observation_specification_sha256
                != checkpoint.observation_specification_sha256
                or self.action_specification_sha256
                != checkpoint.action_specification_sha256
                or self.reward_specification_sha256
                != checkpoint.reward_specification_sha256
                or self.source_data_qualified
                != bool(
                    self._runtime_selection.source_data_qualified
                    and checkpoint_authority.source_data_qualified
                )
            ):
                raise MassiveAdaptiveFrozenRLPolicyV2Error(
                    "frozen PPO runtime state differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _metadata(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    selection: MassiveAdaptiveRLPolicySelectionAuthorityV4,
    checkpoint: MassiveAdaptiveRLCheckpointV1,
) -> dict[str, object]:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(selection) is not MassiveAdaptiveRLPolicySelectionAuthorityV4
    ):
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            "frozen PPO V2 requires exact Manifest V5 and Selection V4"
        )
    manifest.validate()
    selection.validate()
    checkpoint.validate()
    checkpoint_authority = selection.selected_checkpoint_authority
    if type(checkpoint_authority) is not MassiveAdaptiveRLCheckpointAuthorityV1:
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            "frozen PPO V2 requires an exact selected checkpoint authority"
        )
    checkpoint_authority.validate()
    actor, critic = _partition_state(checkpoint.model_state)
    selection_source = _required_digest(
        "selection source", selection.source_receipt_sha256
    )
    selection_commit = _required_digest(
        "selection commit", selection.source_transaction_receipt_sha256
    )
    selection_time = _required_time(
        "selection time", selection.source_transaction_committed_at_ms
    )
    if (
        not selection.development_stage_authorized
        or selection.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
        or checkpoint_authority.runtime_checkpoint is None
        or checkpoint_authority.runtime_checkpoint.semantic_receipt_sha256
        != checkpoint.semantic_receipt_sha256
        or checkpoint.semantic_receipt_sha256
        != selection.selected_checkpoint_receipt_sha256
        or checkpoint.model_state_receipt_sha256
        != selection.selected_model_state_receipt_sha256
    ):
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            "selected PPO checkpoint differs from Selection V4"
        )
    normalization_state = semantic_sha256(
        (
            MASSIVE_ADAPTIVE_RL_IDENTITY_NORMALIZATION_V1_SPEC_SHA256,
            checkpoint.observation_specification_sha256,
        )
    )
    return {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "execution_implementation_registration_receipt_sha256": selection.execution_implementation_registration_receipt_sha256,
        "scientific_execution_fingerprint_sha256": selection.scientific_execution_fingerprint_sha256,
        "validation_release_authority_receipt_sha256": selection.validation_release_authority_receipt_sha256,
        "fold_validation_authority_receipt_sha256": selection.fold_validation_authority_receipt_sha256,
        "policy_selection_authority_receipt_sha256": selection.semantic_receipt_sha256,
        "policy_selection_source_receipt_sha256": selection_source,
        "policy_selection_commit_receipt_sha256": selection_commit,
        "policy_selection_committed_at_ms": selection_time,
        "fold_index": selection.fold_index,
        "selected_checkpoint_authority_receipt_sha256": checkpoint_authority.semantic_receipt_sha256,
        "selected_checkpoint_authority_source_receipt_sha256": checkpoint_authority.checkpoint_source_receipt_sha256,
        "selected_checkpoint_authority_commit_receipt_sha256": checkpoint_authority.loaded_source.commit.receipt_sha256,
        "selected_checkpoint_authority_committed_at_ms": checkpoint_authority.loaded_source.commit.committed_at_ms,
        "selected_checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "selected_model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "selected_update_index": checkpoint.update_index,
        "selected_candidate_validation_eligible": selection.selected_candidate_validation_eligible,
        "validation_eligibility_failures": selection.validation_eligibility_failures,
        "training_forecast_authority_receipt_sha256": selection.training_forecast_authority_receipt_sha256,
        "actor_state_receipt_sha256": _state_receipt(actor),
        "critic_state_receipt_sha256": _state_receipt(critic),
        "frozen_model_state_receipt_sha256": _state_receipt(checkpoint.model_state),
        "actor_state_keys": tuple(sorted(actor)),
        "critic_state_keys": tuple(sorted(critic)),
        "actor_optimizer_state_provenance_receipt_sha256": checkpoint.actor_optimizer_state_receipt_sha256,
        "critic_optimizer_state_provenance_receipt_sha256": checkpoint.critic_optimizer_state_receipt_sha256,
        "normalization_specification_sha256": MASSIVE_ADAPTIVE_RL_IDENTITY_NORMALIZATION_V1_SPEC_SHA256,
        "normalization_state_receipt_sha256": normalization_state,
        "ppo_config_receipt_sha256": checkpoint.ppo_config_receipt_sha256,
        "observation_specification_sha256": checkpoint.observation_specification_sha256,
        "action_specification_sha256": checkpoint.action_specification_sha256,
        "reward_specification_sha256": checkpoint.reward_specification_sha256,
        "source_data_qualified": bool(
            selection.source_data_qualified and checkpoint.source_data_qualified
        ),
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SCHEMA,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("metadata"), Mapping)
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise MassiveAdaptiveFrozenRLPolicyV2Error("frozen PPO payload is malformed")
    metadata = dict(cast(Mapping[str, object], payload["metadata"]))
    for name in (
        "validation_eligibility_failures",
        "actor_state_keys",
        "critic_state_keys",
    ):
        metadata[name] = tuple(cast(Sequence[object], metadata[name]))
    state = dict(cast(Mapping[str, torch.Tensor], payload["model_state"]))
    if not state or any(type(value) is not torch.Tensor for value in state.values()):
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            "frozen PPO model state is malformed"
        )
    return metadata, state


def _parse(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveFrozenRLPolicyV2:
    metadata, _state = _load_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveFrozenRLPolicyV2(
        **metadata,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(metadata),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_frozen_rl_policy_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    fold_index: int,
    verified_at_ms: int | None = None,
) -> MassiveAdaptiveFrozenRLPolicyV2:
    return _parse(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=frozen_rl_policy_relative_path_v2(
                manifest=manifest, fold_index=fold_index
            ),
            verified_at_ms=(
                time.time_ns() // 1_000_000
                if verified_at_ms is None
                else verified_at_ms
            ),
        ),
    )


def run_or_resume_massive_adaptive_frozen_rl_policy_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    selection: MassiveAdaptiveRLPolicySelectionAuthorityV4,
    allow_materialize: bool = True,
) -> MassiveAdaptiveFrozenRLPolicyV2:
    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            "frozen PPO materialization mode differs"
        )
    checkpoint_authority = selection.selected_checkpoint_authority
    checkpoint = checkpoint_authority.runtime_checkpoint
    if checkpoint is None:
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            "selected checkpoint runtime state is absent"
        )
    metadata = _metadata(manifest=manifest, selection=selection, checkpoint=checkpoint)
    relative = frozen_rl_policy_relative_path_v2(
        manifest=manifest, fold_index=selection.fold_index
    )
    complete, partial = _transaction_state(root=root, relative=relative)
    if partial:
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            "frozen PPO transaction is incomplete"
        )
    if not complete:
        if not allow_materialize:
            raise MassiveAdaptiveFrozenRLPolicyV2Error(
                "frozen PPO is absent in read-only mode"
            )
        stream = BytesIO()
        torch.save(
            {
                "metadata": metadata,
                "model_state": {
                    name: value.detach().cpu().clone()
                    for name, value in checkpoint.model_state.items()
                },
            },
            stream,
        )
        committed_at_ms = _wall_clock_after(
            _required_time(
                "selection time", selection.source_transaction_committed_at_ms
            )
        )
        publish_massive_source_object(
            stream=BytesIO(stream.getvalue()),
            root=root,
            relative_payload_path=relative,
            dataset_id=MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_DATASET,
            source_object_key=relative,
            requested_at_ms=committed_at_ms,
            downloaded_at_ms=committed_at_ms,
            schema_sha256=MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SOURCE_SCHEMA_SHA256,
            entitlement_receipt_sha256=selection.semantic_receipt_sha256,
            committed_at_ms=committed_at_ms,
            request_id=f"ADAPTIVE-RL-V5-FROZEN-PPO-{selection.fold_index}",
        )
    persisted = load_massive_adaptive_frozen_rl_policy_v2(
        root=root, manifest=manifest, fold_index=selection.fold_index
    )
    persisted_metadata, state = _load_payload(
        root=root,
        loaded_source=cast(LoadedMassiveSourceObject, persisted._loaded_source),
    )
    if persisted_metadata != metadata or set(state) != set(checkpoint.model_state):
        raise MassiveAdaptiveFrozenRLPolicyV2Error(
            "frozen PPO does not replay from Selection V4"
        )
    if any(
        not torch.equal(state[name], checkpoint.model_state[name]) for name in state
    ):
        raise MassiveAdaptiveFrozenRLPolicyV2Error("frozen PPO tensors differ")
    result = replace(
        persisted,
        runtime_policy_replayed=True,
        development_outer_policy_authorized=persisted.source_data_qualified,
        _runtime_selection=selection,
        _runtime_model_state=state,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_DATASET",
    "MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V2_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_IDENTITY_NORMALIZATION_V1_SPEC_SHA256",
    "MassiveAdaptiveFrozenRLPolicyV2",
    "MassiveAdaptiveFrozenRLPolicyV2Error",
    "frozen_rl_policy_relative_path_v2",
    "load_massive_adaptive_frozen_rl_policy_v2",
    "run_or_resume_massive_adaptive_frozen_rl_policy_v2",
]
