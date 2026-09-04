"""Create-only frozen adaptive RL policy for outer inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
from pathlib import Path
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
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256,
    MassiveAdaptiveRLCheckpointV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    legacy_unscoped_manifest_v5_rejecting_writer_guard_v1,
)


MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-frozen-rl-policy-v1"
)
MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_DATASET = "massive-adaptive-frozen-rl-policy-v1"
MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SCHEMA,
        "payload": "weights-only-torch-state-and-metadata",
        "policy_selection_authority": "exact-v1-only",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source": "committed-inner-validation-policy-selection",
        "policy_selection_authority": "exact-v1-only",
        "state": "actor-and-critic-model-state",
        "updates_after_freeze": False,
        "profitability_reporting": False,
        "lockbox": False,
    }
)


class MassiveAdaptiveFrozenRLPolicyV1Error(ValueError):
    """The selected policy state or selection lineage differs."""


def _tensor_receipt(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_state_receipt(value: Mapping[str, torch.Tensor]) -> str:
    return semantic_sha256(
        tuple((name, _tensor_receipt(tensor)) for name, tensor in sorted(value.items()))
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveFrozenRLPolicyV1:
    fold_index: int
    selected_rl_checkpoint_receipt_sha256: str
    selected_rl_checkpoint_model_state_receipt_sha256: str
    selected_update_index: int
    training_forecast_authority_receipt_sha256: str
    policy_selection_authority_receipt_sha256: str
    policy_selection_receipt_sha256: str
    observation_specification_sha256: str
    action_specification_sha256: str
    reward_specification_sha256: str
    frozen_model_state_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_model_state: dict[str, torch.Tensor] | None
    runtime_policy_replayed: bool
    development_outer_policy_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "selected_rl_checkpoint_receipt_sha256": (
                self.selected_rl_checkpoint_receipt_sha256
            ),
            "selected_rl_checkpoint_model_state_receipt_sha256": (
                self.selected_rl_checkpoint_model_state_receipt_sha256
            ),
            "selected_update_index": self.selected_update_index,
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority_receipt_sha256
            ),
            "policy_selection_authority_receipt_sha256": (
                self.policy_selection_authority_receipt_sha256
            ),
            "policy_selection_receipt_sha256": self.policy_selection_receipt_sha256,
            "observation_specification_sha256": self.observation_specification_sha256,
            "action_specification_sha256": self.action_specification_sha256,
            "reward_specification_sha256": self.reward_specification_sha256,
            "frozen_model_state_receipt_sha256": self.frozen_model_state_receipt_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        runtime = self.runtime_model_state is not None
        expected_authorized = runtime and self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SCHEMA
            or self.fold_index < 0
            or self.selected_update_index < 0
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_policy_replayed != runtime
            or self.development_outer_policy_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.action_specification_sha256
            != MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256
            or self.reward_specification_sha256
            != MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveFrozenRLPolicyV1Error(
                "frozen adaptive RL policy differs"
            )
        if (
            runtime
            and self.runtime_model_state is not None
            and (
                not self.runtime_model_state
                or self.frozen_model_state_receipt_sha256
                != _model_state_receipt(self.runtime_model_state)
            )
        ):
            raise MassiveAdaptiveFrozenRLPolicyV1Error(
                "frozen adaptive RL model state differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.policy_selection_authority_receipt_sha256
        ):
            raise MassiveAdaptiveFrozenRLPolicyV1Error(
                "frozen adaptive RL policy source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _metadata(
    *,
    checkpoint: MassiveAdaptiveRLCheckpointV1,
    selection_authority: MassiveAdaptiveRLPolicySelectionAuthorityV1,
) -> dict[str, object]:
    if type(selection_authority) is not MassiveAdaptiveRLPolicySelectionAuthorityV1:
        raise MassiveAdaptiveFrozenRLPolicyV1Error(
            "frozen policy V1 requires exact policy-selection authority V1"
        )
    selection_authority.validate()
    checkpoint.validate()
    selection = selection_authority.runtime_selection
    if (
        selection is None
        or not selection_authority.runtime_selection_replayed
        or selection.selected_checkpoint_receipt_sha256
        != checkpoint.semantic_receipt_sha256
        or selection.selected_model_state_receipt_sha256
        != checkpoint.model_state_receipt_sha256
        or selection.selected_update_index != checkpoint.update_index
        or selection.training_forecast_authority_receipt_sha256
        != checkpoint.training_forecast_authority_receipt_sha256
    ):
        raise MassiveAdaptiveFrozenRLPolicyV1Error(
            "selected adaptive RL checkpoint differs from policy selection"
        )
    return {
        "schema": MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SCHEMA,
        "fold_index": selection.fold_index,
        "selected_rl_checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "selected_rl_checkpoint_model_state_receipt_sha256": (
            checkpoint.model_state_receipt_sha256
        ),
        "selected_update_index": checkpoint.update_index,
        "training_forecast_authority_receipt_sha256": (
            selection.training_forecast_authority_receipt_sha256
        ),
        "policy_selection_authority_receipt_sha256": (
            selection_authority.semantic_receipt_sha256
        ),
        "policy_selection_receipt_sha256": selection.semantic_receipt_sha256,
        "observation_specification_sha256": checkpoint.observation_specification_sha256,
        "action_specification_sha256": checkpoint.action_specification_sha256,
        "reward_specification_sha256": checkpoint.reward_specification_sha256,
        "frozen_model_state_receipt_sha256": _model_state_receipt(
            checkpoint.model_state
        ),
        "source_data_qualified": bool(
            selection_authority.development_policy_selection_authorized
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SOURCE_SHA256,
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
        raise MassiveAdaptiveFrozenRLPolicyV1Error(
            "frozen adaptive RL policy payload is malformed"
        )
    state = cast(dict[str, torch.Tensor], dict(payload["model_state"]))
    if not state or any(
        not isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise MassiveAdaptiveFrozenRLPolicyV1Error(
            "frozen adaptive RL model state is malformed"
        )
    return dict(payload["metadata"]), state


def parse_massive_adaptive_frozen_rl_policy_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveFrozenRLPolicyV1:
    metadata, _state = _load_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveFrozenRLPolicyV1(
        **metadata,  # type: ignore[arg-type]
        loaded_source=loaded_source,
        runtime_model_state=None,
        runtime_policy_replayed=False,
        development_outer_policy_authorized=False,
    )
    result.validate()
    return result


def authorize_massive_adaptive_frozen_rl_policy_v1(
    *,
    root: str | Path,
    policy: MassiveAdaptiveFrozenRLPolicyV1,
    checkpoint: MassiveAdaptiveRLCheckpointV1,
    selection_authority: MassiveAdaptiveRLPolicySelectionAuthorityV1,
) -> MassiveAdaptiveFrozenRLPolicyV1:
    parsed = parse_massive_adaptive_frozen_rl_policy_v1(
        root=root, loaded_source=policy.loaded_source
    )
    _metadata_payload, state = _load_payload(
        root=root, loaded_source=policy.loaded_source
    )
    expected = _metadata(
        checkpoint=checkpoint,
        selection_authority=selection_authority,
    )
    if (
        parsed.semantic_unsigned() != expected
        or state.keys() != checkpoint.model_state.keys()
    ):
        raise MassiveAdaptiveFrozenRLPolicyV1Error(
            "frozen adaptive RL policy does not replay from its selection"
        )
    if any(
        not torch.equal(state[name], checkpoint.model_state[name]) for name in state
    ):
        raise MassiveAdaptiveFrozenRLPolicyV1Error(
            "frozen adaptive RL policy model tensors differ"
        )
    result = replace(
        parsed,
        runtime_model_state={name: value.clone() for name, value in state.items()},
        runtime_policy_replayed=True,
        development_outer_policy_authorized=parsed.source_data_qualified,
    )
    result.validate()
    return result


@legacy_unscoped_manifest_v5_rejecting_writer_guard_v1
def materialize_massive_adaptive_frozen_rl_policy_v1(
    *,
    root: str | Path,
    artifact_id: str,
    checkpoint: MassiveAdaptiveRLCheckpointV1,
    selection_authority: MassiveAdaptiveRLPolicySelectionAuthorityV1,
    committed_at_ms: int,
) -> MassiveAdaptiveFrozenRLPolicyV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveFrozenRLPolicyV1Error(
            "frozen adaptive RL policy artifact ID is not path safe"
        )
    metadata = _metadata(
        checkpoint=checkpoint,
        selection_authority=selection_authority,
    )
    receipt = semantic_sha256(metadata)
    stream = BytesIO()
    torch.save(
        {
            "metadata": {**metadata, "semantic_receipt_sha256": receipt},
            "model_state": checkpoint.model_state,
        },
        stream,
    )
    stream.seek(0)
    relative = f"massive-adaptive/frozen-rl-policy-v1/{artifact_id}.pt"
    publish_massive_source_object(
        stream=stream,
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_FROZEN_RL_POLICY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=selection_authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-FROZEN-RL-POLICY-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_frozen_rl_policy_v1(
        root=root,
        policy=parse_massive_adaptive_frozen_rl_policy_v1(
            root=root, loaded_source=loaded
        ),
        checkpoint=checkpoint,
        selection_authority=selection_authority,
    )


__all__ = [
    "MassiveAdaptiveFrozenRLPolicyV1",
    "MassiveAdaptiveFrozenRLPolicyV1Error",
    "authorize_massive_adaptive_frozen_rl_policy_v1",
    "materialize_massive_adaptive_frozen_rl_policy_v1",
    "parse_massive_adaptive_frozen_rl_policy_v1",
]
