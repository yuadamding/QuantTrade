"""Create-only exact-resume authority for multi-block adaptive PPO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from rl_quant.training.massive_adaptive_prequential_ppo_runner_v1 import (
    MassiveAdaptivePrequentialPPOCheckpointV1,
    MassiveAdaptivePrequentialPPORunnerV1,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    _checkpoint_payload,
    _parse_checkpoint,
)


MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-prequential-ppo-checkpoint-authority-v1"
)
MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-prequential-ppo-checkpoint-authority-v1"
)
MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SOURCE_SHA256 = (
    file_sha256(Path(__file__))
)
MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": (
                MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SCHEMA
            ),
            "payload": "safe-torch-runner-and-ppo-update-boundary",
            "generic_reload": "runtime-state-stripped",
            "promotion": "restore-runner-rebuild-checkpoint",
        }
    )
)


class MassiveAdaptivePrequentialPPOCheckpointAuthorityV1Error(ValueError):
    """The durable multi-block runner checkpoint failed exact replay."""


def _payload(
    checkpoint: MassiveAdaptivePrequentialPPOCheckpointV1,
) -> dict[str, object]:
    checkpoint.validate()
    metadata = checkpoint.semantic_unsigned()
    metadata["semantic_receipt_sha256"] = checkpoint.semantic_receipt_sha256
    return {
        "metadata": metadata,
        "ppo_checkpoint": _checkpoint_payload(
            checkpoint.ppo_checkpoint,
            source_data_qualified=checkpoint.source_data_qualified,
        ),
    }


def _load_payload(raw: bytes) -> Mapping[str, object]:
    try:
        value = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise MassiveAdaptivePrequentialPPOCheckpointAuthorityV1Error(
            "prequential PPO checkpoint payload is not a safe Torch artifact"
        ) from error
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("metadata"), Mapping)
        or not isinstance(value.get("ppo_checkpoint"), Mapping)
    ):
        raise MassiveAdaptivePrequentialPPOCheckpointAuthorityV1Error(
            "prequential PPO checkpoint payload is malformed"
        )
    return cast(Mapping[str, object], value)


def _parse_runner_checkpoint(
    payload: Mapping[str, object],
) -> MassiveAdaptivePrequentialPPOCheckpointV1:
    metadata = dict(cast(Mapping[str, object], payload["metadata"]))
    metadata.pop("ppo_checkpoint_receipt_sha256", None)
    for name in ("completed_block_receipts", "transition_receipts"):
        metadata[name] = tuple(cast(list[str] | tuple[str, ...], metadata[name]))
    ppo_checkpoint = _parse_checkpoint(
        cast(Mapping[str, object], payload["ppo_checkpoint"])
    )
    source_data_qualified = bool(metadata["source_data_qualified"])
    result = MassiveAdaptivePrequentialPPOCheckpointV1(
        **metadata,  # type: ignore[arg-type]
        ppo_checkpoint=ppo_checkpoint,
        exact_resume_authorized=source_data_qualified,
        development_rl_training_authorized=source_data_qualified,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePrequentialPPOCheckpointAuthorityV1:
    checkpoint_receipt_sha256: str
    ppo_checkpoint_receipt_sha256: str
    training_forecast_authority_receipt_sha256: str
    rl_chronology_authority_receipt_sha256: str
    block_runtime_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_checkpoint: MassiveAdaptivePrequentialPPOCheckpointV1 | None
    runtime_checkpoint_replayed: bool
    exact_resume_authorized: bool
    development_rl_training_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "ppo_checkpoint_receipt_sha256": self.ppo_checkpoint_receipt_sha256,
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority_receipt_sha256
            ),
            "rl_chronology_authority_receipt_sha256": (
                self.rl_chronology_authority_receipt_sha256
            ),
            "block_runtime_inventory_sha256": self.block_runtime_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime = self.runtime_checkpoint is not None
        expected = runtime and self.source_data_qualified
        if self.runtime_checkpoint is not None:
            self.runtime_checkpoint.validate()
        if (
            self.schema
            != MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SCHEMA
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.checkpoint_receipt_sha256
            or self.runtime_checkpoint_replayed != runtime
            or self.exact_resume_authorized != expected
            or self.development_rl_training_authorized != expected
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptivePrequentialPPOCheckpointAuthorityV1Error(
                "prequential PPO checkpoint authority differs"
            )
        if runtime and self.runtime_checkpoint is not None and (
            self.runtime_checkpoint.semantic_receipt_sha256
            != self.checkpoint_receipt_sha256
            or self.runtime_checkpoint.ppo_checkpoint.semantic_receipt_sha256
            != self.ppo_checkpoint_receipt_sha256
            or self.runtime_checkpoint.training_forecast_authority_receipt_sha256
            != self.training_forecast_authority_receipt_sha256
            or self.runtime_checkpoint.rl_chronology_authority_receipt_sha256
            != self.rl_chronology_authority_receipt_sha256
            or self.runtime_checkpoint.block_runtime_inventory_sha256
            != self.block_runtime_inventory_sha256
        ):
            raise MassiveAdaptivePrequentialPPOCheckpointAuthorityV1Error(
                "runtime prequential PPO checkpoint differs from authority"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def parse_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptivePrequentialPPOCheckpointAuthorityV1:
    checkpoint = _parse_runner_checkpoint(
        _load_payload(
            read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SCHEMA,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "ppo_checkpoint_receipt_sha256": (
            checkpoint.ppo_checkpoint.semantic_receipt_sha256
        ),
        "training_forecast_authority_receipt_sha256": (
            checkpoint.training_forecast_authority_receipt_sha256
        ),
        "rl_chronology_authority_receipt_sha256": (
            checkpoint.rl_chronology_authority_receipt_sha256
        ),
        "block_runtime_inventory_sha256": checkpoint.block_runtime_inventory_sha256,
        "source_data_qualified": checkpoint.source_data_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptivePrequentialPPOCheckpointAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_checkpoint=None,
        runtime_checkpoint_replayed=False,
        exact_resume_authorized=False,
        development_rl_training_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def authorize_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptivePrequentialPPOCheckpointAuthorityV1,
    runner: MassiveAdaptivePrequentialPPORunnerV1,
) -> MassiveAdaptivePrequentialPPOCheckpointAuthorityV1:
    parsed = parse_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    checkpoint = _parse_runner_checkpoint(
        _load_payload(
            read_loaded_massive_source_bytes(
                root=root, loaded_source=authority.loaded_source
            )
        )
    )
    runner.restore(checkpoint)
    replayed = runner.checkpoint()
    if (
        replayed.semantic_receipt_sha256
        != checkpoint.semantic_receipt_sha256
        or replayed.ppo_checkpoint.semantic_receipt_sha256
        != checkpoint.ppo_checkpoint.semantic_receipt_sha256
        or replayed.ppo_checkpoint.model_state_receipt_sha256
        != checkpoint.ppo_checkpoint.model_state_receipt_sha256
        or replayed.ppo_checkpoint.actor_optimizer_state_receipt_sha256
        != checkpoint.ppo_checkpoint.actor_optimizer_state_receipt_sha256
        or replayed.ppo_checkpoint.critic_optimizer_state_receipt_sha256
        != checkpoint.ppo_checkpoint.critic_optimizer_state_receipt_sha256
        or replayed.ppo_checkpoint.rng_state_receipt_sha256
        != checkpoint.ppo_checkpoint.rng_state_receipt_sha256
        or replayed.ppo_checkpoint.environment_state_receipt_sha256
        != checkpoint.ppo_checkpoint.environment_state_receipt_sha256
    ):
        raise MassiveAdaptivePrequentialPPOCheckpointAuthorityV1Error(
            "prequential PPO checkpoint did not reproduce after restore"
        )
    result = replace(
        parsed,
        runtime_checkpoint=replayed,
        runtime_checkpoint_replayed=True,
        exact_resume_authorized=parsed.source_data_qualified,
        development_rl_training_authorized=parsed.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    runner: MassiveAdaptivePrequentialPPORunnerV1,
    committed_at_ms: int,
) -> MassiveAdaptivePrequentialPPOCheckpointAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptivePrequentialPPOCheckpointAuthorityV1Error(
            "prequential PPO checkpoint artifact ID is not path safe"
        )
    checkpoint = runner.checkpoint()
    stream = BytesIO()
    torch.save(_payload(checkpoint), stream)
    relative = f"massive-adaptive/prequential-ppo-checkpoint-v1/{artifact_id}.pt"
    publish_massive_source_object(
        stream=BytesIO(stream.getvalue()),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=checkpoint.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-PREQUENTIAL-PPO-CHECKPOINT-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
        root=root,
        authority=parse_massive_adaptive_prequential_ppo_checkpoint_authority_v1(
            root=root, loaded_source=loaded
        ),
        runner=runner,
    )


__all__ = [
    "MassiveAdaptivePrequentialPPOCheckpointAuthorityV1",
    "MassiveAdaptivePrequentialPPOCheckpointAuthorityV1Error",
    "authorize_massive_adaptive_prequential_ppo_checkpoint_authority_v1",
    "materialize_massive_adaptive_prequential_ppo_checkpoint_authority_v1",
    "parse_massive_adaptive_prequential_ppo_checkpoint_authority_v1",
]
