"""Create-only exact-resume checkpoints for adaptive supervised training.

The payload uses Torch's tensor archive only as storage and is reopened with
``weights_only=True``.  Every model, optimizer, scheduler, RNG, cursor, and
loss-trace component has an independent semantic receipt.  A generic reload
is deliberately nonauthorizing; promotion requires the live root-derived
training authority and exact configuration identities.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping, cast

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
from rl_quant.training.massive_adaptive_training_authority_v1 import (
    MassiveAdaptiveTrainingAuthorityV1,
)


MASSIVE_ADAPTIVE_CHECKPOINT_V1_SCHEMA = "rl-quant.massive-adaptive-checkpoint-v1"
MASSIVE_ADAPTIVE_CHECKPOINT_V1_DATASET = "massive-adaptive-checkpoint-v1"
MASSIVE_ADAPTIVE_CHECKPOINT_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_CHECKPOINT_V1_SCHEMA,
        "encoding": "torch-tensor-archive-loaded-weights-only",
        "publication": "create-only-source-transaction",
        "state": "model-optimizer-scheduler-rng-cursor-and-loss-trace",
    }
)
MASSIVE_ADAPTIVE_CHECKPOINT_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_CHECKPOINT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "resume": "exact-next-update",
        "generic_reload": "nonauthorizing",
        "promotion": "root-derived-training-authority-and-state-receipts",
        "decision_roots": "full-chronology-and-target-origin-inventories",
        "pickle_execution": False,
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveCheckpointV1Error(ValueError):
    """Adaptive checkpoint state or root identity differs."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveCheckpointV1Error(
            "adaptive checkpoint artifact ID is not path safe"
        )
    return value


def _tensor_receipt(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_identity(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_dtype": str(value.dtype),
            "tensor_shape": tuple(value.shape),
            "tensor_sha256": _tensor_receipt(value),
        }
    if isinstance(value, Mapping):
        return tuple(
            (
                type(key).__name__,
                str(key),
                _state_identity(item),
            )
            for key, item in sorted(
                value.items(), key=lambda pair: (type(pair[0]).__name__, str(pair[0]))
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_state_identity(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise MassiveAdaptiveCheckpointV1Error(
        f"unsupported adaptive checkpoint state value {type(value).__name__}"
    )


def adaptive_training_state_receipt_v1(value: object) -> str:
    """Hash nested tensor state without relying on archive byte stability."""

    return semantic_sha256(_state_identity(value))


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveCheckpointStateV1:
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    gradient_scaler_state: dict[str, Any]
    torch_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]
    data_order_rng_state: torch.Tensor
    python_rng_state: tuple[Any, ...]
    numpy_rng_state: tuple[Any, ...]
    update_index: int
    epoch_index: int
    window_cursor: int
    window_order: tuple[int, ...]
    loss_trace: tuple[float, ...]

    def validate(self) -> None:
        if (
            not self.model_state
            or not isinstance(self.optimizer_state, dict)
            or not isinstance(self.scheduler_state, dict)
            or not isinstance(self.gradient_scaler_state, dict)
            or self.torch_rng_state.dtype != torch.uint8
            or self.torch_rng_state.ndim != 1
            or any(
                value.dtype != torch.uint8 or value.ndim != 1
                for value in self.cuda_rng_states
            )
            or self.data_order_rng_state.dtype != torch.uint8
            or self.data_order_rng_state.ndim != 1
            or isinstance(self.update_index, bool)
            or self.update_index < 0
            or isinstance(self.epoch_index, bool)
            or self.epoch_index < 0
            or isinstance(self.window_cursor, bool)
            or self.window_cursor < 0
            or not self.window_order
            or set(self.window_order) != set(range(len(self.window_order)))
            or len(self.loss_trace) != self.update_index
            or any(not torch.isfinite(torch.tensor(value)) for value in self.loss_trace)
        ):
            raise MassiveAdaptiveCheckpointV1Error(
                "adaptive checkpoint runtime state is malformed"
            )
        adaptive_training_state_receipt_v1(self.model_state)
        adaptive_training_state_receipt_v1(self.optimizer_state)
        adaptive_training_state_receipt_v1(self.scheduler_state)
        adaptive_training_state_receipt_v1(self.gradient_scaler_state)
        adaptive_training_state_receipt_v1(self.python_rng_state)
        adaptive_training_state_receipt_v1(self.numpy_rng_state)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveCheckpointV1:
    training_authority_receipt_sha256: str
    decision_tensor_receipt_sha256: str
    full_decision_root_inventory_sha256: str
    origin_decision_root_inventory_sha256: str
    target_archive_receipt_sha256: str
    target_root_inventory_sha256: str
    target_experiment_inventory_sha256: str
    split_plan_receipt_sha256: str
    window_plan_receipt_sha256: str
    model_spec_receipt_sha256: str
    training_config_receipt_sha256: str
    update_index: int
    epoch_index: int
    window_cursor: int
    model_state_receipt_sha256: str
    optimizer_state_receipt_sha256: str
    scheduler_state_receipt_sha256: str
    gradient_scaler_state_receipt_sha256: str
    rng_state_receipt_sha256: str
    loss_trace_receipt_sha256: str
    state_receipt_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_state: MassiveAdaptiveCheckpointStateV1 | None
    runtime_checkpoint_replayed: bool
    committed_development_training_authorized: bool
    development_training_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_CHECKPOINT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "training_authority_receipt_sha256": (
                self.training_authority_receipt_sha256
            ),
            "decision_tensor_receipt_sha256": self.decision_tensor_receipt_sha256,
            "full_decision_root_inventory_sha256": (
                self.full_decision_root_inventory_sha256
            ),
            "origin_decision_root_inventory_sha256": (
                self.origin_decision_root_inventory_sha256
            ),
            "target_archive_receipt_sha256": self.target_archive_receipt_sha256,
            "target_root_inventory_sha256": self.target_root_inventory_sha256,
            "target_experiment_inventory_sha256": (
                self.target_experiment_inventory_sha256
            ),
            "split_plan_receipt_sha256": self.split_plan_receipt_sha256,
            "window_plan_receipt_sha256": self.window_plan_receipt_sha256,
            "model_spec_receipt_sha256": self.model_spec_receipt_sha256,
            "training_config_receipt_sha256": self.training_config_receipt_sha256,
            "update_index": self.update_index,
            "epoch_index": self.epoch_index,
            "window_cursor": self.window_cursor,
            "model_state_receipt_sha256": self.model_state_receipt_sha256,
            "optimizer_state_receipt_sha256": self.optimizer_state_receipt_sha256,
            "scheduler_state_receipt_sha256": self.scheduler_state_receipt_sha256,
            "gradient_scaler_state_receipt_sha256": (
                self.gradient_scaler_state_receipt_sha256
            ),
            "rng_state_receipt_sha256": self.rng_state_receipt_sha256,
            "loss_trace_receipt_sha256": self.loss_trace_receipt_sha256,
            "state_receipt_sha256": self.state_receipt_sha256,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "committed_development_training_authorized": (
                self.committed_development_training_authorized
            ),
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
        }

    def validate(self) -> None:
        state_present = self.runtime_state is not None
        if (
            self.schema != MASSIVE_ADAPTIVE_CHECKPOINT_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_CHECKPOINT_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_CHECKPOINT_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
            or self.runtime_checkpoint_replayed != state_present
            or self.development_training_authorized
            != (
                state_present and self.committed_development_training_authorized
            )
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveCheckpointV1Error(
                "adaptive checkpoint identity or authorization differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_CHECKPOINT_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_CHECKPOINT_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.training_authority_receipt_sha256
        ):
            raise MassiveAdaptiveCheckpointV1Error(
                "adaptive checkpoint source transaction differs"
            )
        if self.runtime_state is not None:
            self.runtime_state.validate()
            expected_rng = semantic_sha256(
                {
                    "torch": _tensor_receipt(self.runtime_state.torch_rng_state),
                    "cuda": tuple(
                        _tensor_receipt(value)
                        for value in self.runtime_state.cuda_rng_states
                    ),
                    "data_order": _tensor_receipt(
                        self.runtime_state.data_order_rng_state
                    ),
                    "python": adaptive_training_state_receipt_v1(
                        self.runtime_state.python_rng_state
                    ),
                    "numpy": adaptive_training_state_receipt_v1(
                        self.runtime_state.numpy_rng_state
                    ),
                }
            )
            if (
                self.update_index != self.runtime_state.update_index
                or self.epoch_index != self.runtime_state.epoch_index
                or self.window_cursor != self.runtime_state.window_cursor
                or self.model_state_receipt_sha256
                != adaptive_training_state_receipt_v1(
                    self.runtime_state.model_state
                )
                or self.optimizer_state_receipt_sha256
                != adaptive_training_state_receipt_v1(
                    self.runtime_state.optimizer_state
                )
                or self.scheduler_state_receipt_sha256
                != adaptive_training_state_receipt_v1(
                    self.runtime_state.scheduler_state
                )
                or self.gradient_scaler_state_receipt_sha256
                != adaptive_training_state_receipt_v1(
                    self.runtime_state.gradient_scaler_state
                )
                or self.rng_state_receipt_sha256 != expected_rng
                or self.loss_trace_receipt_sha256
                != semantic_sha256(self.runtime_state.loss_trace)
                or self.state_receipt_sha256
                != adaptive_training_state_receipt_v1(
                    _state_payload(self.runtime_state)
                )
            ):
                raise MassiveAdaptiveCheckpointV1Error(
                    "adaptive checkpoint state does not match its receipts"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _state_payload(state: MassiveAdaptiveCheckpointStateV1) -> dict[str, object]:
    return {
        "model_state": state.model_state,
        "optimizer_state": state.optimizer_state,
        "scheduler_state": state.scheduler_state,
        "gradient_scaler_state": state.gradient_scaler_state,
        "torch_rng_state": state.torch_rng_state,
        "cuda_rng_states": state.cuda_rng_states,
        "data_order_rng_state": state.data_order_rng_state,
        "python_rng_state": state.python_rng_state,
        "numpy_rng_state": state.numpy_rng_state,
        "update_index": state.update_index,
        "epoch_index": state.epoch_index,
        "window_cursor": state.window_cursor,
        "window_order": state.window_order,
        "loss_trace": state.loss_trace,
    }


def _state_from_payload(payload: Mapping[str, object]) -> MassiveAdaptiveCheckpointStateV1:
    def mapping(name: str) -> Mapping[object, object]:
        value = payload.get(name)
        if not isinstance(value, Mapping):
            raise MassiveAdaptiveCheckpointV1Error(
                f"adaptive checkpoint {name} is not a mapping"
            )
        return value

    def sequence(name: str) -> Sequence[object]:
        value = payload.get(name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise MassiveAdaptiveCheckpointV1Error(
                f"adaptive checkpoint {name} is not a sequence"
            )
        return value

    def integer(name: str) -> int:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise MassiveAdaptiveCheckpointV1Error(
                f"adaptive checkpoint {name} is not an integer"
            )
        return value

    torch_rng_state = payload.get("torch_rng_state")
    data_order_rng_state = payload.get("data_order_rng_state")
    if not isinstance(torch_rng_state, torch.Tensor) or not isinstance(
        data_order_rng_state, torch.Tensor
    ):
        raise MassiveAdaptiveCheckpointV1Error(
            "adaptive checkpoint CPU RNG state is malformed"
        )
    result = MassiveAdaptiveCheckpointStateV1(
        model_state=cast(dict[str, torch.Tensor], dict(mapping("model_state"))),
        optimizer_state=cast(dict[str, Any], dict(mapping("optimizer_state"))),
        scheduler_state=cast(dict[str, Any], dict(mapping("scheduler_state"))),
        gradient_scaler_state=cast(
            dict[str, Any], dict(mapping("gradient_scaler_state"))
        ),
        torch_rng_state=torch_rng_state,
        cuda_rng_states=cast(
            tuple[torch.Tensor, ...], tuple(sequence("cuda_rng_states"))
        ),
        data_order_rng_state=data_order_rng_state,
        python_rng_state=tuple(sequence("python_rng_state")),
        numpy_rng_state=tuple(sequence("numpy_rng_state")),
        update_index=integer("update_index"),
        epoch_index=integer("epoch_index"),
        window_cursor=integer("window_cursor"),
        window_order=tuple(
            cast(int, value) for value in sequence("window_order")
        ),
        loss_trace=tuple(
            cast(float, value) for value in sequence("loss_trace")
        ),
    )
    result.validate()
    return result


def _metadata(
    *,
    state: MassiveAdaptiveCheckpointStateV1,
    training_authority: MassiveAdaptiveTrainingAuthorityV1,
    decision_tensor_receipt_sha256: str,
    split_plan_receipt_sha256: str,
    window_plan_receipt_sha256: str,
    model_spec_receipt_sha256: str,
    training_config_receipt_sha256: str,
) -> dict[str, object]:
    rng_receipt = semantic_sha256(
        {
            "torch": _tensor_receipt(state.torch_rng_state),
            "cuda": tuple(_tensor_receipt(value) for value in state.cuda_rng_states),
            "data_order": _tensor_receipt(state.data_order_rng_state),
            "python": adaptive_training_state_receipt_v1(state.python_rng_state),
            "numpy": adaptive_training_state_receipt_v1(state.numpy_rng_state),
        }
    )
    return {
        "schema": MASSIVE_ADAPTIVE_CHECKPOINT_V1_SCHEMA,
        "training_authority_receipt_sha256": (
            training_authority.semantic_receipt_sha256
        ),
        "decision_tensor_receipt_sha256": decision_tensor_receipt_sha256,
        "full_decision_root_inventory_sha256": (
            training_authority.full_decision_root_inventory_sha256
        ),
        "origin_decision_root_inventory_sha256": (
            training_authority.origin_decision_root_inventory_sha256
        ),
        "target_archive_receipt_sha256": (
            training_authority.target_archive_receipt_sha256
        ),
        "target_root_inventory_sha256": (
            training_authority.target_root_inventory_sha256
        ),
        "target_experiment_inventory_sha256": (
            training_authority.target_experiment_inventory_sha256
        ),
        "split_plan_receipt_sha256": split_plan_receipt_sha256,
        "window_plan_receipt_sha256": window_plan_receipt_sha256,
        "model_spec_receipt_sha256": model_spec_receipt_sha256,
        "training_config_receipt_sha256": training_config_receipt_sha256,
        "update_index": state.update_index,
        "epoch_index": state.epoch_index,
        "window_cursor": state.window_cursor,
        "model_state_receipt_sha256": adaptive_training_state_receipt_v1(
            state.model_state
        ),
        "optimizer_state_receipt_sha256": adaptive_training_state_receipt_v1(
            state.optimizer_state
        ),
        "scheduler_state_receipt_sha256": adaptive_training_state_receipt_v1(
            state.scheduler_state
        ),
        "gradient_scaler_state_receipt_sha256": adaptive_training_state_receipt_v1(
            state.gradient_scaler_state
        ),
        "rng_state_receipt_sha256": rng_receipt,
        "loss_trace_receipt_sha256": semantic_sha256(state.loss_trace),
        "state_receipt_sha256": adaptive_training_state_receipt_v1(
            _state_payload(state)
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_CHECKPOINT_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_CHECKPOINT_V1_SOURCE_SHA256,
        "committed_development_training_authorized": (
            training_authority.development_training_authorized
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }


def publish_massive_adaptive_checkpoint_v1(
    *,
    root: str | Path,
    artifact_id: str,
    state: MassiveAdaptiveCheckpointStateV1,
    training_authority: MassiveAdaptiveTrainingAuthorityV1,
    decision_tensor_receipt_sha256: str,
    split_plan_receipt_sha256: str,
    window_plan_receipt_sha256: str,
    model_spec_receipt_sha256: str,
    training_config_receipt_sha256: str,
    committed_at_ms: int,
) -> MassiveAdaptiveCheckpointV1:
    """Publish an immutable state archive and return a generic reload."""

    identifier = _artifact_id(artifact_id)
    state.validate()
    training_authority.validate()
    metadata = _metadata(
        state=state,
        training_authority=training_authority,
        decision_tensor_receipt_sha256=decision_tensor_receipt_sha256,
        split_plan_receipt_sha256=split_plan_receipt_sha256,
        window_plan_receipt_sha256=window_plan_receipt_sha256,
        model_spec_receipt_sha256=model_spec_receipt_sha256,
        training_config_receipt_sha256=training_config_receipt_sha256,
    )
    receipt = semantic_sha256(metadata)
    payload = {
        "metadata": {**metadata, "semantic_receipt_sha256": receipt},
        "state": _state_payload(state),
    }
    stream = BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    relative = f"massive-adaptive/checkpoint-v1/{identifier}.pt"
    publish_massive_source_object(
        stream=stream,
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_CHECKPOINT_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_CHECKPOINT_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=training_authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-CHECKPOINT-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return parse_massive_adaptive_checkpoint_v1(root=root, loaded_source=loaded)


def parse_massive_adaptive_checkpoint_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveCheckpointV1:
    """Safely load checkpoint state while withholding training authority."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise MassiveAdaptiveCheckpointV1Error(
            "adaptive checkpoint payload is malformed"
        )
    metadata = dict(payload["metadata"])
    state_payload = payload.get("state")
    if not isinstance(state_payload, dict):
        raise MassiveAdaptiveCheckpointV1Error(
            "adaptive checkpoint state payload is malformed"
        )
    state = _state_from_payload(state_payload)
    result = MassiveAdaptiveCheckpointV1(
        **metadata,
        loaded_source=loaded_source,
        runtime_state=state,
        runtime_checkpoint_replayed=True,
        development_training_authorized=bool(
            metadata["committed_development_training_authorized"]
        ),
    )
    result.validate()
    generic = replace(
        result,
        runtime_state=None,
        runtime_checkpoint_replayed=False,
        development_training_authorized=False,
    )
    generic.validate()
    return generic


def authorize_massive_adaptive_checkpoint_v1(
    *,
    root: str | Path,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_authority: MassiveAdaptiveTrainingAuthorityV1,
    decision_tensor_receipt_sha256: str,
    split_plan_receipt_sha256: str,
    window_plan_receipt_sha256: str,
    model_spec_receipt_sha256: str,
    training_config_receipt_sha256: str,
) -> MassiveAdaptiveCheckpointV1:
    """Reopen exact state and bind it to the live promoted root authority."""

    training_authority.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=checkpoint.loaded_source
    )
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise MassiveAdaptiveCheckpointV1Error(
            "adaptive checkpoint payload is malformed"
        )
    state = _state_from_payload(payload["state"])
    parsed = parse_massive_adaptive_checkpoint_v1(
        root=root, loaded_source=checkpoint.loaded_source
    )
    if (
        parsed.semantic_receipt_sha256 != checkpoint.semantic_receipt_sha256
        or parsed.training_authority_receipt_sha256
        != training_authority.semantic_receipt_sha256
        or parsed.decision_tensor_receipt_sha256
        != decision_tensor_receipt_sha256
        or parsed.full_decision_root_inventory_sha256
        != training_authority.full_decision_root_inventory_sha256
        or parsed.origin_decision_root_inventory_sha256
        != training_authority.origin_decision_root_inventory_sha256
        or parsed.target_archive_receipt_sha256
        != training_authority.target_archive_receipt_sha256
        or parsed.target_root_inventory_sha256
        != training_authority.target_root_inventory_sha256
        or parsed.target_experiment_inventory_sha256
        != training_authority.target_experiment_inventory_sha256
        or parsed.split_plan_receipt_sha256 != split_plan_receipt_sha256
        or parsed.window_plan_receipt_sha256 != window_plan_receipt_sha256
        or parsed.model_spec_receipt_sha256 != model_spec_receipt_sha256
        or parsed.training_config_receipt_sha256
        != training_config_receipt_sha256
        or parsed.committed_development_training_authorized
        != training_authority.development_training_authorized
    ):
        raise MassiveAdaptiveCheckpointV1Error(
            "adaptive checkpoint roots or training identities differ"
        )
    result = replace(
        parsed,
        runtime_state=state,
        runtime_checkpoint_replayed=True,
        development_training_authorized=(
            training_authority.development_training_authorized
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_CHECKPOINT_V1_SCHEMA",
    "MassiveAdaptiveCheckpointStateV1",
    "MassiveAdaptiveCheckpointV1",
    "MassiveAdaptiveCheckpointV1Error",
    "adaptive_training_state_receipt_v1",
    "authorize_massive_adaptive_checkpoint_v1",
    "parse_massive_adaptive_checkpoint_v1",
    "publish_massive_adaptive_checkpoint_v1",
]
