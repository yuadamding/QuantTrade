"""Create-only durable authority for exact adaptive PPO checkpoint resume."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvStateV1,
)
from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    MassiveAdaptiveEconomicBookV1,
    MassiveAdaptiveHoldingV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import MassiveAdaptiveRLActionV1
from rl_quant.rl.massive_adaptive_rl_observation_v1 import (
    MassiveAdaptiveRLTrailingStateV1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptiveRLCheckpointV1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_protocol_v1 import (
    MassiveAdaptiveRLTrainingForecastAuthorityProtocol,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v2 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV2,
)


MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-checkpoint-authority-v1"
)
MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-checkpoint-authority-v1"
)
MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SCHEMA,
        "encoding": "torch-safe-primitives-v1",
        "generic_reload": "runtime-state-stripped",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "state": (
            "actor-critic",
            "optimizers",
            "all-rng",
            "chronology-and-three-books",
        ),
        "commit_boundary": "ppo-update-only",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer": False,
        "lockbox": False,
    }
)


class MassiveAdaptiveRLCheckpointAuthorityV1Error(ValueError):
    """The durable PPO checkpoint did not replay from its safe payload."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLCheckpointAuthorityV1Error(
            "adaptive RL checkpoint artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLCheckpointAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _numpy_rng_payload(state: tuple[Any, ...]) -> dict[str, object]:
    return {
        "bit_generator": state[0],
        "keys": torch.from_numpy(np.asarray(state[1], dtype=np.uint32).copy()),
        "position": state[2],
        "has_gauss": state[3],
        "cached_gaussian": state[4],
    }


def _checkpoint_payload(
    checkpoint: MassiveAdaptiveRLCheckpointV1,
    *,
    source_data_qualified: bool,
) -> dict[str, object]:
    return {
        "_authority_source_data_qualified": source_data_qualified,
        "update_index": checkpoint.update_index,
        "model_state": checkpoint.model_state,
        "actor_optimizer_state": checkpoint.actor_optimizer_state,
        "critic_optimizer_state": checkpoint.critic_optimizer_state,
        "torch_rng_state": checkpoint.torch_rng_state,
        "cuda_rng_states": checkpoint.cuda_rng_states,
        "python_rng_state": checkpoint.python_rng_state,
        "numpy_rng_state": _numpy_rng_payload(checkpoint.numpy_rng_state),
        "minibatch_rng_state": checkpoint.minibatch_rng_state,
        "environment_state": asdict(checkpoint.environment_state),
        "loss_trace": checkpoint.loss_trace,
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "actor_optimizer_state_receipt_sha256": (
            checkpoint.actor_optimizer_state_receipt_sha256
        ),
        "critic_optimizer_state_receipt_sha256": (
            checkpoint.critic_optimizer_state_receipt_sha256
        ),
        "rng_state_receipt_sha256": checkpoint.rng_state_receipt_sha256,
        "environment_state_receipt_sha256": (
            checkpoint.environment_state_receipt_sha256
        ),
        "loss_trace_receipt_sha256": checkpoint.loss_trace_receipt_sha256,
        "training_source_inventory_sha256": (
            checkpoint.training_source_inventory_sha256
        ),
        "training_forecast_authority_receipt_sha256": (
            checkpoint.training_forecast_authority_receipt_sha256
        ),
        "fit_environment_authority_receipts": (
            checkpoint.fit_environment_authority_receipts
        ),
        "transition_receipts": checkpoint.transition_receipts,
        "transition_decision_session_dates": (
            checkpoint.transition_decision_session_dates
        ),
        "transition_source_data_qualified": (
            checkpoint.transition_source_data_qualified
        ),
        "transition_inventory_sha256": checkpoint.transition_inventory_sha256,
        "source_data_qualified": checkpoint.source_data_qualified,
        "ppo_config_receipt_sha256": checkpoint.ppo_config_receipt_sha256,
        "observation_specification_sha256": (
            checkpoint.observation_specification_sha256
        ),
        "action_specification_sha256": checkpoint.action_specification_sha256,
        "reward_specification_sha256": checkpoint.reward_specification_sha256,
        "semantic_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "exact_resume_authorized": checkpoint.exact_resume_authorized,
        "development_rl_training_authorized": (
            checkpoint.development_rl_training_authorized
        ),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": checkpoint.protocol_receipt_sha256,
        "schema": checkpoint.schema,
    }


def _parse_holding(value: Mapping[str, object]) -> MassiveAdaptiveHoldingV1:
    result = MassiveAdaptiveHoldingV1(**dict(value))  # type: ignore[arg-type]
    result.validate()
    return result


def _parse_book(value: Mapping[str, object]) -> MassiveAdaptiveEconomicBookV1:
    payload = dict(value)
    payload["holdings"] = tuple(
        _parse_holding(cast(Mapping[str, object], item))
        for item in cast(tuple[object, ...], payload["holdings"])
    )
    result = MassiveAdaptiveEconomicBookV1(**payload)  # type: ignore[arg-type]
    result.validate()
    return result


def _parse_action(value: Mapping[str, object]) -> MassiveAdaptiveRLActionV1:
    payload = dict(value)
    payload["bucket_controls"] = tuple(
        cast(tuple[float, ...], payload["bucket_controls"])
    )
    result = MassiveAdaptiveRLActionV1(**payload)  # type: ignore[arg-type]
    result.validate()
    return result


def _parse_trailing(
    value: Mapping[str, object],
) -> MassiveAdaptiveRLTrailingStateV1:
    payload = dict(value)
    payload["strategy_active_log_returns"] = tuple(
        cast(tuple[float, ...], payload["strategy_active_log_returns"])
    )
    payload["incremental_rl_log_returns"] = tuple(
        cast(tuple[float, ...], payload["incremental_rl_log_returns"])
    )
    result = MassiveAdaptiveRLTrailingStateV1(**payload)  # type: ignore[arg-type]
    result.validate()
    return result


def _parse_environment(
    value: Mapping[str, object],
) -> MassiveAdaptiveProfitabilityEnvStateV1:
    payload = dict(value)
    payload["strategy_book"] = _parse_book(
        cast(Mapping[str, object], payload["strategy_book"])
    )
    payload["neutral_book"] = _parse_book(
        cast(Mapping[str, object], payload["neutral_book"])
    )
    payload["benchmark_book"] = _parse_book(
        cast(Mapping[str, object], payload["benchmark_book"])
    )
    payload["previous_action"] = _parse_action(
        cast(Mapping[str, object], payload["previous_action"])
    )
    payload["trailing_state"] = _parse_trailing(
        cast(Mapping[str, object], payload["trailing_state"])
    )
    result = MassiveAdaptiveProfitabilityEnvStateV1(
        **payload  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _load_payload(raw: bytes) -> Mapping[str, object]:
    try:
        value = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise MassiveAdaptiveRLCheckpointAuthorityV1Error(
            "adaptive RL checkpoint payload is not a safe Torch artifact"
        ) from error
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLCheckpointAuthorityV1Error(
            "adaptive RL checkpoint payload is not a mapping"
        )
    return cast(Mapping[str, object], value)


def _parse_checkpoint(payload: Mapping[str, object]) -> MassiveAdaptiveRLCheckpointV1:
    value = dict(payload)
    value.pop("_authority_source_data_qualified", None)
    numpy_payload = cast(Mapping[str, object], value["numpy_rng_state"])
    keys = cast(torch.Tensor, numpy_payload["keys"])
    value["numpy_rng_state"] = (
        numpy_payload["bit_generator"],
        keys.detach().cpu().numpy().astype(np.uint32, copy=True),
        numpy_payload["position"],
        numpy_payload["has_gauss"],
        numpy_payload["cached_gaussian"],
    )
    value["cuda_rng_states"] = tuple(
        cast(tuple[torch.Tensor, ...], value["cuda_rng_states"])
    )
    value["python_rng_state"] = tuple(cast(tuple[Any, ...], value["python_rng_state"]))
    value["loss_trace"] = tuple(
        tuple(float(item) for item in cast(tuple[float, ...], row))
        for row in cast(tuple[object, ...], value["loss_trace"])
    )
    value["fit_environment_authority_receipts"] = tuple(
        cast(
            tuple[str, ...] | list[str],
            value["fit_environment_authority_receipts"],
        )
    )
    value["transition_receipts"] = tuple(
        cast(tuple[str, ...] | list[str], value["transition_receipts"])
    )
    value["transition_decision_session_dates"] = tuple(
        cast(
            tuple[str, ...] | list[str],
            value["transition_decision_session_dates"],
        )
    )
    value["transition_source_data_qualified"] = tuple(
        bool(item)
        for item in cast(
            tuple[bool, ...] | list[bool],
            value["transition_source_data_qualified"],
        )
    )
    value["environment_state"] = _parse_environment(
        cast(Mapping[str, object], value["environment_state"])
    )
    result = MassiveAdaptiveRLCheckpointV1(**value)  # type: ignore[arg-type]
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLCheckpointAuthorityV1:
    checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    training_forecast_authority_receipt_sha256: str
    training_source_inventory_sha256: str
    checkpoint_source_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_checkpoint: MassiveAdaptiveRLCheckpointV1 | None
    runtime_checkpoint_replayed: bool
    exact_resume_authorized: bool
    development_rl_training_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "model_state_receipt_sha256": self.model_state_receipt_sha256,
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority_receipt_sha256
            ),
            "training_source_inventory_sha256": (self.training_source_inventory_sha256),
            "checkpoint_source_receipt_sha256": (self.checkpoint_source_receipt_sha256),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime_present = self.runtime_checkpoint is not None
        expected_authorized = runtime_present and self.source_data_qualified
        if self.runtime_checkpoint is not None:
            self.runtime_checkpoint.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SCHEMA
            or self.checkpoint_source_receipt_sha256
            != self.loaded_source.receipt.receipt_sha256
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.checkpoint_receipt_sha256
            or self.runtime_checkpoint_replayed != runtime_present
            or self.exact_resume_authorized != expected_authorized
            or self.development_rl_training_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLCheckpointAuthorityV1Error(
                "adaptive RL checkpoint replay authority differs"
            )
        if self.runtime_checkpoint is not None and (
            self.runtime_checkpoint.semantic_receipt_sha256
            != self.checkpoint_receipt_sha256
            or self.runtime_checkpoint.model_state_receipt_sha256
            != self.model_state_receipt_sha256
            or self.runtime_checkpoint.training_source_inventory_sha256
            != self.training_source_inventory_sha256
            or self.runtime_checkpoint.training_forecast_authority_receipt_sha256
            != self.training_forecast_authority_receipt_sha256
            or self.runtime_checkpoint.source_data_qualified
            != self.source_data_qualified
        ):
            raise MassiveAdaptiveRLCheckpointAuthorityV1Error(
                "adaptive RL runtime checkpoint differs from authority"
            )
        for value in (
            self.checkpoint_receipt_sha256,
            self.model_state_receipt_sha256,
            self.training_forecast_authority_receipt_sha256,
            self.training_source_inventory_sha256,
            self.checkpoint_source_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL checkpoint authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def parse_massive_adaptive_rl_checkpoint_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLCheckpointAuthorityV1:
    """Reopen checkpoint metadata without exposing runtime state."""

    payload = _load_payload(
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    )
    source_data_qualified = payload.get("_authority_source_data_qualified")
    if not isinstance(source_data_qualified, bool):
        raise MassiveAdaptiveRLCheckpointAuthorityV1Error(
            "adaptive RL checkpoint source qualification is absent"
        )
    checkpoint = _parse_checkpoint(payload)
    assert checkpoint.training_forecast_authority_receipt_sha256 is not None
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SCHEMA,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "training_forecast_authority_receipt_sha256": (
            checkpoint.training_forecast_authority_receipt_sha256
        ),
        "training_source_inventory_sha256": (
            checkpoint.training_source_inventory_sha256
        ),
        "checkpoint_source_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "source_data_qualified": source_data_qualified,
        "loaded_source": loaded_source,
        "runtime_checkpoint": None,
        "runtime_checkpoint_replayed": False,
        "exact_resume_authorized": False,
        "development_rl_training_authorized": False,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLCheckpointAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_checkpoint_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    checkpoint: MassiveAdaptiveRLCheckpointV1,
    training_forecast_authority: MassiveAdaptiveRLTrainingForecastAuthorityProtocol,
    committed_at_ms: int,
) -> MassiveAdaptiveRLCheckpointAuthorityV1:
    """Publish one complete PPO update boundary and replay it immediately."""

    identifier = _artifact_id(artifact_id)
    checkpoint.validate()
    training_forecast_authority.validate()
    if (
        checkpoint.training_forecast_authority_receipt_sha256
        != training_forecast_authority.semantic_receipt_sha256
        or checkpoint.source_data_qualified
        != bool(
            type(training_forecast_authority)
            is MassiveAdaptiveRLTrainingForecastAuthorityV2
            and training_forecast_authority.source_data_qualified
            and training_forecast_authority.reinforcement_learning_authorized
            and checkpoint.fit_environment_authority_receipts
            and checkpoint.transition_receipts
            and all(checkpoint.transition_source_data_qualified)
        )
    ):
        raise MassiveAdaptiveRLCheckpointAuthorityV1Error(
            "checkpoint and RL training forecast authority differ"
        )
    stream = BytesIO()
    torch.save(
        _checkpoint_payload(
            checkpoint,
            source_data_qualified=checkpoint.source_data_qualified,
        ),
        stream,
    )
    relative = f"massive-adaptive/rl-checkpoint-v1/{identifier}.pt"
    publish_massive_source_object(
        stream=BytesIO(stream.getvalue()),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=checkpoint.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-CHECKPOINT-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_rl_checkpoint_authority_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_rl_checkpoint_authority_v1(
        root=root,
        authority=generic,
        training_forecast_authority=training_forecast_authority,
    )


def authorize_massive_adaptive_rl_checkpoint_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    training_forecast_authority: MassiveAdaptiveRLTrainingForecastAuthorityProtocol,
) -> MassiveAdaptiveRLCheckpointAuthorityV1:
    """Restore runtime checkpoint state only after source and root replay."""

    parsed = parse_massive_adaptive_rl_checkpoint_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    training_forecast_authority.validate()
    checkpoint = _parse_checkpoint(
        _load_payload(
            read_loaded_massive_source_bytes(
                root=root, loaded_source=parsed.loaded_source
            )
        )
    )
    if (
        parsed.semantic_receipt_sha256 != authority.semantic_receipt_sha256
        or parsed.training_forecast_authority_receipt_sha256
        != training_forecast_authority.semantic_receipt_sha256
        or parsed.source_data_qualified
        != checkpoint.source_data_qualified
        or parsed.source_data_qualified
        != bool(
            type(training_forecast_authority)
            is MassiveAdaptiveRLTrainingForecastAuthorityV2
            and training_forecast_authority.source_data_qualified
            and training_forecast_authority.reinforcement_learning_authorized
            and checkpoint.fit_environment_authority_receipts
            and checkpoint.transition_receipts
            and all(checkpoint.transition_source_data_qualified)
        )
        or checkpoint.semantic_receipt_sha256 != parsed.checkpoint_receipt_sha256
    ):
        raise MassiveAdaptiveRLCheckpointAuthorityV1Error(
            "committed adaptive RL checkpoint does not replay"
        )
    result = replace(
        parsed,
        runtime_checkpoint=checkpoint,
        runtime_checkpoint_replayed=True,
        exact_resume_authorized=parsed.source_data_qualified,
        development_rl_training_authorized=parsed.source_data_qualified,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_CHECKPOINT_AUTHORITY_V1_SCHEMA",
    "MassiveAdaptiveRLCheckpointAuthorityV1",
    "MassiveAdaptiveRLCheckpointAuthorityV1Error",
    "authorize_massive_adaptive_rl_checkpoint_authority_v1",
    "materialize_massive_adaptive_rl_checkpoint_authority_v1",
    "parse_massive_adaptive_rl_checkpoint_authority_v1",
]
