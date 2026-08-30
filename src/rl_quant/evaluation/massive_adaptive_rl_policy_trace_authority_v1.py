"""Create-only replay authority for checkpoint-generated adaptive RL traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
from pathlib import Path

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.evaluation.massive_adaptive_rl_policy_evaluator_v1 import (
    MassiveAdaptiveRLCheckpointPolicyTraceV1,
    MassiveAdaptiveRLPolicyActionEvidenceV1,
    evaluate_massive_adaptive_rl_checkpoint_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    MassiveAdaptiveRLCheckpointAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
)


MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-policy-trace-authority-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-policy-trace-authority-v1"
)
MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA,
        "payload": "canonical-policy-trace-and-action-evidence",
        "promotion": "reload-checkpoint-rerun-actions-and-economics",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLPolicyTraceAuthorityV1Error(ValueError):
    """The committed policy trace did not replay from its attached checkpoint."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLPolicyTraceAuthorityV1:
    fold_index: int
    evaluation_role: str
    checkpoint_authority_receipt_sha256: str
    checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    policy_trace_receipt_sha256: str
    action_evidence_inventory_sha256: str
    transition_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_trace: MassiveAdaptiveRLCheckpointPolicyTraceV1 | None
    runtime_trace_replayed: bool
    development_policy_evaluation_authorized: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fold_index": self.fold_index,
            "evaluation_role": self.evaluation_role,
            "checkpoint_authority_receipt_sha256": (
                self.checkpoint_authority_receipt_sha256
            ),
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "model_state_receipt_sha256": self.model_state_receipt_sha256,
            "policy_trace_receipt_sha256": self.policy_trace_receipt_sha256,
            "action_evidence_inventory_sha256": self.action_evidence_inventory_sha256,
            "transition_inventory_sha256": self.transition_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.loaded_source.validate()
        runtime = self.runtime_trace is not None
        expected = runtime and self.source_data_qualified
        if self.runtime_trace is not None:
            self.runtime_trace.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA
            or self.evaluation_role not in {"inner_validation", "outer_test"}
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.policy_trace_receipt_sha256
            or self.runtime_trace_replayed != runtime
            or self.development_policy_evaluation_authorized
            != (expected and self.evaluation_role == "inner_validation")
            or self.outer_evaluation_authorized
            != (expected and self.evaluation_role == "outer_test")
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                "adaptive RL policy trace authority differs"
            )
        if runtime and self.runtime_trace is not None and (
            self.runtime_trace.fold_index != self.fold_index
            or self.runtime_trace.evaluation_role != self.evaluation_role
            or self.runtime_trace.checkpoint_authority_receipt_sha256
            != self.checkpoint_authority_receipt_sha256
            or self.runtime_trace.checkpoint_receipt_sha256
            != self.checkpoint_receipt_sha256
            or self.runtime_trace.model_state_receipt_sha256
            != self.model_state_receipt_sha256
            or self.runtime_trace.policy_trace.semantic_receipt_sha256
            != self.policy_trace_receipt_sha256
            or self.runtime_trace.action_evidence_inventory_sha256
            != self.action_evidence_inventory_sha256
            or self.runtime_trace.transition_inventory_sha256
            != self.transition_inventory_sha256
        ):
            raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
                "adaptive runtime policy trace differs from its authority"
            )
        for value in (
            self.checkpoint_authority_receipt_sha256,
            self.checkpoint_receipt_sha256,
            self.model_state_receipt_sha256,
            self.policy_trace_receipt_sha256,
            self.action_evidence_inventory_sha256,
            self.transition_inventory_sha256,
            self.protocol_receipt_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL policy trace authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _payload(trace: MassiveAdaptiveRLCheckpointPolicyTraceV1) -> dict[str, object]:
    return {
        "fold_index": trace.fold_index,
        "evaluation_role": trace.evaluation_role,
        "checkpoint_authority_receipt_sha256": (
            trace.checkpoint_authority_receipt_sha256
        ),
        "checkpoint_receipt_sha256": trace.checkpoint_receipt_sha256,
        "model_state_receipt_sha256": trace.model_state_receipt_sha256,
        "policy_trace": asdict(trace.policy_trace),
        "action_evidence": tuple(asdict(row) for row in trace.action_evidence),
        "action_evidence_inventory_sha256": trace.action_evidence_inventory_sha256,
        "transition_inventory_sha256": trace.transition_inventory_sha256,
        "source_data_qualified": trace.source_data_qualified,
        "checkpoint_policy_trace_receipt_sha256": trace.semantic_receipt_sha256,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace payload is not canonical JSON"
        )
    return dict(value)


def _trace_metadata(payload: Mapping[str, object]) -> tuple[
    MassiveAdaptiveRLPolicyTraceV1,
    tuple[MassiveAdaptiveRLPolicyActionEvidenceV1, ...],
]:
    trace_payload = dict(payload["policy_trace"])  # type: ignore[arg-type]
    for name in (
        "decision_session_dates",
        "transition_receipts",
        "strategy_active_log_returns",
        "incremental_rl_log_returns",
    ):
        trace_payload[name] = tuple(trace_payload[name])
    trace = MassiveAdaptiveRLPolicyTraceV1(**trace_payload)  # type: ignore[arg-type]
    evidence = tuple(
        MassiveAdaptiveRLPolicyActionEvidenceV1(
            **{
                **dict(row),
                "action_values": tuple(dict(row)["action_values"]),
            }
        )
        for row in payload["action_evidence"]  # type: ignore[union-attr]
    )
    trace.validate()
    for row in evidence:
        row.validate()
    return trace, evidence


def parse_massive_adaptive_rl_policy_trace_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLPolicyTraceAuthorityV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    trace, evidence = _trace_metadata(payload)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SCHEMA,
        "fold_index": int(payload["fold_index"]),
        "evaluation_role": str(payload["evaluation_role"]),
        "checkpoint_authority_receipt_sha256": str(
            payload["checkpoint_authority_receipt_sha256"]
        ),
        "checkpoint_receipt_sha256": str(payload["checkpoint_receipt_sha256"]),
        "model_state_receipt_sha256": str(payload["model_state_receipt_sha256"]),
        "policy_trace_receipt_sha256": trace.semantic_receipt_sha256,
        "action_evidence_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in evidence)
        ),
        "transition_inventory_sha256": str(payload["transition_inventory_sha256"]),
        "source_data_qualified": bool(payload["source_data_qualified"]),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLPolicyTraceAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_trace=None,
        runtime_trace_replayed=False,
        development_policy_evaluation_authorized=False,
        outer_evaluation_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_policy_trace_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLPolicyTraceAuthorityV1,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLPolicyTraceAuthorityV1:
    parsed = parse_massive_adaptive_rl_policy_trace_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    committed = _load_payload(root=root, loaded_source=authority.loaded_source)
    replayed = evaluate_massive_adaptive_rl_checkpoint_v1(
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        environment=environment,
        fold_index=parsed.fold_index,
        evaluation_role=parsed.evaluation_role,
        device=device,
    )
    if canonical_json_file_bytes(committed) != canonical_json_file_bytes(
        _payload(replayed)
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace does not replay from its checkpoint"
        )
    result = replace(
        parsed,
        runtime_trace=replayed,
        runtime_trace_replayed=True,
        development_policy_evaluation_authorized=(
            parsed.source_data_qualified and parsed.evaluation_role == "inner_validation"
        ),
        outer_evaluation_authorized=(
            parsed.source_data_qualified and parsed.evaluation_role == "outer_test"
        ),
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_policy_trace_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    checkpoint_authority: MassiveAdaptiveRLCheckpointAuthorityV1,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
    fold_index: int,
    evaluation_role: str,
    committed_at_ms: int,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLPolicyTraceAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveRLPolicyTraceAuthorityV1Error(
            "adaptive RL policy trace artifact ID is not path safe"
        )
    trace = evaluate_massive_adaptive_rl_checkpoint_v1(
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        environment=environment,
        fold_index=fold_index,
        evaluation_role=evaluation_role,
        device=device,
    )
    relative = f"massive-adaptive/rl-policy-trace-authority-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(trace))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_POLICY_TRACE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=trace.policy_trace.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-POLICY-TRACE-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_rl_policy_trace_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_policy_trace_authority_v1(
            root=root, loaded_source=loaded
        ),
        checkpoint_authority=checkpoint_authority,
        chronology_authority=chronology_authority,
        environment=environment,
        device=device,
    )


__all__ = [
    "MassiveAdaptiveRLPolicyTraceAuthorityV1",
    "MassiveAdaptiveRLPolicyTraceAuthorityV1Error",
    "authorize_massive_adaptive_rl_policy_trace_authority_v1",
    "materialize_massive_adaptive_rl_policy_trace_authority_v1",
    "parse_massive_adaptive_rl_policy_trace_authority_v1",
]
