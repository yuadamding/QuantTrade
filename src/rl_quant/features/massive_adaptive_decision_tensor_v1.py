"""Create-only adaptive model inputs rebuilt from committed feature roots.

This artifact closes the boundary between source-derived per-decision feature
cross-sections and the temporal/cross-sectional tensors consumed by the
adaptive alpha model.  The canonical file stores identities and exact array
hashes, not caller-provided tensors.  A generic reload is nonauthorizing; the
runtime tensors exist only after the package rebuilds them from the same
feature and action-origin artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    BARS_MIN_V2_FIELDS,
    TAPE_MIN_V2_FIELDS,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
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


MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SCHEMA = (
    "rl-quant.massive-adaptive-decision-tensor-v1"
)
MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_DATASET = (
    "massive-adaptive-decision-tensor-v1"
)
MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SCHEMA,
        "encoding": "canonical-json-source-and-array-hash-inventory",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "security_axis": (
            "security-id-sorted-union-of-source-qualified-context-rows"
        ),
        "context_membership": "feature-row-present-at-decision",
        "action_mask": "package-owned-adaptive-origin-security-inventory",
        "bars": BARS_MIN_V2_FIELDS,
        "tape": TAPE_MIN_V2_FIELDS,
        "staleness": "source-artifact-session-staleness-repeated-by-modality",
        "missing": "zero-plus-independent-false-mask",
        "intraday": "not-materialized-in-v1",
        "generic_reload": "nonauthorizing",
        "runtime": "package-rebuilt-exact-float32-arrays",
        "duration_prior": False,
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveDecisionTensorV1Error(ValueError):
    """Adaptive decision tensors differ from their committed source rows."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveDecisionTensorV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveDecisionTensorV1Error(
            "adaptive decision tensor artifact ID is not path safe"
        )
    return value


def _tensor_payload(value: torch.Tensor) -> object:
    return value.detach().cpu().tolist()


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveDecisionTensorRuntimeV1:
    decision_session_dates: tuple[str, ...]
    security_ids: tuple[str, ...]
    bars_values: torch.Tensor
    bars_valid: torch.Tensor
    tape_values: torch.Tensor
    tape_valid: torch.Tensor
    source_staleness: torch.Tensor
    context_membership: torch.Tensor
    action_mask: torch.Tensor
    source_array_receipts: tuple[str, ...]
    tensor_inventory_sha256: str

    def validate(self) -> None:
        sessions = len(self.decision_session_dates)
        assets = len(self.security_ids)
        if (
            sessions <= 0
            or assets <= 0
            or self.decision_session_dates
            != tuple(sorted(set(self.decision_session_dates)))
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or self.bars_values.shape
            != (sessions, assets, len(BARS_MIN_V2_FIELDS))
            or self.bars_valid.shape != self.bars_values.shape
            or self.tape_values.shape
            != (sessions, assets, len(TAPE_MIN_V2_FIELDS))
            or self.tape_valid.shape != self.tape_values.shape
            or self.source_staleness.shape != (sessions, assets, 2)
            or self.context_membership.shape != (sessions, assets)
            or self.action_mask.shape != (sessions, assets)
            or self.bars_values.dtype != torch.float32
            or self.tape_values.dtype != torch.float32
            or self.source_staleness.dtype != torch.float32
            or self.bars_valid.dtype != torch.bool
            or self.tape_valid.dtype != torch.bool
            or self.context_membership.dtype != torch.bool
            or self.action_mask.dtype != torch.bool
            or bool((self.action_mask & ~self.context_membership).any().item())
            or bool((self.bars_values[~self.bars_valid] != 0.0).any().item())
            or bool((self.tape_values[~self.tape_valid] != 0.0).any().item())
            or bool(
                (
                    self.source_staleness[
                        ~self.context_membership.unsqueeze(-1).expand_as(
                            self.source_staleness
                        )
                    ]
                    != 0.0
                ).any().item()
            )
            or len(self.source_array_receipts) != sessions
            or self.tensor_inventory_sha256
            != semantic_sha256(self.source_array_receipts)
        ):
            raise MassiveAdaptiveDecisionTensorV1Error(
                "adaptive runtime tensor geometry or missingness differs"
            )
        for value in (*self.source_array_receipts, self.tensor_inventory_sha256):
            _digest("adaptive runtime tensor", value)
        expected = _source_array_receipts(self)
        if self.source_array_receipts != expected:
            raise MassiveAdaptiveDecisionTensorV1Error(
                "adaptive runtime tensor array hashes differ"
            )


def _source_array_receipts(
    runtime: MassiveAdaptiveDecisionTensorRuntimeV1,
) -> tuple[str, ...]:
    return tuple(
        semantic_sha256(
            {
                "decision_session_date": session_date,
                "security_ids": runtime.security_ids,
                "bars_values_float32": _tensor_payload(runtime.bars_values[index]),
                "bars_valid": _tensor_payload(runtime.bars_valid[index]),
                "tape_values_float32": _tensor_payload(runtime.tape_values[index]),
                "tape_valid": _tensor_payload(runtime.tape_valid[index]),
                "source_staleness_float32": _tensor_payload(
                    runtime.source_staleness[index]
                ),
                "context_membership": _tensor_payload(
                    runtime.context_membership[index]
                ),
                "action_mask": _tensor_payload(runtime.action_mask[index]),
            }
        )
        for index, session_date in enumerate(runtime.decision_session_dates)
    )


def _build_runtime(
    *,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    action_origins: Sequence[MassiveAdaptiveOriginAuthorityV1],
) -> MassiveAdaptiveDecisionTensorRuntimeV1:
    ordered_features = tuple(
        sorted(features, key=lambda row: row.decision_session_date)
    )
    ordered_origins = tuple(
        sorted(action_origins, key=lambda row: row.decision_session_date)
    )
    if not ordered_features or len(ordered_features) != len(ordered_origins):
        raise MassiveAdaptiveDecisionTensorV1Error(
            "adaptive feature and action-origin inventories differ"
        )
    feature_dates = tuple(row.decision_session_date for row in ordered_features)
    origin_dates = tuple(row.decision_session_date for row in ordered_origins)
    if (
        feature_dates != tuple(sorted(set(feature_dates)))
        or feature_dates != origin_dates
    ):
        raise MassiveAdaptiveDecisionTensorV1Error(
            "adaptive decision chronology differs"
        )
    for feature, origin in zip(ordered_features, ordered_origins, strict=True):
        if not isinstance(feature, MassiveProfitabilityOriginFeaturesV3) or not isinstance(
            origin, MassiveAdaptiveOriginAuthorityV1
        ):
            raise MassiveAdaptiveDecisionTensorV1Error(
                "adaptive tensor roots use unsupported generations"
            )
        feature.validate()
        origin.validate()
        feature_ids = {row.security_id for row in feature.rows}
        if not set(origin.security_ids) <= feature_ids:
            raise MassiveAdaptiveDecisionTensorV1Error(
                "adaptive action support is absent from the context features"
            )

    security_ids = tuple(
        sorted(
            {
                row.security_id
                for feature in ordered_features
                for row in feature.rows
            }
        )
    )
    index_by_id = {security_id: index for index, security_id in enumerate(security_ids)}
    sessions = len(ordered_features)
    assets = len(security_ids)
    bars_values = torch.zeros(
        (sessions, assets, len(BARS_MIN_V2_FIELDS)), dtype=torch.float32
    )
    bars_valid = torch.zeros_like(bars_values, dtype=torch.bool)
    tape_values = torch.zeros(
        (sessions, assets, len(TAPE_MIN_V2_FIELDS)), dtype=torch.float32
    )
    tape_valid = torch.zeros_like(tape_values, dtype=torch.bool)
    source_staleness = torch.zeros((sessions, assets, 2), dtype=torch.float32)
    context_membership = torch.zeros((sessions, assets), dtype=torch.bool)
    action_mask = torch.zeros((sessions, assets), dtype=torch.bool)
    for session_index, (feature, origin) in enumerate(
        zip(ordered_features, ordered_origins, strict=True)
    ):
        for row in feature.rows:
            asset_index = index_by_id[row.security_id]
            bars_values[session_index, asset_index] = torch.tensor(
                row.bars_values, dtype=torch.float32
            )
            bars_valid[session_index, asset_index] = torch.tensor(
                row.bars_valid, dtype=torch.bool
            )
            tape_values[session_index, asset_index] = torch.tensor(
                row.tape_values, dtype=torch.float32
            )
            tape_valid[session_index, asset_index] = torch.tensor(
                row.tape_valid, dtype=torch.bool
            )
            source_staleness[session_index, asset_index] = float(
                row.source_staleness_sessions
            )
            context_membership[session_index, asset_index] = True
        for security_id in origin.security_ids:
            action_mask[session_index, index_by_id[security_id]] = True
    bars_values = torch.where(bars_valid, bars_values, torch.zeros_like(bars_values))
    tape_values = torch.where(tape_valid, tape_values, torch.zeros_like(tape_values))
    provisional = MassiveAdaptiveDecisionTensorRuntimeV1(
        decision_session_dates=feature_dates,
        security_ids=security_ids,
        bars_values=bars_values,
        bars_valid=bars_valid,
        tape_values=tape_values,
        tape_valid=tape_valid,
        source_staleness=source_staleness,
        context_membership=context_membership,
        action_mask=action_mask,
        source_array_receipts=(),
        tensor_inventory_sha256="0" * 64,
    )
    receipts = _source_array_receipts(provisional)
    result = replace(
        provisional,
        source_array_receipts=receipts,
        tensor_inventory_sha256=semantic_sha256(receipts),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveDecisionTensorV1:
    decision_session_dates: tuple[str, ...]
    security_ids: tuple[str, ...]
    feature_semantic_receipts: tuple[str, ...]
    feature_audit_receipts: tuple[str, ...]
    action_origin_receipts: tuple[str, ...]
    source_array_receipts: tuple[str, ...]
    feature_inventory_sha256: str
    action_origin_inventory_sha256: str
    tensor_inventory_sha256: str
    committed_source_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_tensor: MassiveAdaptiveDecisionTensorRuntimeV1 | None
    runtime_source_replayed: bool
    model_input_authorized: bool
    development_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "decision_session_dates": self.decision_session_dates,
            "security_ids": self.security_ids,
            "feature_semantic_receipts": self.feature_semantic_receipts,
            "action_origin_receipts": self.action_origin_receipts,
            "source_array_receipts": self.source_array_receipts,
            "feature_inventory_sha256": self.feature_inventory_sha256,
            "action_origin_inventory_sha256": self.action_origin_inventory_sha256,
            "tensor_inventory_sha256": self.tensor_inventory_sha256,
            "committed_source_data_qualified": self.committed_source_data_qualified,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "development_training_authorized": self.development_training_authorized,
            "profitability_reporting_authorized": self.profitability_reporting_authorized,
            "lockbox_access_authorized": self.lockbox_access_authorized,
            "reinforcement_learning_authorized": self.reinforcement_learning_authorized,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            **self.semantic_unsigned(),
            "feature_audit_receipts": self.feature_audit_receipts,
            "semantic_receipt_sha256": self.semantic_receipt_sha256,
            "audit_receipt_sha256": self.audit_receipt_sha256,
        }

    def validate(self) -> None:
        runtime_present = self.runtime_tensor is not None
        if (
            self.schema != MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SCHEMA
            or not self.decision_session_dates
            or self.decision_session_dates
            != tuple(sorted(set(self.decision_session_dates)))
            or not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or len(self.feature_semantic_receipts)
            != len(self.decision_session_dates)
            or len(self.feature_audit_receipts) != len(self.decision_session_dates)
            or len(self.action_origin_receipts) != len(self.decision_session_dates)
            or len(self.source_array_receipts) != len(self.decision_session_dates)
            or self.feature_inventory_sha256
            != semantic_sha256(self.feature_semantic_receipts)
            or self.action_origin_inventory_sha256
            != semantic_sha256(self.action_origin_receipts)
            or self.tensor_inventory_sha256
            != semantic_sha256(self.source_array_receipts)
            or not isinstance(self.committed_source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or self.audit_receipt_sha256
            != semantic_sha256(
                {
                    "semantic_receipt_sha256": self.semantic_receipt_sha256,
                    "feature_audit_receipts": self.feature_audit_receipts,
                    "action_origin_receipts": self.action_origin_receipts,
                }
            )
            or self.runtime_source_replayed != runtime_present
            or self.model_input_authorized
            != (runtime_present and self.committed_source_data_qualified)
            or self.development_training_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveDecisionTensorV1Error(
                "adaptive decision tensor identity or authorization differs"
            )
        for value in (
            *self.feature_semantic_receipts,
            *self.feature_audit_receipts,
            *self.action_origin_receipts,
            *self.source_array_receipts,
            self.feature_inventory_sha256,
            self.action_origin_inventory_sha256,
            self.tensor_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("adaptive decision tensor", value)
        if self.runtime_tensor is not None:
            self.runtime_tensor.validate()
            if (
                self.runtime_tensor.decision_session_dates
                != self.decision_session_dates
                or self.runtime_tensor.security_ids != self.security_ids
                or self.runtime_tensor.source_array_receipts
                != self.source_array_receipts
                or self.runtime_tensor.tensor_inventory_sha256
                != self.tensor_inventory_sha256
            ):
                raise MassiveAdaptiveDecisionTensorV1Error(
                    "adaptive runtime tensor differs from its commitment"
                )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.feature_inventory_sha256
        ):
            raise MassiveAdaptiveDecisionTensorV1Error(
                "adaptive decision tensor committed source differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def materialize_massive_adaptive_decision_tensor_v1(
    *,
    root: str | Path,
    artifact_id: str,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    action_origins: Sequence[MassiveAdaptiveOriginAuthorityV1],
    committed_at_ms: int,
) -> MassiveAdaptiveDecisionTensorV1:
    """Publish and immediately replay one adaptive model-input inventory."""

    identifier = _artifact_id(artifact_id)
    ordered_features = tuple(
        sorted(features, key=lambda row: row.decision_session_date)
    )
    ordered_origins = tuple(
        sorted(action_origins, key=lambda row: row.decision_session_date)
    )
    runtime = _build_runtime(
        features=ordered_features, action_origins=ordered_origins
    )
    feature_receipts = tuple(
        row.semantic_receipt_sha256 for row in ordered_features
    )
    feature_audits = tuple(row.audit_receipt_sha256 for row in ordered_features)
    origin_receipts = tuple(row.semantic_receipt_sha256 for row in ordered_origins)
    semantic: dict[str, object] = {
        "schema": MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SCHEMA,
        "decision_session_dates": runtime.decision_session_dates,
        "security_ids": runtime.security_ids,
        "feature_semantic_receipts": feature_receipts,
        "action_origin_receipts": origin_receipts,
        "source_array_receipts": runtime.source_array_receipts,
        "feature_inventory_sha256": semantic_sha256(feature_receipts),
        "action_origin_inventory_sha256": semantic_sha256(origin_receipts),
        "tensor_inventory_sha256": runtime.tensor_inventory_sha256,
        "committed_source_data_qualified": all(
            row.source_inputs_data_qualified for row in ordered_features
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SOURCE_SHA256,
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    receipt = semantic_sha256(semantic)
    payload = {
        **semantic,
        "feature_audit_receipts": feature_audits,
        "semantic_receipt_sha256": receipt,
        "audit_receipt_sha256": semantic_sha256(
            {
                "semantic_receipt_sha256": receipt,
                "feature_audit_receipts": feature_audits,
                "action_origin_receipts": origin_receipts,
            }
        ),
    }
    relative = f"massive-adaptive/decision-tensor-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=semantic_sha256(feature_receipts),
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-DECISION-TENSOR-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_adaptive_decision_tensor_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_decision_tensor_v1(
        root=root,
        tensor=parsed,
        features=ordered_features,
        action_origins=ordered_origins,
    )


def parse_massive_adaptive_decision_tensor_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveDecisionTensorV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveAdaptiveDecisionTensorV1Error(
            "adaptive decision tensor is not canonical JSON"
        )
    for name in (
        "decision_session_dates",
        "security_ids",
        "feature_semantic_receipts",
        "feature_audit_receipts",
        "action_origin_receipts",
        "source_array_receipts",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveAdaptiveDecisionTensorV1(
        **payload,
        loaded_source=loaded_source,
        runtime_tensor=None,
        runtime_source_replayed=False,
        model_input_authorized=False,
    )
    result.validate()
    if canonical_json_file_bytes(result.canonical_payload()) != raw:
        raise MassiveAdaptiveDecisionTensorV1Error(
            "adaptive decision tensor canonical bytes differ"
        )
    return result


def authorize_massive_adaptive_decision_tensor_v1(
    *,
    root: str | Path,
    tensor: MassiveAdaptiveDecisionTensorV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    action_origins: Sequence[MassiveAdaptiveOriginAuthorityV1],
) -> MassiveAdaptiveDecisionTensorV1:
    """Reopen the commitment and reconstruct every runtime input array."""

    parsed = parse_massive_adaptive_decision_tensor_v1(
        root=root, loaded_source=tensor.loaded_source
    )
    ordered_features = tuple(
        sorted(features, key=lambda row: row.decision_session_date)
    )
    ordered_origins = tuple(
        sorted(action_origins, key=lambda row: row.decision_session_date)
    )
    rebuilt = _build_runtime(
        features=ordered_features, action_origins=ordered_origins
    )
    if (
        parsed.semantic_receipt_sha256 != tensor.semantic_receipt_sha256
        or parsed.decision_session_dates != rebuilt.decision_session_dates
        or parsed.security_ids != rebuilt.security_ids
        or parsed.feature_semantic_receipts
        != tuple(row.semantic_receipt_sha256 for row in ordered_features)
        or parsed.feature_audit_receipts
        != tuple(row.audit_receipt_sha256 for row in ordered_features)
        or parsed.action_origin_receipts
        != tuple(row.semantic_receipt_sha256 for row in ordered_origins)
        or parsed.source_array_receipts != rebuilt.source_array_receipts
        or parsed.tensor_inventory_sha256 != rebuilt.tensor_inventory_sha256
        or parsed.committed_source_data_qualified
        != all(row.source_inputs_data_qualified for row in ordered_features)
    ):
        raise MassiveAdaptiveDecisionTensorV1Error(
            "adaptive decision tensor does not replay from its source roots"
        )
    result = replace(
        parsed,
        runtime_tensor=rebuilt,
        runtime_source_replayed=True,
        model_input_authorized=parsed.committed_source_data_qualified,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_DECISION_TENSOR_V1_SCHEMA",
    "MassiveAdaptiveDecisionTensorRuntimeV1",
    "MassiveAdaptiveDecisionTensorV1",
    "MassiveAdaptiveDecisionTensorV1Error",
    "authorize_massive_adaptive_decision_tensor_v1",
    "materialize_massive_adaptive_decision_tensor_v1",
    "parse_massive_adaptive_decision_tensor_v1",
]
