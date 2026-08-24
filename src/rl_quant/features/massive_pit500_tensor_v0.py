"""Typed PIT-500 decision tensors for finalized Massive validation V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_rolling_features_v0 import (
    MASSIVE_ROLLING_BARS_V0_FIELDS,
    MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256,
    MASSIVE_ROLLING_TAPE_V0_FIELDS,
    MassiveRollingFeatureArtifactV0,
    validate_massive_rolling_features_v0,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_PIT500_TENSOR_V0_SCHEMA = "rl-quant.massive-pit500-decision-tensor-v0"
MASSIVE_PIT500_TENSOR_V0_DATASET = "massive-finalized-pit500-tensor-v0"
MASSIVE_PIT500_TENSOR_V0_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PIT500_TENSOR_V0_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "universe_rule_receipt": MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256,
        "rolling_feature_spec": MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256,
        "membership": "latest-effective-complete-PIT-group-known-at-decision",
        "security_order": "universe-rank-then-security-id",
        "missing_member_features": "zero-value-plus-false-mask",
        "source_staleness_context": True,
        "position_age_input": False,
    }
)
MASSIVE_PIT500_TENSOR_V0_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PIT500_TENSOR_V0_SCHEMA,
        "bars_fields": MASSIVE_ROLLING_BARS_V0_FIELDS,
        "tape_fields": MASSIVE_ROLLING_TAPE_V0_FIELDS,
        "value_type": "finite-float64",
        "mask_type": "boolean",
    }
)


class MassivePIT500TensorV0Error(ValueError):
    """PIT membership or decision tensor bytes differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassivePIT500TensorV0Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassivePIT500DecisionTensorV0:
    decision_session_date: str
    decision_at_ms: int
    source_session_date: str
    source_staleness_sessions: int
    security_ids: tuple[str, ...]
    universe_ranks: tuple[int, ...]
    bars_values: tuple[tuple[float, ...], ...]
    bars_valid: tuple[tuple[bool, ...], ...]
    tape_values: tuple[tuple[float, ...], ...]
    tape_valid: tuple[tuple[bool, ...], ...]
    rolling_feature_artifact_receipt_sha256: str
    identity_authority_receipt_sha256: str
    universe_rule_receipt_sha256: str
    membership_group_receipt_sha256: str
    tensor_spec_receipt_sha256: str
    tensor_source_sha256: str
    semantic_tensor_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_PIT500_TENSOR_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PIT500_TENSOR_V0_SCHEMA:
            raise MassivePIT500TensorV0Error("PIT500 tensor schema drifted")
        if (
            isinstance(self.decision_at_ms, bool)
            or not isinstance(self.decision_at_ms, int)
            or self.decision_at_ms <= 0
            or isinstance(self.source_staleness_sessions, bool)
            or not isinstance(self.source_staleness_sessions, int)
            or not 1 <= self.source_staleness_sessions <= 3
        ):
            raise MassivePIT500TensorV0Error("PIT500 tensor chronology is invalid")
        if (
            not self.security_ids
            or len(self.security_ids) > 500
            or len(set(self.security_ids)) != len(self.security_ids)
            or len(self.universe_ranks) != len(self.security_ids)
            or tuple(sorted(self.universe_ranks)) != self.universe_ranks
            or any(rank <= 0 or rank > 500 for rank in self.universe_ranks)
        ):
            raise MassivePIT500TensorV0Error("PIT500 membership inventory differs")
        expected_shapes = (
            (self.bars_values, self.bars_valid, len(MASSIVE_ROLLING_BARS_V0_FIELDS)),
            (self.tape_values, self.tape_valid, len(MASSIVE_ROLLING_TAPE_V0_FIELDS)),
        )
        for values, valid, width in expected_shapes:
            if len(values) != len(self.security_ids) or len(valid) != len(values):
                raise MassivePIT500TensorV0Error("PIT500 tensor row count differs")
            for row_values, row_valid in zip(values, valid, strict=True):
                if (
                    len(row_values) != width
                    or len(row_valid) != width
                    or any(not isinstance(flag, bool) for flag in row_valid)
                    or any(not math.isfinite(float(value)) for value in row_values)
                ):
                    raise MassivePIT500TensorV0Error("PIT500 tensor values differ")
        for name in (
            "rolling_feature_artifact_receipt_sha256",
            "identity_authority_receipt_sha256",
            "universe_rule_receipt_sha256",
            "membership_group_receipt_sha256",
            "tensor_spec_receipt_sha256",
            "tensor_source_sha256",
            "semantic_tensor_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.universe_rule_receipt_sha256
            != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
            or self.tensor_spec_receipt_sha256 != MASSIVE_PIT500_TENSOR_V0_SPEC_SHA256
            or self.tensor_source_sha256 != MASSIVE_PIT500_TENSOR_V0_SOURCE_SHA256
        ):
            raise MassivePIT500TensorV0Error("PIT500 tensor implementation drifted")
        expected_semantic = semantic_sha256(
            {
                "security_ids": self.security_ids,
                "universe_ranks": self.universe_ranks,
                "bars_values": self.bars_values,
                "bars_valid": self.bars_valid,
                "tape_values": self.tape_values,
                "tape_valid": self.tape_valid,
                "source_staleness_sessions": self.source_staleness_sessions,
            }
        )
        if self.semantic_tensor_sha256 != expected_semantic:
            raise MassivePIT500TensorV0Error("PIT500 tensor semantic hash differs")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_PIT500_TENSOR_V0_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PIT500_TENSOR_V0_SOURCE_SCHEMA_SHA256
        ):
            raise MassivePIT500TensorV0Error("PIT500 tensor source differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassivePIT500TensorV0Error("PIT500 tensor receipt differs")


def _payload(tensor: MassivePIT500DecisionTensorV0) -> dict[str, object]:
    return {
        "schema": tensor.schema,
        "decision_session_date": tensor.decision_session_date,
        "decision_at_ms": tensor.decision_at_ms,
        "source_session_date": tensor.source_session_date,
        "source_staleness_sessions": tensor.source_staleness_sessions,
        "security_ids": tensor.security_ids,
        "universe_ranks": tensor.universe_ranks,
        "bars_feature_names": MASSIVE_ROLLING_BARS_V0_FIELDS,
        "tape_feature_names": MASSIVE_ROLLING_TAPE_V0_FIELDS,
        "bars_values": tensor.bars_values,
        "bars_valid": tensor.bars_valid,
        "tape_values": tensor.tape_values,
        "tape_valid": tensor.tape_valid,
        "rolling_feature_artifact_receipt_sha256": tensor.rolling_feature_artifact_receipt_sha256,
        "identity_authority_receipt_sha256": tensor.identity_authority_receipt_sha256,
        "universe_rule_receipt_sha256": tensor.universe_rule_receipt_sha256,
        "membership_group_receipt_sha256": tensor.membership_group_receipt_sha256,
        "tensor_spec_receipt_sha256": tensor.tensor_spec_receipt_sha256,
        "tensor_source_sha256": tensor.tensor_source_sha256,
        "semantic_tensor_sha256": tensor.semantic_tensor_sha256,
    }


def materialize_massive_pit500_tensor_v0(
    *,
    rolling_root: str | Path,
    output_root: str | Path,
    rolling: MassiveRollingFeatureArtifactV0,
    identity_authority: PITSecurityUniverseAuthority,
    decision_session_date: str,
    decision_at_ms: int,
    source_staleness_sessions: int,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassivePIT500DecisionTensorV0:
    validate_massive_rolling_features_v0(root=rolling_root, artifact=rolling)
    identity_authority.validate()
    if (
        identity_authority.rule.receipt_sha256
        != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
    ):
        raise MassivePIT500TensorV0Error("PIT authority is not the frozen V0 universe")
    effective_times = tuple(
        sorted(
            {
                row.effective_at_ms
                for row in identity_authority.membership_events
                if row.effective_at_ms <= decision_at_ms
                and row.available_at_ms <= decision_at_ms
            }
        )
    )
    if not effective_times:
        raise MassivePIT500TensorV0Error("no PIT membership is known at decision")
    effective = effective_times[-1]
    group = tuple(
        row
        for row in identity_authority.membership_events
        if row.effective_at_ms == effective
    )
    members = tuple(
        sorted(
            (row for row in group if row.is_member),
            key=lambda row: (row.universe_rank or 10**9, row.security_id),
        )
    )
    if not members or any(row.available_at_ms > decision_at_ms for row in group):
        raise MassivePIT500TensorV0Error(
            "PIT membership group is incomplete at decision"
        )
    by_security = {row.security_id: row for row in rolling.rows}
    bars_width = len(MASSIVE_ROLLING_BARS_V0_FIELDS)
    tape_width = len(MASSIVE_ROLLING_TAPE_V0_FIELDS)
    security_ids = tuple(row.security_id for row in members)
    ranks = tuple(int(row.universe_rank or 0) for row in members)
    bars_values = tuple(
        by_security[security].bars_values
        if security in by_security
        else (0.0,) * bars_width
        for security in security_ids
    )
    bars_valid = tuple(
        by_security[security].bars_valid
        if security in by_security
        else (False,) * bars_width
        for security in security_ids
    )
    tape_values = tuple(
        by_security[security].tape_values
        if security in by_security
        else (0.0,) * tape_width
        for security in security_ids
    )
    tape_valid = tuple(
        by_security[security].tape_valid
        if security in by_security
        else (False,) * tape_width
        for security in security_ids
    )
    semantic_tensor = semantic_sha256(
        {
            "security_ids": security_ids,
            "universe_ranks": ranks,
            "bars_values": bars_values,
            "bars_valid": bars_valid,
            "tape_values": tape_values,
            "tape_valid": tape_valid,
            "source_staleness_sessions": source_staleness_sessions,
        }
    )
    relative = (
        f"massive-finalized-v0/decision={decision_session_date}/pit500-tensor.json"
    )
    placeholder = MassivePIT500DecisionTensorV0(
        decision_session_date=decision_session_date,
        decision_at_ms=decision_at_ms,
        source_session_date=rolling.source_session_date,
        source_staleness_sessions=source_staleness_sessions,
        security_ids=security_ids,
        universe_ranks=ranks,
        bars_values=bars_values,
        bars_valid=bars_valid,
        tape_values=tape_values,
        tape_valid=tape_valid,
        rolling_feature_artifact_receipt_sha256=rolling.receipt_sha256,
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        universe_rule_receipt_sha256=identity_authority.rule.receipt_sha256,
        membership_group_receipt_sha256=semantic_sha256(
            tuple(asdict(row) for row in group)
        ),
        tensor_spec_receipt_sha256=MASSIVE_PIT500_TENSOR_V0_SPEC_SHA256,
        tensor_source_sha256=MASSIVE_PIT500_TENSOR_V0_SOURCE_SHA256,
        semantic_tensor_sha256=semantic_tensor,
        loaded_source=rolling.loaded_source,
        receipt_sha256="0" * 64,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(placeholder))),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PIT500_TENSOR_V0_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_PIT500_TENSOR_V0_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=published_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=output_root, relative_payload_path=relative, verified_at_ms=published_at_ms
    )
    provisional = replace(placeholder, loaded_source=loaded)
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    validate_massive_pit500_tensor_v0(root=output_root, tensor=result)
    return result


def validate_massive_pit500_tensor_v0(
    *, root: str | Path, tensor: MassivePIT500DecisionTensorV0
) -> None:
    tensor.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=tensor.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassivePIT500TensorV0Error("PIT500 tensor source is not JSON") from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _payload(tensor)
    ):
        raise MassivePIT500TensorV0Error("PIT500 tensor bytes differ")


__all__ = [
    "MASSIVE_PIT500_TENSOR_V0_DATASET",
    "MASSIVE_PIT500_TENSOR_V0_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PIT500_TENSOR_V0_SPEC_SHA256",
    "MassivePIT500DecisionTensorV0",
    "MassivePIT500TensorV0Error",
    "materialize_massive_pit500_tensor_v0",
    "validate_massive_pit500_tensor_v0",
]
