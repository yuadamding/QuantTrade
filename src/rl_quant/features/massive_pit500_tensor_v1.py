"""Authority-bound PIT-500 tensors for the finalized validation typed lane."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.finalized_typed_decision_origin import (
    MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
    MassiveTypedDecisionOriginV1,
)
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

MASSIVE_PIT500_TENSOR_V1_SCHEMA = "rl-quant.massive-pit500-decision-tensor-v1"
MASSIVE_PIT500_TENSOR_V1_DATASET = "massive-finalized-pit500-tensor-v1"
MASSIVE_PIT500_TENSOR_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "rolling_feature_spec": MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256,
        "decision_origin_spec": MASSIVE_TYPED_DECISION_ORIGIN_V1_SPEC_SHA256,
        "chronology": "derived-exclusively-from-typed-decision-origin-v1",
        "membership": "latest-effective-complete-PIT-group-known-at-decision",
        "security_order": "universe-rank-then-security-id",
        "missing_member_features": "zero-value-plus-false-mask",
        "source_staleness": "explicit-value-plus-valid-mask-per-security",
        "position_age_input": False,
    }
)
MASSIVE_PIT500_TENSOR_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PIT500_TENSOR_V1_SCHEMA,
        "bars_fields": MASSIVE_ROLLING_BARS_V0_FIELDS,
        "tape_fields": MASSIVE_ROLLING_TAPE_V0_FIELDS,
        "staleness_fields": ("source_staleness_sessions", "staleness_valid"),
    }
)


class MassivePIT500TensorV1Error(ValueError):
    """The authority-bound tensor or its committed bytes differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassivePIT500TensorV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassivePIT500DecisionTensorV1:
    decision_session_date: str
    decision_at_ms: int
    source_session_date: str
    security_ids: tuple[str, ...]
    universe_ranks: tuple[int, ...]
    bars_values: tuple[tuple[float, ...], ...]
    bars_valid: tuple[tuple[bool, ...], ...]
    tape_values: tuple[tuple[float, ...], ...]
    tape_valid: tuple[tuple[bool, ...], ...]
    source_staleness_sessions: tuple[float, ...]
    staleness_valid: tuple[bool, ...]
    rolling_feature_artifact_receipt_sha256: str
    decision_origin_receipt_sha256: str
    identity_authority_receipt_sha256: str
    universe_rule_receipt_sha256: str
    membership_group_receipt_sha256: str
    tensor_spec_receipt_sha256: str
    tensor_source_sha256: str
    semantic_tensor_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_PIT500_TENSOR_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PIT500_TENSOR_V1_SCHEMA:
            raise MassivePIT500TensorV1Error("PIT500 tensor v1 schema drifted")
        count = len(self.security_ids)
        if (
            not self.security_ids
            or count > 500
            or len(set(self.security_ids)) != count
            or len(self.universe_ranks) != count
            or tuple(sorted(self.universe_ranks)) != self.universe_ranks
            or len(self.source_staleness_sessions) != count
            or len(self.staleness_valid) != count
            or any(value != 1.0 for value in self.source_staleness_sessions)
            or any(flag is not True for flag in self.staleness_valid)
        ):
            raise MassivePIT500TensorV1Error("PIT500 tensor v1 inventory differs")
        for values, valid, width in (
            (self.bars_values, self.bars_valid, len(MASSIVE_ROLLING_BARS_V0_FIELDS)),
            (self.tape_values, self.tape_valid, len(MASSIVE_ROLLING_TAPE_V0_FIELDS)),
        ):
            if len(values) != count or len(valid) != count:
                raise MassivePIT500TensorV1Error("PIT500 tensor v1 shape differs")
            for row_values, row_valid in zip(values, valid, strict=True):
                if (
                    len(row_values) != width
                    or len(row_valid) != width
                    or any(not math.isfinite(float(value)) for value in row_values)
                    or any(not isinstance(flag, bool) for flag in row_valid)
                ):
                    raise MassivePIT500TensorV1Error("PIT500 tensor v1 values differ")
        for name in (
            "rolling_feature_artifact_receipt_sha256",
            "decision_origin_receipt_sha256",
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
            self.tensor_spec_receipt_sha256 != MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256
            or self.tensor_source_sha256 != MASSIVE_PIT500_TENSOR_V1_SOURCE_SHA256
            or self.universe_rule_receipt_sha256
            != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
        ):
            raise MassivePIT500TensorV1Error("PIT500 tensor v1 contract drifted")
        expected_semantic = semantic_sha256(
            {
                "security_ids": self.security_ids,
                "universe_ranks": self.universe_ranks,
                "bars_values": self.bars_values,
                "bars_valid": self.bars_valid,
                "tape_values": self.tape_values,
                "tape_valid": self.tape_valid,
                "source_staleness_sessions": self.source_staleness_sessions,
                "staleness_valid": self.staleness_valid,
                "decision_origin_receipt_sha256": self.decision_origin_receipt_sha256,
            }
        )
        if self.semantic_tensor_sha256 != expected_semantic:
            raise MassivePIT500TensorV1Error("PIT500 tensor v1 semantic hash differs")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_PIT500_TENSOR_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PIT500_TENSOR_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassivePIT500TensorV1Error("PIT500 tensor v1 source differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassivePIT500TensorV1Error("PIT500 tensor v1 receipt differs")


def _payload(tensor: MassivePIT500DecisionTensorV1) -> dict[str, object]:
    return {
        key: value for key, value in tensor.unsigned().items() if key != "loaded_source"
    }


def materialize_massive_pit500_tensor_v1(
    *,
    rolling_root: str | Path,
    output_root: str | Path,
    rolling: MassiveRollingFeatureArtifactV0,
    identity_authority: PITSecurityUniverseAuthority,
    decision_origin: MassiveTypedDecisionOriginV1,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassivePIT500DecisionTensorV1:
    validate_massive_rolling_features_v0(root=rolling_root, artifact=rolling)
    identity_authority.validate()
    decision_origin.validate()
    if rolling.source_session_date != decision_origin.source_session_date:
        raise MassivePIT500TensorV1Error("rolling source differs from decision origin")
    if (
        identity_authority.rule.receipt_sha256
        != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
    ):
        raise MassivePIT500TensorV1Error("PIT authority is not the frozen universe")
    decision_at_ms = decision_origin.decision_at_ms
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
        raise MassivePIT500TensorV1Error("no PIT membership is known at decision")
    effective = effective_times[-1]
    group = tuple(
        row for row in identity_authority.membership_events if row.effective_at_ms == effective
    )
    members = tuple(
        sorted(
            (row for row in group if row.is_member),
            key=lambda row: (row.universe_rank or 10**9, row.security_id),
        )
    )
    if not members or any(row.available_at_ms > decision_at_ms for row in group):
        raise MassivePIT500TensorV1Error("PIT membership group is incomplete")
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
    staleness = tuple(float(decision_origin.source_staleness_sessions) for _ in members)
    staleness_valid = tuple(True for _ in members)
    semantic_tensor = semantic_sha256(
        {
            "security_ids": security_ids,
            "universe_ranks": ranks,
            "bars_values": bars_values,
            "bars_valid": bars_valid,
            "tape_values": tape_values,
            "tape_valid": tape_valid,
            "source_staleness_sessions": staleness,
            "staleness_valid": staleness_valid,
            "decision_origin_receipt_sha256": decision_origin.receipt_sha256,
        }
    )
    relative = f"massive-finalized-v1/decision={decision_origin.decision_session_date}/pit500-tensor-v1.json"
    placeholder = MassivePIT500DecisionTensorV1(
        decision_session_date=decision_origin.decision_session_date,
        decision_at_ms=decision_origin.decision_at_ms,
        source_session_date=rolling.source_session_date,
        security_ids=security_ids,
        universe_ranks=ranks,
        bars_values=bars_values,
        bars_valid=bars_valid,
        tape_values=tape_values,
        tape_valid=tape_valid,
        source_staleness_sessions=staleness,
        staleness_valid=staleness_valid,
        rolling_feature_artifact_receipt_sha256=rolling.receipt_sha256,
        decision_origin_receipt_sha256=decision_origin.receipt_sha256,
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        universe_rule_receipt_sha256=identity_authority.rule.receipt_sha256,
        membership_group_receipt_sha256=semantic_sha256(
            tuple(asdict(row) for row in group)
        ),
        tensor_spec_receipt_sha256=MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256,
        tensor_source_sha256=MASSIVE_PIT500_TENSOR_V1_SOURCE_SHA256,
        semantic_tensor_sha256=semantic_tensor,
        loaded_source=rolling.loaded_source,
        receipt_sha256="0" * 64,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(placeholder))),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PIT500_TENSOR_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_PIT500_TENSOR_V1_SOURCE_SCHEMA_SHA256,
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
    validate_massive_pit500_tensor_v1(root=output_root, tensor=result)
    return result


def validate_massive_pit500_tensor_v1(
    *, root: str | Path, tensor: MassivePIT500DecisionTensorV1
) -> None:
    tensor.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=tensor.loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassivePIT500TensorV1Error("PIT500 tensor v1 source is not JSON") from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _payload(tensor)
    ):
        raise MassivePIT500TensorV1Error("PIT500 tensor v1 bytes differ")


__all__ = [
    "MASSIVE_PIT500_TENSOR_V1_DATASET",
    "MASSIVE_PIT500_TENSOR_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PIT500_TENSOR_V1_SOURCE_SHA256",
    "MASSIVE_PIT500_TENSOR_V1_SPEC_SHA256",
    "MassivePIT500DecisionTensorV1",
    "MassivePIT500TensorV1Error",
    "materialize_massive_pit500_tensor_v1",
    "validate_massive_pit500_tensor_v1",
]
