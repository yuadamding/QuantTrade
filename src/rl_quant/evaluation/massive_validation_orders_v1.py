"""Seed-distinct diagnostic requested orders for finalized validation V1."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.finalized_typed_decision_origin import (
    MassiveTypedDecisionOriginV1,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_validation_inference_v1 import (
    MassiveValidationInferenceArtifactV1,
    validate_massive_validation_inference_v1,
)
from rl_quant.evaluation.massive_validation_orders_v0 import (
    MassiveRequestedOrderRowV0,
)
from rl_quant.features.massive_pit500_tensor_v1 import (
    MassivePIT500DecisionTensorV1,
    validate_massive_pit500_tensor_v1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)

MASSIVE_VALIDATION_ORDERS_V1_SCHEMA = "rl-quant.massive-validation-requested-orders-v1"
MASSIVE_VALIDATION_ORDERS_V1_DATASET = (
    "massive-finalized-validation-requested-orders-v1"
)
MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_VALIDATION_ORDERS_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "purpose": "readiness-canary-diagnostic-orders-only",
        "score": "equal-mean-across-H01-H05-H21-H63",
        "selection": "top-20-percent-at-least-one",
        "weights": "equal-long-only-plus-explicit-cash",
        "path_identity": ("decision-session", "setting-id", "seed"),
        "decision_origin_required": True,
        "position_age_input": False,
        "duration_field": False,
        "portfolio_optimization": False,
    }
)
MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_VALIDATION_ORDERS_V1_SCHEMA,
        "identity": ("decision_session_date", "setting_id", "seed"),
        "fields": ("security_id", "requested_weight"),
        "cash": "explicit",
    }
)


class MassiveValidationOrdersV1Error(ValueError):
    """Requested-order V1 bytes or authority links differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveValidationOrdersV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveRequestedOrdersArtifactV1:
    decision_session_date: str
    decision_at_ms: int
    setting_id: str
    seed: int
    decision_origin_receipt_sha256: str
    tensor_receipt_sha256: str
    inference_receipt_sha256: str
    order_spec_receipt_sha256: str
    order_source_sha256: str
    diagnostic_fill_rule: str
    rows: tuple[MassiveRequestedOrderRowV0, ...]
    order_inventory_sha256: str
    loaded_source: LoadedMassiveSourceObject
    panel_materialization_authorized: bool
    portfolio_evaluation_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_VALIDATION_ORDERS_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_VALIDATION_ORDERS_V1_SCHEMA
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or isinstance(self.decision_at_ms, bool)
            or not isinstance(self.decision_at_ms, int)
            or self.decision_at_ms <= 0
            or self.diagnostic_fill_rule
            != "[15:50:00,16:00:00)-qualifying-trade-vwap"
        ):
            raise MassiveValidationOrdersV1Error("requested orders v1 identity differs")
        for name in (
            "decision_origin_receipt_sha256",
            "tensor_receipt_sha256",
            "inference_receipt_sha256",
            "order_spec_receipt_sha256",
            "order_source_sha256",
            "order_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.order_spec_receipt_sha256 != MASSIVE_VALIDATION_ORDERS_V1_SPEC_SHA256
            or self.order_source_sha256 != MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SHA256
        ):
            raise MassiveValidationOrdersV1Error("requested orders v1 contract drifted")
        keys = tuple(row.security_id for row in self.rows)
        expected_keys = tuple(sorted(set(keys), key=lambda value: (value != "CASH", value)))
        if not keys or keys != expected_keys or keys.count("CASH") != 1:
            raise MassiveValidationOrdersV1Error("requested orders v1 rows differ")
        for row in self.rows:
            row.validate()
        if not math.isclose(
            sum(row.requested_weight for row in self.rows), 1.0, abs_tol=1e-12
        ):
            raise MassiveValidationOrdersV1Error("requested orders v1 are not funded")
        if self.order_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveValidationOrdersV1Error("requested orders v1 inventory differs")
        if self.panel_materialization_authorized or self.portfolio_evaluation_authorized:
            raise MassiveValidationOrdersV1Error(
                "requested orders v1 cannot authorize performance work"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_VALIDATION_ORDERS_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveValidationOrdersV1Error("requested orders v1 source differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveValidationOrdersV1Error("requested orders v1 receipt differs")


def _payload(artifact: MassiveRequestedOrdersArtifactV1) -> dict[str, object]:
    return {
        key: value for key, value in artifact.unsigned().items() if key != "loaded_source"
    }


def materialize_massive_requested_orders_v1(
    *,
    tensor_root: str | Path,
    inference_root: str | Path,
    output_root: str | Path,
    tensor: MassivePIT500DecisionTensorV1,
    inference: MassiveValidationInferenceArtifactV1,
    decision_origin: MassiveTypedDecisionOriginV1,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveRequestedOrdersArtifactV1:
    validate_massive_pit500_tensor_v1(root=tensor_root, tensor=tensor)
    validate_massive_validation_inference_v1(root=inference_root, artifact=inference)
    decision_origin.validate()
    if (
        inference.tensor_receipt_sha256 != tensor.receipt_sha256
        or inference.decision_session_date != tensor.decision_session_date
        or inference.seed < 0
        or tensor.decision_origin_receipt_sha256 != decision_origin.receipt_sha256
        or tensor.decision_session_date != decision_origin.decision_session_date
        or tensor.decision_at_ms != decision_origin.decision_at_ms
    ):
        raise MassiveValidationOrdersV1Error("requested orders v1 inputs differ")
    scores: dict[str, list[float]] = {security: [] for security in tensor.security_ids}
    for row in inference.rows:
        scores[row.security_id].append(row.mean)
    if any(len(values) != 4 for values in scores.values()):
        raise MassiveValidationOrdersV1Error("orders v1 require four horizons")
    ranked = tuple(
        sorted(scores, key=lambda security: (-sum(scores[security]) / 4.0, security))
    )
    selected_count = max(1, math.ceil(0.20 * len(ranked)))
    selected = ranked[:selected_count]
    row_values = []
    cash_body = {"security_id": "CASH", "requested_weight": 0.0}
    row_values.append(
        MassiveRequestedOrderRowV0(
            **cash_body, receipt_sha256=semantic_sha256(cash_body)
        )
    )
    for security_id in sorted(selected):
        body = {
            "security_id": security_id,
            "requested_weight": 1.0 / selected_count,
        }
        row_values.append(
            MassiveRequestedOrderRowV0(
                **body,
                receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
            )
        )
    rows = tuple(row_values)
    relative = (
        f"massive-finalized-v1/decision={tensor.decision_session_date}/"
        f"requested-orders-v1-{inference.setting_id}-seed{inference.seed}.json"
    )
    placeholder = MassiveRequestedOrdersArtifactV1(
        decision_session_date=tensor.decision_session_date,
        decision_at_ms=tensor.decision_at_ms,
        setting_id=inference.setting_id,
        seed=inference.seed,
        decision_origin_receipt_sha256=decision_origin.receipt_sha256,
        tensor_receipt_sha256=tensor.receipt_sha256,
        inference_receipt_sha256=inference.receipt_sha256,
        order_spec_receipt_sha256=MASSIVE_VALIDATION_ORDERS_V1_SPEC_SHA256,
        order_source_sha256=MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SHA256,
        diagnostic_fill_rule="[15:50:00,16:00:00)-qualifying-trade-vwap",
        rows=rows,
        order_inventory_sha256=semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        loaded_source=inference.loaded_source,
        panel_materialization_authorized=False,
        portfolio_evaluation_authorized=False,
        receipt_sha256="0" * 64,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(placeholder))),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_VALIDATION_ORDERS_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SCHEMA_SHA256,
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
    validate_massive_requested_orders_v1(root=output_root, artifact=result)
    return result


def validate_massive_requested_orders_v1(
    *, root: str | Path, artifact: MassiveRequestedOrdersArtifactV1
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=artifact.loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveValidationOrdersV1Error("requested orders v1 source is not JSON") from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _payload(artifact)
    ):
        raise MassiveValidationOrdersV1Error("requested orders v1 bytes differ")


__all__ = [
    "MASSIVE_VALIDATION_ORDERS_V1_DATASET",
    "MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_VALIDATION_ORDERS_V1_SOURCE_SHA256",
    "MASSIVE_VALIDATION_ORDERS_V1_SPEC_SHA256",
    "MassiveRequestedOrdersArtifactV1",
    "MassiveValidationOrdersV1Error",
    "materialize_massive_requested_orders_v1",
    "validate_massive_requested_orders_v1",
]
