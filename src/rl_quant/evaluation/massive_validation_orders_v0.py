"""Typed diagnostic requested orders for finalized Massive validation V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path

from rl_quant.features.massive_pit500_tensor_v0 import (
    MassivePIT500DecisionTensorV0,
    validate_massive_pit500_tensor_v0,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_validation_inference_v0 import (
    MassiveValidationInferenceArtifactV0,
    validate_massive_validation_inference_v0,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_VALIDATION_ORDERS_V0_SCHEMA = "rl-quant.massive-validation-requested-orders-v0"
MASSIVE_VALIDATION_ORDERS_V0_DATASET = (
    "massive-finalized-validation-requested-orders-v0"
)
MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_VALIDATION_ORDERS_V0_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "purpose": "readiness-canary-diagnostic-orders-only",
        "score": "equal-mean-across-H01-H05-H21-H63",
        "selection": "top-20-percent-at-least-one",
        "weights": "equal-long-only-plus-explicit-cash",
        "fill": "same-session-[15:50,16:00)-qualifying-trade-vwap-diagnostic",
        "position_age_input": False,
        "duration_field": False,
        "portfolio_optimization": False,
    }
)
MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_VALIDATION_ORDERS_V0_SCHEMA,
        "fields": ("security_id", "requested_weight"),
        "cash": "explicit",
    }
)


class MassiveValidationOrdersV0Error(ValueError):
    """Requested-order bytes or diagnostic semantics differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveValidationOrdersV0Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveRequestedOrderRowV0:
    security_id: str
    requested_weight: float
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not self.security_id
            or not math.isfinite(self.requested_weight)
            or not 0.0 <= self.requested_weight <= 1.0
        ):
            raise MassiveValidationOrdersV0Error("requested order row is invalid")
        _digest("requested order receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveValidationOrdersV0Error("requested order receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveRequestedOrdersArtifactV0:
    decision_session_date: str
    decision_at_ms: int
    setting_id: str
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
    schema: str = MASSIVE_VALIDATION_ORDERS_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_VALIDATION_ORDERS_V0_SCHEMA:
            raise MassiveValidationOrdersV0Error("requested orders schema drifted")
        if isinstance(self.decision_at_ms, bool) or self.decision_at_ms <= 0:
            raise MassiveValidationOrdersV0Error(
                "requested order decision time differs"
            )
        if self.diagnostic_fill_rule != "[15:50:00,16:00:00)-qualifying-trade-vwap":
            raise MassiveValidationOrdersV0Error("diagnostic fill rule drifted")
        for name in (
            "tensor_receipt_sha256",
            "inference_receipt_sha256",
            "order_spec_receipt_sha256",
            "order_source_sha256",
            "order_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.order_spec_receipt_sha256 != MASSIVE_VALIDATION_ORDERS_V0_SPEC_SHA256
            or self.order_source_sha256 != MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SHA256
        ):
            raise MassiveValidationOrdersV0Error(
                "requested order implementation drifted"
            )
        keys = tuple(row.security_id for row in self.rows)
        expected_keys = tuple(
            sorted(set(keys), key=lambda value: (value != "CASH", value))
        )
        if not keys or keys != expected_keys or keys.count("CASH") != 1:
            raise MassiveValidationOrdersV0Error("requested order inventory differs")
        for row in self.rows:
            row.validate()
        if not math.isclose(
            sum(row.requested_weight for row in self.rows), 1.0, abs_tol=1e-12
        ):
            raise MassiveValidationOrdersV0Error("requested weights are not funded")
        if self.order_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveValidationOrdersV0Error("requested order hash differs")
        if (
            self.panel_materialization_authorized
            or self.portfolio_evaluation_authorized
        ):
            raise MassiveValidationOrdersV0Error(
                "readiness orders cannot authorize performance work"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_VALIDATION_ORDERS_V0_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveValidationOrdersV0Error("requested order source differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveValidationOrdersV0Error("requested orders receipt differs")


def _payload(artifact: MassiveRequestedOrdersArtifactV0) -> dict[str, object]:
    return {
        key: value
        for key, value in artifact.unsigned().items()
        if key != "loaded_source"
    }


def materialize_massive_requested_orders_v0(
    *,
    tensor_root: str | Path,
    inference_root: str | Path,
    output_root: str | Path,
    tensor: MassivePIT500DecisionTensorV0,
    inference: MassiveValidationInferenceArtifactV0,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveRequestedOrdersArtifactV0:
    validate_massive_pit500_tensor_v0(root=tensor_root, tensor=tensor)
    validate_massive_validation_inference_v0(root=inference_root, artifact=inference)
    if (
        inference.tensor_receipt_sha256 != tensor.receipt_sha256
        or inference.decision_session_date != tensor.decision_session_date
    ):
        raise MassiveValidationOrdersV0Error("orders input authorities differ")
    scores: dict[str, list[float]] = {security: [] for security in tensor.security_ids}
    for row in inference.rows:
        scores[row.security_id].append(row.mean)
    if any(len(values) != 4 for values in scores.values()):
        raise MassiveValidationOrdersV0Error("orders require all four horizon scores")
    ranked = tuple(
        sorted(
            scores,
            key=lambda security: (-sum(scores[security]) / 4.0, security),
        )
    )
    selected_count = max(1, math.ceil(0.20 * len(ranked)))
    selected = ranked[:selected_count]
    security_weight = 1.0 / selected_count
    order_rows = []
    cash_body = {"security_id": "CASH", "requested_weight": 0.0}
    order_rows.append(
        MassiveRequestedOrderRowV0(
            **cash_body, receipt_sha256=semantic_sha256(cash_body)
        )
    )
    for security_id in sorted(selected):
        body = {"security_id": security_id, "requested_weight": security_weight}
        order_rows.append(
            MassiveRequestedOrderRowV0(**body, receipt_sha256=semantic_sha256(body))
        )
    rows = tuple(order_rows)
    relative = f"massive-finalized-v0/decision={tensor.decision_session_date}/requested-orders-{inference.setting_id}.json"
    placeholder = MassiveRequestedOrdersArtifactV0(
        decision_session_date=tensor.decision_session_date,
        decision_at_ms=tensor.decision_at_ms,
        setting_id=inference.setting_id,
        tensor_receipt_sha256=tensor.receipt_sha256,
        inference_receipt_sha256=inference.receipt_sha256,
        order_spec_receipt_sha256=MASSIVE_VALIDATION_ORDERS_V0_SPEC_SHA256,
        order_source_sha256=MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SHA256,
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
        dataset_id=MASSIVE_VALIDATION_ORDERS_V0_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SCHEMA_SHA256,
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
    validate_massive_requested_orders_v0(root=output_root, artifact=result)
    return result


def validate_massive_requested_orders_v0(
    *, root: str | Path, artifact: MassiveRequestedOrdersArtifactV0
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveValidationOrdersV0Error(
            "requested order source is not JSON"
        ) from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _payload(artifact)
    ):
        raise MassiveValidationOrdersV0Error("requested order bytes differ")


__all__ = [
    "MASSIVE_VALIDATION_ORDERS_V0_DATASET",
    "MASSIVE_VALIDATION_ORDERS_V0_SOURCE_SCHEMA_SHA256",
    "MASSIVE_VALIDATION_ORDERS_V0_SPEC_SHA256",
    "MassiveRequestedOrderRowV0",
    "MassiveRequestedOrdersArtifactV0",
    "MassiveValidationOrdersV0Error",
    "materialize_massive_requested_orders_v0",
    "validate_massive_requested_orders_v0",
]
