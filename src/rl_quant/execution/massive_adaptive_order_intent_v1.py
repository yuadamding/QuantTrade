"""Decision-close share intents for next-session adaptive execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Mapping, Sequence

from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    MassiveAdaptiveEconomicBookV1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioDecisionV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_ORDER_INTENT_V1_SCHEMA = "rl-quant.massive-adaptive-order-intent-v1"


class MassiveAdaptiveOrderIntentV1Error(ValueError):
    """A pending adaptive order is detached from its decision-close state."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOrderRowV1:
    security_id: str
    target_weight: float
    decision_mark: float
    requested_shares: float
    decision_mark_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            not self.security_id
            or not math.isfinite(self.target_weight)
            or self.target_weight < 0.0
            or not math.isfinite(self.decision_mark)
            or self.decision_mark <= 0.0
            or not math.isfinite(self.requested_shares)
            or len(self.decision_mark_receipt_sha256) != 64
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveOrderIntentV1Error("adaptive order row differs")


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOrderIntentV1:
    decision_session_date: str
    scheduled_fill_session_date: str
    rows: tuple[MassiveAdaptiveOrderRowV1, ...]
    pretrade_book_receipt_sha256: str
    portfolio_decision_receipt_sha256: str
    target_receipt_sha256: str
    row_inventory_sha256: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_ORDER_INTENT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_ORDER_INTENT_V1_SCHEMA
            or not self.decision_session_date
            or not self.scheduled_fill_session_date
            or self.scheduled_fill_session_date <= self.decision_session_date
            or tuple(row.security_id for row in self.rows)
            != tuple(sorted(set(row.security_id for row in self.rows)))
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveOrderIntentV1Error("adaptive order intent differs")
        for row in self.rows:
            row.validate()
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_order_intent_v1(
    *,
    book: MassiveAdaptiveEconomicBookV1,
    decision: MassiveAdaptivePortfolioDecisionV1,
    scheduled_fill_session_date: str,
    decision_marks: Mapping[str, float],
    decision_mark_receipts: Mapping[str, str],
) -> MassiveAdaptiveOrderIntentV1:
    """Convert target weights to shares using only decision-close marks."""

    book.validate()
    decision.validate()
    if decision.decision_id != book.decision_session_date:
        raise MassiveAdaptiveOrderIntentV1Error("decision and book dates differ")
    existing = book.shares_by_security()
    rows: list[MassiveAdaptiveOrderRowV1] = []
    for security_id, target_weight in zip(
        decision.security_ids, decision.target_weights, strict=True
    ):
        if target_weight <= 1.0e-12 and existing.get(security_id, 0.0) <= 1.0e-12:
            continue
        mark = float(decision_marks[security_id])
        requested = target_weight * book.marked_equity / mark - existing.get(
            security_id, 0.0
        )
        body = {
            "security_id": security_id,
            "target_weight": float(target_weight),
            "decision_mark": mark,
            "requested_shares": float(requested),
            "decision_mark_receipt_sha256": decision_mark_receipts[security_id],
        }
        row = MassiveAdaptiveOrderRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        rows.append(row)
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "schema": MASSIVE_ADAPTIVE_ORDER_INTENT_V1_SCHEMA,
        "decision_session_date": book.decision_session_date,
        "scheduled_fill_session_date": scheduled_fill_session_date,
        "rows": tuple(rows),
        "pretrade_book_receipt_sha256": book.semantic_receipt_sha256,
        "portfolio_decision_receipt_sha256": decision.semantic_receipt_sha256,
        "target_receipt_sha256": semantic_sha256(
            (decision.security_ids, decision.target_weights)
        ),
        "row_inventory_sha256": row_inventory,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveOrderIntentV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def build_massive_adaptive_target_order_intent_v1(
    *,
    decision_session_date: str,
    scheduled_fill_session_date: str,
    book: MassiveAdaptiveEconomicBookV1,
    security_ids: Sequence[str],
    target_weights: Sequence[float],
    target_receipt_sha256: str,
    decision_marks: Mapping[str, float],
    decision_mark_receipts: Mapping[str, str],
) -> MassiveAdaptiveOrderIntentV1:
    """Package-owned path for a fixed benchmark target using the same execution."""

    if decision_session_date != book.decision_session_date:
        raise MassiveAdaptiveOrderIntentV1Error("benchmark decision and book differ")
    ordered = tuple(security_ids)
    weights = tuple(float(value) for value in target_weights)
    if ordered != tuple(sorted(set(ordered))) or len(ordered) != len(weights):
        raise MassiveAdaptiveOrderIntentV1Error("benchmark target support differs")
    existing = book.shares_by_security()
    rows: list[MassiveAdaptiveOrderRowV1] = []
    for security_id, target_weight in zip(ordered, weights, strict=True):
        if target_weight <= 1.0e-12 and existing.get(security_id, 0.0) <= 1.0e-12:
            continue
        mark = float(decision_marks[security_id])
        body = {
            "security_id": security_id,
            "target_weight": target_weight,
            "decision_mark": mark,
            "requested_shares": target_weight * book.marked_equity / mark
            - existing.get(security_id, 0.0),
            "decision_mark_receipt_sha256": decision_mark_receipts[security_id],
        }
        rows.append(
            MassiveAdaptiveOrderRowV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "schema": MASSIVE_ADAPTIVE_ORDER_INTENT_V1_SCHEMA,
        "decision_session_date": decision_session_date,
        "scheduled_fill_session_date": scheduled_fill_session_date,
        "rows": tuple(rows),
        "pretrade_book_receipt_sha256": book.semantic_receipt_sha256,
        "portfolio_decision_receipt_sha256": target_receipt_sha256,
        "target_receipt_sha256": target_receipt_sha256,
        "row_inventory_sha256": row_inventory,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveOrderIntentV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveOrderIntentV1",
    "MassiveAdaptiveOrderIntentV1Error",
    "MassiveAdaptiveOrderRowV1",
    "build_massive_adaptive_order_intent_v1",
    "build_massive_adaptive_target_order_intent_v1",
]
