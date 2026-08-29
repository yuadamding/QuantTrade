"""Source-bound next-morning fills and next-close book transition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    MassiveAdaptiveEconomicBookV1,
    revalue_massive_adaptive_economic_book_v1,
)
from rl_quant.execution.massive_adaptive_order_intent_v1 import (
    MassiveAdaptiveOrderIntentV1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import (
    MassiveAdaptiveFillSourceV1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_EXECUTION_RESULT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-execution-result-v1"
)


class MassiveAdaptiveExecutionResultV1Error(ValueError):
    """Requested, filled, cash, and marked quantities do not reconcile."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveExecutionRowV1:
    security_id: str
    requested_shares: float
    filled_shares: float
    unfilled_shares: float
    fill_price: float
    executed_notional: float
    transaction_cost: float
    fill_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        values = (
            self.requested_shares,
            self.filled_shares,
            self.unfilled_shares,
            self.fill_price,
            self.executed_notional,
            self.transaction_cost,
        )
        if (
            not self.security_id
            or any(not math.isfinite(value) for value in values)
            or self.fill_price < 0.0
            or self.executed_notional < 0.0
            or self.transaction_cost < 0.0
            or abs(self.requested_shares - self.filled_shares - self.unfilled_shares)
            > 1.0e-10
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveExecutionResultV1Error(
                "adaptive execution row differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveExecutionResultV1:
    decision_session_date: str
    fill_session_date: str
    rows: tuple[MassiveAdaptiveExecutionRowV1, ...]
    order_intent_receipt_sha256: str
    pretrade_book_receipt_sha256: str
    posttrade_book: MassiveAdaptiveEconomicBookV1
    gross_traded_notional: float
    total_transaction_cost: float
    unfilled_absolute_shares: float
    fill_source_receipt_sha256: str
    daily_input_receipt_sha256: str
    identity_authority_receipt_sha256: str
    event_inventory_receipt_sha256: str
    row_inventory_sha256: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_EXECUTION_RESULT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_EXECUTION_RESULT_V1_SCHEMA
            or not self.decision_session_date
            or self.fill_session_date <= self.decision_session_date
            or tuple(row.security_id for row in self.rows)
            != tuple(sorted(set(row.security_id for row in self.rows)))
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or not math.isfinite(self.gross_traded_notional)
            or not math.isfinite(self.total_transaction_cost)
            or not math.isfinite(self.unfilled_absolute_shares)
            or min(
                self.gross_traded_notional,
                self.total_transaction_cost,
                self.unfilled_absolute_shares,
            )
            < 0.0
            or self.posttrade_book.decision_session_date != self.fill_session_date
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveExecutionResultV1Error(
                "adaptive execution result differs"
            )
        for row in self.rows:
            row.validate()
        self.posttrade_book.validate()
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def execute_massive_adaptive_order_intent_v1(
    *,
    order_intent: MassiveAdaptiveOrderIntentV1,
    book: MassiveAdaptiveEconomicBookV1,
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    transaction_cost_basis_points: float = 20.0,
    maximum_fill_participation: float = 0.02,
) -> MassiveAdaptiveExecutionResultV1:
    """Fill one pending intent, charge costs, and mark the resulting book."""

    order_intent.validate()
    book.validate()
    fill_source.validate()
    daily_input_authority.validate()
    identity_authority.validate()
    if (
        order_intent.pretrade_book_receipt_sha256 != book.semantic_receipt_sha256
        or order_intent.decision_session_date != book.decision_session_date
        or fill_source.daily_input_authority_semantic_receipt_sha256
        != daily_input_authority.semantic_receipt_sha256
        or not math.isfinite(transaction_cost_basis_points)
        or transaction_cost_basis_points < 0.0
        or not 0.0 < maximum_fill_participation <= 1.0
    ):
        raise MassiveAdaptiveExecutionResultV1Error("execution roots or policy differ")
    masters = {row.security_id: row for row in identity_authority.security_master}
    existing = book.shares_by_security()
    tentative: dict[str, float] = {}
    fill_rows = {}
    for order in order_intent.rows:
        if order.security_id not in masters:
            raise MassiveAdaptiveExecutionResultV1Error("order identity is unknown")
        fill = fill_source.row(
            session_date=order_intent.scheduled_fill_session_date,
            security_id=order.security_id,
        )
        fill_rows[order.security_id] = fill
        capacity = maximum_fill_participation * fill.qualifying_share_volume
        requested = order.requested_shares
        quantity = (
            0.0
            if not fill.valid
            else math.copysign(min(abs(requested), capacity), requested)
        )
        if quantity < 0.0:
            quantity = -min(abs(quantity), existing.get(order.security_id, 0.0))
        tentative[order.security_id] = quantity

    cost_rate = transaction_cost_basis_points / 10_000.0
    sell_cash = sum(
        -quantity * fill_rows[security_id].fill_vwap * (1.0 - cost_rate)
        for security_id, quantity in tentative.items()
        if quantity < 0.0
    )
    available_cash = book.cash + sell_cash
    requested_buy_cash = sum(
        quantity * fill_rows[security_id].fill_vwap * (1.0 + cost_rate)
        for security_id, quantity in tentative.items()
        if quantity > 0.0
    )
    buy_scale = (
        1.0
        if requested_buy_cash <= available_cash + 1.0e-10
        else available_cash / requested_buy_cash
    )
    executed = {
        security_id: quantity * buy_scale if quantity > 0.0 else quantity
        for security_id, quantity in tentative.items()
    }
    rows: list[MassiveAdaptiveExecutionRowV1] = []
    gross_notional = 0.0
    total_cost = 0.0
    shares = dict(existing)
    cash = book.cash
    for order in order_intent.rows:
        security_id = order.security_id
        fill = fill_rows[security_id]
        quantity = executed[security_id]
        notional = abs(quantity) * fill.fill_vwap
        cost = notional * cost_rate
        cash -= quantity * fill.fill_vwap + cost
        shares[security_id] = shares.get(security_id, 0.0) + quantity
        if shares[security_id] <= 1.0e-12:
            shares.pop(security_id, None)
        gross_notional += notional
        total_cost += cost
        body = {
            "security_id": security_id,
            "requested_shares": order.requested_shares,
            "filled_shares": quantity,
            "unfilled_shares": order.requested_shares - quantity,
            "fill_price": fill.fill_vwap if fill.valid else 0.0,
            "executed_notional": notional,
            "transaction_cost": cost,
            "fill_row_receipt_sha256": fill.receipt_sha256,
        }
        row = MassiveAdaptiveExecutionRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        rows.append(row)
    if cash < -1.0e-7:
        raise MassiveAdaptiveExecutionResultV1Error("execution spent unavailable cash")
    cash = max(0.0, cash)
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    marks: dict[str, float] = {}
    mark_receipts: dict[str, str] = {}
    issuer_ids: dict[str, str] = {}
    identity_receipts: dict[str, str] = {}
    for security_id in shares:
        daily = daily_input_authority.row(
            session_date=order_intent.scheduled_fill_session_date,
            security_id=security_id,
        )
        if not daily.bars_valid[close_index] or daily.bars_values[close_index] <= 0.0:
            raise MassiveAdaptiveExecutionResultV1Error(
                "held security lacks a source-qualified close mark"
            )
        master = masters[security_id]
        marks[security_id] = float(daily.bars_values[close_index])
        mark_receipts[security_id] = daily.receipt_sha256
        issuer_ids[security_id] = master.issuer_id
        identity_receipts[security_id] = master.identity_source_receipt_sha256
    event_inventory = semantic_sha256(())
    source_state = semantic_sha256(
        {
            "order": order_intent.semantic_receipt_sha256,
            "fill_source": fill_source.semantic_receipt_sha256,
            "daily_input": daily_input_authority.semantic_receipt_sha256,
            "identity": identity_authority.receipt_sha256,
            "events": event_inventory,
        }
    )
    posttrade = revalue_massive_adaptive_economic_book_v1(
        book=book,
        decision_session_date=order_intent.scheduled_fill_session_date,
        shares=shares,
        cash=cash,
        issuer_ids=issuer_ids,
        marks=marks,
        identity_receipts=identity_receipts,
        mark_receipts=mark_receipts,
        source_state_receipt_sha256=source_state,
    )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "schema": MASSIVE_ADAPTIVE_EXECUTION_RESULT_V1_SCHEMA,
        "decision_session_date": order_intent.decision_session_date,
        "fill_session_date": order_intent.scheduled_fill_session_date,
        "rows": tuple(rows),
        "order_intent_receipt_sha256": order_intent.semantic_receipt_sha256,
        "pretrade_book_receipt_sha256": book.semantic_receipt_sha256,
        "posttrade_book": posttrade,
        "gross_traded_notional": gross_notional,
        "total_transaction_cost": total_cost,
        "unfilled_absolute_shares": sum(abs(row.unfilled_shares) for row in rows),
        "fill_source_receipt_sha256": fill_source.semantic_receipt_sha256,
        "daily_input_receipt_sha256": daily_input_authority.semantic_receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "event_inventory_receipt_sha256": event_inventory,
        "row_inventory_sha256": row_inventory,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveExecutionResultV1(
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
    "MassiveAdaptiveExecutionResultV1",
    "MassiveAdaptiveExecutionResultV1Error",
    "MassiveAdaptiveExecutionRowV1",
    "execute_massive_adaptive_order_intent_v1",
]
