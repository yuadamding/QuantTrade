"""Continuous cash/share book for the no-duration adaptive profit path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Mapping

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_ECONOMIC_BOOK_V1_SCHEMA = "rl-quant.massive-adaptive-economic-book-v1"


class MassiveAdaptiveEconomicBookV1Error(ValueError):
    """The adaptive cash/share ledger does not reconcile."""


def _finite(name: str, value: object, *, nonnegative: bool = False) -> float:
    observed = float(value)  # type: ignore[arg-type]
    if not math.isfinite(observed) or (nonnegative and observed < 0.0):
        raise MassiveAdaptiveEconomicBookV1Error(f"{name} is invalid")
    return observed


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveEconomicBookV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveHoldingV1:
    security_id: str
    issuer_id: str
    shares: float
    last_mark: float
    market_value: float
    identity_receipt_sha256: str
    mark_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.security_id or not self.issuer_id:
            raise MassiveAdaptiveEconomicBookV1Error("holding identity is absent")
        shares = _finite("shares", self.shares, nonnegative=True)
        mark = _finite("last mark", self.last_mark, nonnegative=True)
        value = _finite("market value", self.market_value, nonnegative=True)
        if (
            shares <= 0.0
            or mark <= 0.0
            or abs(value - shares * mark) > max(1.0e-8, value * 1.0e-12)
        ):
            raise MassiveAdaptiveEconomicBookV1Error("holding value does not reconcile")
        for name in (
            "identity_receipt_sha256",
            "mark_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveAdaptiveEconomicBookV1Error("holding receipt differs")


def build_massive_adaptive_holding_v1(
    *,
    security_id: str,
    issuer_id: str,
    shares: float,
    mark: float,
    identity_receipt_sha256: str,
    mark_receipt_sha256: str,
) -> MassiveAdaptiveHoldingV1:
    body = {
        "security_id": security_id,
        "issuer_id": issuer_id,
        "shares": float(shares),
        "last_mark": float(mark),
        "market_value": float(shares) * float(mark),
        "identity_receipt_sha256": identity_receipt_sha256,
        "mark_receipt_sha256": mark_receipt_sha256,
    }
    result = MassiveAdaptiveHoldingV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveEconomicBookV1:
    decision_session_date: str
    cash: float
    holdings: tuple[MassiveAdaptiveHoldingV1, ...]
    marked_equity: float
    high_water_mark: float
    source_state_receipt_sha256: str
    holding_inventory_sha256: str
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_ECONOMIC_BOOK_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        cash = _finite("cash", self.cash, nonnegative=True)
        equity = _finite("marked equity", self.marked_equity, nonnegative=True)
        high_water = _finite("high-water mark", self.high_water_mark, nonnegative=True)
        if (
            self.schema != MASSIVE_ADAPTIVE_ECONOMIC_BOOK_V1_SCHEMA
            or not self.decision_session_date
            or tuple(row.security_id for row in self.holdings)
            != tuple(sorted(set(row.security_id for row in self.holdings)))
            or high_water + 1.0e-10 < equity
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
        ):
            raise MassiveAdaptiveEconomicBookV1Error("economic book identity differs")
        for holding in self.holdings:
            holding.validate()
        expected_equity = cash + sum(row.market_value for row in self.holdings)
        if abs(equity - expected_equity) > max(1.0e-8, equity * 1.0e-12):
            raise MassiveAdaptiveEconomicBookV1Error(
                "cash and holdings do not equal equity"
            )
        if self.holding_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.holdings)
        ):
            raise MassiveAdaptiveEconomicBookV1Error("holding inventory differs")
        for name in (
            "source_state_receipt_sha256",
            "holding_inventory_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveAdaptiveEconomicBookV1Error("economic book receipt differs")
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())

    def shares_by_security(self) -> dict[str, float]:
        self.validate()
        return {row.security_id: row.shares for row in self.holdings}

    def weights(self, security_ids: tuple[str, ...]) -> tuple[float, ...]:
        self.validate()
        values = {row.security_id: row.market_value for row in self.holdings}
        if self.marked_equity <= 0.0:
            return (0.0,) * len(security_ids)
        return tuple(
            values.get(security_id, 0.0) / self.marked_equity
            for security_id in security_ids
        )


def build_massive_adaptive_economic_book_v1(
    *,
    decision_session_date: str,
    cash: float,
    holdings: tuple[MassiveAdaptiveHoldingV1, ...],
    prior_high_water_mark: float,
    source_state_receipt_sha256: str,
) -> MassiveAdaptiveEconomicBookV1:
    ordered = tuple(sorted(holdings, key=lambda row: row.security_id))
    equity = float(cash) + sum(row.market_value for row in ordered)
    body = {
        "schema": MASSIVE_ADAPTIVE_ECONOMIC_BOOK_V1_SCHEMA,
        "decision_session_date": decision_session_date,
        "cash": float(cash),
        "holdings": ordered,
        "marked_equity": equity,
        "high_water_mark": max(float(prior_high_water_mark), equity),
        "source_state_receipt_sha256": source_state_receipt_sha256,
        "holding_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveEconomicBookV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def initial_massive_adaptive_economic_book_v1(
    *, decision_session_date: str, capital: float, initialization_receipt_sha256: str
) -> MassiveAdaptiveEconomicBookV1:
    return build_massive_adaptive_economic_book_v1(
        decision_session_date=decision_session_date,
        cash=capital,
        holdings=(),
        prior_high_water_mark=capital,
        source_state_receipt_sha256=initialization_receipt_sha256,
    )


def revalue_massive_adaptive_economic_book_v1(
    *,
    book: MassiveAdaptiveEconomicBookV1,
    decision_session_date: str,
    shares: Mapping[str, float],
    cash: float,
    issuer_ids: Mapping[str, str],
    marks: Mapping[str, float],
    identity_receipts: Mapping[str, str],
    mark_receipts: Mapping[str, str],
    source_state_receipt_sha256: str,
) -> MassiveAdaptiveEconomicBookV1:
    book.validate()
    holdings = tuple(
        build_massive_adaptive_holding_v1(
            security_id=security_id,
            issuer_id=issuer_ids[security_id],
            shares=quantity,
            mark=marks[security_id],
            identity_receipt_sha256=identity_receipts[security_id],
            mark_receipt_sha256=mark_receipts[security_id],
        )
        for security_id, quantity in sorted(shares.items())
        if quantity > 1.0e-12
    )
    return build_massive_adaptive_economic_book_v1(
        decision_session_date=decision_session_date,
        cash=cash,
        holdings=holdings,
        prior_high_water_mark=book.high_water_mark,
        source_state_receipt_sha256=source_state_receipt_sha256,
    )


__all__ = [
    "MassiveAdaptiveEconomicBookV1",
    "MassiveAdaptiveEconomicBookV1Error",
    "MassiveAdaptiveHoldingV1",
    "build_massive_adaptive_economic_book_v1",
    "build_massive_adaptive_holding_v1",
    "initial_massive_adaptive_economic_book_v1",
    "revalue_massive_adaptive_economic_book_v1",
]
