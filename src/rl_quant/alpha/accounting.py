"""Corporate-action-complete economic position and target accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from rl_quant.alpha.contracts import (
    CorporateActionKind,
    CorporateActionRecord,
    PITAlphaDataError,
    TerminalEventKind,
    TerminalEventRecord,
)


@dataclass(frozen=True, slots=True)
class PositionHolding:
    security_id: str
    shares: float

    def validate(self) -> None:
        if not self.security_id or self.security_id != self.security_id.strip():
            raise PITAlphaDataError("position security ID is invalid")
        if not math.isfinite(self.shares) or self.shares <= 0.0:
            raise PITAlphaDataError("position shares must be finite and positive")


@dataclass(frozen=True, slots=True)
class EconomicPosition:
    holdings: tuple[PositionHolding, ...]
    cash: float = 0.0
    applied_event_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if not math.isfinite(self.cash):
            raise PITAlphaDataError("position cash must be finite")
        security_ids: list[str] = []
        for row in self.holdings:
            row.validate()
            security_ids.append(row.security_id)
        if security_ids != sorted(security_ids) or len(set(security_ids)) != len(security_ids):
            raise PITAlphaDataError("position holdings must be sorted and unique")
        if (
            tuple(sorted(self.applied_event_ids)) != self.applied_event_ids
            or len(set(self.applied_event_ids)) != len(self.applied_event_ids)
        ):
            raise PITAlphaDataError("applied economic events must be sorted and unique")

    @classmethod
    def from_mapping(
        cls,
        holdings: Mapping[str, float],
        *,
        cash: float = 0.0,
        applied_event_ids: Sequence[str] = (),
    ) -> "EconomicPosition":
        value = cls(
            holdings=tuple(
                PositionHolding(security_id, float(shares))
                for security_id, shares in sorted(holdings.items())
                if float(shares) != 0.0
            ),
            cash=float(cash),
            applied_event_ids=tuple(sorted(applied_event_ids)),
        )
        value.validate()
        return value

    def as_mapping(self) -> dict[str, float]:
        return {row.security_id: row.shares for row in self.holdings}


def apply_corporate_action(
    position: EconomicPosition,
    event: CorporateActionRecord | TerminalEventRecord,
) -> EconomicPosition:
    """Apply one economic event exactly once to permanent-ID holdings."""

    position.validate()
    event.validate()
    if event.event_id in position.applied_event_ids:
        raise PITAlphaDataError("corporate action would be booked more than once")
    holdings = position.as_mapping()
    shares = holdings.get(event.security_id, 0.0)
    if shares <= 0.0:
        raise PITAlphaDataError("corporate action does not match a held security")
    cash = position.cash

    if isinstance(event, TerminalEventRecord):
        holdings.pop(event.security_id)
        if event.kind is TerminalEventKind.MERGER_STOCK:
            assert event.successor_security_id is not None
            holdings[event.successor_security_id] = (
                holdings.get(event.successor_security_id, 0.0)
                + shares * event.successor_ratio
            )
            cash += shares * event.cash_per_share
        elif event.kind is not TerminalEventKind.WORTHLESS:
            cash += shares * event.cash_per_share
    elif event.kind in {CorporateActionKind.SPLIT, CorporateActionKind.REVERSE_SPLIT}:
        holdings[event.security_id] = shares * event.share_ratio
    elif event.kind in {
        CorporateActionKind.CASH_DIVIDEND,
        CorporateActionKind.SPECIAL_DIVIDEND,
        CorporateActionKind.RETURN_OF_CAPITAL,
    }:
        cash += shares * event.cash_per_share
    elif event.kind is CorporateActionKind.SPIN_OFF:
        assert event.successor_security_id is not None
        holdings[event.successor_security_id] = (
            holdings.get(event.successor_security_id, 0.0)
            + shares * event.successor_ratio
        )
        cash += shares * event.cash_per_share
    elif event.kind is CorporateActionKind.MERGER_STOCK:
        assert event.successor_security_id is not None
        affected = shares * event.affected_fraction
        holdings[event.security_id] = shares - affected
        holdings[event.successor_security_id] = (
            holdings.get(event.successor_security_id, 0.0)
            + affected * event.successor_ratio
        )
        cash += affected * event.cash_per_share
    elif event.kind in {CorporateActionKind.MERGER_CASH, CorporateActionKind.TENDER_OFFER}:
        affected = shares * event.affected_fraction
        holdings[event.security_id] = shares - affected
        cash += affected * event.cash_per_share
    elif event.kind is CorporateActionKind.RIGHTS_DISTRIBUTION:
        cash += shares * event.cash_per_share
        if event.successor_ratio > 0.0:
            assert event.successor_security_id is not None
            holdings[event.successor_security_id] = (
                holdings.get(event.successor_security_id, 0.0)
                + shares * event.successor_ratio
            )
    elif event.kind is not CorporateActionKind.TICKER_EXCHANGE_CHANGE:
        raise PITAlphaDataError("corporate action has no accounting implementation")

    cleaned = {
        security_id: remaining
        for security_id, remaining in holdings.items()
        if remaining > 1e-15
    }
    return EconomicPosition.from_mapping(
        cleaned,
        cash=cash,
        applied_event_ids=(*position.applied_event_ids, event.event_id),
    )


def apply_cash_return(position: EconomicPosition, one_step_return: float) -> EconomicPosition:
    """Accrue the declared causal cash return without changing risky holdings."""

    position.validate()
    if not math.isfinite(one_step_return) or one_step_return <= -1.0:
        raise PITAlphaDataError("cash return is outside its economic domain")
    return EconomicPosition.from_mapping(
        position.as_mapping(),
        cash=position.cash * (1.0 + one_step_return),
        applied_event_ids=position.applied_event_ids,
    )


def mark_position(position: EconomicPosition, marks: Mapping[str, float]) -> float:
    """Mark every risky holding; missing marks never become zero returns."""

    position.validate()
    total = position.cash
    for holding in position.holdings:
        if holding.security_id not in marks:
            raise PITAlphaDataError(
                f"held security {holding.security_id!r} has no economic mark"
            )
        mark = float(marks[holding.security_id])
        if not math.isfinite(mark) or mark < 0.0:
            raise PITAlphaDataError("economic marks must be finite and nonnegative")
        total += holding.shares * mark
    if not math.isfinite(total) or total < 0.0:
        raise PITAlphaDataError("marked economic value is invalid")
    return total


@dataclass(frozen=True, slots=True)
class EconomicValuePoint:
    session_index: int
    economic_at_ms: int
    available_at_ms: int
    value: float
    mark_kind: str
    terminal: bool = False

    def validate(self) -> None:
        if isinstance(self.session_index, bool) or not isinstance(self.session_index, int) or self.session_index < 0:
            raise PITAlphaDataError("economic session index must be nonnegative")
        for name in ("economic_at_ms", "available_at_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PITAlphaDataError(f"{name} must be nonnegative epoch milliseconds")
        if self.available_at_ms < self.economic_at_ms:
            raise PITAlphaDataError("an economic value cannot be available before it exists")
        if not math.isfinite(self.value) or self.value < 0.0:
            raise PITAlphaDataError("economic value must be finite and nonnegative")
        if self.mark_kind not in {"market", "validated-fallback", "terminal-disposition"}:
            raise PITAlphaDataError("economic mark kind is unsupported")
        if self.terminal != (self.mark_kind == "terminal-disposition"):
            raise PITAlphaDataError("terminal points must use terminal-disposition marks")


@dataclass(frozen=True, slots=True)
class TotalReturnTarget:
    fill_session_index: int
    horizon_sessions: int
    start_value: float
    end_value: float
    simple_return: float
    log_return: float | None
    terminal_zero_value: bool

    def validate(self) -> None:
        if self.fill_session_index < 0 or self.horizon_sessions <= 0:
            raise PITAlphaDataError("post-fill target indices are invalid")
        if self.start_value <= 0.0 or self.end_value < 0.0:
            raise PITAlphaDataError("post-fill target values are invalid")
        expected_simple = self.end_value / self.start_value - 1.0
        if not math.isclose(self.simple_return, expected_simple, rel_tol=0.0, abs_tol=1e-12):
            raise PITAlphaDataError("simple return does not reconcile to economic values")
        if self.end_value == 0.0:
            if not self.terminal_zero_value or self.log_return is not None:
                raise PITAlphaDataError("total loss must remain explicit rather than fabricated finite")
        else:
            expected_log = math.log(self.end_value / self.start_value)
            if self.terminal_zero_value or self.log_return is None or not math.isclose(
                self.log_return, expected_log, rel_tol=0.0, abs_tol=1e-12
            ):
                raise PITAlphaDataError("log return does not reconcile to economic values")


def compute_post_fill_total_return(
    points: Sequence[EconomicValuePoint],
    *,
    fill_session_index: int,
    horizon_sessions: int,
) -> TotalReturnTarget:
    """Compute a target beginning at the actual fill, with terminal carry."""

    if fill_session_index < 0 or horizon_sessions <= 0:
        raise PITAlphaDataError("fill and horizon must be positive session coordinates")
    by_session: dict[int, EconomicValuePoint] = {}
    terminal_seen = False
    terminal_value: float | None = None
    previous_time = -1
    for point in points:
        point.validate()
        if point.session_index in by_session or point.economic_at_ms <= previous_time:
            raise PITAlphaDataError("economic value path must be unique and chronological")
        if terminal_seen and (
            not point.terminal
            or point.value != terminal_value
            or point.mark_kind != "terminal-disposition"
        ):
            raise PITAlphaDataError("post-terminal economic value must be carried exactly")
        by_session[point.session_index] = point
        previous_time = point.economic_at_ms
        if point.terminal:
            terminal_seen = True
            terminal_value = point.value
    end_index = fill_session_index + horizon_sessions
    required = range(fill_session_index, end_index + 1)
    missing = [index for index in required if index not in by_session]
    if missing:
        raise PITAlphaDataError(
            "target requires a complete economic path or explicit terminal carry"
        )
    start = by_session[fill_session_index].value
    end = by_session[end_index].value
    if start <= 0.0:
        raise PITAlphaDataError("fill-time economic value must be positive")
    result = TotalReturnTarget(
        fill_session_index=fill_session_index,
        horizon_sessions=horizon_sessions,
        start_value=start,
        end_value=end,
        simple_return=end / start - 1.0,
        log_return=None if end == 0.0 else math.log(end / start),
        terminal_zero_value=end == 0.0,
    )
    result.validate()
    return result


__all__ = [
    "EconomicPosition",
    "EconomicValuePoint",
    "PositionHolding",
    "TotalReturnTarget",
    "apply_corporate_action",
    "apply_cash_return",
    "compute_post_fill_total_return",
    "mark_position",
]
