"""Structural 30-sleeve comparator for the Hold-30 mechanism screen.

The state owns thirty independent sleeve books. Exactly one sleeve reaches its
scheduled review each session; the other twenty-nine remain locked except for
mandatory membership, availability, and risk repairs. Sleeve weights are
fractions of total portfolio NAV, so their sums may drift away from ``1/30``
and are never recapitalized.

This module intentionally owns no economic cohort-age ledger. It emits one
cross-netted asset delta, allowing the shared cohort ledger to preserve age for
same-name renewals and for the unexecuted remainder of cap-censored reviews.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_quant.execution.hold30 import (
    HOLD30_MAX_DISCRETIONARY_TURNOVER,
    HOLD30_MAX_STOCK_WEIGHT,
    capped_waterfill,
    centered_benchmark_tilt,
)


HOLD30_SLEEVE_COUNT = 30


def _finite_float(name: str, value: torch.Tensor, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point {ndim}-D tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _matrix_like(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = _finite_float(name, value, 2)
    if value.shape != reference.shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} must match {tuple(reference.shape)}")
    return value


def _mask_like(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool tensor with shape {tuple(reference.shape)}")
    return value


def _vector_like(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.shape != reference.shape[:1] or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite [batch] tensor")
    return value


def _one_way_turnover(delta: torch.Tensor) -> torch.Tensor:
    return 0.5 * delta.abs().sum(dim=-1)


@dataclass(frozen=True)
class Hold30SleeveSnapshot:
    """Restart-complete H3 economic state, excluding the external cohort ledger."""

    books: torch.Tensor
    session_index: int
    last_review_session: torch.Tensor
    review_count: torch.Tensor
    cash_index: int


@dataclass(frozen=True)
class Hold30SleeveRepair:
    """Cause-separated mandatory sleeve deltas, each shaped ``[B,30,A]``."""

    membership_forced_delta: torch.Tensor
    availability_forced_delta: torch.Tensor
    risk_forced_delta: torch.Tensor

    @property
    def total_sleeve_delta(self) -> torch.Tensor:
        return self.membership_forced_delta + self.availability_forced_delta + self.risk_forced_delta

    @property
    def aggregate_delta(self) -> torch.Tensor:
        return self.total_sleeve_delta.sum(dim=1)


@dataclass(frozen=True)
class Hold30SleeveReview:
    """One scheduled H3 review and its single cross-netted executable order."""

    session_index: int
    maturing_sleeve: int
    review_age: int
    sleeve_nav_before: torch.Tensor
    entry_direction: torch.Tensor
    residual_asset_capacity: torch.Tensor
    residual_risky_capacity: torch.Tensor
    target_sleeve: torch.Tensor
    requested_sleeve_delta: torch.Tensor
    constructed_sleeve_delta: torch.Tensor
    requested_delta: torch.Tensor
    constructed_delta: torch.Tensor
    requested_turnover: torch.Tensor
    constructed_turnover: torch.Tensor
    same_name_cross_net_notional: torch.Tensor
    unallocated_sleeve_cash: torch.Tensor
    maturity_cap_censored: torch.Tensor


class Hold30SleeveState:
    """Mutable economic state for the staggered 30-sleeve H3 comparator."""

    def __init__(
        self,
        books: torch.Tensor,
        *,
        session_index: int = 0,
        last_review_session: torch.Tensor | None = None,
        review_count: torch.Tensor | None = None,
        cash_index: int = 0,
    ) -> None:
        books = _finite_float("books", books, 3)
        if books.shape[1] != HOLD30_SLEEVE_COUNT:
            raise ValueError(f"books must contain exactly {HOLD30_SLEEVE_COUNT} sleeves")
        if books.shape[2] < 1 or not 0 <= int(cash_index) < books.shape[2]:
            raise ValueError("cash_index is outside the asset axis")
        if bool((books < -1e-12).any()):
            raise ValueError("sleeve books cannot be negative")
        if isinstance(session_index, bool) or int(session_index) < 0:
            raise ValueError("session_index must be a nonnegative integer")
        self.books = books
        self.session_index = int(session_index)
        self.cash_index = int(cash_index)
        # Sleeve s first reviews at session s. Treat its preceding structural
        # review as s-30 so every first and subsequent review age is exactly 30.
        default_last = torch.arange(HOLD30_SLEEVE_COUNT, device=books.device, dtype=torch.int64) - 30
        default_count = torch.zeros(HOLD30_SLEEVE_COUNT, device=books.device, dtype=torch.int64)
        self.last_review_session = self._phase_vector(
            "last_review_session", default_last if last_review_session is None else last_review_session
        )
        self.review_count = self._phase_vector(
            "review_count", default_count if review_count is None else review_count
        )
        self._assert_reconciled()

    def _phase_vector(self, name: str, value: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(value, device=self.books.device)
        if value.shape != (HOLD30_SLEEVE_COUNT,) or value.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"{name} must be an integer [{HOLD30_SLEEVE_COUNT}] tensor")
        return value.to(dtype=torch.int64)

    @classmethod
    def from_portfolio(cls, weights: torch.Tensor, *, cash_index: int = 0) -> "Hold30SleeveState":
        """Split one normalized initial book equally across all fixed phases."""

        weights = _finite_float("weights", weights, 2)
        if bool((weights < -1e-12).any()):
            raise ValueError("initial weights cannot be negative")
        total = weights.sum(dim=-1)
        if not bool(torch.allclose(total, torch.ones_like(total), rtol=0.0, atol=1e-6)):
            raise ValueError("initial portfolio weights must sum to one")
        books = weights.unsqueeze(1).expand(-1, HOLD30_SLEEVE_COUNT, -1).clone()
        books = books / float(HOLD30_SLEEVE_COUNT)
        return cls(books, cash_index=cash_index)

    @classmethod
    def from_snapshot(cls, snapshot: Hold30SleeveSnapshot) -> "Hold30SleeveState":
        """Restore a numerically identical state from a captured receipt."""

        if not isinstance(snapshot, Hold30SleeveSnapshot):
            raise TypeError("snapshot must be a Hold30SleeveSnapshot")
        return cls(
            snapshot.books.clone(),
            session_index=snapshot.session_index,
            last_review_session=snapshot.last_review_session.clone(),
            review_count=snapshot.review_count.clone(),
            cash_index=snapshot.cash_index,
        )

    def capture(self, *, detach: bool = False) -> Hold30SleeveSnapshot:
        """Capture all sleeve/phase state; optionally sever the autograd graph."""

        books = self.books.detach().clone() if detach else self.books.clone()
        return Hold30SleeveSnapshot(
            books=books,
            session_index=self.session_index,
            last_review_session=self.last_review_session.detach().clone(),
            review_count=self.review_count.detach().clone(),
            cash_index=self.cash_index,
        )

    def restore_(self, snapshot: Hold30SleeveSnapshot) -> "Hold30SleeveState":
        """Replace this instance with a validated captured state."""

        restored = self.from_snapshot(snapshot)
        self.books = restored.books
        self.session_index = restored.session_index
        self.last_review_session = restored.last_review_session
        self.review_count = restored.review_count
        self.cash_index = restored.cash_index
        return self

    def detach_(self) -> "Hold30SleeveState":
        """Detach economic tensor state at a computational, not economic, boundary."""

        self.books = self.books.detach()
        self.last_review_session = self.last_review_session.detach()
        self.review_count = self.review_count.detach()
        return self

    @property
    def maturing_sleeve(self) -> int:
        return self.session_index % HOLD30_SLEEVE_COUNT

    def aggregate_weights(self) -> torch.Tensor:
        return self.books.sum(dim=1)

    def sleeve_navs(self) -> torch.Tensor:
        return self.books.sum(dim=-1)

    def _assert_reconciled(self) -> None:
        if not bool(torch.isfinite(self.books).all()) or bool((self.books < -1e-9).any()):
            raise RuntimeError("non-finite or negative H3 sleeve state")
        aggregate = self.aggregate_weights()
        total = aggregate.sum(dim=-1)
        if not bool(torch.allclose(total, torch.ones_like(total), rtol=0.0, atol=2e-6)):
            raise RuntimeError("H3 sleeve books must reconcile to one unit of current portfolio NAV")

    def drift_(self, returns: torch.Tensor) -> torch.Tensor:
        """Drift every locked/maturing sleeve independently and renormalize total NAV.

        Returns the batch portfolio gross growth factor. The common
        normalization changes no sleeve's value relative to another sleeve;
        it is not recapitalization.
        """

        returns = _matrix_like("returns", returns, self.aggregate_weights())
        if bool((returns <= -1.0).any()):
            raise ValueError("returns must be greater than -1")
        grown = self.books * (1.0 + returns).unsqueeze(1)
        growth = grown.sum(dim=(1, 2))
        if bool((growth <= 0).any()):
            raise RuntimeError("portfolio NAV became nonpositive")
        self.books = grown / growth[:, None, None]
        self._assert_reconciled()
        return growth

    def _liquidate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Move selected risky holdings to cash inside their original sleeves."""

        mask = mask.clone()
        mask[:, self.cash_index] = False
        removed = torch.where(mask.unsqueeze(1), self.books, torch.zeros_like(self.books))
        after = self.books - removed
        cash = torch.zeros_like(after)
        cash[:, :, self.cash_index] = removed.sum(dim=-1)
        after = after + cash
        delta = after - self.books
        self.books = after
        return delta

    def apply_forced_repairs_(
        self,
        membership: torch.Tensor,
        available: torch.Tensor,
        risk_asset_caps: torch.Tensor,
        risk_gross_max: torch.Tensor,
    ) -> Hold30SleeveRepair:
        """Apply mandatory repairs without advancing or resetting sleeve phases.

        Membership and availability exits occur first. Aggregate name-cap and
        gross-risk excess is then removed pro rata from every sleeve holding
        the affected risk. Every proceeds dollar remains as cash in its source
        sleeve.
        """

        aggregate = self.aggregate_weights()
        membership = _mask_like("membership", membership, aggregate)
        available = _mask_like("available", available, aggregate)
        risk_asset_caps = _matrix_like("risk_asset_caps", risk_asset_caps, aggregate)
        risk_gross_max = _vector_like("risk_gross_max", risk_gross_max, aggregate)
        if bool((risk_asset_caps < 0).any()) or bool((risk_gross_max < 0).any()):
            raise ValueError("risk ceilings cannot be negative")

        membership_delta = self._liquidate_mask(~membership)
        availability_delta = self._liquidate_mask(membership & ~available)

        before_risk = self.books
        risky = torch.ones_like(aggregate, dtype=torch.bool)
        risky[:, self.cash_index] = False
        cap = torch.minimum(
            risk_asset_caps.clamp_min(0.0),
            aggregate.new_tensor(HOLD30_MAX_STOCK_WEIGHT),
        )
        cap[:, self.cash_index] = 0.0
        held_by_name = torch.where(risky, self.aggregate_weights(), torch.zeros_like(aggregate))
        name_scale = torch.minimum(
            torch.ones_like(held_by_name),
            cap / held_by_name.clamp_min(1e-18),
        )
        name_scale = torch.where(risky, name_scale, torch.ones_like(name_scale))
        after_name = self.books * name_scale.unsqueeze(1)
        name_removed = self.books - after_name
        name_cash = torch.zeros_like(after_name)
        name_cash[:, :, self.cash_index] = name_removed.sum(dim=-1)
        self.books = after_name + name_cash

        gross = torch.where(
            risky.unsqueeze(1), self.books, torch.zeros_like(self.books)
        ).sum(dim=(1, 2))
        hard_gross = torch.minimum(torch.ones_like(risk_gross_max), risk_gross_max)
        gross_scale = torch.minimum(torch.ones_like(gross), hard_gross / gross.clamp_min(1e-18))
        risky_books = torch.where(risky.unsqueeze(1), self.books, torch.zeros_like(self.books))
        gross_removed = risky_books * (1.0 - gross_scale[:, None, None])
        after_gross = self.books - gross_removed
        gross_cash = torch.zeros_like(after_gross)
        gross_cash[:, :, self.cash_index] = gross_removed.sum(dim=-1)
        self.books = after_gross + gross_cash
        risk_delta = self.books - before_risk
        self._assert_reconciled()
        return Hold30SleeveRepair(membership_delta, availability_delta, risk_delta)

    def review_maturing_(
        self,
        entry_scores: torch.Tensor,
        benchmark_weights: torch.Tensor,
        trade_mask: torch.Tensor,
        risk_asset_caps: torch.Tensor,
        risk_gross_max: torch.Tensor,
        *,
        max_turnover: float = HOLD30_MAX_DISCRETIONARY_TURNOVER,
    ) -> Hold30SleeveReview:
        """Review one scheduled sleeve and apply one cross-netted capped order.

        The review always advances the fixed phase, including when the common
        portfolio turnover cap censors the requested rebuild. There is no
        next-session retry and no transfer from a locked sleeve.
        """

        aggregate = self.aggregate_weights()
        entry_scores = _matrix_like("entry_scores", entry_scores, aggregate)
        benchmark_weights = _matrix_like("benchmark_weights", benchmark_weights, aggregate)
        trade_mask = _mask_like("trade_mask", trade_mask, aggregate)
        risk_asset_caps = _matrix_like("risk_asset_caps", risk_asset_caps, aggregate)
        risk_gross_max = _vector_like("risk_gross_max", risk_gross_max, aggregate)
        if not 0.0 <= float(max_turnover) <= 1.0:
            raise ValueError("max_turnover must lie in [0, 1]")
        if bool((risk_asset_caps < 0).any()) or bool((risk_gross_max < 0).any()):
            raise ValueError("risk ceilings cannot be negative")

        session = self.session_index
        sleeve_index = self.maturing_sleeve
        review_age = session - int(self.last_review_session[sleeve_index].item())
        if review_age != HOLD30_SLEEVE_COUNT:
            raise RuntimeError(
                f"sleeve {sleeve_index} reviewed after {review_age} sessions; expected {HOLD30_SLEEVE_COUNT}"
            )
        current = self.books[:, sleeve_index]
        sleeve_nav = current.sum(dim=-1)
        locked = self.books.sum(dim=1) - current
        risky = torch.ones_like(aggregate, dtype=torch.bool)
        risky[:, self.cash_index] = False

        absolute_cap = torch.minimum(
            risk_asset_caps.clamp_min(0.0),
            aggregate.new_tensor(HOLD30_MAX_STOCK_WEIGHT),
        )
        absolute_cap[:, self.cash_index] = 0.0
        locked_risky = torch.where(risky, locked, torch.zeros_like(locked))
        residual_cap = (absolute_cap - locked_risky).clamp_min(0.0)
        residual_cap = torch.where(trade_mask & risky, residual_cap, torch.zeros_like(residual_cap))
        locked_gross = locked_risky.sum(dim=-1)
        gross_ceiling = torch.minimum(torch.ones_like(risk_gross_max), risk_gross_max.clamp_min(0.0))
        residual_gross = (gross_ceiling - locked_gross).clamp_min(0.0)

        direction = centered_benchmark_tilt(
            entry_scores,
            benchmark_weights,
            trade_mask,
            cash_index=self.cash_index,
        )
        requested_risky = torch.minimum(sleeve_nav, residual_gross)
        allocation, effective = capped_waterfill(requested_risky, direction, residual_cap)
        target = allocation.clone()
        target[:, self.cash_index] = sleeve_nav - allocation.sum(dim=-1)
        requested_sleeve_delta = target - current

        # The target and current sleeve are crossed at asset level before cost;
        # same-name renewal therefore produces no synthetic sell/rebuy or age
        # reset in the external economic cohort ledger.
        requested_delta = requested_sleeve_delta
        requested_turnover = _one_way_turnover(requested_delta)
        max_turnover_t = requested_turnover.new_tensor(float(max_turnover))
        scale = torch.where(
            requested_turnover > max_turnover_t,
            max_turnover_t / requested_turnover.clamp_min(1e-18),
            torch.ones_like(requested_turnover),
        )
        constructed_sleeve_delta = requested_sleeve_delta * scale.unsqueeze(-1)
        constructed_delta = constructed_sleeve_delta
        constructed_turnover = _one_way_turnover(constructed_delta)
        cap_censored = requested_turnover > max_turnover_t

        current_risky = torch.where(risky, current, torch.zeros_like(current))
        target_risky = torch.where(risky, target, torch.zeros_like(target))
        cross_net = torch.minimum(current_risky, target_risky).sum(dim=-1)

        updated = current + constructed_sleeve_delta
        next_books = self.books.clone()
        next_books[:, sleeve_index] = updated
        self.books = next_books
        self.last_review_session = self.last_review_session.clone()
        self.review_count = self.review_count.clone()
        self.last_review_session[sleeve_index] = session
        self.review_count[sleeve_index] += 1
        self.session_index += 1
        self._assert_reconciled()

        return Hold30SleeveReview(
            session_index=session,
            maturing_sleeve=sleeve_index,
            review_age=review_age,
            sleeve_nav_before=sleeve_nav,
            entry_direction=direction,
            residual_asset_capacity=residual_cap,
            residual_risky_capacity=residual_gross,
            target_sleeve=target,
            requested_sleeve_delta=requested_sleeve_delta,
            constructed_sleeve_delta=constructed_sleeve_delta,
            requested_delta=requested_delta,
            constructed_delta=constructed_delta,
            requested_turnover=requested_turnover,
            constructed_turnover=constructed_turnover,
            same_name_cross_net_notional=cross_net,
            unallocated_sleeve_cash=sleeve_nav - effective,
            maturity_cap_censored=cap_censored,
        )


__all__ = [
    "HOLD30_SLEEVE_COUNT",
    "Hold30SleeveRepair",
    "Hold30SleeveReview",
    "Hold30SleeveSnapshot",
    "Hold30SleeveState",
]
