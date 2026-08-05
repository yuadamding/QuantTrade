"""Differentiable position-age and return-neutral cohort accounting.

The economic ledger stores portfolio-weight notional in exact age bins ``0``
through ``59`` and a terminal ``60+`` bin.  A parallel retention ledger stores
entry-notional units.  Economic notional drifts with asset returns; retention
units do not.  Sales remove the same fraction from both ledgers, which makes
holding-period statistics insensitive to subsequent winner/loser returns.

The helpers are deliberately functional: every transition returns a new
``CohortLedger`` and leaves the input untouched.  This lets direct optimizers
differentiate through the same accounting contract that the historical
environment uses, while callers can explicitly ``detach`` at a computational
graph boundary without creating an economic reset.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import torch

MAX_EXACT_AGE = 60
AGE_BIN_COUNT = MAX_EXACT_AGE + 1
TARGET_HOLDING_DAYS = 30


class TurnoverCause(str, Enum):
    """Disjoint causes used by the authoritative portfolio ledger."""

    STARTUP = "startup"
    DISCRETIONARY = "discretionary"
    MEMBERSHIP_FORCED = "membership_forced"
    AVAILABILITY_FORCED = "availability_forced"
    RISK_FORCED = "risk_forced"
    TERMINAL = "terminal"

    @property
    def early_exit_exempt(self) -> bool:
        """Only an ordinary discretionary sale can pay an early-exit term."""

        return self is not TurnoverCause.DISCRETIONARY


TURNOVER_CAUSES: tuple[TurnoverCause, ...] = tuple(TurnoverCause)


def _as_cause(value: TurnoverCause | str) -> TurnoverCause:
    try:
        return value if isinstance(value, TurnoverCause) else TurnoverCause(value)
    except ValueError as exc:
        allowed = ", ".join(cause.value for cause in TURNOVER_CAUSES)
        raise ValueError(
            f"Unknown turnover cause {value!r}; expected one of {allowed}."
        ) from exc


def _validate_weight_tensor(weights: torch.Tensor, *, name: str) -> None:
    if weights.ndim != 2:
        raise ValueError(
            f"{name} must have shape [batch, asset]; got {tuple(weights.shape)}."
        )
    if not weights.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype; got {weights.dtype}.")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    if bool((weights < -1e-7).any().item()):
        raise ValueError(f"{name} must be nonnegative.")
    totals = weights.sum(dim=-1)
    if not bool(torch.allclose(totals, torch.ones_like(totals), atol=1e-6, rtol=1e-6)):
        raise ValueError(f"{name} must sum to one along the asset dimension.")


def net_trade_legs(
    proposed_buys: torch.Tensor,
    proposed_sells: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross same-name buy and sell legs before cohort attribution.

    The returned tensors are nonnegative and have disjoint support.  Calling
    this function before touching the cohort state prevents a sell-and-rebuy
    pair from manufacturing a younger holding age.
    """

    if proposed_buys.shape != proposed_sells.shape:
        raise ValueError("proposed_buys and proposed_sells must have identical shapes.")
    if (
        proposed_buys.dtype != proposed_sells.dtype
        or proposed_buys.device != proposed_sells.device
    ):
        raise ValueError(
            "proposed_buys and proposed_sells must share dtype and device."
        )
    if bool((proposed_buys < 0).any().item()) or bool(
        (proposed_sells < 0).any().item()
    ):
        raise ValueError("Trade legs must be nonnegative.")
    net = proposed_buys - proposed_sells
    return net.clamp_min(0.0), (-net).clamp_min(0.0)


@dataclass(frozen=True)
class CohortTradeAccounting:
    """Accounting emitted by one cause-specific, pre-cost portfolio trade."""

    cause: TurnoverCause
    turnover: torch.Tensor
    net_buys: torch.Tensor
    net_sells: torch.Tensor
    sold_value_by_age: torch.Tensor
    sold_units_by_age: torch.Tensor
    entry_units_added: torch.Tensor
    early_exit_notional: torch.Tensor
    early_exit_units: torch.Tensor

    @property
    def early_exit_exempt(self) -> bool:
        return self.cause.early_exit_exempt


@dataclass(frozen=True)
class CohortLedger:
    """Economic value and return-neutral entry units by asset and age.

    ``economic_value`` and ``retention_units`` have shape
    ``[batch, asset, 61]``.  CASH is represented implicitly in ``weights`` and
    its cohort bins are always zero.
    """

    economic_value: torch.Tensor
    retention_units: torch.Tensor
    cash_index: int

    def __post_init__(self) -> None:
        if (
            self.economic_value.ndim != 3
            or self.economic_value.shape[-1] != AGE_BIN_COUNT
        ):
            raise ValueError(
                "economic_value must have shape [batch, asset, 61]; "
                f"got {tuple(self.economic_value.shape)}."
            )
        if self.retention_units.shape != self.economic_value.shape:
            raise ValueError("retention_units must match economic_value shape.")
        if (
            self.retention_units.dtype != self.economic_value.dtype
            or self.retention_units.device != self.economic_value.device
        ):
            raise ValueError("Both cohort ledgers must share dtype and device.")
        if not 0 <= self.cash_index < self.economic_value.shape[1]:
            raise ValueError("cash_index is outside the asset axis.")
        if not bool(torch.isfinite(self.economic_value).all()) or not bool(
            torch.isfinite(self.retention_units).all()
        ):
            raise ValueError("cohort ledgers must contain only finite values.")
        if bool((self.economic_value < -1e-7).any()) or bool(
            (self.retention_units < -1e-7).any()
        ):
            raise ValueError("cohort ledgers cannot contain negative mass.")
        if bool((self.economic_value[:, self.cash_index] != 0).any()) or bool(
            (self.retention_units[:, self.cash_index] != 0).any()
        ):
            raise ValueError("CASH cannot carry economic or retention age cohorts.")

    @classmethod
    def from_weights(
        cls,
        weights: torch.Tensor,
        *,
        cash_index: int,
        initial_age: int = MAX_EXACT_AGE,
        track_initial_units: bool = False,
    ) -> CohortLedger:
        """Create an endowed ledger without manufacturing a startup trade.

        Initial holdings default to the ``60+`` bin and are excluded from the
        experimental retention population.  ``track_initial_units=True`` is
        available for controlled fixtures that intentionally treat the
        endowment as an entry cohort.
        """

        _validate_weight_tensor(weights, name="weights")
        if not 0 <= cash_index < weights.shape[1]:
            raise ValueError("cash_index is outside the asset axis.")
        if isinstance(initial_age, bool) or not 0 <= int(initial_age) <= MAX_EXACT_AGE:
            raise ValueError(f"initial_age must be an integer in [0, {MAX_EXACT_AGE}].")
        if int(initial_age) != initial_age:
            raise ValueError(f"initial_age must be an integer in [0, {MAX_EXACT_AGE}].")

        economic = weights.new_zeros((*weights.shape, AGE_BIN_COUNT))
        endowed = weights.clone()
        endowed[:, cash_index] = 0.0
        economic[..., int(initial_age)] = endowed
        units = economic.clone() if track_initial_units else torch.zeros_like(economic)
        return cls(
            economic_value=economic, retention_units=units, cash_index=int(cash_index)
        )

    @classmethod
    def from_staggered_endowment(
        cls,
        weights: torch.Tensor,
        *,
        cash_index: int,
        youngest_age: int = 0,
        oldest_age: int = TARGET_HOLDING_DAYS - 1,
        track_initial_units: bool = False,
    ) -> CohortLedger:
        """Split endowed risky holdings evenly over an explicit age range.

        The pre-lockbox evaluation uses ages ``0..29`` so the common C1
        endowment does not give every position the same artificial maturity.
        Endowed units remain excluded from experimental survival statistics by
        default.
        """

        _validate_weight_tensor(weights, name="weights")
        if not 0 <= cash_index < weights.shape[1]:
            raise ValueError("cash_index is outside the asset axis.")
        for name, value in (("youngest_age", youngest_age), ("oldest_age", oldest_age)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer in [0, {MAX_EXACT_AGE}].")
            if not 0 <= value <= MAX_EXACT_AGE:
                raise ValueError(f"{name} must be an integer in [0, {MAX_EXACT_AGE}].")
        if youngest_age > oldest_age:
            raise ValueError("youngest_age cannot exceed oldest_age.")
        economic = weights.new_zeros((*weights.shape, AGE_BIN_COUNT))
        risky = weights.clone()
        risky[:, cash_index] = 0.0
        count = oldest_age - youngest_age + 1
        economic[..., youngest_age : oldest_age + 1] = risky.unsqueeze(-1) / count
        units = economic.clone() if track_initial_units else torch.zeros_like(economic)
        return cls(economic, units, int(cash_index))

    @property
    def batch_size(self) -> int:
        return self.economic_value.shape[0]

    @property
    def num_assets(self) -> int:
        return self.economic_value.shape[1]

    @property
    def weights(self) -> torch.Tensor:
        risky = self.economic_value.sum(dim=-1)
        cash = 1.0 - risky.sum(dim=-1)
        full = risky.clone()
        full[:, self.cash_index] = cash
        return full

    def clone(self) -> CohortLedger:
        return CohortLedger(
            economic_value=self.economic_value.clone(),
            retention_units=self.retention_units.clone(),
            cash_index=self.cash_index,
        )

    def detach(self) -> CohortLedger:
        return CohortLedger(
            economic_value=self.economic_value.detach(),
            retention_units=self.retention_units.detach(),
            cash_index=self.cash_index,
        )

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        return {
            "economic_value": self.economic_value.clone(),
            "retention_units": self.retention_units.clone(),
            "cash_index": self.cash_index,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor | int]) -> CohortLedger:
        economic = state.get("economic_value")
        units = state.get("retention_units")
        cash_index = state.get("cash_index")
        if not isinstance(economic, torch.Tensor) or not isinstance(
            units, torch.Tensor
        ):
            raise TypeError(
                "Cohort state must contain tensor economic_value and retention_units."
            )
        if isinstance(cash_index, bool) or not isinstance(cash_index, int):
            raise TypeError("Cohort state cash_index must be an integer.")
        return cls(economic.clone(), units.clone(), cash_index)

    def assert_reconciles(
        self,
        weights: torch.Tensor,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        if weights.shape != (self.batch_size, self.num_assets):
            raise ValueError("weights do not match the cohort batch and asset axes.")
        if not bool(torch.allclose(self.weights, weights, atol=atol, rtol=rtol)):
            maximum = (self.weights - weights).abs().max().item()
            raise RuntimeError(
                f"Cohort economic value does not reconcile with weights (max diff {maximum:g})."
            )
        cash_economic = self.economic_value[:, self.cash_index]
        cash_units = self.retention_units[:, self.cash_index]
        if bool((cash_economic != 0).any().item()) or bool(
            (cash_units != 0).any().item()
        ):
            raise RuntimeError("CASH cannot carry economic or retention age cohorts.")

    def age_and_drift(self, asset_returns: torch.Tensor) -> CohortLedger:
        """Apply one holding return, normalize weights, then advance age.

        Cohorts in age 59 and the existing ``60+`` bin merge into ``60+``.
        Retention units advance in age but remain return-neutral.
        """

        expected = (self.batch_size, self.num_assets)
        if asset_returns.shape != expected:
            raise ValueError(f"asset_returns must have shape {expected}.")
        if (
            asset_returns.dtype != self.economic_value.dtype
            or asset_returns.device != self.economic_value.device
        ):
            raise ValueError("asset_returns must share cohort dtype and device.")
        if bool((asset_returns <= -1.0).any().item()):
            raise ValueError("asset_returns must be greater than -1.")

        current = self.weights
        growth = 1.0 + (current * asset_returns).sum(dim=-1)
        if bool((growth <= 0).any().item()):
            raise RuntimeError(
                "Portfolio gross return reached -100%; cohort weights cannot be drifted."
            )
        grown = self.economic_value * (1.0 + asset_returns).unsqueeze(-1)
        normalized = grown / growth[:, None, None]

        aged_value = torch.zeros_like(normalized)
        aged_value[..., 1:MAX_EXACT_AGE] = normalized[..., : MAX_EXACT_AGE - 1]
        aged_value[..., MAX_EXACT_AGE] = (
            normalized[..., MAX_EXACT_AGE - 1] + normalized[..., MAX_EXACT_AGE]
        )
        aged_units = torch.zeros_like(self.retention_units)
        aged_units[..., 1:MAX_EXACT_AGE] = self.retention_units[
            ..., : MAX_EXACT_AGE - 1
        ]
        aged_units[..., MAX_EXACT_AGE] = (
            self.retention_units[..., MAX_EXACT_AGE - 1]
            + self.retention_units[..., MAX_EXACT_AGE]
        )
        return CohortLedger(aged_value, aged_units, self.cash_index)

    def trade_to(
        self,
        target_weights: torch.Tensor,
        *,
        cause: TurnoverCause | str,
        proposed_release: torch.Tensor | None = None,
        track_new_entries: bool = True,
    ) -> tuple[CohortLedger, CohortTradeAccounting]:
        """Apply one cause-specific pre-cost target after same-name netting.

        ``proposed_release`` optionally supplies an H2 cohort release proposal
        with shape ``[batch, asset, 61]``.  Actual net sales consume proposed
        release proportionally first and any residual sale pro rata from the
        remaining economic cohorts.  Without it, all sales are pro rata.
        """

        resolved_cause = _as_cause(cause)
        _validate_weight_tensor(target_weights, name="target_weights")
        expected = (self.batch_size, self.num_assets)
        if target_weights.shape != expected:
            raise ValueError(f"target_weights must have shape {expected}.")
        if (
            target_weights.dtype != self.economic_value.dtype
            or target_weights.device != self.economic_value.device
        ):
            raise ValueError("target_weights must share cohort dtype and device.")

        current = self.weights
        delta = target_weights - current
        buys = delta.clamp_min(0.0)
        sells = (-delta).clamp_min(0.0)
        buys = buys.clone()
        sells = sells.clone()
        buys[:, self.cash_index] = 0.0
        sells[:, self.cash_index] = 0.0

        value = self.economic_value
        if proposed_release is None:
            total_value = value.sum(dim=-1)
            sold_value = value * (
                sells / total_value.clamp_min(torch.finfo(value.dtype).eps)
            ).unsqueeze(-1)
            sold_value = torch.where(
                total_value.unsqueeze(-1) > 0, sold_value, torch.zeros_like(sold_value)
            )
        else:
            if proposed_release.shape != value.shape:
                raise ValueError("proposed_release must match the cohort ledger shape.")
            if (
                proposed_release.dtype != value.dtype
                or proposed_release.device != value.device
            ):
                raise ValueError("proposed_release must share cohort dtype and device.")
            proposed = torch.minimum(proposed_release.clamp_min(0.0), value)
            proposed_total = proposed.sum(dim=-1)
            primary_scale = torch.minimum(
                sells / proposed_total.clamp_min(torch.finfo(value.dtype).eps),
                torch.ones_like(sells),
            )
            primary = proposed * primary_scale.unsqueeze(-1)
            residual_sale = (sells - primary.sum(dim=-1)).clamp_min(0.0)
            residual_value = (value - primary).clamp_min(0.0)
            residual_total = residual_value.sum(dim=-1)
            secondary = residual_value * (
                residual_sale / residual_total.clamp_min(torch.finfo(value.dtype).eps)
            ).unsqueeze(-1)
            secondary = torch.where(
                residual_total.unsqueeze(-1) > 0,
                secondary,
                torch.zeros_like(secondary),
            )
            sold_value = primary + secondary

        removal_fraction = sold_value / value.clamp_min(torch.finfo(value.dtype).eps)
        removal_fraction = torch.where(
            value > 0, removal_fraction, torch.zeros_like(removal_fraction)
        )
        sold_units = self.retention_units * removal_fraction
        remaining_value = (value - sold_value).clamp_min(0.0)
        remaining_units = (self.retention_units - sold_units).clamp_min(0.0)

        entry_value = torch.zeros_like(value)
        entry_value[..., 0] = buys
        entry_units = torch.zeros_like(value)
        if track_new_entries:
            entry_units[..., 0] = buys
        next_ledger = CohortLedger(
            economic_value=remaining_value + entry_value,
            retention_units=remaining_units + entry_units,
            cash_index=self.cash_index,
        )

        age = torch.arange(AGE_BIN_COUNT, dtype=value.dtype, device=value.device)
        early_weight = ((TARGET_HOLDING_DAYS - age) / TARGET_HOLDING_DAYS).clamp(
            min=0.0, max=1.0
        )
        if resolved_cause.early_exit_exempt:
            early_notional = torch.zeros(
                self.batch_size, dtype=value.dtype, device=value.device
            )
            early_units = torch.zeros_like(early_notional)
        else:
            early_notional = (sold_value * early_weight).sum(dim=(-1, -2))
            early_units = (sold_units * early_weight).sum(dim=(-1, -2))
        accounting = CohortTradeAccounting(
            cause=resolved_cause,
            turnover=0.5 * delta.abs().sum(dim=-1),
            net_buys=buys,
            net_sells=sells,
            sold_value_by_age=sold_value,
            sold_units_by_age=sold_units,
            entry_units_added=entry_units.sum(dim=-1),
            early_exit_notional=early_notional,
            early_exit_units=early_units,
        )
        return next_ledger, accounting

    def age_summaries(self) -> torch.Tensor:
        """Return five compact age features per asset.

        Features are mean age divided by 60, fraction younger than 10, 20,
        and 30 sessions, and fraction at least 30 sessions.  Empty positions
        receive all-zero summaries; CASH always has an all-zero summary.
        """

        value = self.economic_value
        total = value.sum(dim=-1)
        denominator = total.clamp_min(torch.finfo(value.dtype).eps)
        age = torch.arange(AGE_BIN_COUNT, dtype=value.dtype, device=value.device)
        mean_age = (value * age).sum(dim=-1) / denominator / MAX_EXACT_AGE
        young10 = value[..., :10].sum(dim=-1) / denominator
        young20 = value[..., :20].sum(dim=-1) / denominator
        young30 = value[..., :30].sum(dim=-1) / denominator
        at_least30 = value[..., 30:].sum(dim=-1) / denominator
        summaries = torch.stack(
            (mean_age, young10, young20, young30, at_least30), dim=-1
        )
        summaries = torch.where(
            total[..., None] > 0, summaries, torch.zeros_like(summaries)
        )
        summaries = summaries.clone()
        summaries[:, self.cash_index] = 0.0
        return summaries


def zero_turnover_by_cause(
    reference: torch.Tensor,
) -> dict[TurnoverCause, torch.Tensor]:
    """Create a fresh batch vector for each disjoint turnover cause."""

    if reference.ndim != 1:
        raise ValueError("reference must have shape [batch].")
    return {cause: torch.zeros_like(reference) for cause in TURNOVER_CAUSES}
