"""Sealed deterministic C0--C4 controls and common Hold-30 cost ladder.

Control construction is deliberately separated from pricing.  A constructor
produces one gross-return/action trace with continuing holdings and disjoint
turnover causes.  :func:`price_hold30_cost_ladder` then applies 10, 20, and 40
basis points to that same trace; no stress rung may alter an action.

C1 is not rebuilt here.  The evaluator accepts receipt-bound C1 portfolio
weights, derives and verifies their exact mandatory repairs and monthly action
schedule, checks the bound 20-bp return, and then re-prices the resulting
trace.  Outcome-null C1 reconstruction remains solely owned by
``hold30_null_rebuild``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from rl_quant.datasets.hold30 import Hold30DatasetSequence
from rl_quant.envs.hold30 import CohortLedger, TurnoverCause
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.training.hold30_runtime import Hold30ChronologicalRuntime, Hold30Sequence

HOLD30_CONTROL_IDS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
HOLD30_COST_RUNGS_BPS = (10, 20, 40)
HOLD30_PRIMARY_COST_BPS = 20
HOLD30_C1_ACTIVE_COUNT = 300
HOLD30_CONTROL_TOLERANCE = 1e-6
HOLD30_MOMENTUM_SESSIONS = 21

_CAUSES = (
    TurnoverCause.STARTUP,
    TurnoverCause.MEMBERSHIP_FORCED,
    TurnoverCause.AVAILABILITY_FORCED,
    TurnoverCause.RISK_FORCED,
    TurnoverCause.DISCRETIONARY,
    TurnoverCause.TERMINAL,
)


class Hold30ControlError(ValueError):
    """A control trace is incomplete, contaminated, or economically invalid."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30ControlError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_float64(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise Hold30ControlError(f"{name} must have shape {shape}")
    if value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
        raise Hold30ControlError(f"{name} must be a finite float64 tensor")


def _validate_fitting_rows(
    fitting_rows: Iterable[int],
    *,
    outer_start: int,
    rows: int,
) -> tuple[int, ...]:
    if isinstance(outer_start, bool) or not isinstance(outer_start, int):
        raise Hold30ControlError("outer_start must be an integer decision row")
    if not 0 <= outer_start <= rows:
        raise Hold30ControlError("outer_start lies outside the trace")
    values = tuple(fitting_rows)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise Hold30ControlError("fitting_rows must contain integer decision rows")
    if tuple(sorted(set(values))) != values:
        raise Hold30ControlError("fitting_rows must be strictly increasing and unique")
    if any(value < 0 or value >= outer_start for value in values):
        raise Hold30ControlError("outer data are forbidden during control fitting")
    return values


@dataclass(frozen=True, slots=True)
class Hold30ControlGrossTrace:
    """One immutable gross/action path shared by every transaction-cost rung."""

    control_id: str
    axis_id: str
    asset_ids: tuple[str, ...]
    weights: torch.Tensor
    pretrade_weights: torch.Tensor
    gross_returns: torch.Tensor
    startup_delta: torch.Tensor
    membership_forced_delta: torch.Tensor
    availability_forced_delta: torch.Tensor
    risk_forced_delta: torch.Tensor
    discretionary_delta: torch.Tensor
    terminal_delta: torch.Tensor
    score_mask: torch.Tensor
    outer_start: int
    fitting_rows: tuple[int, ...]
    source_receipt_sha256: str
    strategy_inputs_sha256: str

    def __post_init__(self) -> None:
        if self.control_id not in HOLD30_CONTROL_IDS:
            raise Hold30ControlError(f"unknown control ID {self.control_id!r}")
        _require_digest("axis_id", self.axis_id)
        _require_digest("source_receipt_sha256", self.source_receipt_sha256)
        _require_digest("strategy_inputs_sha256", self.strategy_inputs_sha256)
        if (
            not isinstance(self.asset_ids, tuple)
            or len(self.asset_ids) < 1
            or len(set(self.asset_ids)) != len(self.asset_ids)
            or self.asset_ids[0] != "CASH"
        ):
            raise Hold30ControlError("asset_ids must be unique with CASH at index zero")
        if not isinstance(self.weights, torch.Tensor) or self.weights.ndim != 3:
            raise Hold30ControlError("weights must have shape [position, batch, asset]")
        positions, batch, assets = self.weights.shape
        if positions < 2 or assets != len(self.asset_ids):
            raise Hold30ControlError("weights do not match the position/asset contract")
        rows = positions - 1
        matrix_shape = (rows, batch, assets)
        _require_float64("weights", self.weights, (positions, batch, assets))
        _require_float64("pretrade_weights", self.pretrade_weights, matrix_shape)
        _require_float64("gross_returns", self.gross_returns, (rows, batch))
        for cause, value in self.action_deltas.items():
            _require_float64(f"{cause.value}_delta", value, matrix_shape)
        if (
            not isinstance(self.score_mask, torch.Tensor)
            or self.score_mask.dtype != torch.bool
            or tuple(self.score_mask.shape) != (rows,)
        ):
            raise Hold30ControlError("score_mask must be boolean [decision]")
        fitting = _validate_fitting_rows(
            self.fitting_rows,
            outer_start=self.outer_start,
            rows=rows,
        )
        if fitting != self.fitting_rows:
            raise AssertionError("fitting-row normalization drifted")
        for name, value in (
            ("weights", self.weights),
            ("pretrade_weights", self.pretrade_weights),
        ):
            if bool((value < -HOLD30_CONTROL_TOLERANCE).any()) or not bool(
                torch.allclose(
                    value.sum(dim=-1),
                    torch.ones_like(value.sum(dim=-1)),
                    atol=HOLD30_CONTROL_TOLERANCE,
                    rtol=HOLD30_CONTROL_TOLERANCE,
                )
            ):
                raise Hold30ControlError(f"{name} must contain long-only simplexes")
        total_delta = sum(
            self.action_deltas.values(), torch.zeros_like(self.startup_delta)
        )
        if not bool(
            torch.allclose(
                self.pretrade_weights + total_delta,
                self.weights[1:],
                atol=HOLD30_CONTROL_TOLERANCE,
                rtol=HOLD30_CONTROL_TOLERANCE,
            )
        ):
            raise Hold30ControlError(
                "cause-separated actions do not reconcile to holdings"
            )
        if bool(self.startup_delta.abs().gt(HOLD30_CONTROL_TOLERANCE).any()):
            raise Hold30ControlError(
                "common evaluation endowment cannot book startup turnover"
            )
        if bool(self.terminal_delta.abs().gt(HOLD30_CONTROL_TOLERANCE).any()):
            raise Hold30ControlError(
                "continuing trace cannot book terminal liquidation"
            )

    @property
    def action_deltas(self) -> dict[TurnoverCause, torch.Tensor]:
        return {
            TurnoverCause.STARTUP: self.startup_delta,
            TurnoverCause.MEMBERSHIP_FORCED: self.membership_forced_delta,
            TurnoverCause.AVAILABILITY_FORCED: self.availability_forced_delta,
            TurnoverCause.RISK_FORCED: self.risk_forced_delta,
            TurnoverCause.DISCRETIONARY: self.discretionary_delta,
            TurnoverCause.TERMINAL: self.terminal_delta,
        }

    @property
    def turnover_by_cause(self) -> dict[TurnoverCause, torch.Tensor]:
        return {
            cause: 0.5 * delta.abs().sum(dim=-1)
            for cause, delta in self.action_deltas.items()
        }

    @property
    def total_turnover(self) -> torch.Tensor:
        values = self.turnover_by_cause
        return sum(values.values(), torch.zeros_like(next(iter(values.values()))))

    @property
    def trace_sha256(self) -> str:
        return _payload_sha256(self.receipt_payload)

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "control_id": self.control_id,
            "axis_id": self.axis_id,
            "asset_ids": list(self.asset_ids),
            "source_receipt_sha256": self.source_receipt_sha256,
            "strategy_inputs_sha256": self.strategy_inputs_sha256,
            "outer_start": self.outer_start,
            "fitting_rows": list(self.fitting_rows),
            "state_convention": "continuing_no_terminal_liquidation",
            "canonical_closed_loop_cost_bps": HOLD30_PRIMARY_COST_BPS,
            "tensor_sha256s": {
                "weights": _tensor_sha256(self.weights),
                "pretrade_weights": _tensor_sha256(self.pretrade_weights),
                "gross_returns": _tensor_sha256(self.gross_returns),
                "score_mask": _tensor_sha256(self.score_mask),
                **{
                    f"{cause.value}_delta": _tensor_sha256(delta)
                    for cause, delta in self.action_deltas.items()
                },
            },
        }


@dataclass(frozen=True, slots=True)
class Hold30CostRung:
    cost_bps: int
    net_returns: torch.Tensor
    costs_by_cause: dict[TurnoverCause, torch.Tensor]
    total_cost: torch.Tensor
    continuing_wealth: torch.Tensor

    @property
    def terminal_wealth(self) -> torch.Tensor:
        return self.continuing_wealth[-1]


@dataclass(frozen=True, slots=True)
class Hold30CostLadder:
    trace_sha256: str
    rungs: tuple[Hold30CostRung, ...]
    receipt_sha256: str


def price_hold30_cost_ladder(trace: Hold30ControlGrossTrace) -> Hold30CostLadder:
    """Price one fixed gross/action trace at the frozen 10/20/40-bp ladder."""

    if not isinstance(trace, Hold30ControlGrossTrace):
        raise TypeError("trace must be Hold30ControlGrossTrace")
    turnover = trace.turnover_by_cause
    results: list[Hold30CostRung] = []
    receipt_rows: list[dict[str, Any]] = []
    for cost_bps in HOLD30_COST_RUNGS_BPS:
        rate = trace.gross_returns.new_tensor(cost_bps / 10_000.0)
        costs = {cause: value * rate for cause, value in turnover.items()}
        total_cost = sum(costs.values(), torch.zeros_like(trace.gross_returns))
        net = trace.gross_returns - total_cost
        if bool((net <= -1.0).any()):
            raise Hold30ControlError(f"{cost_bps}-bp net return reached -100%")
        scored_growth = torch.where(
            trace.score_mask.to(device=net.device).unsqueeze(-1),
            1.0 + net,
            torch.ones_like(net),
        )
        wealth = torch.cat(
            (
                torch.ones((1, net.shape[1]), dtype=torch.float64, device=net.device),
                torch.cumprod(scored_growth, dim=0),
            ),
            dim=0,
        )
        result = Hold30CostRung(cost_bps, net, costs, total_cost, wealth)
        results.append(result)
        receipt_rows.append(
            {
                "cost_bps": cost_bps,
                "net_returns_sha256": _tensor_sha256(net),
                "total_cost_sha256": _tensor_sha256(total_cost),
                "continuing_wealth_sha256": _tensor_sha256(wealth),
                "terminal_wealth_sha256": _tensor_sha256(result.terminal_wealth),
                "cost_by_cause_sha256": {
                    cause.value: _tensor_sha256(value) for cause, value in costs.items()
                },
            }
        )
    payload = {
        "schema_version": 1,
        "trace_sha256": trace.trace_sha256,
        "cost_rungs_bps": list(HOLD30_COST_RUNGS_BPS),
        "rungs": receipt_rows,
    }
    return Hold30CostLadder(
        trace_sha256=trace.trace_sha256,
        rungs=tuple(results),
        receipt_sha256=_payload_sha256(payload),
    )


def _trace(
    control_id: str,
    sequence: Hold30DatasetSequence,
    *,
    weights: torch.Tensor,
    pretrade_weights: torch.Tensor,
    gross_returns: torch.Tensor,
    deltas: Mapping[TurnoverCause, torch.Tensor],
    score_mask: torch.Tensor,
    outer_start: int,
    fitting_rows: Iterable[int],
    source_receipt_sha256: str,
    strategy_inputs_sha256: str,
) -> Hold30ControlGrossTrace:
    rows, batch, assets = sequence.asset_returns.shape
    frozen_score = _score_mask(sequence, score_mask)
    scored_rows = torch.where(frozen_score.to(device="cpu"))[0]
    if scored_rows.numel() == 0:
        raise Hold30ControlError("frozen control trace has no score-bearing row")
    if outer_start != int(scored_rows[0]):
        raise Hold30ControlError(
            "outer_start must equal the first true row of the frozen score mask"
        )
    zero = sequence.asset_returns.new_zeros((rows, batch, assets))
    complete = {cause: deltas.get(cause, zero) for cause in _CAUSES}
    return Hold30ControlGrossTrace(
        control_id=control_id,
        axis_id=sequence.axis_id,
        asset_ids=sequence.asset_ids,
        weights=weights,
        pretrade_weights=pretrade_weights,
        gross_returns=gross_returns,
        startup_delta=complete[TurnoverCause.STARTUP],
        membership_forced_delta=complete[TurnoverCause.MEMBERSHIP_FORCED],
        availability_forced_delta=complete[TurnoverCause.AVAILABILITY_FORCED],
        risk_forced_delta=complete[TurnoverCause.RISK_FORCED],
        discretionary_delta=complete[TurnoverCause.DISCRETIONARY],
        terminal_delta=complete[TurnoverCause.TERMINAL],
        score_mask=frozen_score,
        outer_start=outer_start,
        fitting_rows=_validate_fitting_rows(
            fitting_rows,
            outer_start=outer_start,
            rows=rows,
        ),
        source_receipt_sha256=source_receipt_sha256,
        strategy_inputs_sha256=strategy_inputs_sha256,
    )


def assemble_hold30_control_trace(
    control_id: str,
    sequence: Hold30DatasetSequence,
    *,
    weights: torch.Tensor,
    pretrade_weights: torch.Tensor,
    gross_returns: torch.Tensor,
    deltas: Mapping[TurnoverCause, torch.Tensor],
    score_mask: torch.Tensor,
    outer_start: int,
    fitting_rows: Iterable[int],
    source_receipt_sha256: str,
    strategy_inputs_sha256: str,
) -> Hold30ControlGrossTrace:
    """Public fail-closed assembler for later sealed control tranches.

    Control-specific modules own intent generation and the common runtime owns
    execution.  This boundary centralizes score-role, fitting-row, simplex,
    cause-reconciliation, continuing-wealth, and receipt validation without
    requiring those modules to duplicate C0--C4 accounting logic.
    """

    return _trace(
        control_id,
        sequence,
        weights=weights,
        pretrade_weights=pretrade_weights,
        gross_returns=gross_returns,
        deltas=deltas,
        score_mask=score_mask,
        outer_start=outer_start,
        fitting_rows=fitting_rows,
        source_receipt_sha256=source_receipt_sha256,
        strategy_inputs_sha256=strategy_inputs_sha256,
    )


def _check_market(sequence: Hold30DatasetSequence) -> None:
    if not isinstance(sequence, Hold30DatasetSequence):
        raise TypeError("sequence must be a validated Hold30DatasetSequence")
    if sequence.asset_returns.dtype != torch.float64:
        raise Hold30ControlError(
            "sealed control accounting requires float64 market returns"
        )


def _drift(
    weights: torch.Tensor, returns: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    gross_return = (weights * returns).sum(dim=-1)
    growth = 1.0 + gross_return
    if bool((growth <= 0).any()):
        raise Hold30ControlError("gross control wealth reached zero")
    return weights * (1.0 + returns) / growth.unsqueeze(-1), gross_return


def _repair_mask(
    weights: torch.Tensor, allowed: torch.Tensor, cash_index: int
) -> torch.Tensor:
    target = torch.where(allowed, weights, torch.zeros_like(weights))
    target = target.clone()
    target[:, cash_index] = 0.0
    target[:, cash_index] = 1.0 - target.sum(dim=-1)
    return target


def _repair_risk(
    weights: torch.Tensor,
    caps: torch.Tensor,
    gross_max: torch.Tensor,
    cash_index: int,
) -> torch.Tensor:
    risky = torch.ones_like(weights, dtype=torch.bool)
    risky[:, cash_index] = False
    held = torch.where(risky, weights, torch.zeros_like(weights))
    held = torch.minimum(
        held, torch.minimum(caps.clamp_min(0.0), weights.new_tensor(0.01))
    )
    gross = held.sum(dim=-1)
    hard = gross_max.clamp(min=0.0, max=1.0)
    scale = torch.where(gross > hard, hard / gross.clamp_min(1e-18), 1.0)
    held = held * scale.unsqueeze(-1)
    target = held.clone()
    target[:, cash_index] = 1.0 - held.sum(dim=-1)
    return target


def _forced_stages(
    pretrade: torch.Tensor,
    sequence: Hold30DatasetSequence,
    fill: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[TurnoverCause, torch.Tensor]]:
    membership = _repair_mask(
        pretrade, sequence.fill_membership[fill], sequence.cash_index
    )
    availability = _repair_mask(
        membership,
        sequence.fill_tradability[fill],
        sequence.cash_index,
    )
    risk = _repair_risk(
        availability,
        sequence.risk_asset_caps[fill],
        sequence.risk_gross_max[fill],
        sequence.cash_index,
    )
    return (
        membership,
        availability,
        risk,
        {
            TurnoverCause.MEMBERSHIP_FORCED: membership - pretrade,
            TurnoverCause.AVAILABILITY_FORCED: availability - membership,
            TurnoverCause.RISK_FORCED: risk - availability,
        },
    )


def _score_mask(
    sequence: Hold30DatasetSequence, value: torch.Tensor | None
) -> torch.Tensor:
    expected = sequence.roles.score[:-1].to(device=sequence.asset_returns.device)
    if value is None:
        return expected
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or value.shape != expected.shape
    ):
        raise Hold30ControlError("score_mask must be boolean [decision]")
    if not torch.equal(value.to(device=expected.device), expected):
        raise Hold30ControlError("score_mask differs from the frozen sequence roles")
    return value.to(device=expected.device)


def construct_c0_cash(
    sequence: Hold30DatasetSequence,
    *,
    outer_start: int,
    score_mask: torch.Tensor | None = None,
    fitting_rows: Iterable[int] = (),
) -> Hold30ControlGrossTrace:
    """Construct C0 from the frozen point-in-time cash-return coordinate."""

    _check_market(sequence)
    rows, batch, assets = sequence.asset_returns.shape
    weights = sequence.asset_returns.new_zeros((rows + 1, batch, assets))
    weights[..., sequence.cash_index] = 1.0
    pretrade = weights[:-1].clone()
    gross = sequence.asset_returns[..., sequence.cash_index].clone()
    return _trace(
        "C0",
        sequence,
        weights=weights,
        pretrade_weights=pretrade,
        gross_returns=gross,
        deltas={},
        score_mask=_score_mask(sequence, score_mask),
        outer_start=outer_start,
        fitting_rows=fitting_rows,
        source_receipt_sha256=sequence.provenance.receipt_id,
        strategy_inputs_sha256=_payload_sha256(
            {
                "rule": "frozen_cash_returns",
                "cash_return_sha256": _tensor_sha256(gross),
            }
        ),
    )


def accept_c1_bound_trace(
    sequence: Hold30DatasetSequence,
    *,
    monthly_rebalance: torch.Tensor,
    bound_receipt_sha256: str,
    outer_start: int,
    score_mask: torch.Tensor | None = None,
    fitting_rows: Iterable[int] = (),
) -> Hold30ControlGrossTrace:
    """Accept and verify the already-bound C1 action trace without rebuilding it."""

    _check_market(sequence)
    if not bool(
        torch.equal(
            sequence.cost_rate,
            torch.full_like(sequence.cost_rate, 0.002),
        )
    ):
        raise Hold30ControlError("the bound primary C1 trace must use exactly 20 bp")
    if bound_receipt_sha256 != sequence.provenance.c1_benchmark_trace_sha256:
        raise Hold30ControlError("C1 trace does not match the bound benchmark receipt")
    _require_digest("bound_receipt_sha256", bound_receipt_sha256)
    if (
        not isinstance(monthly_rebalance, torch.Tensor)
        or monthly_rebalance.dtype != torch.bool
        or tuple(monthly_rebalance.shape) != (sequence.n_positions,)
        or not bool(monthly_rebalance[0])
    ):
        raise Hold30ControlError(
            "monthly_rebalance must be bound boolean [position] with row zero true"
        )
    weights = sequence.c1_benchmark_weights
    rows = sequence.n_positions - 1
    pretrade_weights = sequence.asset_returns.new_zeros(
        (rows, sequence.batch_size, sequence.num_assets)
    )
    gross_returns = sequence.asset_returns.new_zeros((rows, sequence.batch_size))
    membership_forced_delta = torch.zeros_like(pretrade_weights)
    availability_forced_delta = torch.zeros_like(pretrade_weights)
    risk_forced_delta = torch.zeros_like(pretrade_weights)
    discretionary_delta = torch.zeros_like(pretrade_weights)
    for row in range(rows):
        expected_pretrade, expected_gross = _drift(
            weights[row], sequence.asset_returns[row]
        )
        pretrade_weights[row] = expected_pretrade
        gross_returns[row] = expected_gross
        membership, availability, risk, expected = _forced_stages(
            expected_pretrade,
            sequence,
            row + 1,
        )
        membership_forced_delta[row] = expected[TurnoverCause.MEMBERSHIP_FORCED]
        availability_forced_delta[row] = expected[TurnoverCause.AVAILABILITY_FORCED]
        risk_forced_delta[row] = expected[TurnoverCause.RISK_FORCED]
        expected_discretionary = weights[row + 1] - risk
        discretionary_delta[row] = expected_discretionary
        if not bool(monthly_rebalance[row + 1]) and bool(
            discretionary_delta[row].abs().gt(HOLD30_CONTROL_TOLERANCE).any()
        ):
            raise Hold30ControlError(
                "C1 traded discretionarily outside a monthly event"
            )
        del membership, availability
    membership_changes = (
        sequence.fill_membership[1:] != sequence.fill_membership[:-1]
    ).any(dim=-1)
    if bool((membership_changes & ~monthly_rebalance[1:].view(-1, 1)).any()):
        raise Hold30ControlError(
            "C1 membership changed outside the bound monthly schedule"
        )
    deltas = {
        TurnoverCause.MEMBERSHIP_FORCED: membership_forced_delta,
        TurnoverCause.AVAILABILITY_FORCED: availability_forced_delta,
        TurnoverCause.RISK_FORCED: risk_forced_delta,
        TurnoverCause.DISCRETIONARY: discretionary_delta,
    }
    trace = _trace(
        "C1",
        sequence,
        weights=weights,
        pretrade_weights=pretrade_weights,
        gross_returns=gross_returns,
        deltas=deltas,
        score_mask=_score_mask(sequence, score_mask),
        outer_start=outer_start,
        fitting_rows=fitting_rows,
        source_receipt_sha256=bound_receipt_sha256,
        strategy_inputs_sha256=_payload_sha256(
            {
                "rule": "bound_monthly_pit_equal_weight_buy_and_drift",
                "monthly_rebalance_sha256": _tensor_sha256(monthly_rebalance),
            }
        ),
    )
    expected_primary = gross_returns - sequence.cost_rate * trace.total_turnover
    if not bool(
        torch.allclose(
            expected_primary,
            sequence.c1_benchmark_net_returns,
            atol=HOLD30_CONTROL_TOLERANCE,
            rtol=HOLD30_CONTROL_TOLERANCE,
        )
    ):
        raise Hold30ControlError(
            "C1 bound net-return trace does not match its actions and costs"
        )
    return trace


def _equal_weight_target(
    sequence: Hold30DatasetSequence,
    *,
    decision: int,
    fill: int,
) -> torch.Tensor:
    risky = torch.ones_like(sequence.fill_membership[fill], dtype=torch.bool)
    risky[:, sequence.cash_index] = False
    active_count = (sequence.fill_membership[fill] & risky).sum(dim=-1)
    if bool((active_count != HOLD30_C1_ACTIVE_COUNT).any()):
        raise Hold30ControlError("C2 requires exactly 300 PIT active risky names")
    trade = sequence.decision_trade[decision] & sequence.fill_trade[fill] & risky
    target = torch.where(
        trade,
        sequence.asset_returns.new_full(trade.shape, 1.0 / HOLD30_C1_ACTIVE_COUNT),
        torch.zeros_like(sequence.asset_returns[decision]),
    )
    target = torch.minimum(
        target,
        torch.minimum(sequence.risk_asset_caps[fill], target.new_tensor(0.01)),
    )
    gross = target.sum(dim=-1)
    hard = sequence.risk_gross_max[fill].clamp(min=0.0, max=1.0)
    scale = torch.where(gross > hard, hard / gross.clamp_min(1e-18), 1.0)
    target = target * scale.unsqueeze(-1)
    target[:, sequence.cash_index] = 1.0 - target.sum(dim=-1)
    return target


def _construct_path(
    control_id: str,
    sequence: Hold30DatasetSequence,
    *,
    discretionary_target: Callable[[int, int, torch.Tensor], torch.Tensor] | None,
    outer_start: int,
    score_mask: torch.Tensor | None,
    fitting_rows: Iterable[int],
) -> Hold30ControlGrossTrace:
    _check_market(sequence)
    rows, batch, assets = sequence.asset_returns.shape
    weights = sequence.asset_returns.new_zeros((rows + 1, batch, assets))
    weights[0] = sequence.c1_benchmark_weights[0]
    pretrade = sequence.asset_returns.new_zeros((rows, batch, assets))
    gross = sequence.asset_returns.new_zeros((rows, batch))
    deltas = {
        cause: sequence.asset_returns.new_zeros((rows, batch, assets))
        for cause in _CAUSES
    }
    for row in range(rows):
        pretrade[row], gross[row] = _drift(weights[row], sequence.asset_returns[row])
        membership, availability, repaired, forced = _forced_stages(
            pretrade[row],
            sequence,
            row + 1,
        )
        del membership, availability
        for cause, value in forced.items():
            deltas[cause][row] = value
        final = repaired
        if discretionary_target is not None:
            desired = discretionary_target(row, row + 1, repaired)
            requested = desired - repaired
            turnover = 0.5 * requested.abs().sum(dim=-1)
            scale = torch.where(
                turnover > 0.10,
                requested.new_tensor(0.10) / turnover.clamp_min(1e-18),
                1.0,
            )
            discretionary = requested * scale.unsqueeze(-1)
            deltas[TurnoverCause.DISCRETIONARY][row] = discretionary
            final = repaired + discretionary
        weights[row + 1] = final
    strategy = {
        "C2": "daily_pit_equal_weight_common_execution",
        "C3": "initial_universe_hold_until_forced_exit",
    }[control_id]
    return _trace(
        control_id,
        sequence,
        weights=weights,
        pretrade_weights=pretrade,
        gross_returns=gross,
        deltas=deltas,
        score_mask=_score_mask(sequence, score_mask),
        outer_start=outer_start,
        fitting_rows=fitting_rows,
        source_receipt_sha256=sequence.provenance.receipt_id,
        strategy_inputs_sha256=_payload_sha256(
            {
                "rule": strategy,
                "axis_id": sequence.axis_id,
                "initial_weights_sha256": _tensor_sha256(weights[0]),
            }
        ),
    )


def construct_c2_daily_equal_weight(
    sequence: Hold30DatasetSequence,
    *,
    outer_start: int,
    score_mask: torch.Tensor | None = None,
    fitting_rows: Iterable[int] = (),
) -> Hold30ControlGrossTrace:
    """Construct daily PIT equal weight through the common forced/action stages."""

    return _construct_path(
        "C2",
        sequence,
        discretionary_target=lambda decision, fill, _repaired: _equal_weight_target(
            sequence,
            decision=decision,
            fill=fill,
        ),
        outer_start=outer_start,
        score_mask=score_mask,
        fitting_rows=fitting_rows,
    )


def construct_c3_initial_universe_hold(
    sequence: Hold30DatasetSequence,
    *,
    outer_start: int,
    score_mask: torch.Tensor | None = None,
    fitting_rows: Iterable[int] = (),
) -> Hold30ControlGrossTrace:
    """Hold the initial C1 book; send every mandatory-exit proceed to CASH."""

    return _construct_path(
        "C3",
        sequence,
        discretionary_target=None,
        outer_start=outer_start,
        score_mask=score_mask,
        fitting_rows=fitting_rows,
    )


@dataclass(frozen=True, slots=True)
class Hold30MomentumScores:
    values: torch.Tensor
    valid: torch.Tensor
    stable_rank: torch.Tensor
    receipt_sha256: str


def c4_momentum_scores(
    sequence: Hold30DatasetSequence,
    split_adjusted_close: torch.Tensor,
    close_valid: torch.Tensor,
    close_known_at_ms: torch.Tensor,
) -> Hold30MomentumScores:
    """Frozen trailing-21-session z-score with stable-ID tie ranks.

    The close tensors must include at least 21 sessions preceding the first
    economic position.  This prevents the warm-up portfolio from receiving an
    invented zero signal merely because the evaluation slice began.
    """

    _check_market(sequence)
    if (
        not isinstance(split_adjusted_close, torch.Tensor)
        or split_adjusted_close.ndim != 3
    ):
        raise Hold30ControlError(
            "split_adjusted_close must have shape [history_position, batch, asset]"
        )
    history_positions = int(split_adjusted_close.shape[0])
    history_shape = (history_positions, sequence.batch_size, sequence.num_assets)
    _require_float64("split_adjusted_close", split_adjusted_close, history_shape)
    history_offset = history_positions - sequence.n_positions
    if history_offset < HOLD30_MOMENTUM_SESSIONS:
        raise Hold30ControlError(
            "C4 requires at least 21 pre-evaluation close sessions"
        )
    if (
        not isinstance(close_valid, torch.Tensor)
        or close_valid.dtype != torch.bool
        or tuple(close_valid.shape) != history_shape
    ):
        raise Hold30ControlError(
            "close_valid must be boolean [history_position, batch, asset]"
        )
    if (
        not isinstance(close_known_at_ms, torch.Tensor)
        or close_known_at_ms.dtype != torch.int64
        or tuple(close_known_at_ms.shape) != history_shape
    ):
        raise Hold30ControlError(
            "close_known_at_ms must be int64 [history_position, batch, asset]"
        )
    if bool((split_adjusted_close.masked_select(close_valid) <= 0).any()):
        raise Hold30ControlError("every valid split-adjusted close must be positive")
    current_known = close_known_at_ms[history_offset:]
    current_valid = close_valid[history_offset:]
    lag_known = close_known_at_ms[
        history_offset - HOLD30_MOMENTUM_SESSIONS : history_positions
        - HOLD30_MOMENTUM_SESSIONS
    ]
    lag_valid = close_valid[
        history_offset - HOLD30_MOMENTUM_SESSIONS : history_positions
        - HOLD30_MOMENTUM_SESSIONS
    ]
    legally_used = current_valid & lag_valid
    decision_times = sequence.decision_timestamps_ms.view(-1, 1, 1).expand(
        sequence.n_positions,
        sequence.batch_size,
        sequence.num_assets,
    )
    if bool(
        (
            torch.maximum(current_known, lag_known)
            .to(device="cpu")
            .masked_select(legally_used.to(device="cpu"))
            > decision_times.masked_select(legally_used.to(device="cpu"))
        ).any()
    ):
        raise Hold30ControlError(
            "C4 close was not legally available at its decision timestamp"
        )

    output_shape = (sequence.n_positions, sequence.batch_size, sequence.num_assets)
    scores = split_adjusted_close.new_zeros(output_shape)
    valid = torch.zeros(
        output_shape, dtype=torch.bool, device=split_adjusted_close.device
    )
    ranks = torch.full(
        output_shape, -1, dtype=torch.int64, device=split_adjusted_close.device
    )
    risky = torch.ones(
        (sequence.num_assets,), dtype=torch.bool, device=split_adjusted_close.device
    )
    risky[sequence.cash_index] = False
    stable_order = tuple(
        sorted(range(sequence.num_assets), key=lambda index: sequence.asset_ids[index])
    )
    for position in range(sequence.n_positions):
        close_position = history_offset + position
        lag_position = close_position - HOLD30_MOMENTUM_SESSIONS
        eligible = (
            close_valid[close_position]
            & close_valid[lag_position]
            & sequence.decision_trade[position]
            & risky.view(1, -1)
        )
        momentum = torch.log(split_adjusted_close[close_position]) - torch.log(
            split_adjusted_close[lag_position]
        )
        for batch_index in range(sequence.batch_size):
            indices = [
                index for index in stable_order if bool(eligible[batch_index, index])
            ]
            if not indices:
                continue
            valid[position, batch_index, indices] = True
            ranked = sorted(
                indices,
                key=lambda index: (
                    -float(momentum[batch_index, index].detach().cpu()),
                    sequence.asset_ids[index],
                ),
            )
            ranks[position, batch_index, ranked] = torch.arange(
                len(ranked), device=ranks.device
            )
            if len(indices) < 2:
                continue
            values = momentum[batch_index, indices]
            mean = values.mean()
            variance = ((values - mean) ** 2).mean()
            if bool(variance <= 0):
                continue
            scores[position, batch_index, indices] = (
                (values - mean) / variance.sqrt()
            ).clamp(-2.0, 2.0)
    payload = {
        "schema_version": 1,
        "rule": "trailing21_split_adjusted_log_return_within_date_zscore",
        "winsor": [-2.0, 2.0],
        "stable_asset_ids": list(sequence.asset_ids),
        "axis_id": sequence.axis_id,
        "history_offset": history_offset,
        "close_sha256": _tensor_sha256(split_adjusted_close),
        "close_valid_sha256": _tensor_sha256(close_valid),
        "close_known_at_sha256": _tensor_sha256(close_known_at_ms),
        "scores_sha256": _tensor_sha256(scores),
        "stable_rank_sha256": _tensor_sha256(ranks),
    }
    return Hold30MomentumScores(scores, valid, ranks, _payload_sha256(payload))


class _C4Policy(nn.Module):
    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del available, age_summaries
        return Hold30Intent(
            entry_scores=state_t[..., 0],
            hazard_residual=torch.zeros_like(prev_weights),
            exposure_residual=torch.zeros(
                prev_weights.shape[0],
                dtype=prev_weights.dtype,
                device=prev_weights.device,
            ),
        )


def construct_c4_momentum(
    sequence: Hold30DatasetSequence,
    split_adjusted_close: torch.Tensor,
    close_valid: torch.Tensor,
    close_known_at_ms: torch.Tensor,
    *,
    outer_start: int,
    score_mask: torch.Tensor | None = None,
    fitting_rows: Iterable[int] = (),
) -> Hold30ControlGrossTrace:
    """Run frozen C4 scores through the canonical H2 builder with zero residuals."""

    momentum = c4_momentum_scores(
        sequence,
        split_adjusted_close,
        close_valid,
        close_known_at_ms,
    )
    initial = CohortLedger.from_staggered_endowment(
        sequence.c1_benchmark_weights[0],
        cash_index=sequence.cash_index,
        youngest_age=0,
        oldest_age=29,
        track_initial_units=False,
    )
    c4_available = (sequence.decision_trade & momentum.valid).clone()
    c4_available[..., sequence.cash_index] = True
    runtime_sequence = Hold30Sequence(
        decision_state=momentum.values.unsqueeze(-1),
        asset_returns=sequence.asset_returns,
        decision_available=c4_available,
        fill_membership=sequence.fill_membership,
        fill_availability=sequence.fill_tradability,
        benchmark_weights=sequence.c1_benchmark_weights,
        risk_asset_caps=sequence.risk_asset_caps,
        risk_gross_max=sequence.risk_gross_max,
        benchmark_net_returns=sequence.c1_benchmark_net_returns,
        initial_ledger=initial,
        # The primary 20-bp chronology is canonical.  Under the v1 common
        # execution contract, proportional cost financing changes NAV but not
        # normalized portfolio weights, so this exact action trace can be
        # repriced at 10 and 40 bp without re-evaluating the rule.
        cost_rate=HOLD30_PRIMARY_COST_BPS / 10_000.0,
        track_entry_units=sequence.roles.score[:-1],
        axis_id=sequence.axis_id,
    )
    runtime = Hold30ChronologicalRuntime("H2")
    policy = _C4Policy()
    state = runtime.initial_state(runtime_sequence)
    weights = [state.ledger.weights]
    pretrade: list[torch.Tensor] = []
    gross: list[torch.Tensor] = []
    deltas: dict[TurnoverCause, list[torch.Tensor]] = {cause: [] for cause in _CAUSES}
    for row in range(runtime_sequence.n_positions - 1):
        state = runtime.decide(
            policy,
            runtime_sequence,
            state,
            decision_state=momentum.values[row].unsqueeze(-1),
        )
        state, transition = runtime.advance(runtime_sequence, state)
        pretrade.append(transition.execution_pretrade_weights)
        gross.append(transition.holding_return)
        deltas[TurnoverCause.STARTUP].append(torch.zeros_like(transition.filled_delta))
        deltas[TurnoverCause.MEMBERSHIP_FORCED].append(
            transition.membership_repaired_weights
            - transition.execution_pretrade_weights
        )
        deltas[TurnoverCause.AVAILABILITY_FORCED].append(
            transition.availability_repaired_weights
            - transition.membership_repaired_weights
        )
        deltas[TurnoverCause.RISK_FORCED].append(
            transition.risk_repaired_weights - transition.availability_repaired_weights
        )
        deltas[TurnoverCause.DISCRETIONARY].append(
            transition.pre_cost_weights - transition.risk_repaired_weights
        )
        deltas[TurnoverCause.TERMINAL].append(torch.zeros_like(transition.filled_delta))
        weights.append(state.ledger.weights)
    return _trace(
        "C4",
        sequence,
        weights=torch.stack(weights),
        pretrade_weights=torch.stack(pretrade),
        gross_returns=torch.stack(gross),
        deltas={cause: torch.stack(values) for cause, values in deltas.items()},
        score_mask=_score_mask(sequence, score_mask),
        outer_start=outer_start,
        fitting_rows=fitting_rows,
        source_receipt_sha256=sequence.provenance.receipt_id,
        strategy_inputs_sha256=momentum.receipt_sha256,
    )


__all__ = [
    "HOLD30_CONTROL_IDS",
    "HOLD30_COST_RUNGS_BPS",
    "HOLD30_MOMENTUM_SESSIONS",
    "HOLD30_PRIMARY_COST_BPS",
    "Hold30ControlError",
    "Hold30ControlGrossTrace",
    "Hold30CostLadder",
    "Hold30CostRung",
    "Hold30MomentumScores",
    "accept_c1_bound_trace",
    "assemble_hold30_control_trace",
    "c4_momentum_scores",
    "construct_c0_cash",
    "construct_c2_daily_equal_weight",
    "construct_c3_initial_universe_hold",
    "construct_c4_momentum",
    "price_hold30_cost_ladder",
]
