"""Scalar SIGNED-POSITION DOLLAR transition-P&L path: one instrument, P&L = units * mid-diff * scale.

``transition_pnl`` is torch-free arithmetic (runs on tensors OR python scalars) so the env step, eval loop,
and pretraining-target builder share one engine; ``fill_indices`` is the single tensor-only helper
(vectorized counterpart of scalar ``fill_index``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_quant.execution.fills import (
    MarketSnapshot,
    _fill_price,
)
from rl_quant.execution.types import (
    ExecutionConfig,
    FillLevel,
    TerminalPolicy,
)
from rl_quant.execution.validation import (
    _coerce_finite,
    _coerce_positive_price,
    _require_int_allow_negative,
    _require_nonnegative_int,
    _require_positive_int,
)


@dataclass(frozen=True)
class PositionState:
    position: float  # signed units; intraday {-1, 0, 1}; weight-aware paths may use a weight
    bars_held: int = 0
    entry_price: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _coerce_finite("position", self.position))
        object.__setattr__(self, "bars_held", _require_nonnegative_int("bars_held", self.bars_held))
        if self.entry_price is not None:
            object.__setattr__(self, "entry_price", _coerce_positive_price("entry_price", self.entry_price))
        # A flat book holds no open position, so it carries no entry price -- the same invariant the
        # terminal-liquidation fix enforces, applied generally so no path can leave stale entry state.
        if self.position == 0.0 and self.entry_price is not None:
            raise ValueError("a flat position (0) must not carry an entry_price.")


@dataclass(frozen=True)
class TransitionOutcome:
    old_latency_return: float  # old * (mid_fill - mid_now) * scale   (held leg, now -> fill)
    new_interval_return: float  # new * (mid_next - mid_fill) * scale  (new leg, fill -> next)
    gross_return: float
    entry_fill_price: float | None  # price the NEW leg is acquired at (None for proxy fills)
    exit_fill_price: float | None  # close-out price at terminal liquidation (None otherwise)
    entry_cost: float  # turnover spread+fees cost at the fill bar (>= 0)
    exit_cost: float  # terminal liquidation spread+fees cost at the next bar (>= 0)
    impact_cost: float  # market-impact dollars (0 unless a *_plus_impact level)
    total_cost: float  # entry_cost + exit_cost + impact_cost
    net_return: float  # gross_return - total_cost  == the trainer reward
    order_legs: float  # |new - old| turnover units; a full reversal (-1 -> +1) is 2
    real_executable_fill_model: bool
    next_state: PositionState


def fill_index(now_index: int, *, step_horizon: int, latency_steps: int) -> int:
    """Bar at which a decision fills: ``min(now + latency, next)``, capped at the next decision bar.

    ``latency_steps <= 0`` collapses the fill to the current bar (decision and fill coincide). For valid
    (non-negative) bar indices this mirrors the vectorized ``fill_indices`` element-wise, so a fill can never
    be pushed past the holding horizon. ``now_index`` must be a non-negative integer bar index -- a negative
    value would silently index from the end of the array (PyTorch negative-indexing footgun) downstream."""
    now = _require_nonnegative_int("now_index", now_index)
    horizon = _require_positive_int("step_horizon", step_horizon)
    latency = _require_int_allow_negative("latency_steps", latency_steps)
    next_index = now + horizon
    if latency <= 0:
        return now
    return min(now + latency, next_index)


# Only wide signed integer dtypes are safe as bar indices: uint8/int8/int16 (and unsigned generally)
# overflow once a fill bar exceeds their tiny range when adding the horizon/latency offset.
_INTEGER_INDEX_DTYPES = (torch.int32, torch.int64)


def fill_indices(now_indices: torch.Tensor, *, step_horizon: int, latency_steps: int) -> torch.Tensor:
    """Vectorized counterpart of scalar :func:`fill_index`: the per-element fill bar for a batch of
    decision bars, ``min(now + latency, now + step_horizon)`` with ``latency <= 0`` collapsing to ``now``.

    The single source of truth for vectorized fill timing (intraday env step + pretraining targets). It is
    byte-for-byte equal to ``fill_index`` applied element-wise (``torch.minimum`` is the per-element min),
    preserves the input tensor's device and dtype, validates the same integer-like bar/latency arguments as
    the scalar version (no silent fractional truncation), and requires an integer index tensor."""
    horizon = _require_positive_int("step_horizon", step_horizon)
    latency = _require_int_allow_negative("latency_steps", latency_steps)
    if now_indices.dtype not in _INTEGER_INDEX_DTYPES:
        raise ValueError(
            f"now_indices must be an int32/int64 index tensor (smaller/unsigned dtypes overflow); "
            f"got dtype {now_indices.dtype}."
        )
    if latency <= 0:
        return now_indices
    next_indices = now_indices + horizon
    return torch.minimum(now_indices + latency, next_indices)


def transition_pnl(
    old_position,
    new_position,
    mid_now,
    mid_fill,
    mid_next,
    half_spread_fill,
    half_spread_next,
    terminal,
    *,
    trade_scale: float,
    commission_per_share: float,
    extra_cost_per_share: float,
):
    """Vectorized ``delayed_close`` transition reward in scaled dollars -- the single source of truth for
    the intraday env-step / eval / pretraining-target reward (previously inlined three times).

    Operates on torch tensors OR python scalars via ``+ - * abs`` only (no torch import, no ``.float()``
    casts -- long*float and bool*float promote identically), so the same function serves the vectorized
    env/grid and the scalar eval loop. The held (old) position earns ``mid_now -> mid_fill`` (the latency
    leg) and the new position earns ``mid_fill -> mid_next``; turnover pays the fill-bar cost and a true
    terminal pays an extra ``|new|`` liquidation cost at the next bar. Equals
    ``simulate_transition(...).net_return`` for the delayed_close fill level (cross-checked in tests)."""
    cost_fill = half_spread_fill + extra_cost_per_share + commission_per_share
    cost_next = half_spread_next + extra_cost_per_share + commission_per_share
    turnover = abs(new_position - old_position)
    return (
        old_position * (mid_fill - mid_now)
        + new_position * (mid_next - mid_fill)
        - turnover * cost_fill
        - terminal * abs(new_position) * cost_next
    ) * trade_scale


def _base_cost_per_share(snapshot: MarketSnapshot, *, buying: bool, config: ExecutionConfig) -> float:
    """Spread + fees cost per share (NO impact). Proxy levels use the half-spread; quote-side levels
    use the actual crossing distance |fill_price - mid|."""
    if config.fill_level in (FillLevel.DELAYED_CLOSE, FillLevel.MID_PLUS_SPREAD):
        spread_cost = config.spread_multiplier * snapshot.half_spread
    else:
        fill_px = _fill_price(snapshot, buying=buying, config=config)
        spread_cost = abs(float(fill_px) - snapshot.mid)
    return spread_cost + config.extra_cost_per_share + config.commission_per_share


def _impact_per_share(turnover_units: float, config: ExecutionConfig) -> float:
    if config.fill_level == FillLevel.QUOTE_SIDE_PLUS_IMPACT and config.impact_model.kind == "linear":
        return config.impact_model.coef_per_unit * float(turnover_units)
    return 0.0


def simulate_transition(
    state: PositionState,
    action_position: float,
    now: MarketSnapshot,
    fill: MarketSnapshot,
    nxt: MarketSnapshot,
    *,
    is_terminal: bool,
    config: ExecutionConfig,
) -> TransitionOutcome:
    """Pure per-step transition P&L. ``now``/``fill``/``nxt`` are point-in-time dataset snapshots at the
    current, fill, and next bars; ``state``/``action_position``/``is_terminal`` come from the env. With
    ``fill_level=delayed_close`` this reproduces the intraday reward exactly."""
    scale = config.trade_scale
    old = float(state.position)
    new = float(action_position)
    turnover_units = abs(new - old)

    old_latency_return = old * (fill.mid - now.mid) * scale  # old leg keeps earning until its fill
    new_interval_return = new * (nxt.mid - fill.mid) * scale  # new leg earns from fill to next
    gross_return = old_latency_return + new_interval_return

    entry_cost = 0.0
    impact_cost = 0.0
    entry_fill_price: float | None = None
    if turnover_units > 0.0:
        buying = new > old
        entry_cost = turnover_units * _base_cost_per_share(fill, buying=buying, config=config) * scale
        impact_cost += _impact_per_share(turnover_units, config) * turnover_units * scale
        entry_fill_price = _fill_price(fill, buying=buying, config=config)

    exit_cost = 0.0
    exit_fill_price: float | None = None
    liquidating = is_terminal and config.terminal_policy == TerminalPolicy.LIQUIDATE_AT_NEXT and new != 0.0
    if liquidating:
        units = abs(new)
        buying = new < 0.0  # closing a short is a buy; closing a long is a sell
        exit_cost = units * _base_cost_per_share(nxt, buying=buying, config=config) * scale
        impact_cost += _impact_per_share(units, config) * units * scale
        exit_fill_price = _fill_price(nxt, buying=buying, config=config)

    total_cost = entry_cost + exit_cost + impact_cost
    net_return = gross_return - total_cost

    if liquidating:
        # A terminal liquidation flattens the book: the (now-closed) leg carries no held bars or entry
        # price forward -- otherwise a hold-into-terminal would leave a flat position with stale state.
        next_state = PositionState(position=0.0, bars_held=0, entry_price=None)
    else:
        held = new == old
        # A transition that ends FLAT carries no entry price -- the close-out price is recorded on the
        # outcome (entry_fill_price), not on the next state. (For quote-side fills entry_fill_price is a
        # real bid/ask, so without this a flat next_state would carry a stale exit price.)
        if new == 0.0:
            next_entry_price: float | None = None
        else:
            next_entry_price = state.entry_price if held else entry_fill_price
        next_state = PositionState(
            position=new,
            bars_held=state.bars_held + 1 if held else 0,
            entry_price=next_entry_price,
        )
    return TransitionOutcome(
        old_latency_return=old_latency_return,
        new_interval_return=new_interval_return,
        gross_return=gross_return,
        entry_fill_price=entry_fill_price,
        exit_fill_price=exit_fill_price,
        entry_cost=entry_cost,
        exit_cost=exit_cost,
        impact_cost=impact_cost,
        total_cost=total_cost,
        net_return=net_return,
        order_legs=turnover_units,
        real_executable_fill_model=config.real_executable_fill_model,
        next_state=next_state,
    )
