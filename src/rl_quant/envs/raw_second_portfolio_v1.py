"""One shared cash/share ledger for raw-second PPO collection and evaluation.

Second-OPEN/whole-share/volume-participation fills are an explicit research
proxy, NOT observed quotes, queue position, or guaranteed capacity. Market
responses are never converted to policy features. Corporate actions reuse the
existing exact-decimal ledger and leave archived unadjusted OHLCV untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, localcontext
import math
from typing import Sequence
from zoneinfo import ZoneInfo

import torch

from rl_quant.datasets.massive_raw_seconds_v1 import RawSecondCatalog, _rows
from rl_quant.datasets.raw_second_economics_v1 import SecondSession
from rl_quant.execution.qt200_aggregate_execution_v1 import (
    Book, Dividend, Split, apply_actions, equity, liquidate,
)
from rl_quant.rl.types import ActionBatch, ObservationBatch, RewardComponents, TransitionBatch

D = Decimal


def _session(stamp: int) -> str:
    return datetime.fromtimestamp(stamp / 1000, timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()


@dataclass(frozen=True)
class SecondExecutionConfig:
    capital: str = "10000000"
    cost_basis_points: int = 20
    maximum_fill_participation: str = "0.02"
    maximum_asset_weight: str = "0.10"
    maximum_gross_exposure: str = "1"
    maximum_drawdown: str = "0.25"
    decision_interval_seconds: int = 300
    observation_session: str = "all-captured"
    execution_session: str = "all-captured"
    order_expiry: str = "next-decision"

    def __post_init__(self) -> None:
        for name in ("capital", "maximum_fill_participation", "maximum_asset_weight",
                     "maximum_gross_exposure", "maximum_drawdown"):
            value = getattr(self, name)
            if not isinstance(value, str) or not D(value).is_finite() or D(value) <= 0:
                raise ValueError(f"Invalid decimal execution setting {name}")
            if name != "capital" and D(value) > 1:
                raise ValueError(f"{name} exceeds one")
        if type(self.cost_basis_points) is not int or self.cost_basis_points not in (10, 20, 40):
            raise ValueError("Use the registered 10/20/40-bp sensitivity rungs")
        if type(self.decision_interval_seconds) is not int or self.decision_interval_seconds <= 0:
            raise ValueError("Decision interval must be positive, independently of one-second data")
        if (self.observation_session not in ("all-captured", "regular-only")
                or self.execution_session not in ("all-captured", "regular-only")
                or self.order_expiry not in ("next-decision", "session-close")):
            raise ValueError("Explicit observation/execution/expiry policy required")


@dataclass(frozen=True)
class SecondFill:
    asset_id: str
    second_start_ms: int
    signed_shares: str
    price: str
    fee: str


class RawSecondPortfolioEnv:
    """Chronological, carried-book engineering environment, one episode.

    Unresolved identity/corporate-action coverage cannot be authorized by this
    numerical interface. Real-source registration is a separate prerequisite.
    No source-ready/profitability-authorizing flag is exposed here.
    """

    def __init__(self, catalog: RawSecondCatalog, *, config: SecondExecutionConfig | None = None,
                 device: str | torch.device, splits: Sequence[Split] = (), dividends: Sequence[Dividend] = (),
                 sessions: Sequence[SecondSession] = ()):
        if len(catalog.windows) < 2:
            raise ValueError("At least one decision and one terminal window required")
        self.catalog = catalog
        self.config = SecondExecutionConfig() if config is None else config
        self.device = torch.device(device)
        self.splits, self.dividends = tuple(splits), tuple(dividends)
        self.sessions = tuple(sessions)
        if (self.config.observation_session == "regular-only" or self.config.execution_session == "regular-only"
                or self.config.order_expiry == "session-close") and not self.sessions:
            raise ValueError("A bound session calendar is required, including early closes")
        ids = [e.event_id for e in (*self.splits, *self.dividends)]
        if len(ids) != len(set(ids)) or any(e.instrument not in catalog.asset_ids for e in (*self.splits, *self.dividends)):
            raise ValueError("Corporate-action identity differs")
        times = [w.decision_ms for w in catalog.windows]
        if times != sorted(set(times)):
            raise ValueError("Decision windows must be strictly chronological")
        for left, right in zip(times, times[1:]):
            if _session(left) == _session(right) and right - left != self.config.decision_interval_seconds * 1000:
                raise ValueError("Decision frequency differs from registered clock")
        if self.sessions and any(not any(s.open_ms <= t <= s.close_ms for s in self.sessions) for t in times):
            raise ValueError("Decision lies outside the declared session calendar")
        if self.config.observation_session == "regular-only":
            for window in catalog.windows:
                if not any(s.open_ms <= window.start_ms and window.start_ms + window.seconds * 1000 <= s.close_ms
                           for s in self.sessions):
                    raise ValueError("Raw observation context extends outside a regular session")
        self.reset()

    def _market(self, index: int) -> dict[str, list]:
        ref = self.catalog.windows[index]
        result = {}
        for asset, capture in zip(ref.asset_ids, ref.captures, strict=True):
            query, pages = capture.load()
            result[asset] = _rows(query, pages)
        return result

    def _marks(self, index: int, *, available_only: bool) -> dict[str, Decimal]:
        ref = self.catalog.windows[index]
        prices = {}
        for asset, rows in self._market(index).items():
            eligible = []
            for stamp, values, received in rows:
                scope = self.config.observation_session if available_only else self.config.execution_session
                if scope == "regular-only" and not any(s.open_ms <= stamp and stamp + 1000 <= s.close_ms for s in self.sessions):
                    continue
                arrival = received if ref.contract.availability_assumption == "captured-receipt-time" else stamp + 1000 + ref.contract.availability_delay_ms
                if stamp + 1000 <= ref.decision_ms and (not available_only or arrival <= ref.decision_ms):
                    eligible.append(values[3])
            if eligible:
                prices[asset] = D(str(eligible[-1]))
        # A last-known mark belongs only in the ledger, never the market tensor.
        return {**(self.known_marks if available_only else self.last_marks), **prices}

    def _actions_through(self, stamp: int) -> None:
        target = _session(stamp)
        dates = {target}
        dates.update(e.effective_date for e in self.splits if (self.book.action_date or "") < e.effective_date <= target)
        dates.update(e.ex_date for e in self.dividends if (self.book.action_date or "") < e.ex_date <= target)
        for session in sorted(dates):
            if self.book.action_date is None or session > self.book.action_date:
                split_rows = [e for e in self.splits if e.effective_date == session]
                self.book = apply_actions(self.book, session, splits=split_rows,
                                           dividends=[e for e in self.dividends if e.ex_date == session])
                for event in split_rows:
                    if event.instrument in self.last_marks:
                        self.last_marks[event.instrument] *= event.shares_from / event.shares_to
                    if event.instrument in self.known_marks:
                        self.known_marks[event.instrument] *= event.shares_from / event.shares_to

    def observation(self) -> ObservationBatch:
        self.catalog.load(self.index, device=self.device)  # validate coverage before acting
        prices = self._marks(self.index, available_only=True)
        self.known_marks = prices
        shares = dict(self.book.holdings)
        account = [float(self.book.cash), *(float(shares.get(asset, D(0))) for asset in self.catalog.asset_ids)]
        mask = [True, *(asset in prices and not self.risk_halted for asset in self.catalog.asset_ids)]
        return ObservationBatch(
            tensors={"raw_window_index": torch.tensor([[self.index]], dtype=torch.int64, device=self.device),
                     "raw_catalog_sha256": self.catalog.identity_tensor(device=self.device)[None],
                     "account_state": torch.tensor([account], dtype=torch.float32, device=self.device)},
            action_mask=torch.tensor([mask], dtype=torch.bool, device=self.device),
            episode_start=torch.tensor([self.index == 0], dtype=torch.bool, device=self.device),
        )

    def reset(self) -> tuple[ObservationBatch, dict]:
        self.index = 0
        self.book = Book(D(self.config.capital))
        self.last_marks: dict[str, Decimal] = {}
        self.known_marks: dict[str, Decimal] = {}
        self.peak = D(self.config.capital)
        self.risk_halted = False
        self.fills: list[SecondFill] = []
        self.audit: list[dict] = []
        self._actions_through(self.catalog.windows[0].decision_ms)
        self.last_marks = self._marks(0, available_only=False)
        self.current_equity = equity(self.book, self.last_marks)
        return self.observation(), {}

    def _execute(self, requested: torch.Tensor, decision: int, end: int, *, submit_orders: bool = True) -> tuple[list[SecondFill], Decimal]:
        # Weights -> fixed whole-share orders at decision-known ledger marks.
        weights = [D(str(v)) for v in requested.detach().cpu().tolist()]
        prices = self._marks(self.index, available_only=True)
        wealth = equity(self.book, prices)
        risky = [min(w, D(self.config.maximum_asset_weight)) for w in weights[1:]]
        gross = sum(risky, D(0))
        if gross > D(self.config.maximum_gross_exposure):
            risky = [w * D(self.config.maximum_gross_exposure) / gross for w in risky]
        shares = dict(self.book.holdings)
        remaining = {}
        for asset, weight in zip(self.catalog.asset_ids, risky, strict=True):
            desired = (wealth * weight / prices[asset]).to_integral_value(rounding=ROUND_FLOOR) if asset in prices else D(0)
            remaining[asset] = desired - shares.get(asset, D(0))
        if not submit_orders:
            remaining = {a: D(0) for a in remaining}
        requested_orders = {a: str(q) for a, q in remaining.items()}
        requested_notional = sum((abs(q) * prices.get(a, D(0)) for a, q in remaining.items()), D(0))
        expiry = end
        if self.config.order_expiry == "session-close":
            expiry = min(end, next(s.close_ms for s in self.sessions if s.open_ms <= decision <= s.close_ms))
        market = self._market(self.index + 1)
        events: dict[int, dict] = {}
        for asset, rows in market.items():
            query, _ = self.catalog.windows[self.index + 1].captures[self.catalog.asset_ids.index(asset)].load()
            if query.start_ms > decision or query.end_ms + 1000 < end:
                raise ValueError("Unknown execution coverage; do not invent zero liquidity")
            for stamp, values, _ in rows:
                eligible_session = self.config.execution_session == "all-captured" or any(
                    s.open_ms <= stamp and stamp + 1000 <= s.close_ms for s in self.sessions)
                if decision < stamp and stamp + 1000 <= expiry and eligible_session:
                    events.setdefault(stamp, {})[asset] = (D(str(values[0])), D(str(values[4])))
        cash, fills = self.book.cash, []
        rate = D(self.config.cost_basis_points) / 10_000
        for stamp, bars in sorted(events.items()):
            if _session(stamp) != self.book.action_date:
                # Carry the actual book across sessions; rebase pending shares
                # for splits before this session. No implicit daily CASH reset.
                self.book = replace(self.book, cash=cash, holdings=tuple(sorted((a, q) for a, q in shares.items() if q)))
                previous_action_date = self.book.action_date or ""
                self._actions_through(stamp)
                shares = dict(self.book.holdings)
                cash = self.book.cash
                for event in self.splits:
                    if previous_action_date < event.effective_date <= self.book.action_date:
                        remaining[event.instrument] *= event.shares_to / event.shares_from
            caps = {a: (volume * D(self.config.maximum_fill_participation)).to_integral_value(rounding=ROUND_FLOOR)
                    for a, (_, volume) in bars.items()}
            for asset, (price, _) in sorted(bars.items()):
                sold = min(max(-remaining[asset], D(0)), caps[asset], shares.get(asset, D(0)))
                if sold:
                    fee = sold * price * rate
                    cash += sold * price - fee
                    shares[asset] -= sold
                    remaining[asset] += sold
                    fills.append(SecondFill(asset, stamp, str(-sold), str(price), str(fee)))
            buys = {a: min(max(remaining[a], D(0)), caps[a]) for a in bars}
            required = sum((q * bars[a][0] * (1 + rate) for a, q in buys.items()), D(0))
            scale = min(D(1), cash / required) if required else D(0)
            for asset, quantity in sorted(buys.items()):
                bought = (quantity * scale).to_integral_value(rounding=ROUND_FLOOR)
                if bought:
                    price = bars[asset][0]
                    fee = bought * price * rate
                    cash -= bought * price + fee
                    shares[asset] = shares.get(asset, D(0)) + bought
                    remaining[asset] -= bought
                    fills.append(SecondFill(asset, stamp, str(bought), str(price), str(fee)))
        self.book = replace(self.book, cash=cash, holdings=tuple(sorted((a, q) for a, q in shares.items() if q)))
        self._last_orders = dict(requested_orders=requested_orders, unfilled_shares={a: str(q) for a, q in remaining.items()},
                                 requested_notional=str(requested_notional), expiry_ms=expiry)
        return fills, sum((D(f.fee) for f in fills), D(0))

    @torch.no_grad()
    def step(self, action: ActionBatch, *, submit_orders: bool = True) -> TransitionBatch:
        if self.index >= len(self.catalog.windows) - 1:
            raise ValueError("Episode is already complete")
        observation = self.observation()
        requested = action.action
        if (requested.shape != observation.action_mask.shape or requested.device != self.device
                or bool((requested < 0).any()) or not bool(torch.isclose(requested.sum(-1), torch.ones(1, device=self.device), atol=1e-6).all())
                or bool((requested[~observation.action_mask] != 0).any())):
            raise ValueError("Requested allocation is outside the masked simplex")
        before = self.current_equity
        decision = self.catalog.windows[self.index].decision_ms
        end = self.catalog.windows[self.index + 1].decision_ms
        with localcontext() as context:
            context.prec = 34
            fills, fees = self._execute(requested[0], decision, end, submit_orders=submit_orders)
            self.index += 1
            self._actions_through(end)
            self.last_marks = self._marks(self.index, available_only=False)
            after = equity(self.book, self.last_marks)
            self.peak = max(self.peak, after)
            self.risk_halted |= 1 - after / self.peak > D(self.config.maximum_drawdown)
            terminal = self.index == len(self.catalog.windows) - 1
            liquidation_fee = D(0)
            pre_liquidation_holdings = self.book.holdings
            liquidation_notional = sum((q * self.last_marks[a] for a, q in pre_liquidation_holdings), D(0)) if terminal else D(0)
            if terminal:
                self.book, liquidation_fee = liquidate(self.book, self.last_marks, cost_bps=D(self.config.cost_basis_points))
                after = equity(self.book, self.last_marks)
            self.current_equity = after
        self.fills.extend(fills)
        self.audit.append(dict(decision_ms=decision, end_ms=end, requested_weights=requested[0].cpu().tolist(),
            **self._last_orders, fills=[vars(f) for f in fills], equity_before=str(before), equity_after=str(after),
            cash=str(self.book.cash), holdings=[(a, str(q)) for a, q in self.book.holdings],
            receivables=[dict(event_id=r.event_id, amount=str(r.amount), payable_date=r.payable_date) for r in self.book.receivables],
            applied_events=self.book.applied_events, transaction_costs=str(fees), terminal_liquidation_cost=str(liquidation_fee),
            terminal_liquidation_notional=str(liquidation_notional),
            pre_liquidation_holdings=[(a, str(q)) for a, q in pre_liquidation_holdings] if terminal else [],
            marks={a: str(p) for a, p in self.last_marks.items()},
            reward_net_log_equity=math.log(float(after / before))))
        if before <= 0 or after <= 0:
            raise ValueError("Nonpositive equity")
        def scalar(v):
            return torch.tensor([float(v)], dtype=torch.float32, device=self.device)
        zero = scalar(0)
        # The canonical shared reward container is additive. All accounting
        # diagnostics stay in info; its ONLY reward term here is net log equity.
        rewards = RewardComponents(scalar(math.log(float(after / before))), zero, zero, zero, zero, zero)
        shares = dict(self.book.holdings)
        executed = [self.book.cash / after, *(shares.get(a, D(0)) * self.last_marks.get(a, D(0)) / after for a in self.catalog.asset_ids)]
        return TransitionBatch(observation, action, torch.tensor([executed], dtype=torch.float32, device=self.device),
                               rewards, self.observation(), torch.tensor([terminal], device=self.device),
                               torch.tensor([False], device=self.device), scalar(0 if terminal else 1),
                               info={"equity_before": scalar(before), "equity_after": scalar(after),
                                     "transaction_costs": scalar(fees), "terminal_liquidation_cost": scalar(liquidation_fee),
                                     "fill_count": scalar(len(fills)), "gross_traded_notional": scalar(sum(abs(D(f.signed_shares)) * D(f.price) for f in fills))})
