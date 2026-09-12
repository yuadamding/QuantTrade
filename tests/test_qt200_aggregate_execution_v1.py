"""Hand-calculable aggregate accounting oracles; execute on LSF GPUs only."""

import math
from dataclasses import replace
from datetime import datetime
from decimal import Decimal as D

import pytest

from rl_quant.execution import qt200_aggregate_execution_v1 as model

DAY = "2026-03-06"


def window(ticker="AAPL", day=DAY, price="10", volume="50"):
    return [dict(ticker=ticker, bar_start_ms=int(datetime.fromisoformat(day + f"T09:{minute}:00").replace(
        tzinfo=model.ET).timestamp()) * 1000, o=price, h=price, l=price, c=price, v=volume)
        for minute in range(35, 45)]


def run(book, orders, windows, *, day=DAY, bps="20"):
    return model.execute_window(book, orders, windows, session=day, committed_at_ns=1, cost_bps=D(bps))


def book(cash="1000", holdings=(), day=DAY):
    return model.apply_actions(model.Book(D(cash), holdings), day)


def test_ten_whole_share_fills_reconcile_cash_cost_and_unfilled_orders():
    result = run(book(), {"AAPL": 12}, {"AAPL": window()})
    assert len(result.fills) == 10
    assert all(f.signed_shares == 1 and f.fee == D("0.02") for f in result.fills)
    assert result.book.cash == D("899.8")
    assert result.book.holdings == (("AAPL", D(10)),)
    assert result.unfilled_shares == (("AAPL", D(2)),)
    assert result.gross_traded_notional == 100 and result.transaction_costs == D("0.2")
    assert model.equity(result.book, {"AAPL": D(10)}) == D("999.8")


def test_no_order_has_no_turnover_or_cost_and_does_not_read_a_price():
    original = book(holdings=(("AAPL", D(10)),))
    result = run(original, {}, {})
    assert result.book == original and result.fills == ()
    assert result.transaction_costs == result.gross_traded_notional == 0


def test_missing_minute_is_unsupported_not_zero_volume():
    rows = window()
    missing = run(book(), {"AAPL": 1}, {"AAPL": rows[:-1]})
    zero = run(book(), {"AAPL": 1}, {"AAPL": window(volume="0")})
    assert missing.fills == zero.fills == ()
    assert missing.unsupported_windows == ("AAPL",) and zero.unsupported_windows == ()
    assert missing.unfilled_shares == zero.unfilled_shares == (("AAPL", D(1)),)


def test_buys_share_cash_proportionally_not_by_dictionary_order():
    orders = {"MSFT": 10, "AAPL": 10}
    windows = {t: window(t, volume="10000") for t in orders}
    first = run(book("100.2"), orders, windows)
    second = run(book("100.2"), dict(reversed(list(orders.items()))), windows)
    assert first == second
    assert first.book.holdings == (("AAPL", D(5)), ("MSFT", D(5)))
    assert first.book.cash == 0 and first.transaction_costs == D("0.2")


def test_sell_proceeds_are_available_before_proportional_buys():
    result = run(book("0", (("AAPL", D(10)),)), {"AAPL": -10, "MSFT": 10},
                 {"AAPL": window(volume="1000"), "MSFT": window("MSFT", volume="1000")})
    assert result.book.holdings == (("MSFT", D(9)),)
    assert result.book.cash == D("9.62")
    assert result.transaction_costs == D("0.38")


def test_nonmonotone_cost_outcome_remains_a_valid_execution():
    totals = []
    for rung in model.COST_RUNGS:
        filled = run(book("100.2"), {"AAPL": 10}, {"AAPL": window(volume="1000")}, bps=str(rung))
        final, _ = model.liquidate(filled.book, {"AAPL": D(1)}, cost_bps=rung)
        totals.append(model.equity(final, {}))
    assert totals[2] == D("18.804") > totals[1] == D("9.98")


def test_terminal_liquidation_compounding_uses_cash_shares_and_fees():
    filled = run(book(), {"AAPL": 10}, {"AAPL": window()})
    marked = model.equity(filled.book, {"AAPL": D(12)})
    final, fee = model.liquidate(filled.book, {"AAPL": D(12)})
    assert marked == D("1019.8") and fee == D("0.24")
    assert final.cash == D("1019.56") and final.holdings == ()
    logs = [math.log(float(marked / D(1000))), math.log(float(final.cash / marked))]
    assert math.expm1(math.fsum(logs)) == pytest.approx(float(final.cash / D(1000) - 1), abs=1e-14)


def test_split_and_reverse_split_preserve_marked_equity():
    original = model.Book(D(100), (("AAPL", D(10)),))
    event = model.Split("split-a", "AAPL", DAY, D(1), D(2))
    after = model.apply_actions(original, DAY, splits=(event,))
    assert after.holdings == (("AAPL", D(20)),)
    assert model.equity(original, {"AAPL": D(20)}) == model.equity(after, {"AAPL": D(10)})
    event = model.Split("split-b", "AAPL", "2026-03-09", D(4), D(1))
    reverse = model.apply_actions(after, "2026-03-09", splits=(event,))
    assert reverse.holdings == (("AAPL", D(5)),)
    assert model.equity(reverse, {"AAPL": D(40)}) == D(300)


def test_dividend_is_receivable_on_ex_date_cash_only_on_pay_date_even_after_sale():
    original = model.Book(D(100), (("AAPL", D(10)),))
    event = model.Dividend("div-a", "AAPL", DAY, "2026-03-10", D(1))
    ex = model.apply_actions(original, DAY, dividends=(event,))
    assert ex.cash == 100 and len(ex.receivables) == 1 and ex.receivables[0].amount == 10
    assert model.equity(ex, {"AAPL": D(9)}) == model.equity(original, {"AAPL": D(10)})
    sold = run(ex, {"AAPL": -10}, {"AAPL": window(price="9", volume="1000")})
    assert sold.book.cash == D("189.82") and model.equity(sold.book, {}) == D("199.82")
    settled = model.apply_actions(sold.book, "2026-03-10")
    assert settled.cash == D("199.82") and settled.receivables == ()


def test_terminal_valuation_does_not_pretend_unpaid_dividend_is_cash():
    ex = model.apply_actions(model.Book(D(0), (("AAPL", D(10)),)), DAY,
        dividends=(model.Dividend("div-a", "AAPL", DAY, "2026-03-10", D(1)),))
    result, fee = model.liquidate(ex, {"AAPL": D(9)})
    assert result.cash == D("89.82") and fee == D("0.18")
    assert model.equity(result, {}) == D("99.82")


@pytest.mark.parametrize("change", ["duplicate", "wrong_day", "negative_volume", "zero_price", "wrong_instrument"])
def test_unsupported_input_is_rejected_before_economic_mutation(change):
    rows = window()
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "wrong_day":
        rows = window(day="2026-03-09")
    elif change == "negative_volume":
        rows[0]["v"] = "-1"
    elif change == "zero_price":
        rows[0]["o"] = "0"
    else:
        rows[0]["ticker"] = "MSFT"
    original = book()
    with pytest.raises(ValueError):
        run(original, {"AAPL": 1}, {"AAPL": rows})
    assert original.cash == 1000 and original.holdings == ()


def test_orders_cannot_be_committed_after_window_started_or_shorted():
    rows = window()
    with pytest.raises(ValueError, match="committed"):
        model.execute_window(book(), {"AAPL": 1}, {"AAPL": rows}, session=DAY,
                             committed_at_ns=rows[0]["bar_start_ms"] * 1_000_000)
    for q in (-1, 1.5, True):
        with pytest.raises(ValueError):
            run(book(), {"AAPL": q}, {"AAPL": rows})


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "garbage", True, 1.2])
def test_exact_finite_accounting_input(value):
    with pytest.raises(ValueError):
        model.number(value)


def test_event_replay_and_ambiguous_same_day_basis_fail():
    event = model.Split("split-a", "AAPL", DAY, D(1), D(2))
    initial = model.Book(D(100), (("AAPL", D(1)),))
    after = model.apply_actions(initial, DAY, splits=(event,))
    with pytest.raises(ValueError):
        model.apply_actions(after, "2026-03-09", splits=(replace(event, effective_date="2026-03-09"),))
    with pytest.raises(ValueError):
        model.apply_actions(after, DAY)
    with pytest.raises(ValueError, match="share basis"):
        model.apply_actions(initial, DAY, splits=(event,),
            dividends=(model.Dividend("div-a", "AAPL", DAY, DAY, D(1)),))


def test_dst_changes_timestamp_not_ten_minute_slot_semantics():
    friday = model.validate_window(window(), DAY)
    monday = model.validate_window(window(day="2026-03-09"), "2026-03-09")
    assert len(friday) == len(monday) == 10
    assert monday[0][0] - friday[0][0] == (72 - 1) * 3600 * 10**9
