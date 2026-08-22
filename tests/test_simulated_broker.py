"""The backtest broker.

Two lies a backtest tells, both pinned here: filling at the close of the bar
that produced the signal, and pretending statutory charges do not exist. On NSE
the second one alone is roughly 0.6% of a delivery round trip -- more than most
15-minute edges are worth.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.brokers.base import Broker, round_qty
from claude_trader.brokers.simulated import FillModel, SimulatedBroker
from claude_trader.costs import IndianEquityCosts, NoCosts
from claude_trader.data.sources import HistoricalMarketData
from claude_trader.markets.india import INDIA_MARKET
from claude_trader.markets.us import US_MARKET
from claude_trader.models import Bar, OrderRequest, Side
from tests.conftest import make_bars

START = datetime(2026, 3, 2, 4, 0, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def market(**series) -> HistoricalMarketData:
    if not series:
        series = {"RELIANCE": [100.0, 100.0, 100.0, 100.0, 100.0]}
    return HistoricalMarketData(
        {sym: make_bars(sym, closes, start=START, step=STEP) for sym, closes in series.items()}
    )


def broker(data=None, *, cash=100_000.0, slippage=0.0, costs=None, profile=INDIA_MARKET,
           at=0) -> SimulatedBroker:
    data = data or market()
    b = SimulatedBroker(data, starting_cash=cash, fill_model=FillModel(slippage),
                        costs=costs or NoCosts(), profile=profile)
    b.set_clock(START + at * STEP)
    return b


def buy(symbol="RELIANCE", **kwargs) -> OrderRequest:
    return OrderRequest(symbol=symbol, side=Side.BUY, intent="entry", **kwargs)


def sell(symbol="RELIANCE", **kwargs) -> OrderRequest:
    return OrderRequest(symbol=symbol, side=Side.SELL, intent="exit", **kwargs)


# ------------------------------------------------------------------- clock
def test_the_clock_must_be_set_before_anything_happens():
    """A broker with no clock would silently read the last bar of the whole
    dataset, which is the future."""
    with pytest.raises(RuntimeError, match="set_clock"):
        _ = SimulatedBroker(market()).now


def test_the_day_start_equity_is_captured_at_each_new_session():
    """The daily-loss breaker measures against this; if it drifted forward with
    every bar, it would never fire."""
    data = market(RELIANCE=[100.0] * 200)
    b = broker(data, cash=50_000.0)
    assert b.account().last_equity == 50_000.0
    b.submit(buy(notional=20_000.0))
    b.set_clock(START + timedelta(days=1))
    assert b.account().last_equity == pytest.approx(b.equity())


def test_the_clock_moving_inside_one_day_does_not_reset_the_baseline():
    b = broker(cash=10_000.0)
    b.submit(buy(notional=5_000.0))
    b.set_clock(START + 3 * STEP)
    assert b.account().last_equity == 10_000.0


# -------------------------------------------------------------------- fills
def test_a_fill_happens_at_the_open_of_the_next_bar():
    """Filling at the decision bar's close is the most common way a backtest
    lies about its own returns."""
    bars = (
        Bar(symbol="X", t=START, o=100.0, h=101.0, l=99.0, c=100.0, v=1e4),
        Bar(symbol="X", t=START + STEP, o=130.0, h=131.0, l=129.0, c=132.0, v=1e4),
    )
    b = broker(HistoricalMarketData({"X": bars}), profile=None)
    result = b.submit(buy("X", qty=1))
    assert result.price == pytest.approx(130.0)   # not the 100.0 it decided on


def test_slippage_is_paid_in_the_direction_that_hurts():
    data = HistoricalMarketData({"X": make_bars("X", [100.0, 100.0], start=START, step=STEP)})
    reference = data.all_bars("X")[1].o
    bought = broker(data, profile=None, slippage=10.0).submit(buy("X", qty=1)).price
    b = broker(data, profile=None, slippage=10.0)
    b._holdings["X"] = (5.0, 100.0)
    sold = b.submit(sell("X", qty=1)).price
    assert bought > reference > sold


def test_the_last_bar_has_no_next_bar_so_the_close_is_used():
    """Rejecting here would silently drop every square-off on the final bar of
    a backtest, flattering the result."""
    data = HistoricalMarketData({"X": make_bars("X", [100.0, 110.0], start=START, step=STEP)})
    b = broker(data, profile=None, at=1)
    assert b.submit(buy("X", qty=1)).price == pytest.approx(110.0)


def test_no_price_at_all_is_a_recorded_rejection_not_a_crash():
    b = broker()
    assert b.submit(buy("GHOST", qty=1)) is None
    assert b.rejections == [("GHOST", "no reference price")]


def test_fill_prices_are_rounded_to_the_market_tick():
    data = HistoricalMarketData({"X": make_bars("X", [100.0, 100.0], start=START, step=STEP)})
    price = broker(data, slippage=3.0).submit(buy("X", qty=1)).price
    assert price == round(price, 2)


# ------------------------------------------------------------------ sizing
def test_a_notional_order_is_converted_to_whole_shares_on_nse():
    """NSE has no fractional shares; a 0.7-share order is not an order."""
    b = broker(market(X=[1_000.0] * 5))
    result = b.submit(buy("X", notional=3_500.0))
    assert result.qty == 3.0


def test_a_notional_smaller_than_one_share_is_a_visible_rejection():
    """The ordinary outcome of a small budget on an expensive share -- but if it
    were silent the backtest would look like it simply had no signals."""
    b = broker(market(X=[10_000.0] * 5))
    assert b.submit(buy("X", notional=500.0)) is None
    assert b.rejections == [("X", "quantity rounds to zero")]


def test_a_us_order_may_be_fractional():
    b = broker(market(AAPL=[190.0] * 5), profile=US_MARKET)
    assert 0 < b.submit(buy("AAPL", notional=100.0)).qty < 1.0


def test_an_order_must_carry_exactly_one_of_qty_or_notional():
    """Caught at the model boundary, so the broker never has to guess."""
    with pytest.raises(ValueError, match="exactly one"):
        OrderRequest(symbol="X", side=Side.BUY, intent="entry")


# ------------------------------------------------------------------- cash
def test_cash_is_reduced_by_the_full_cost_including_charges():
    b = broker(market(X=[1_000.0] * 5), cash=50_000.0, costs=IndianEquityCosts("intraday"))
    b.submit(buy("X", qty=10))
    assert b._cash < 50_000.0 - 10_000.0
    assert b.charges_paid > 0


def test_an_order_larger_than_the_cash_is_shrunk_not_rejected():
    """A partial fill is a real broker outcome; rejecting outright would make
    the backtest skip trades a live account would have taken."""
    b = broker(market(X=[1_000.0] * 5), cash=5_400.0)
    result = b.submit(buy("X", notional=100_000.0))
    assert result.qty == 5.0
    assert b._cash >= 0


def test_an_account_too_small_for_one_share_is_rejected():
    b = broker(market(X=[1_000.0] * 5), cash=500.0)
    assert b.submit(buy("X", notional=100_000.0)) is None
    assert b.rejections == [("X", "insufficient cash")]


def test_charges_can_make_an_otherwise_affordable_order_unaffordable():
    """The edge nobody models: cash covers the shares but not the STT."""
    b = broker(market(X=[100.0] * 5), cash=1_000.0, costs=IndianEquityCosts("delivery"))
    b.submit(buy("X", notional=1_000.0))
    assert b._cash >= -1e-9


# -------------------------------------------------------------- positions
def test_a_position_appears_after_a_buy():
    b = broker(market(X=[1_000.0] * 5))
    b.submit(buy("X", qty=4))
    position = b.positions()[0]
    assert (position.symbol, position.qty) == ("X", 4.0)
    assert position.avg_entry_price > 0


def test_averaging_up_blends_the_entry_price():
    data = HistoricalMarketData(
        {"X": make_bars("X", [100.0, 100.0, 200.0, 200.0], start=START, step=STEP)}
    )
    b = broker(data, profile=None)
    b.submit(buy("X", qty=1))
    b.set_clock(START + 2 * STEP)
    b.submit(buy("X", qty=1))
    avg = b.positions()[0].avg_entry_price
    assert 100.0 < avg < 200.0


def test_positions_are_sorted_for_reproducible_reports():
    b = broker(market(B=[100.0] * 5, A=[100.0] * 5))
    b.submit(buy("B", qty=1))
    b.submit(buy("A", qty=1))
    assert [p.symbol for p in b.positions()] == ["A", "B"]


def test_selling_more_than_is_held_sells_only_what_is_held():
    """Otherwise the backtest silently goes short on a long-only strategy."""
    b = broker(market(X=[1_000.0] * 5))
    b.submit(buy("X", qty=2))
    result = b.submit(sell("X", qty=99))
    assert result.qty == 2.0
    assert b.positions() == ()


def test_selling_nothing_is_a_recorded_rejection():
    b = broker()
    assert b.submit(sell("RELIANCE", qty=1)) is None
    assert b.rejections == [("RELIANCE", "no position to sell")]


def test_a_partial_sale_keeps_the_original_entry_price():
    """Re-basing the average on a partial exit would fabricate P&L."""
    b = broker(market(X=[1_000.0] * 5))
    b.submit(buy("X", qty=4))
    entry = b.positions()[0].avg_entry_price
    b.submit(sell("X", qty=1))
    assert b.positions()[0].avg_entry_price == entry
    assert b.positions()[0].qty == 3.0


def test_closing_a_position_flattens_it_entirely():
    b = broker(market(X=[1_000.0] * 5))
    b.submit(buy("X", qty=3))
    result = b.close_position("X")
    assert result.qty == 3.0 and result.side is Side.SELL
    assert b.holdings() == {}


def test_closing_nothing_is_a_no_op():
    b = broker()
    assert b.close_position("RELIANCE") is None
    assert b.rejections == []


# ---------------------------------------------------------------- equity
def test_equity_marks_holdings_at_the_current_bar():
    data = HistoricalMarketData(
        {"X": make_bars("X", [100.0, 100.0, 200.0], start=START, step=STEP)}
    )
    b = broker(data, cash=10_000.0, profile=None)
    b.submit(buy("X", qty=10))
    b.set_clock(START + 2 * STEP)
    assert b.equity() > 10_000.0


def test_a_flat_book_is_all_cash():
    b = broker(cash=7_500.0)
    assert b.equity() == 7_500.0
    assert b.account().buying_power == 7_500.0


def test_a_round_trip_at_a_flat_price_loses_exactly_the_costs():
    """The control experiment: with no edge, the strategy's return is the
    negative of its cost model."""
    b = broker(market(X=[1_000.0] * 5), cash=100_000.0, costs=IndianEquityCosts("intraday"))
    b.submit(buy("X", qty=10))
    b.submit(sell("X", qty=10))
    assert b.equity() == pytest.approx(100_000.0 - b.charges_paid, abs=0.01)


def test_the_charge_breakdown_accumulates_by_component():
    """A backtest that reports only a total cannot tell you STT from brokerage,
    which is the difference between switching segment and switching broker."""
    b = broker(market(X=[1_000.0] * 5), costs=IndianEquityCosts("delivery"))
    b.submit(buy("X", qty=5))
    b.submit(sell("X", qty=5))
    assert sum(b.charge_breakdown.values()) == pytest.approx(b.charges_paid, abs=0.05)
    assert len(b.charge_breakdown) > 1


def test_no_costs_means_no_charges_recorded():
    b = broker(market(X=[1_000.0] * 5))
    b.submit(buy("X", qty=1))
    assert b.charges_paid == 0.0 and b.charge_breakdown == {}


# --------------------------------------------------------------- metadata
def test_orders_are_stamped_as_simulated_with_the_cycle_time():
    b = broker(market(X=[1_000.0] * 5))
    result = b.submit(buy("X", qty=1))
    assert result.simulated is True
    assert result.status == "filled"
    assert result.submitted_at == b.now
    assert result.order_id == "sim-1"


def test_order_ids_are_unique_within_a_run():
    b = broker(market(X=[1_000.0] * 5))
    b.submit(buy("X", qty=1))
    b.submit(buy("X", qty=1))
    assert [o.order_id for o in b.orders] == ["sim-1", "sim-2"]


def test_the_starting_cash_and_cost_model_are_reported_for_the_summary():
    b = broker(cash=25_000.0, costs=IndianEquityCosts("intraday"))
    assert b.starting_cash == 25_000.0
    assert "intraday" in b.cost_model_name.lower() or b.cost_model_name


def test_the_holdings_view_is_a_copy():
    b = broker(market(X=[1_000.0] * 5))
    b.submit(buy("X", qty=1))
    b.holdings()["X"] = (999.0, 1.0)
    assert b.positions()[0].qty == 1.0


# -------------------------------------------------------------- calendar
def test_the_presence_of_bars_is_the_trading_calendar():
    """The backtest has no holiday file; a session with no data is a session
    that did not happen."""
    b = broker()
    assert b.is_market_open(START + STEP) is True
    assert b.is_market_open(START - timedelta(days=5)) is False


# --------------------------------------------------------------- protocol
def test_the_simulator_satisfies_the_broker_protocol():
    """If this fails, the backtest is no longer exercising the live code path."""
    assert isinstance(broker(), Broker)


def test_quantity_rounding_never_rounds_up():
    """Rounding up would let a close order try to sell more than is held."""
    assert round_qty(1.9999999) < 2.0
    assert round_qty(-5.0) == 0.0
