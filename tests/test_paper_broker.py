"""The journal-backed paper broker.

For NSE there is no Alpaca-style paper account, so the journal *is* the account.
That makes these tests the only thing standing between a modelled book and a
book that quietly invents money.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping

import pytest

from claude_trader.brokers.paper import PaperBroker, SlippageModel
from claude_trader.costs import IndianEquityCosts, NoCosts
from claude_trader.errors import BrokerError
from claude_trader.markets import INDIA_MARKET, US_MARKET
from claude_trader.models import OrderRequest, Side

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)     # 10:30 IST


class StubPrices:
    """Just the slice of the data source a broker touches."""

    def __init__(self, prices: Mapping[str, float] | None = None, fail: bool = False):
        self.prices = dict(prices or {"RELIANCE": 1_000.0, "TCS": 3_000.0})
        self.fail = fail
        self.calls = 0

    def latest_prices(self, symbols: Iterable[str], as_of: datetime) -> dict[str, float]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("yahoo timed out")
        return {s: self.prices[s] for s in symbols if s in self.prices}


def broker(
    journal,
    prices: StubPrices | None = None,
    *,
    profile=INDIA_MARKET,
    costs=None,
    cash: float | None = 100_000.0,
    slippage_bps: float = 5.0,
    clock: datetime | None = NOW,
    account: str = "default",
) -> PaperBroker:
    return PaperBroker(
        journal=journal,
        market=prices or StubPrices(),
        profile=profile,
        costs=costs if costs is not None else IndianEquityCosts("intraday"),
        account=account,
        starting_cash=cash,
        slippage=SlippageModel(slippage_bps),
        clock=clock,
    )


# --------------------------------------------------------------- lifecycle
def test_a_new_account_opens_with_the_configured_cash(journal):
    account = broker(journal).account()
    assert account.cash == 100_000.0
    assert account.equity == 100_000.0
    assert account.buying_power == 100_000.0


def test_trading_without_a_clock_is_refused_rather_than_guessed(journal):
    """Every fill price, charge and day boundary is stamped with this clock.
    Defaulting it to 'now' would make a backtest silently non-reproducible."""
    with pytest.raises(BrokerError, match="set_clock"):
        broker(journal, clock=None)


def test_the_book_survives_a_new_process(journal):
    """On GitHub Actions the process exits after every cycle. If the book did
    not reload from the journal, every run would start from opening cash."""
    first = broker(journal)
    first.submit(OrderRequest("RELIANCE", Side.BUY, notional=10_000))
    reopened = broker(journal, cash=None)
    assert reopened.account().cash == pytest.approx(first.account().cash)
    assert len(reopened.positions()) == 1


def test_the_two_markets_keep_separate_books(journal):
    """One journal, two accounts. A rupee balance and a dollar balance sharing a
    row would be nonsense."""
    india = broker(journal)
    india.submit(OrderRequest("RELIANCE", Side.BUY, notional=10_000))
    us = broker(journal, profile=US_MARKET, costs=NoCosts(), cash=50_000.0)
    assert us.account().cash == 50_000.0
    assert us.positions() == ()


# ------------------------------------------------------------------- fills
def test_a_buy_pays_the_spread_and_the_charges(journal):
    costs = IndianEquityCosts("intraday")
    result = broker(journal, costs=costs).submit(
        OrderRequest("RELIANCE", Side.BUY, notional=10_000)
    )
    assert result is not None
    assert result.price == pytest.approx(1_000.5)       # 5 bps above the mark
    assert result.qty == 9                              # 10,000 / 1,000.5, floored
    assert result.simulated is True
    assert result.status == "filled"


def test_the_cash_movement_matches_the_fill_exactly(journal):
    costs = IndianEquityCosts("intraday")
    b = broker(journal, costs=costs)
    result = b.submit(OrderRequest("RELIANCE", Side.BUY, notional=10_000))
    charges = costs.charges(Side.BUY, result.qty, result.price)
    expected = 100_000.0 - (result.qty * result.price + charges.total)
    assert b.account().cash == pytest.approx(expected)


def test_a_sell_fills_below_the_mark(journal):
    b = broker(journal)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    sell = b.submit(OrderRequest("RELIANCE", Side.SELL, qty=10))
    assert sell.price == pytest.approx(999.5)
    assert b.positions() == ()


def test_slippage_is_always_against_the_trader(journal):
    b = broker(journal, slippage_bps=50.0)
    buy = b.submit(OrderRequest("RELIANCE", Side.BUY, qty=1))
    sell = b.submit(OrderRequest("RELIANCE", Side.SELL, qty=1))
    assert buy.price > 1_000.0 > sell.price


def test_adding_to_a_position_averages_the_entry(journal):
    prices = StubPrices({"RELIANCE": 1_000.0})
    b = broker(journal, prices)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    prices.prices["RELIANCE"] = 1_200.0
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    position = b.positions()[0]
    assert position.qty == 20
    assert position.avg_entry_price == pytest.approx((1_000.5 + 1_200.6) / 2)


def test_a_partial_sell_keeps_the_remainder(journal):
    b = broker(journal)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    b.submit(OrderRequest("RELIANCE", Side.SELL, qty=4))
    assert b.positions()[0].qty == 6


def test_selling_more_than_is_held_sells_what_is_held(journal):
    """A short position on a cash segment account is not a thing. Silently
    creating one would make every subsequent number wrong."""
    b = broker(journal)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=5))
    sell = b.submit(OrderRequest("RELIANCE", Side.SELL, qty=50))
    assert sell.qty == 5
    assert b.positions() == ()


def test_selling_nothing_is_refused(journal):
    assert broker(journal).submit(OrderRequest("TCS", Side.SELL, qty=5)) is None


def test_realised_pnl_is_net_of_charges(journal):
    prices = StubPrices({"RELIANCE": 1_000.0})
    costs = IndianEquityCosts("intraday")
    b = broker(journal, prices, costs=costs)
    buy = b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    prices.prices["RELIANCE"] = 1_100.0
    sell = b.submit(OrderRequest("RELIANCE", Side.SELL, qty=10))
    gross = 10 * (sell.price - buy.price)
    realized = float(
        journal.query(
            "SELECT realized_pnl FROM paper_account WHERE account = 'default:in'"
        )[0][0]
    )
    assert realized < gross
    assert realized == pytest.approx(gross - costs.charges(Side.SELL, 10, sell.price).total)


def test_charges_paid_accumulates(journal):
    b = broker(journal)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    b.submit(OrderRequest("RELIANCE", Side.SELL, qty=10))
    paid = float(
        journal.query(
            "SELECT charges_paid FROM paper_account WHERE account = 'default:in'"
        )[0][0]
    )
    assert paid > 0


# ------------------------------------------------------------- constraints
def test_an_order_with_no_price_is_dropped(journal):
    assert broker(journal).submit(OrderRequest("NOSUCH", Side.BUY, notional=5_000)) is None


def test_a_budget_below_one_share_buys_nothing(journal):
    """On NSE this is not an edge case: a Rs 500 budget against a Rs 3,000 share
    is a common outcome of position sizing."""
    assert broker(journal).submit(OrderRequest("TCS", Side.BUY, notional=500)) is None


def test_an_order_is_trimmed_to_what_the_cash_covers(journal):
    b = broker(journal, cash=20_000.0)
    result = b.submit(OrderRequest("RELIANCE", Side.BUY, qty=100))
    assert result is not None
    assert result.qty < 100
    assert b.account().cash >= 0


def test_an_account_that_cannot_afford_one_share_buys_nothing(journal):
    b = broker(journal, cash=500.0)
    assert b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10)) is None
    assert b.account().cash == 500.0


def test_the_book_never_goes_cash_negative(journal):
    b = broker(journal, cash=5_000.0)
    for _ in range(5):
        b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    assert b.account().cash >= 0


# ---------------------------------------------------------- mark to market
def test_positions_are_marked_at_the_live_price(journal):
    prices = StubPrices({"RELIANCE": 1_000.0})
    b = broker(journal, prices)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    prices.prices["RELIANCE"] = 1_100.0
    assert b.positions()[0].current_price == 1_100.0
    assert b.account().equity > 100_000.0 - 20   # up on the position, less charges


def test_a_data_outage_falls_back_to_cost_basis_instead_of_zero(journal):
    """Marking a live position to zero because Yahoo timed out would trip the
    drawdown breaker and liquidate a perfectly good book."""
    prices = StubPrices({"RELIANCE": 1_000.0})
    b = broker(journal, prices)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    prices.fail = True
    account = b.account()
    assert account.equity == pytest.approx(account.cash + 10 * 1_000.5)


# ------------------------------------------------------------ day boundary
def test_the_days_opening_equity_is_stamped_once(journal):
    """It anchors the daily loss breaker. Recomputing it as the day moves would
    make the breaker unable to ever fire."""
    prices = StubPrices({"RELIANCE": 1_000.0})
    b = broker(journal, prices)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    opening = b.account().last_equity
    prices.prices["RELIANCE"] = 800.0
    assert b.account().last_equity == pytest.approx(opening)
    assert b.account().day_pl_pct < 0


def test_a_new_session_restamps_the_opening_equity(journal):
    prices = StubPrices({"RELIANCE": 1_000.0})
    b = broker(journal, prices)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    prices.prices["RELIANCE"] = 800.0
    b.set_clock(NOW + timedelta(days=1))
    account = b.account()
    assert account.last_equity == pytest.approx(account.equity)
    assert account.day_pl_pct == 0.0


# --------------------------------------------------------------- utilities
def test_close_position_flattens_the_holding(journal):
    b = broker(journal)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    assert b.close_position("RELIANCE").qty == 10
    assert b.positions() == ()
    assert b.close_position("RELIANCE") is None


def test_market_open_delegates_to_the_data_source_when_it_can_answer(journal):
    class WithClock(StubPrices):
        def is_trading_now(self, as_of: datetime) -> bool:
            return False

    assert broker(journal, WithClock()).is_market_open(NOW) is False
    # No probe on the source: fall back to the profile calendar.
    assert broker(journal).is_market_open(NOW) is True
    assert broker(journal).is_market_open(NOW.replace(hour=20)) is False


def test_summary_reports_in_the_market_currency(journal):
    b = broker(journal)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    summary = b.summary()
    assert summary["currency"] == "INR"
    assert summary["equity"].startswith("₹")
    assert summary["open positions"] == 1
    assert summary["cost model"] == "NSE cash (intraday)"


def test_reset_is_explicit_and_total(journal):
    b = broker(journal)
    b.submit(OrderRequest("RELIANCE", Side.BUY, qty=10))
    b.reset()
    assert b.account().cash == 100_000.0
    assert b.positions() == ()
    assert b.account().equity == 100_000.0
