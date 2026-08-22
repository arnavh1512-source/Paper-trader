"""Domain types. Everything here is frozen, so the tests are mostly about one
question: does a state transition return a correct NEW object and leave the old
one untouched?"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.models import (
    Account,
    Action,
    Bar,
    Decision,
    OrderRequest,
    PortfolioState,
    Position,
    PositionRisk,
    Quote,
    Side,
    utcnow,
)

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)


def _risk(symbol="RELIANCE", entry=1_000.0, stop=980.0, target=1_030.0, atr=10.0):
    return PositionRisk(
        symbol=symbol,
        entry_price=entry,
        entry_time=NOW,
        stop_price=stop,
        target_price=target,
        high_water=entry,
        atr_at_entry=atr,
    )


def test_domain_types_are_frozen():
    position = Position("TCS", 10, 100.0, 110.0)
    with pytest.raises(FrozenInstanceError):
        position.qty = 20


# ---------------------------------------------------------------------- bars
def test_bar_derivations():
    bar = Bar("TCS", NOW, o=100, h=110, l=90, c=105, v=1_000)
    assert bar.typical_price == pytest.approx((110 + 90 + 105) / 3)
    assert bar.true_range_vs == 20


# -------------------------------------------------------------------- quotes
def test_quote_mid_and_spread():
    quote = Quote("TCS", NOW, bid=99.0, ask=101.0)
    assert quote.mid == 100.0
    assert quote.spread_bps == pytest.approx(200.0)


def test_quote_falls_back_to_whichever_side_exists():
    assert Quote("TCS", NOW, bid=0.0, ask=101.0).mid == 101.0
    assert Quote("TCS", NOW, bid=99.0, ask=0.0).mid == 99.0
    assert Quote("TCS", NOW, bid=0.0, ask=0.0).is_tradable is False


def test_one_sided_quote_has_an_infinite_spread():
    """Better to score it as untradeable than to compute a flattering number
    from half a book."""
    assert Quote("TCS", NOW, bid=0.0, ask=101.0).spread_bps == float("inf")


def test_quote_age_never_goes_negative():
    quote = Quote("TCS", NOW, 99.0, 101.0)
    assert quote.age_seconds(NOW + timedelta(seconds=90)) == 90
    assert quote.age_seconds(NOW - timedelta(seconds=90)) == 0


# ----------------------------------------------------------------- positions
def test_position_pnl():
    position = Position("TCS", 10, avg_entry_price=100.0, current_price=110.0)
    assert position.market_value == 1_100.0
    assert position.cost_basis == 1_000.0
    assert position.unrealized_pl == 100.0
    assert position.unrealized_plpc == pytest.approx(0.10)
    assert position.repriced(90.0).unrealized_pl == -100.0


def test_position_with_no_basis_reports_no_return():
    assert Position("TCS", 0, 0.0, 0.0).unrealized_plpc == 0.0


# ---------------------------------------------------------------- stop ratchet
def test_trailing_stop_only_ever_ratchets_up():
    risk = _risk()
    advanced = risk.advanced(1_050.0, trail_atr=2.0)
    assert advanced.high_water == 1_050.0
    assert advanced.stop_price == 1_030.0     # 1050 - 2*10
    assert advanced.bars_held == 1
    # A pullback must not loosen the stop.
    pulled_back = advanced.advanced(1_010.0, trail_atr=2.0)
    assert pulled_back.stop_price == 1_030.0
    assert pulled_back.high_water == 1_050.0
    assert risk.stop_price == 980.0           # original untouched


def test_trailing_without_atr_leaves_the_stop_where_it_was():
    risk = _risk(atr=0.0)
    assert risk.advanced(1_100.0, 2.0).stop_price == 980.0


# ------------------------------------------------------------------- account
def test_day_pl_is_zero_without_a_reference():
    assert Account(equity=100.0, cash=100.0, buying_power=100.0).day_pl_pct == 0.0
    account = Account(equity=99.0, cash=99.0, buying_power=99.0, last_equity=100.0)
    assert account.day_pl_pct == pytest.approx(-0.01)


# ----------------------------------------------------------------- portfolio
def test_exposure_ratio_is_a_fraction_not_a_currency_amount():
    """This is what gets journalled. Writing the absolute figure produced an
    'average exposure' of 1,580,350% in the first Indian backtest."""
    state = PortfolioState.build(
        Account(equity=100_000.0, cash=75_000.0, buying_power=75_000.0),
        [Position("RELIANCE", 25, 1_000.0, 1_000.0)],
    )
    assert state.gross_exposure == 25_000.0
    assert state.exposure_ratio == pytest.approx(0.25)
    assert state.exposure_pct("RELIANCE") == pytest.approx(0.25)
    assert state.exposure_pct("TCS") == 0.0


def test_exposure_ratio_survives_a_wiped_out_account():
    state = PortfolioState.build(Account(0.0, 0.0, 0.0), [])
    assert state.exposure_ratio == 0.0


def test_buy_fill_creates_a_position_and_spends_cash():
    state = PortfolioState.build(Account(100_000.0, 100_000.0, 100_000.0), [])
    after = state.with_fill("RELIANCE", Side.BUY, 10, 1_000.0, _risk())

    assert after.position_count == 1
    assert after.account.cash == 90_000.0
    assert after.account.equity == 100_000.0
    assert after.risks["RELIANCE"].stop_price == 980.0
    assert state.position_count == 0          # original untouched


def test_adding_to_a_position_averages_the_entry():
    state = PortfolioState.build(Account(100_000.0, 100_000.0, 100_000.0), [])
    state = state.with_fill("RELIANCE", Side.BUY, 10, 1_000.0)
    state = state.with_fill("RELIANCE", Side.BUY, 10, 1_200.0)
    position = state.positions["RELIANCE"]
    assert position.qty == 20
    assert position.avg_entry_price == pytest.approx(1_100.0)


def test_partial_sell_keeps_the_remainder():
    state = PortfolioState.build(Account(100_000.0, 100_000.0, 100_000.0), [])
    state = state.with_fill("RELIANCE", Side.BUY, 10, 1_000.0, _risk())
    state = state.with_fill("RELIANCE", Side.SELL, 4, 1_100.0)
    assert state.positions["RELIANCE"].qty == 6
    assert "RELIANCE" in state.risks


def test_full_sell_drops_the_position_and_its_risk_record():
    state = PortfolioState.build(Account(100_000.0, 100_000.0, 100_000.0), [])
    state = state.with_fill("RELIANCE", Side.BUY, 10, 1_000.0, _risk())
    state = state.with_fill("RELIANCE", Side.SELL, 10, 1_100.0)
    assert state.positions == {}
    assert state.risks == {}
    assert state.account.cash == pytest.approx(101_000.0)


def test_nonsense_fills_are_no_ops():
    state = PortfolioState.build(Account(1_000.0, 1_000.0, 1_000.0), [])
    assert state.with_fill("X", Side.BUY, 0, 100.0) is state
    assert state.with_fill("X", Side.BUY, 10, 0.0) is state
    assert state.with_fill("X", Side.SELL, 10, 100.0) is state   # nothing held


def test_open_symbols_are_sorted():
    state = PortfolioState.build(
        Account(1_000.0, 1_000.0, 1_000.0),
        [Position("TCS", 1, 1, 1), Position("INFY", 1, 1, 1)],
    )
    assert state.open_symbols == ("INFY", "TCS")


# -------------------------------------------------------------------- orders
def test_order_needs_exactly_one_of_qty_or_notional():
    """Sending both to a broker is ambiguous; sending neither is a no-op that
    would fail silently at the API."""
    with pytest.raises(ValueError):
        OrderRequest("TCS", Side.BUY)
    with pytest.raises(ValueError):
        OrderRequest("TCS", Side.BUY, qty=1, notional=100)
    assert OrderRequest("TCS", Side.BUY, qty=1).qty == 1
    assert OrderRequest("TCS", Side.BUY, notional=100).notional == 100


def test_decision_knows_whether_it_is_actionable():
    assert Decision("TCS", Action.BUY, 8, "").is_trade is True
    assert Decision("TCS", Action.HOLD, 8, "").is_trade is False


def test_utcnow_is_timezone_aware():
    assert utcnow().tzinfo is timezone.utc
