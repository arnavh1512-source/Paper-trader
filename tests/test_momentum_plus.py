"""The filtered momentum arm.

``momentum_plus`` is the control with a fixed set of filters stacked on top.
None of them is tuned; each is a textbook idea stated before any data was
looked at. They are what is under test here, not a source of returns.
Every test below claims that one filter does exactly one thing.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from claude_trader.costs import IndianEquityCosts, NoCosts
from claude_trader.markets import INDIA_MARKET, US_MARKET
from claude_trader.models import Action, Indicators, Position
from claude_trader.strategies.momentum import MomentumStrategy
from claude_trader.strategies.momentum_plus import MomentumPlusStrategy
from tests.conftest import make_snapshot, make_state

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)   # 10:30 IST
UNIVERSE = ("RELIANCE", "TCS", "INFY")


def ind(price=1_000.0, fast=1_010.0, slow=1_000.0, rsi=55.0, ret_5=0.01,
        ret_20=0.02, volume_ratio=1.0, atr_pct=1.0) -> Indicators:
    return Indicators(
        last_price=price, sma_fast=fast, sma_slow=slow, rsi=rsi,
        atr=price * (atr_pct or 0) / 100, atr_pct=atr_pct, ret_5=ret_5,
        ret_20=ret_20, volume_ratio=volume_ratio, trend="up",
    )


def strategy(costs=None, profile=INDIA_MARKET) -> MomentumPlusStrategy:
    return MomentumPlusStrategy(UNIVERSE, profile=profile, costs=costs or NoCosts(),
                                position_pct=0.2)


def snapshot(indicators: Indicators, position: Position | None = None):
    return replace(make_snapshot(position=position), indicators=indicators)


def held(symbol="RELIANCE") -> Position:
    return Position(symbol=symbol, qty=10, avg_entry_price=1_000.0, current_price=1_000.0)


# ------------------------------------------------------------- selection
def test_it_ranks_by_trend_per_unit_of_volatility_not_raw_trend():
    """TCS has the bigger raw trend but is four times as volatile; per ATR it
    is the weaker setup, so the quieter name goes first."""
    picks = strategy().pick(NOW, make_state(), {
        "RELIANCE": ind(fast=1_010.0, ret_5=0.01, atr_pct=0.5),
        "TCS": ind(fast=1_020.0, ret_5=0.02, atr_pct=2.0),
        "INFY": ind(fast=1_005.0, ret_5=0.005, atr_pct=1.0),
    }, max_new_positions=2)
    assert picks.symbols == ("RELIANCE", "TCS")


def test_it_abstains_when_most_of_the_universe_is_falling():
    """One uptrend in a falling market is the trade most likely to be dragged
    down with it. Breadth is the regime check, and it needs no benchmark."""
    down = ind(fast=990.0, slow=1_000.0)
    picks = strategy().pick(NOW, make_state(), {
        "RELIANCE": ind(), "TCS": down, "INFY": down,
    }, max_new_positions=3)
    assert picks.abstain and picks.symbols == ()
    assert "breadth" in picks.rationale


def test_half_the_universe_rising_is_enough_breadth():
    picks = strategy().pick(NOW, make_state(), {
        "RELIANCE": ind(), "TCS": ind(fast=990.0),
    }, max_new_positions=2)
    assert picks.symbols == ("RELIANCE",)


def test_breadth_ignores_symbols_without_moving_averages():
    picks = strategy().pick(NOW, make_state(), {
        "RELIANCE": ind(), "TCS": ind(fast=None, slow=None), "INFY": ind(fast=None),
    }, max_new_positions=3)
    assert picks.symbols == ("RELIANCE",)


def test_no_moving_averages_anywhere_means_no_breadth_and_no_picks():
    picks = strategy().pick(NOW, make_state(), {"RELIANCE": ind(fast=None)}, 3)
    assert picks.abstain and "breadth" in picks.rationale


@pytest.mark.parametrize("minute", [45, 50, 59])
def test_it_sits_out_the_first_quarter_hour_of_the_session(minute):
    """At 09:17 the newest bar is a two-minute stub, and the overnight gap
    dominates every short return in it."""
    early = datetime(2026, 3, 2, 3, minute, tzinfo=timezone.utc)  # 09:15-09:29 IST
    picks = strategy().pick(early, make_state(), {"RELIANCE": ind()}, 1)
    assert picks.abstain and "opening" in picks.rationale


def test_the_opening_window_ends_fifteen_minutes_after_the_open():
    at_0930 = datetime(2026, 3, 2, 4, 0, tzinfo=timezone.utc)
    assert strategy().pick(at_0930, make_state(), {"RELIANCE": ind()}, 1).symbols == ("RELIANCE",)


def test_the_opening_window_comes_from_the_market_profile():
    """Nothing here knows which market it is in; 09:35 New York is early."""
    ny_0935 = datetime(2026, 3, 2, 14, 35, tzinfo=timezone.utc)
    picks = strategy(profile=US_MARKET).pick(ny_0935, make_state(), {"RELIANCE": ind()}, 1)
    assert picks.abstain and "opening" in picks.rationale


def test_it_never_picks_a_held_name_or_one_outside_the_universe():
    picks = strategy().pick(NOW, make_state(positions=[held("TCS")]), {
        "TCS": ind(), "WIPRO": ind(), "RELIANCE": ind(),
    }, max_new_positions=3)
    assert picks.symbols == ("RELIANCE",)


def test_it_abstains_when_there_is_no_room():
    picks = strategy().pick(NOW, make_state(), {"RELIANCE": ind()}, 0)
    assert picks.abstain and picks.symbols == ()


def test_it_abstains_when_every_candidate_is_filtered_out():
    picks = strategy().pick(NOW, make_state(), {"RELIANCE": ind(ret_20=-0.01)}, 1)
    assert picks.abstain and picks.symbols == ()
    assert "0 of 1" in picks.rationale


def test_it_only_ever_picks_a_subset_of_what_the_control_would():
    """Strictly a filter: every name it buys also clears the control's own
    entry rule. That keeps the two arms comparable."""
    overview = {
        "RELIANCE": ind(),
        "TCS": ind(fast=1_001.0, ret_5=0.0005),        # below the control's floor
        "INFY": ind(volume_ratio=0.5),                  # control buys, plus does not
    }
    control = MomentumStrategy(UNIVERSE).pick(NOW, make_state(), overview, 3)
    plus = strategy().pick(NOW, make_state(), overview, 3)
    assert set(plus.symbols) <= set(control.symbols)
    assert plus.symbols == ("RELIANCE",)


# ------------------------------------------------------------- entry filters
@pytest.mark.parametrize("override, reason", [
    ({"fast": 1_001.0, "ret_5": 0.0}, "trend score"),
    ({"rsi": 72.0}, "overbought"),
    ({"ret_20": 0.0}, "20-bar return"),
    ({"ret_20": None}, "20-bar return"),
    ({"volume_ratio": 0.79}, "volume"),
    ({"volume_ratio": None}, "volume"),
    ({"atr_pct": None}, "volatility"),
    ({"atr_pct": 0.0}, "volatility"),
])
def test_each_filter_blocks_an_entry_and_says_why(override, reason):
    decision = strategy().decide(snapshot(ind(**override)), "", make_state())
    assert decision.action is Action.HOLD
    assert reason in decision.reason


def test_a_move_smaller_than_the_round_trip_cost_is_not_worth_taking():
    """Intraday NSE costs about 0.11% in and out. A name whose typical bar moves
    0.05% cannot pay that back from one ATR of favourable movement."""
    costs = IndianEquityCosts(segment="intraday")
    quiet = strategy(costs).decide(snapshot(ind(atr_pct=0.05)), "", make_state())
    lively = strategy(costs).decide(snapshot(ind(atr_pct=0.5)), "", make_state())
    assert quiet.action is Action.HOLD and "round-trip cost" in quiet.reason
    assert lively.action is Action.BUY


def test_the_cost_floor_is_priced_at_the_ticket_this_book_would_actually_send():
    """Delivery charges a flat DP fee, so a small ticket costs more per rupee.
    The same name passes on a large book and fails on a tiny one."""
    costs = IndianEquityCosts(segment="delivery")
    setup = snapshot(ind(atr_pct=1.5))
    assert strategy(costs).decide(setup, "", make_state(cash=100_000)).action is Action.BUY
    assert strategy(costs).decide(setup, "", make_state(cash=2_000)).action is Action.HOLD


# ------------------------------------------------------------- entries
def test_it_buys_a_setup_that_clears_every_filter():
    decision = strategy().decide(snapshot(ind()), "", make_state())
    assert decision.action is Action.BUY
    assert decision.source == "rule"


@pytest.mark.parametrize("atr_pct, confidence", [(2.0, 7), (1.0, 8), (0.5, 8)])
def test_confidence_rises_with_trend_measured_in_atrs(atr_pct, confidence):
    """score 0.02: 1 ATR at 2%, 2 ATRs at 1%, 4 ATRs at 0.5%."""
    decision = strategy().decide(snapshot(ind(atr_pct=atr_pct)), "", make_state())
    assert (decision.action, decision.confidence) == (Action.BUY, confidence)


def test_every_buy_clears_the_default_confidence_gate():
    from claude_trader.config import RiskConfig

    decision = strategy().decide(snapshot(ind(atr_pct=5.0, fast=1_002.0, ret_5=0.001)), "",
                                 make_state())
    assert decision.action is Action.BUY
    assert decision.confidence >= RiskConfig().min_confidence


# ------------------------------------------------------------- exits
def test_a_held_name_is_sold_when_its_trend_breaks():
    decision = strategy().decide(snapshot(ind(fast=990.0), held()), "", make_state())
    assert (decision.action, decision.confidence) == (Action.SELL, 8)
    assert "trend broken" in decision.reason


def test_a_hot_winner_is_left_to_the_trailing_stop_not_sold_on_rsi():
    """The control sells at RSI 78, which caps every winner. Here the risk
    layer's trailing stop and target do that job."""
    decision = strategy().decide(snapshot(ind(rsi=85.0), held()), "", make_state())
    assert decision.action is Action.HOLD
    control = MomentumStrategy(UNIVERSE).decide(snapshot(ind(rsi=85.0), held()), "", make_state())
    assert control.action is Action.SELL


def test_entry_filters_never_force_an_exit():
    """Volume drying up is a reason not to buy, not a reason to sell."""
    decision = strategy().decide(snapshot(ind(volume_ratio=0.1, ret_20=-0.05), held()), "",
                                 make_state())
    assert decision.action is Action.HOLD


def test_a_held_name_without_moving_averages_is_held_not_sold():
    decision = strategy().decide(snapshot(ind(fast=None, slow=None), held()), "", make_state())
    assert decision.action is Action.HOLD


def test_it_has_its_own_name_so_it_gets_its_own_paper_account():
    assert MomentumPlusStrategy.name == "momentum_plus"
