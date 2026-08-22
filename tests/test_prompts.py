"""Prompt construction.

Two things are being defended here. First, that doing nothing is presented as an
available answer -- a prompt that says "pick the best 3-5 stocks to trade RIGHT
NOW" cannot return "nothing here", so it never does. Second, that the model is
told which market it is looking at: rupee prices read as dollar prices produce
confidently wrong reasoning about what is expensive.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from claude_trader.data.indicators import compute
from claude_trader.llm.prompts import (
    DECIDER_SYSTEM,
    PICKER_SYSTEM,
    build_decision_prompt,
    build_picker_prompt,
    format_indicators,
    market_brief,
)
from claude_trader.markets import INDIA_MARKET, US_MARKET
from claude_trader.models import Indicators, Position, PositionRisk
from tests.conftest import make_bars, make_quote, make_snapshot, make_state, ramp

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)   # 10:30 IST


# ------------------------------------------------------------ system prompts
def test_the_picker_is_allowed_to_return_nothing():
    assert "empty list" in PICKER_SYSTEM
    assert "never on how many" in PICKER_SYSTEM


def test_the_decider_defaults_to_hold_and_demands_an_invalidation():
    assert "HOLD is the default" in DECIDER_SYSTEM
    assert "invalidation" in DECIDER_SYSTEM
    assert "calibrated" in DECIDER_SYSTEM


def test_the_decider_is_told_sizing_is_not_its_job():
    """Otherwise it argues for a bigger position instead of a better one."""
    assert "handled downstream by" in DECIDER_SYSTEM
    assert "advisory" in DECIDER_SYSTEM


# ------------------------------------------------------------- market brief
def test_the_indian_brief_names_the_currency_and_the_session():
    brief = market_brief(INDIA_MARKET, "intraday")
    assert "INR" in brief and "₹" in brief
    assert "09:15-15:30" in brief and "Asia/Kolkata" in brief


def test_the_indian_brief_warns_that_shares_are_indivisible():
    """A model that thinks it can buy 0.4 shares proposes budgets that round to
    nothing on NSE."""
    assert "indivisible" in market_brief(INDIA_MARKET)


def test_the_us_brief_does_not_mention_indivisible_shares():
    assert "indivisible" not in market_brief(US_MARKET)


@pytest.mark.parametrize(
    "segment,expected",
    [
        ("intraday", "closed before the session ends"),
        ("delivery", "carried overnight"),
    ],
)
def test_the_segment_changes_what_counts_as_tradable(segment, expected):
    assert expected in market_brief(INDIA_MARKET, segment)


def test_no_segment_line_when_there_is_no_segment():
    brief = market_brief(INDIA_MARKET, "")
    assert "INTRADAY" not in brief and "DELIVERY" not in brief


# -------------------------------------------------------------- formatting
def test_missing_indicators_are_printed_as_na_not_as_zero():
    """Zero is a number the model will reason about. 'n/a' is not."""
    line = format_indicators(Indicators(last_price=100.0))
    assert "rsi=n/a" in line and "atr=n/a" in line
    assert "last=100.00" in line


def test_indicator_line_is_one_line_per_symbol():
    line = format_indicators(compute(make_bars("TCS", ramp(40, 3_000.0, 5.0))))
    assert "\n" not in line
    assert "trend=" in line


# ------------------------------------------------------------ picker prompt
def test_the_picker_prompt_carries_the_portfolio_and_the_room_left():
    state = make_state(cash=90_000.0, positions=[Position("TCS", 3, 3_000.0, 3_000.0)])
    prompt = build_picker_prompt(
        NOW,
        state,
        {"RELIANCE": compute(make_bars("RELIANCE", ramp(40, 1_000.0)))},
        universe=("RELIANCE", "TCS"),
        max_new_positions=2,
        profile=INDIA_MARKET,
        segment="intraday",
    )
    assert "room for new positions: 2" in prompt
    assert "TCS" in prompt
    assert "₹99,000.00" in prompt            # equity, grouped in lakhs
    assert "Choose only from: RELIANCE, TCS" in prompt


def test_the_picker_prompt_is_stamped_in_exchange_time():
    prompt = build_picker_prompt(
        NOW, make_state(), {}, ("RELIANCE",), 3, profile=INDIA_MARKET
    )
    assert "2026-03-02 10:30 Asia/Kolkata" in prompt


def test_an_empty_universe_snapshot_says_so_rather_than_looking_normal():
    prompt = build_picker_prompt(
        NOW, make_state(), {}, ("RELIANCE",), 3, profile=INDIA_MARKET
    )
    assert "(no data available)" in prompt


def test_the_picker_prompt_states_when_there_is_no_room():
    prompt = build_picker_prompt(
        NOW, make_state(), {}, ("RELIANCE",), 0, profile=INDIA_MARKET
    )
    assert "room for new positions: 0" in prompt
    assert "abstain=true" in prompt


# ---------------------------------------------------------- decision prompt
def _snapshot(**kwargs):
    bars = make_bars("RELIANCE", ramp(20, 1_000.0, 2.0))
    return make_snapshot("RELIANCE", price=bars[-1].c, bars=bars, when=NOW, **kwargs)


def test_the_decision_prompt_shows_the_book_and_the_bars():
    prompt = build_decision_prompt(
        _snapshot(), "momentum regime", make_state(), profile=INDIA_MARKET
    )
    assert "Symbol: RELIANCE" in prompt
    assert "Regime note: momentum regime" in prompt
    assert "NOT HELD" in prompt
    assert prompt.count("O:") == 10          # only the last ten bars
    assert prompt.rstrip().endswith("Decide: buy, sell, or hold.")


def test_a_held_position_shows_its_stop_and_target():
    risk = PositionRisk(
        symbol="RELIANCE",
        entry_price=1_000.0,
        entry_time=NOW,
        stop_price=980.0,
        target_price=1_030.0,
        high_water=1_010.0,
        atr_at_entry=10.0,
        bars_held=4,
    )
    prompt = build_decision_prompt(
        _snapshot(position=Position("RELIANCE", 10, 1_000.0, 1_038.0), risk=risk),
        "",
        make_state(),
        profile=INDIA_MARKET,
    )
    assert "HELD: 10 shares @ ₹1,000.00" in prompt
    assert "active stop ₹980.00" in prompt
    assert "bars held 4" in prompt
    assert "+3.80%" in prompt


def test_a_modelled_quote_is_labelled_as_such():
    """Free NSE data has no order book. If the model is shown a 4 bps spread it
    will read liquidity that nobody actually quoted."""
    quote = make_quote("RELIANCE", 1_038.0, when=NOW, modelled=True)
    prompt = build_decision_prompt(
        _snapshot(quote=quote), "", make_state(), profile=INDIA_MARKET
    )
    assert "[modelled, not an observed order book]" in prompt


def test_a_real_quote_is_not_labelled():
    quote = make_quote("RELIANCE", 1_038.0, when=NOW, modelled=False)
    prompt = build_decision_prompt(
        _snapshot(quote=quote), "", make_state(), profile=INDIA_MARKET
    )
    assert "modelled" not in prompt


def test_the_cost_floor_is_stated_when_there_is_one():
    """Without this the model proposes 0.05% scalps that lose money when they
    are right."""
    prompt = build_decision_prompt(
        _snapshot(), "", make_state(), profile=INDIA_MARKET,
        segment="intraday", round_trip_cost_pct=0.00106,
    )
    assert "0.106%" in prompt
    assert "loses money even if you" in prompt


def test_no_cost_line_when_costs_are_not_modelled():
    prompt = build_decision_prompt(
        _snapshot(), "", make_state(), profile=US_MARKET, round_trip_cost_pct=0.0
    )
    assert "Cost floor" not in prompt


def test_prices_are_rendered_in_the_market_currency():
    india = build_decision_prompt(_snapshot(), "", make_state(), profile=INDIA_MARKET)
    us = build_decision_prompt(_snapshot(), "", make_state(), profile=US_MARKET)
    assert "₹" in india and "$" not in india
    assert "$" in us and "₹" not in us


def test_a_symbol_with_no_bars_or_quote_says_so():
    snapshot = replace(
        make_snapshot("RELIANCE", price=1_000.0, bars=(), when=NOW), quote=None
    )
    prompt = build_decision_prompt(snapshot, "", make_state(), profile=INDIA_MARKET)
    assert "(no bars available)" in prompt
    assert "Quote: unavailable" in prompt
