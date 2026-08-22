"""Market profiles: the layer that makes one engine trade two exchanges."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from claude_trader.markets import (
    INDIA_MARKET,
    MARKETS,
    US_MARKET,
    format_money,
    get_market,
)

IST = timezone(timedelta(minutes=330))
ET = timezone(timedelta(hours=-5))


def test_registry_exposes_both_markets():
    assert set(MARKETS) == {"in", "us"}
    assert get_market("IN ") is INDIA_MARKET
    assert get_market("us") is US_MARKET


def test_unknown_market_names_the_alternatives():
    with pytest.raises(ValueError) as exc:
        get_market("uk")
    assert "in" in str(exc.value) and "us" in str(exc.value)


# ------------------------------------------------------------------- symbols
def test_india_data_symbol_carries_the_yahoo_suffix():
    assert INDIA_MARKET.data_symbol("RELIANCE") == "RELIANCE.NS"
    assert INDIA_MARKET.native_symbol("RELIANCE.NS") == "RELIANCE"
    # Idempotent on a symbol that never had the suffix.
    assert INDIA_MARKET.native_symbol("RELIANCE") == "RELIANCE"


def test_us_symbols_are_unchanged():
    assert US_MARKET.data_symbol("AAPL") == "AAPL"
    assert US_MARKET.native_symbol("AAPL") == "AAPL"


def test_unmapped_symbol_never_looks_diversifying():
    """An unknown ticker must not share a sector with anything, or the
    concentration limit would let the book pile into one theme."""
    assert INDIA_MARKET.sector_of("NOSUCH") == "unknown:NOSUCH"
    assert INDIA_MARKET.sector_of("nosuch") == "unknown:NOSUCH"
    assert INDIA_MARKET.sector_of("RELIANCE") != INDIA_MARKET.sector_of("NOSUCH")


# ---------------------------------------------------------------- quantities
@pytest.mark.parametrize(
    "wanted,expected",
    [(7.9, 7.0), (1.0, 1.0), (0.99, 0.0), (0.0, 0.0), (-5.0, 0.0), (250.7, 250.0)],
)
def test_nse_sizes_round_down_to_whole_shares(wanted, expected):
    assert INDIA_MARKET.round_qty(wanted) == expected


def test_us_keeps_fractional_size():
    assert US_MARKET.round_qty(7.9) == pytest.approx(7.9)
    assert US_MARKET.round_qty(0.25) == pytest.approx(0.25)
    assert US_MARKET.round_qty(-1.0) == 0.0


def test_prices_snap_to_the_tick():
    assert INDIA_MARKET.round_price(1234.5678) == 1234.55
    assert INDIA_MARKET.round_price(100.02) == 100.0
    assert US_MARKET.round_price(123.4567) == 123.46


# --------------------------------------------------------------------- money
@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "₹0.00"),
        (100, "₹100.00"),
        (1_000, "₹1,000.00"),
        (100_000, "₹1,00,000.00"),
        (1_234_567.891, "₹12,34,567.89"),
        (10_000_000, "₹1,00,00,000.00"),
        (-1_500.5, "-₹1,500.50"),
    ],
)
def test_rupees_group_in_lakhs_not_thousands(value, expected):
    assert format_money(value, INDIA_MARKET) == expected


@pytest.mark.parametrize(
    "value,expected", [(1_234_567.891, "$1,234,567.89"), (-20.1, "-$20.10")]
)
def test_dollars_group_in_thousands(value, expected):
    assert format_money(value, US_MARKET) == expected


def test_money_without_a_profile_falls_back_to_dollars():
    assert format_money(5) == "$5.00"


# ------------------------------------------------------------------ sessions
def test_nse_session_window():
    # 2026-03-02 is a Monday and not a listed holiday.
    assert INDIA_MARKET.is_session_time(datetime(2026, 3, 2, 10, 0, tzinfo=IST))
    assert INDIA_MARKET.is_session_time(datetime(2026, 3, 2, 9, 15, tzinfo=IST))
    assert INDIA_MARKET.is_session_time(datetime(2026, 3, 2, 15, 30, tzinfo=IST))
    assert not INDIA_MARKET.is_session_time(datetime(2026, 3, 2, 9, 14, tzinfo=IST))
    assert not INDIA_MARKET.is_session_time(datetime(2026, 3, 2, 15, 31, tzinfo=IST))


def test_session_is_read_in_local_time_not_utc():
    """04:30 UTC is 10:00 IST -- inside the session. Getting this wrong is how
    a bot trades the wrong six hours of the day."""
    assert INDIA_MARKET.is_session_time(datetime(2026, 3, 2, 4, 30, tzinfo=timezone.utc))
    assert not INDIA_MARKET.is_session_time(
        datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    )


def test_weekends_and_holidays_are_closed():
    assert not INDIA_MARKET.is_session_time(datetime(2026, 3, 7, 10, 0, tzinfo=IST))
    assert not INDIA_MARKET.is_session_time(datetime(2026, 3, 8, 10, 0, tzinfo=IST))
    assert date(2026, 1, 26) in INDIA_MARKET.holidays
    assert not INDIA_MARKET.is_session_time(datetime(2026, 1, 26, 10, 0, tzinfo=IST))


def test_us_session_window():
    assert US_MARKET.is_session_time(datetime(2026, 3, 2, 10, 0, tzinfo=ET))
    assert not US_MARKET.is_session_time(datetime(2026, 3, 2, 8, 0, tzinfo=ET))


@pytest.mark.parametrize(
    "when,expected",
    [
        (time(9, 15), 0.0),
        (time(12, 22), pytest.approx(0.5, abs=0.01)),
        (time(15, 30), 1.0),
        (time(23, 0), 1.0),  # clamped, never above 1
        (time(1, 0), 0.0),   # clamped, never below 0
    ],
)
def test_session_progress_is_clamped_to_the_session(when, expected):
    moment = datetime(2026, 3, 2, when.hour, when.minute, tzinfo=IST)
    assert INDIA_MARKET.session_progress(moment) == expected


def test_local_converts_into_exchange_time():
    utc = datetime(2026, 3, 2, 4, 30, tzinfo=timezone.utc)
    local = INDIA_MARKET.local(utc)
    assert (local.hour, local.minute) == (10, 0)


# ------------------------------------------------------------------- profile
def test_india_profile_states_the_facts_the_engine_depends_on():
    assert INDIA_MARKET.currency == "INR"
    assert INDIA_MARKET.fractional_shares is False
    assert INDIA_MARKET.benchmark == "NIFTYBEES"
    assert INDIA_MARKET.timeframe == "15m"
    # 09:15-15:30 is 375 minutes; 25 fifteen-minute bars.
    assert INDIA_MARKET.bars_per_session == 25


def test_us_profile_still_allows_fractional_shares():
    assert US_MARKET.fractional_shares is True
    assert US_MARKET.currency == "USD"
    assert US_MARKET.benchmark == "SPY"
