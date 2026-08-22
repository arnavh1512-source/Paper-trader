"""NSE (India) profile.

Universe is a liquid slice of the NIFTY 50 -- large caps with tight spreads and
deep order books, because a bot placing market orders on a mid-cap is mostly
measuring its own impact.

Benchmark is ``NIFTYBEES``, the Nippon India Nifty 50 ETF, rather than the
``NIFTY 50`` index itself. The benchmark has to be something the bot could
actually have bought, otherwise "we beat the index" is a claim about an
untradable number.
"""

from __future__ import annotations

from datetime import date, time
from types import MappingProxyType

# Coarse sector buckets, used only to stop the book holding five names that are
# really one bet. RELIANCE is a conglomerate; it sits in energy because that is
# still what moves it.
SECTORS = MappingProxyType(
    {
        # Banks -- by far the heaviest weight in the index, hence its own bucket
        "HDFCBANK": "banks",
        "ICICIBANK": "banks",
        "SBIN": "banks",
        "AXISBANK": "banks",
        "KOTAKBANK": "banks",
        "INDUSINDBK": "banks",
        # Non-bank financials
        "BAJFINANCE": "nbfc",
        "BAJAJFINSV": "nbfc",
        "HDFCLIFE": "nbfc",
        "SBILIFE": "nbfc",
        # IT services -- one export-revenue, USD-INR sensitive bet
        "TCS": "it",
        "INFY": "it",
        "HCLTECH": "it",
        "WIPRO": "it",
        "TECHM": "it",
        "LTIM": "it",
        # Energy and utilities
        "RELIANCE": "energy",
        "ONGC": "energy",
        "NTPC": "utilities",
        "POWERGRID": "utilities",
        "COALINDIA": "energy",
        "BPCL": "energy",
        # Telecom
        "BHARTIARTL": "telecom",
        # Consumer staples
        "ITC": "fmcg",
        "HINDUNILVR": "fmcg",
        "NESTLEIND": "fmcg",
        "BRITANNIA": "fmcg",
        "TATACONSUM": "fmcg",
        # Autos
        "MARUTI": "auto",
        "TATAMOTORS": "auto",
        "M&M": "auto",
        "BAJAJ-AUTO": "auto",
        "EICHERMOT": "auto",
        "HEROMOTOCO": "auto",
        # Pharma and healthcare
        "SUNPHARMA": "pharma",
        "CIPLA": "pharma",
        "DRREDDY": "pharma",
        "DIVISLAB": "pharma",
        "APOLLOHOSP": "pharma",
        # Metals -- a single commodity-cycle bet
        "TATASTEEL": "metals",
        "JSWSTEEL": "metals",
        "HINDALCO": "metals",
        # Materials / infrastructure
        "ULTRACEMCO": "materials",
        "GRASIM": "materials",
        "LT": "infra",
        "ADANIPORTS": "infra",
        # Consumer discretionary
        "TITAN": "consumer_disc",
        "ASIANPAINT": "consumer_disc",
        "TRENT": "consumer_disc",
        # Index ETF. Bucketed alone: holding it alongside anything else is not
        # diversification, it is leverage on the same book.
        "NIFTYBEES": "index_etf",
    }
)

# Thirty names, not the full fifty: every extra symbol is another HTTP round
# trip per cycle and another candidate the model has to reason about.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "BHARTIARTL",
    "ITC",
    "SBIN",
    "LT",
    "AXISBANK",
    "KOTAKBANK",
    "HINDUNILVR",
    "MARUTI",
    "TATAMOTORS",
    "SUNPHARMA",
    "TITAN",
    "ASIANPAINT",
    "BAJFINANCE",
    "HCLTECH",
    "TATASTEEL",
    "JSWSTEEL",
    "NTPC",
    "POWERGRID",
    "ONGC",
    "M&M",
    "WIPRO",
    "ADANIPORTS",
    "CIPLA",
    "TECHM",
    "ULTRACEMCO",
)

BENCHMARK_SYMBOL = "NIFTYBEES"

# NSE trading holidays.
#
# IMPORTANT: most Indian market holidays follow lunar calendars and are fixed
# only when NSE publishes its annual circular, so this list needs updating every
# December from https://www.nseindia.com/resources/exchange-communication-holidays
# Nothing critical depends on it being complete: session detection asks the data
# feed whether today actually printed bars. The list only avoids pointless API
# calls and keeps synthetic calendars plausible.
HOLIDAYS_2025 = (
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-ul-Fitr
    date(2025, 4, 10),   # Mahavir Jayanti
    date(2025, 4, 14),   # Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti / Dussehra
    date(2025, 10, 21),  # Diwali Laxmi Pujan (muhurat session only)
    date(2025, 10, 22),  # Balipratipada
    date(2025, 11, 5),   # Prakash Gurpurb
    date(2025, 12, 25),  # Christmas
)

# 2026: only the fixed-date national holidays are known ahead of the circular.
# Festival dates are deliberately absent rather than guessed.
HOLIDAYS_2026 = (
    date(2026, 1, 26),   # Republic Day
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 12, 25),  # Christmas
)

HOLIDAYS = frozenset(HOLIDAYS_2025 + HOLIDAYS_2026)


def _profile():
    from . import MarketProfile

    return MarketProfile(
        key="in",
        name="NSE India",
        currency="INR",
        currency_symbol="₹",
        tz_name="Asia/Kolkata",
        utc_offset_minutes=330,
        open_time=time(9, 15),
        close_time=time(15, 30),
        benchmark=BENCHMARK_SYMBOL,
        universe=DEFAULT_UNIVERSE,
        sectors=SECTORS,
        fractional_shares=False,   # NSE equities trade in whole shares only
        lot_size=1,
        data_suffix=".NS",
        holidays=HOLIDAYS,
        starting_cash=100_000.0,   # one lakh: a realistic retail starting book
        max_per_trade=10_000.0,
        tick_size=0.05,
        timeframe="15m",
        bars_per_session=25,       # 09:15-15:30 is 375 minutes
    )


INDIA_MARKET = _profile()
