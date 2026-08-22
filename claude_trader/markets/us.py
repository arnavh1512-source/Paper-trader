"""US equities profile (Alpaca)."""

from __future__ import annotations

from datetime import date, time

from ..universe import BENCHMARK_SYMBOL, DEFAULT_UNIVERSE, SECTORS

# US market holidays are stable and well known, but the same rule applies as for
# NSE: the feed is the authority, this list is an optimisation.
HOLIDAYS = frozenset(
    (
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    )
)


def _profile():
    from . import MarketProfile

    return MarketProfile(
        key="us",
        name="US equities",
        currency="USD",
        currency_symbol="$",
        tz_name="America/New_York",
        utc_offset_minutes=-300,
        open_time=time(9, 30),
        close_time=time(16, 0),
        benchmark=BENCHMARK_SYMBOL,
        universe=DEFAULT_UNIVERSE,
        sectors=SECTORS,
        fractional_shares=True,
        lot_size=1,
        data_suffix="",
        holidays=HOLIDAYS,
        starting_cash=10_000.0,
        max_per_trade=100.0,
        tick_size=0.01,
        timeframe="15Min",
        bars_per_session=26,
    )


US_MARKET = _profile()
