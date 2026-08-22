"""Yahoo Finance as the market data source.

This is the only free feed that covers NSE without a KYC'd broker account, and
everything downstream trusts what it returns. Two properties matter more than
the rest: a gap in the payload must never become a fabricated bar, and the
"current price" in the metadata must never leak backwards into a replay.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.data.yahoo import (
    DEFAULT_HALF_SPREAD_BPS,
    LARGE_CAP_HALF_SPREAD_BPS,
    MAX_RANGE_DAYS,
    YahooMarketData,
    chart_meta,
    parse_chart,
)
from claude_trader.errors import MarketDataError
from claude_trader.markets import INDIA_MARKET, US_MARKET

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)


def stamps(n: int, step_minutes: int = 15, end: datetime = NOW) -> list[int]:
    return [
        int((end - timedelta(minutes=step_minutes * (n - 1 - i))).timestamp())
        for i in range(n)
    ]


def chart(
    closes,
    *,
    times=None,
    opens=None,
    highs=None,
    lows=None,
    volumes=None,
    meta=None,
    error=None,
    result=None,
) -> dict:
    n = len(closes)
    times = times if times is not None else stamps(n)
    body = {
        "timestamp": times,
        "indicators": {"quote": [{
            "open": opens if opens is not None else list(closes),
            "high": highs if highs is not None else [c * 1.01 if c else c for c in closes],
            "low": lows if lows is not None else [c * 0.99 if c else c for c in closes],
            "close": list(closes),
            "volume": volumes if volumes is not None else [1_000] * n,
        }]},
        "meta": meta or {},
    }
    return {"chart": {"result": result if result is not None else [body], "error": error}}


class FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = ""
        self.content = b"{}"

    def json(self):
        return self._payload


class FakeSession:
    """Serves a payload per vendor symbol and records every request."""

    def __init__(self, payloads=None, default=None):
        self.payloads = dict(payloads or {})
        self.default = default
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        symbol = url.rsplit("/", 1)[-1]
        payload = self.payloads.get(symbol, self.default)
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload if payload is not None else chart([100.0]))


def source(session, *, profile=INDIA_MARKET, interval="15m", **kwargs) -> YahooMarketData:
    return YahooMarketData(profile, interval=interval, session=session, **kwargs)


# ------------------------------------------------------------- parse_chart
def test_bars_are_built_from_the_payload():
    bars = parse_chart("RELIANCE", chart([100.0, 101.0, 102.0]))
    assert [b.c for b in bars] == [100.0, 101.0, 102.0]
    assert bars[0].symbol == "RELIANCE"
    assert bars[0].t.tzinfo is timezone.utc


def test_a_null_row_is_dropped_not_forward_filled():
    """A fabricated bar is worse than a missing one: the indicators cannot tell
    them apart, and a flat synthetic candle reads as low volatility."""
    bars = parse_chart("RELIANCE", chart([100.0, None, 102.0]))
    assert [b.c for b in bars] == [100.0, 102.0]


def test_a_null_in_any_leg_drops_the_row():
    bars = parse_chart("X", chart([100.0, 101.0], highs=[101.0, None]))
    assert [b.c for b in bars] == [100.0]


def test_a_zero_close_is_rejected():
    """Yahoo occasionally prints a zero on a halted scrip; dividing by it later
    produces an infinite return."""
    assert [b.c for b in parse_chart("X", chart([100.0, 0.0, 102.0]))] == [100.0, 102.0]


def test_a_null_timestamp_drops_the_row():
    payload = chart([100.0, 101.0])
    payload["chart"]["result"][0]["timestamp"][1] = None
    assert len(parse_chart("X", payload)) == 1


def test_bars_come_back_in_time_order():
    payload = chart([100.0, 101.0, 102.0])
    payload["chart"]["result"][0]["timestamp"].reverse()
    bars = parse_chart("X", payload)
    assert [b.t for b in bars] == sorted(b.t for b in bars)


def test_a_short_ohlc_array_stops_rather_than_misaligning():
    """Reading past the end would pair today's close with last week's open."""
    payload = chart([100.0, 101.0, 102.0])
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"] = [100.0, 101.0]
    assert len(parse_chart("X", payload)) == 2


def test_a_missing_volume_reads_as_zero():
    payload = chart([100.0])
    payload["chart"]["result"][0]["indicators"]["quote"][0]["volume"] = [None]
    assert parse_chart("X", payload)[0].v == 0.0


def test_unparseable_numbers_are_skipped_not_crashed():
    payload = chart([100.0, 101.0, 102.0])
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][1] = "n/a"
    assert [b.c for b in parse_chart("X", payload)] == [100.0, 102.0]


def test_an_empty_result_is_no_bars_not_an_error():
    """A symbol that has not traded today is a normal condition."""
    assert parse_chart("X", chart([], result=[])) == ()


def test_a_vendor_error_is_raised_as_a_typed_error():
    with pytest.raises(MarketDataError, match="RELIANCE"):
        parse_chart("RELIANCE", chart([], error={"code": "Not Found"}))


def test_a_payload_that_is_not_a_chart_is_rejected():
    with pytest.raises(MarketDataError, match="unexpected chart payload"):
        parse_chart("X", {"nope": 1})


def test_chart_meta_is_empty_rather_than_none_when_absent():
    assert chart_meta({"chart": {"result": []}}) == {}
    assert chart_meta(chart([100.0], meta={"regularMarketPrice": 1.0})) \
        == {"regularMarketPrice": 1.0}


# ------------------------------------------------------------------- setup
def test_an_unsupported_interval_is_refused_at_construction():
    """Failing here beats discovering mid-backtest that Yahoo silently served a
    different interval."""
    with pytest.raises(MarketDataError, match="unsupported"):
        YahooMarketData(INDIA_MARKET, interval="7m")


def test_the_interval_defaults_to_the_market_profile():
    assert YahooMarketData(INDIA_MARKET)._interval == INDIA_MARKET.timeframe.lower()


def test_the_vendor_suffix_comes_from_the_profile():
    """RELIANCE is RELIANCE.NS to Yahoo, and AAPL is just AAPL."""
    session = FakeSession()
    source(session).bars("RELIANCE", 5, NOW)
    assert session.calls[0]["url"].endswith("RELIANCE.NS")
    session = FakeSession()
    source(session, profile=US_MARKET, interval="15m").bars("AAPL", 5, NOW)
    assert session.calls[0]["url"].endswith("AAPL")


def test_the_request_range_matches_what_the_vendor_will_serve():
    session = FakeSession()
    source(session).bars("RELIANCE", 5, NOW)
    assert session.calls[0]["params"]["range"] == f"{MAX_RANGE_DAYS['15m']}d"


def test_a_browser_user_agent_is_sent():
    """Yahoo rejects the default python-requests agent on some edges."""
    session = FakeSession()
    source(session).bars("RELIANCE", 5, NOW)
    assert "Mozilla" in session.calls[0]["headers"]["User-Agent"]


# -------------------------------------------------------------------- bars
def test_bars_are_trimmed_to_the_requested_limit():
    session = FakeSession(default=chart([100.0 + i for i in range(30)]))
    assert len(source(session).bars("RELIANCE", 10, NOW)) == 10


def test_a_zero_limit_returns_everything_visible():
    session = FakeSession(default=chart([100.0 + i for i in range(30)]))
    assert len(source(session).bars("RELIANCE", 0, NOW)) == 30


def test_bars_after_the_as_of_moment_are_invisible():
    """Without this the backtester would trade on bars that had not printed."""
    session = FakeSession(default=chart([100.0, 101.0, 102.0]))
    bars = source(session).bars("RELIANCE", 10, NOW - timedelta(minutes=16))
    assert [b.c for b in bars] == [100.0]


def test_repeated_asks_inside_the_ttl_hit_the_cache():
    """One cycle asks for the same symbol four times -- overview, snapshot,
    quote, benchmark -- and Yahoo throttles callers who do not notice."""
    session = FakeSession()
    data = source(session, cache_ttl=600.0)
    data.bars("RELIANCE", 5, NOW)
    data.bars("RELIANCE", 5, NOW)
    data.quote("RELIANCE", NOW)
    assert len(session.calls) == 1


def test_the_cache_can_be_cleared():
    session = FakeSession()
    data = source(session, cache_ttl=600.0)
    data.bars("RELIANCE", 5, NOW)
    data.clear_cache()
    data.bars("RELIANCE", 5, NOW)
    assert len(session.calls) == 2


def test_a_zero_ttl_disables_the_cache():
    session = FakeSession()
    data = source(session, cache_ttl=0.0)
    data.bars("RELIANCE", 5, NOW)
    data.bars("RELIANCE", 5, NOW)
    assert len(session.calls) == 2


def test_two_symbols_do_not_share_a_cache_entry():
    session = FakeSession({
        "RELIANCE.NS": chart([1_000.0]),
        "TCS.NS": chart([3_000.0]),
    })
    data = source(session, cache_ttl=600.0)
    assert data.bars("RELIANCE", 1, NOW)[0].c == 1_000.0
    assert data.bars("TCS", 1, NOW)[0].c == 3_000.0


# ----------------------------------------------------------------- history
def test_history_requests_an_explicit_window():
    session = FakeSession()
    start, end = NOW - timedelta(days=5), NOW
    source(session).history("RELIANCE", start, end)
    params = session.calls[0]["params"]
    assert params["period1"] == int(start.timestamp())
    assert params["period2"] == int(end.timestamp())
    assert "range" not in params


def test_history_is_filtered_to_the_window():
    session = FakeSession(default=chart([100.0, 101.0, 102.0]))
    bars = source(session).history("RELIANCE", NOW - timedelta(minutes=16), NOW)
    assert [b.c for b in bars] == [101.0, 102.0]


def test_an_over_long_intraday_window_is_trimmed_and_logged(caplog):
    """Intraday history beyond the vendor window does not exist. Silently
    returning a shorter series would look like a data gap in the backtest."""
    caplog.set_level("WARNING")
    session = FakeSession()
    start = NOW - timedelta(days=400)
    source(session).history("RELIANCE", start, NOW)
    requested = NOW.timestamp() - session.calls[0]["params"]["period1"]
    assert requested <= MAX_RANGE_DAYS["15m"] * 86_400 + 60
    assert "capped" in caplog.text


def test_daily_history_is_not_trimmed():
    session = FakeSession()
    start = NOW - timedelta(days=400)
    source(session).history("RELIANCE", start, NOW, interval="1d")
    assert session.calls[0]["params"]["period1"] == int(start.timestamp())


def test_history_never_reads_the_per_cycle_cache():
    """The live cache is keyed for a rolling window; serving a backtest from it
    would leak a different date range into the replay."""
    session = FakeSession()
    data = source(session, cache_ttl=600.0)
    data.history("RELIANCE", NOW - timedelta(days=2), NOW)
    data.history("RELIANCE", NOW - timedelta(days=2), NOW)
    assert len(session.calls) == 2


# ------------------------------------------------------------------ quotes
def test_a_quote_is_modelled_around_the_last_price():
    session = FakeSession(default=chart(
        [1_000.0], meta={"regularMarketPrice": 1_000.0, "regularMarketTime": int(NOW.timestamp())}
    ))
    quote = source(session).quote("RELIANCE", NOW)
    assert quote.modelled is True
    half = 1_000.0 * LARGE_CAP_HALF_SPREAD_BPS / 10_000.0
    assert quote.bid == pytest.approx(1_000.0 - half)
    assert quote.ask == pytest.approx(1_000.0 + half)


def test_an_unlisted_symbol_gets_the_pessimistic_spread():
    """A strategy that only works on optimistic spreads does not work."""
    session = FakeSession(default=chart(
        [500.0], meta={"regularMarketPrice": 500.0, "regularMarketTime": int(NOW.timestamp())}
    ))
    quote = source(session).quote("SMALLCAP", NOW)
    half = 500.0 * DEFAULT_HALF_SPREAD_BPS / 10_000.0
    assert quote.ask - quote.bid == pytest.approx(2 * half)


def test_a_future_meta_price_is_not_leaked_into_a_replay():
    """The meta price is the *current* print. Trusting it while replaying last
    Tuesday is look-ahead bias, and it makes any backtest worthless."""
    past = NOW - timedelta(days=1)
    session = FakeSession(default=chart(
        [900.0], times=[int((past - timedelta(minutes=15)).timestamp())],
        meta={"regularMarketPrice": 1_500.0, "regularMarketTime": int(NOW.timestamp())},
    ))
    quote = source(session).quote("RELIANCE", past)
    assert (quote.bid + quote.ask) / 2 == pytest.approx(900.0, rel=1e-3)
    assert quote.t < past


def test_a_missing_meta_price_falls_back_to_the_last_bar():
    session = FakeSession(default=chart([950.0]))
    quote = source(session).quote("RELIANCE", NOW)
    assert (quote.bid + quote.ask) / 2 == pytest.approx(950.0, rel=1e-3)


def test_a_corrupt_meta_price_falls_back_rather_than_crashing():
    session = FakeSession(default=chart(
        [950.0], meta={"regularMarketPrice": "n/a", "regularMarketTime": "later"}
    ))
    assert source(session).quote("RELIANCE", NOW) is not None


def test_no_data_at_all_is_no_quote():
    session = FakeSession(default=chart([], result=[]))
    assert source(session).quote("RELIANCE", NOW) is None


# ----------------------------------------------------------- latest prices
def test_latest_prices_returns_only_what_is_available():
    session = FakeSession({
        "RELIANCE.NS": chart([1_000.0]),
        "TCS.NS": chart([], result=[]),
    })
    prices = source(session).latest_prices(["RELIANCE", "TCS"], NOW)
    assert prices == {"RELIANCE": 1_000.0}


def test_one_broken_symbol_does_not_lose_the_others(caplog):
    """The mark-to-market of a whole book must not depend on every symbol."""
    caplog.set_level("WARNING")
    session = FakeSession({
        "RELIANCE.NS": chart([1_000.0]),
        "TCS.NS": {"nope": 1},
    })
    prices = source(session).latest_prices(["TCS", "RELIANCE"], NOW)
    assert prices == {"RELIANCE": 1_000.0}
    assert "TCS" in caplog.text


# ------------------------------------------------------------ session probe
def test_the_market_is_shut_outside_session_hours():
    session = FakeSession()
    assert source(session).is_trading_now(NOW.replace(hour=20)) is False
    assert session.calls == []      # no need to ask the vendor


def test_a_recent_benchmark_print_means_the_market_is_open():
    """Asking the data rather than a hardcoded holiday list is the only way to
    stay correct through NSE's lunar-calendar holidays."""
    session = FakeSession(default=chart([25_000.0], times=[int(NOW.timestamp())]))
    assert source(session).is_trading_now(NOW) is True


def test_a_stale_benchmark_print_means_a_holiday():
    old = int((NOW - timedelta(days=1)).timestamp())
    session = FakeSession(default=chart([25_000.0], times=[old]))
    assert source(session).is_trading_now(NOW) is False


def test_no_benchmark_data_is_treated_as_closed():
    session = FakeSession(default=chart([], result=[]))
    assert source(session).is_trading_now(NOW) is False


def test_a_vendor_failure_is_treated_as_closed_rather_than_open():
    """Trading blind because the feed is down is the worse of the two errors."""
    session = FakeSession(default={"chart": {"error": {"code": "boom"}}})
    assert source(session).is_trading_now(NOW) is False


def test_the_reference_symbol_can_be_overridden():
    session = FakeSession({"TCS.NS": chart([3_000.0], times=[int(NOW.timestamp())])})
    assert source(session).is_trading_now(NOW, reference="TCS") is True
    assert session.calls[0]["url"].endswith("TCS.NS")


def test_the_benchmark_is_the_default_reference():
    session = FakeSession(default=chart([25_000.0], times=[int(NOW.timestamp())]))
    source(session).is_trading_now(NOW)
    vendor = INDIA_MARKET.data_symbol(INDIA_MARKET.benchmark)
    assert session.calls[0]["url"].endswith(vendor.replace("^", "%5E"))


@pytest.mark.parametrize("interval,minutes", [("1m", 1), ("15m", 15), ("60m", 60), ("1h", 60)])
def test_the_staleness_tolerance_tracks_the_bar_width(interval, minutes):
    assert YahooMarketData(INDIA_MARKET, interval=interval)._bar_minutes() == minutes
