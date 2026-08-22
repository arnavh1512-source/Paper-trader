"""Market data sources.

Both sources satisfy one Protocol, which is what lets the backtester exercise
the exact decision path that trades. The property these tests exist to protect
is ``as_of``: a historical source must be physically unable to hand back a bar
from the future, because a backtest that peeks is worse than no backtest -- it
produces a confident number that is wrong.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.config import AppConfig
from claude_trader.data.sources import (
    AlpacaMarketData,
    HistoricalMarketData,
    MarketDataSource,
    bar_from_payload,
)
from claude_trader.errors import MarketDataError
from claude_trader.models import Bar
from tests.conftest import make_bars, ramp

NOW = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)
START = NOW - timedelta(hours=6)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.text = ""
        self.content = b"{}"

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        payload = self.payloads.pop(0) if self.payloads else {}
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload)


def alpaca(session, **overrides) -> AlpacaMarketData:
    config = AppConfig(
        market="us", alpaca_key="key", alpaca_secret="secret",
        feed="iex", timeframe="15Min", **overrides,
    )
    return AlpacaMarketData(config, session=session)


def series(symbol="RELIANCE", closes=None, start=START, step=timedelta(minutes=15)):
    return make_bars(symbol, closes or ramp(20, 1_000.0, 1.0), start=start, step=step)


# ------------------------------------------------------------------ payload
def test_a_bar_payload_is_converted():
    bar = bar_from_payload("AAPL", {
        "t": "2026-03-02T15:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100
    })
    assert bar.c == 1.5 and bar.v == 100.0
    assert bar.t == datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)


def test_a_naive_timestamp_is_read_as_utc():
    """Alpaca sends Z-suffixed times; a naive one would compare wrongly against
    an aware ``as_of`` and raise mid-cycle."""
    bar = bar_from_payload("AAPL", {"t": "2026-03-02T15:00:00", "o": 1, "h": 1, "l": 1, "c": 1})
    assert bar.t.tzinfo is timezone.utc


def test_a_missing_volume_reads_as_zero():
    assert bar_from_payload("A", {"t": "2026-03-02T15:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1}).v == 0.0


@pytest.mark.parametrize("payload", [
    {},
    {"t": "2026-03-02T15:00:00Z"},
    {"t": "not a time", "o": 1, "h": 1, "l": 1, "c": 1},
    {"t": "2026-03-02T15:00:00Z", "o": "x", "h": 1, "l": 1, "c": 1},
    {"t": None, "o": 1, "h": 1, "l": 1, "c": 1},
])
def test_a_malformed_bar_is_dropped_rather_than_guessed(payload):
    assert bar_from_payload("A", payload) is None


# ------------------------------------------------------------------ alpaca
def test_alpaca_bars_are_parsed_and_sorted():
    session = FakeSession({"bars": [
        {"t": "2026-03-02T15:00:00Z", "o": 2, "h": 2, "l": 2, "c": 2},
        {"t": "2026-03-02T14:45:00Z", "o": 1, "h": 1, "l": 1, "c": 1},
    ]})
    bars = alpaca(session).bars("AAPL", 10, NOW)
    assert [b.c for b in bars] == [1.0, 2.0]


def test_alpaca_bars_after_the_as_of_moment_are_dropped():
    """Belt and braces: the request already sends ``end``, but a vendor that
    ignores it must not be able to leak a future bar into a replay."""
    session = FakeSession({"bars": [
        {"t": "2026-03-02T15:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1},
        {"t": "2026-03-02T16:00:00Z", "o": 9, "h": 9, "l": 9, "c": 9},
    ]})
    assert [b.c for b in alpaca(session).bars("AAPL", 10, NOW)] == [1.0]


def test_alpaca_requests_carry_the_credentials_and_the_feed():
    session = FakeSession({"bars": []})
    alpaca(session).bars("AAPL", 10, NOW)
    call = session.calls[0]
    assert call["headers"]["APCA-API-KEY-ID"] == "key"
    assert call["headers"]["APCA-API-SECRET-KEY"] == "secret"
    assert call["params"]["feed"] == "iex"
    assert call["params"]["adjustment"] == "raw"
    assert call["params"]["end"] == NOW.isoformat()


def test_a_missing_bars_key_is_no_bars():
    assert alpaca(FakeSession({})).bars("AAPL", 10, NOW) == ()


def test_a_bad_row_does_not_lose_the_good_ones():
    session = FakeSession({"bars": [
        {"t": "2026-03-02T14:45:00Z", "o": 1, "h": 1, "l": 1, "c": 1},
        {"garbage": True},
    ]})
    assert len(alpaca(session).bars("AAPL", 10, NOW)) == 1


def test_an_alpaca_quote_is_parsed():
    session = FakeSession({"quote": {
        "t": "2026-03-02T15:00:00Z", "bp": 100.0, "ap": 100.1, "bs": 3, "as": 4
    }})
    quote = alpaca(session).quote("AAPL", NOW)
    assert (quote.bid, quote.ask) == (100.0, 100.1)
    assert quote.modelled is False       # a real book, unlike Yahoo's


def test_a_quote_with_no_book_is_none():
    """An empty book means nobody is willing to trade; sizing off it would
    invent a price."""
    assert alpaca(FakeSession({"quote": {"bp": 0, "ap": 0}})).quote("AAPL", NOW) is None


def test_a_missing_quote_object_is_none():
    assert alpaca(FakeSession({})).quote("AAPL", NOW) is None
    assert alpaca(FakeSession({"quote": "nope"})).quote("AAPL", NOW) is None


def test_a_quote_without_a_timestamp_is_stamped_now():
    quote = alpaca(FakeSession({"quote": {"bp": 1.0, "ap": 1.1}})).quote("AAPL", NOW)
    assert quote is not None and quote.t.tzinfo is not None


def test_a_corrupt_quote_is_none_rather_than_an_exception():
    session = FakeSession({"quote": {"bp": "x", "ap": 1.0}})
    assert alpaca(session).quote("AAPL", NOW) is None


def test_snapshots_are_batched_into_one_request():
    """One request for the whole universe instead of thirty is the difference
    between a cycle that finishes and one that gets rate limited."""
    session = FakeSession({"AAPL": {}, "MSFT": {}})
    alpaca(session).snapshots(["AAPL", "MSFT"])
    assert len(session.calls) == 1
    assert session.calls[0]["params"]["symbols"] == "AAPL,MSFT"


def test_no_symbols_makes_no_request():
    session = FakeSession()
    assert alpaca(session).snapshots([]) == {}
    assert session.calls == []


def test_an_unexpected_snapshot_shape_is_a_typed_error():
    with pytest.raises(MarketDataError, match="unexpected shape"):
        alpaca(FakeSession([1, 2, 3])).snapshots(["AAPL"])


def test_latest_prices_prefer_the_last_trade():
    session = FakeSession({
        "AAPL": {"latestTrade": {"p": 190.0}, "dailyBar": {"c": 185.0}},
        "MSFT": {"dailyBar": {"c": 400.0}},
    })
    assert alpaca(session).latest_prices(["AAPL", "MSFT"], NOW) == \
        {"AAPL": 190.0, "MSFT": 400.0}


@pytest.mark.parametrize("data", [
    {"latestTrade": {"p": 0}},
    {"latestTrade": {"p": "x"}},
    {"latestTrade": {}},
    {},
    "not a dict",
])
def test_an_unusable_snapshot_price_is_omitted(data):
    """A zero here would mark a live position to nothing and trip the drawdown
    breaker."""
    assert alpaca(FakeSession({"AAPL": data})).latest_prices(["AAPL"], NOW) == {}


# -------------------------------------------------------------- historical
def test_history_is_sliced_strictly_at_the_as_of_moment():
    """This slice is the mechanism that makes lookahead structurally impossible
    rather than merely discouraged."""
    bars = series(closes=[100.0, 101.0, 102.0, 103.0])
    data = HistoricalMarketData({"RELIANCE": bars})
    visible = data.bars("RELIANCE", 10, bars[1].t)
    assert [b.c for b in visible] == [100.0, 101.0]


def test_the_limit_takes_the_most_recent_bars():
    data = HistoricalMarketData({"X": series("X", ramp(20, 100.0, 1.0))})
    visible = data.bars("X", 3, NOW)
    assert [b.c for b in visible] == [117.0, 118.0, 119.0]


def test_a_zero_limit_returns_the_whole_visible_series():
    data = HistoricalMarketData({"X": series("X", ramp(20, 100.0, 1.0))})
    assert len(data.bars("X", 0, NOW)) == 20


def test_an_unknown_symbol_is_empty_not_an_error():
    assert HistoricalMarketData({}).bars("NOPE", 5, NOW) == ()
    assert HistoricalMarketData({}).all_bars("NOPE") == ()


def test_input_is_sorted_on_construction():
    bars = series("X", [100.0, 101.0, 102.0])
    data = HistoricalMarketData({"X": tuple(reversed(bars))})
    assert [b.c for b in data.all_bars("X")] == [100.0, 101.0, 102.0]


def test_the_timeline_is_the_union_of_every_symbol():
    """The backtest clock steps over this, so a symbol that only trades on some
    bars must not shorten the run."""
    a = series("A", [1.0, 2.0], start=START)
    b = series("B", [3.0], start=START + timedelta(minutes=30))
    timeline = HistoricalMarketData({"A": a, "B": b}).timeline()
    assert timeline == tuple(sorted({a[0].t, a[1].t, b[0].t}))


def test_the_symbol_list_is_sorted_for_reproducibility():
    assert HistoricalMarketData({"TCS": (), "INFY": ()}).symbols == ("INFY", "TCS")


def test_a_bar_can_be_looked_up_by_its_exact_stamp():
    bars = series("X", [100.0, 101.0])
    data = HistoricalMarketData({"X": bars})
    assert data.bar_at("X", bars[1].t).c == 101.0
    assert data.bar_at("X", NOW + timedelta(days=1)) is None


def test_the_next_bar_is_the_fill_price_for_a_signal():
    """Filling at the close of the bar that produced the signal is the classic
    backtest lie; the executor needs the *next* bar."""
    bars = series("X", [100.0, 101.0, 102.0])
    data = HistoricalMarketData({"X": bars})
    assert data.next_bar_after("X", bars[0].t).c == 101.0
    assert data.next_bar_after("X", bars[-1].t) is None


def test_a_synthetic_quote_costs_a_spread():
    """A backtest that fills at mid enjoys an edge live trading never gets."""
    bars = series("X", [1_000.0])
    quote = HistoricalMarketData({"X": bars}).quote("X", NOW)
    assert quote.bid < 1_000.0 < quote.ask
    assert quote.spread_bps > 0


def test_the_synthetic_spread_has_a_floor_in_absolute_terms():
    """On a penny stock a proportional spread rounds to nothing."""
    quote = HistoricalMarketData({"X": series("X", [1.0])}).quote("X", NOW)
    assert quote.ask - quote.bid == pytest.approx(0.02)


def test_no_visible_bar_is_no_quote():
    bars = series("X", [100.0])
    data = HistoricalMarketData({"X": bars})
    assert data.quote("X", bars[0].t - timedelta(minutes=1)) is None


def test_latest_prices_skip_symbols_with_nothing_visible_yet():
    a = series("A", [100.0], start=START)
    b = series("B", [200.0], start=START + timedelta(hours=1))
    data = HistoricalMarketData({"A": a, "B": b})
    assert data.latest_prices(["A", "B"], a[0].t) == {"A": 100.0}


# ---------------------------------------------------------- forward return
def test_a_forward_return_is_measured_over_the_full_horizon():
    """This is what answers 'did the 9s actually beat the 5s'."""
    bars = series("X", [100.0, 101.0, 102.0, 110.0])
    data = HistoricalMarketData({"X": bars})
    entry, exit_price, ret = data.forward_return("X", bars[0].t, 3)
    assert (entry, exit_price) == (100.0, 110.0)
    assert ret == pytest.approx(0.10)


def test_a_partly_elapsed_horizon_returns_nothing():
    """Truncating it would bias the sample towards recent price action, which
    is exactly the period a live bot is most curious about."""
    bars = series("X", [100.0, 101.0])
    assert HistoricalMarketData({"X": bars}).forward_return("X", bars[0].t, 5) is None


def test_the_first_bar_at_or_after_the_decision_is_the_entry():
    bars = series("X", [100.0, 101.0, 102.0, 103.0])
    data = HistoricalMarketData({"X": bars})
    entry, _, _ = data.forward_return("X", bars[1].t - timedelta(minutes=1), 2)
    assert entry == 101.0


def test_a_decision_after_the_last_bar_has_no_forward_return():
    bars = series("X", [100.0])
    assert HistoricalMarketData({"X": bars}).forward_return("X", NOW + timedelta(days=1), 1) is None


def test_an_unknown_symbol_has_no_forward_return():
    assert HistoricalMarketData({}).forward_return("NOPE", NOW, 1) is None


# ------------------------------------------------------------------ shared
def test_both_sources_satisfy_the_protocol():
    """If this ever fails, the backtest is no longer testing the live path."""
    assert isinstance(HistoricalMarketData({}), MarketDataSource)
    assert isinstance(alpaca(FakeSession()), MarketDataSource)


def test_a_plain_object_does_not_satisfy_the_protocol():
    assert not isinstance(object(), MarketDataSource)
