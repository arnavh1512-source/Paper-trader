"""Historical bar loading and the synthetic generator.

The cache key is the part worth guarding: two datasets that differ in market,
timeframe or window but share a cache slot would silently backtest the wrong
prices, and nothing downstream could detect it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.backtest.dataset import (
    DatasetSpec,
    fetch_bars,
    fetch_bars_yahoo,
    from_json,
    load_dataset,
    synthetic_bars,
    to_json,
)
from claude_trader.config import AppConfig
from claude_trader.errors import MarketDataError
from claude_trader.markets.india import INDIA_MARKET
from claude_trader.markets.us import US_MARKET
from claude_trader.models import Bar

START = datetime(2026, 1, 5, tzinfo=timezone.utc)
END = datetime(2026, 3, 2, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
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


def spec(**overrides) -> DatasetSpec:
    return DatasetSpec(**{
        "symbols": ("AAPL", "MSFT"), "start": START, "end": END,
        "timeframe": "15Min", "feed": "iex", "market": "us", **overrides,
    })


def row(minute: int, close: float = 100.0) -> dict:
    stamp = (START + timedelta(minutes=minute)).isoformat().replace("+00:00", "Z")
    return {"t": stamp, "o": close, "h": close, "l": close, "c": close, "v": 1_000}


def us_config(**overrides) -> AppConfig:
    return AppConfig(market="us", alpaca_key="key", alpaca_secret="secret", **overrides)


# ------------------------------------------------------------------ cache key
def test_the_cache_name_separates_the_two_markets():
    """RELIANCE and AAPL 15m bars would otherwise collide on symbol count
    alone, and the backtest would replay the wrong country's prices."""
    assert spec(market="us").cache_name() != spec(market="in").cache_name()


@pytest.mark.parametrize("field,value", [
    ("timeframe", "60Min"),
    ("feed", "sip"),
    ("start", START - timedelta(days=30)),
    ("end", END + timedelta(days=30)),
    ("symbols", ("AAPL",)),
])
def test_every_part_of_the_spec_changes_the_cache_name(field, value):
    assert spec().cache_name() != spec(**{field: value}).cache_name()


def test_the_cache_name_is_stable_for_the_same_spec():
    assert spec().cache_name() == spec().cache_name()


def test_the_cache_name_is_a_safe_filename():
    name = spec().cache_name()
    assert name.endswith(".json")
    assert not set(name) & set('/\\:*?"<>|')


# ------------------------------------------------------------ serialisation
def test_bars_survive_a_round_trip_through_json():
    bars = {"AAPL": (Bar(symbol="AAPL", t=START, o=1.0, h=2.0, l=0.5, c=1.5, v=99.0),)}
    restored = from_json(to_json(bars))
    assert restored["AAPL"][0] == bars["AAPL"][0]


def test_a_corrupt_row_in_the_cache_is_dropped_not_fatal():
    """A truncated download should cost you one bar, not the whole file."""
    raw = json.dumps({"AAPL": [row(0), {"t": "nonsense"}, row(15)]})
    assert len(from_json(raw)["AAPL"]) == 2


def test_a_symbol_with_no_usable_rows_is_omitted_entirely():
    assert from_json(json.dumps({"AAPL": [{"broken": True}]})) == {}


def test_cached_bars_are_sorted_on_load():
    """Cache files are appended to by paging; order is not guaranteed."""
    raw = json.dumps({"AAPL": [row(30), row(0), row(15)]})
    stamps = [b.t for b in from_json(raw)["AAPL"]]
    assert stamps == sorted(stamps)


# -------------------------------------------------------------------- alpaca
def test_alpaca_bars_are_requested_with_the_full_window():
    session = FakeSession({"bars": {"AAPL": [row(0)]}})
    fetch_bars(us_config(), spec(symbols=("AAPL",)), session=session)
    params = session.calls[0]["params"]
    assert params["symbols"] == "AAPL"
    assert params["timeframe"] == "15Min"
    assert params["feed"] == "iex"
    assert params["adjustment"] == "split"   # backtests need split-adjusted history
    assert params["start"].endswith("Z") and params["end"].endswith("Z")


def test_paging_continues_until_the_cursor_runs_out():
    """Six months of 15-minute bars for thirty symbols does not fit in one
    page, and a loader that stops at the first one silently truncates history."""
    session = FakeSession(
        {"bars": {"AAPL": [row(0)]}, "next_page_token": "p2"},
        {"bars": {"AAPL": [row(15)]}},
    )
    bars = fetch_bars(us_config(), spec(symbols=("AAPL",)), session=session)
    assert len(bars["AAPL"]) == 2
    assert session.calls[1]["params"]["page_token"] == "p2"


def test_a_runaway_cursor_is_stopped():
    """A vendor that keeps handing back the same token would download forever."""
    session = FakeSession(*[{"bars": {"AAPL": [row(0)]}, "next_page_token": "same"}] * 600)
    fetch_bars(us_config(), spec(symbols=("AAPL",)), session=session)
    assert len(session.calls) <= 502


def test_an_unexpected_payload_shape_is_a_typed_error():
    with pytest.raises(MarketDataError, match="unexpected shape"):
        fetch_bars(us_config(), spec(), session=FakeSession([1, 2]))


def test_a_symbol_with_no_bars_is_left_out_of_the_dataset():
    session = FakeSession({"bars": {"AAPL": [row(0)], "MSFT": []}})
    assert set(fetch_bars(us_config(), spec(), session=session)) == {"AAPL"}


def test_bars_are_sorted_across_pages():
    session = FakeSession(
        {"bars": {"AAPL": [row(30)]}, "next_page_token": "p2"},
        {"bars": {"AAPL": [row(0)]}},
    )
    stamps = [b.t for b in fetch_bars(us_config(), spec(symbols=("AAPL",)), session=session)["AAPL"]]
    assert stamps == sorted(stamps)


def test_credentials_are_sent_on_the_download():
    session = FakeSession({"bars": {"AAPL": [row(0)]}})
    fetch_bars(us_config(), spec(symbols=("AAPL",)), session=session)
    assert session.calls[0]["headers"]["APCA-API-KEY-ID"] == "key"


# --------------------------------------------------------------------- yahoo
def test_yahoo_is_fetched_one_symbol_at_a_time():
    """Yahoo has no batch endpoint, so this is a per-symbol loop by necessity."""
    chart = _chart("RELIANCE.NS")
    session = FakeSession(chart, chart)
    config = AppConfig(market="in")
    bars = fetch_bars_yahoo(config, spec(symbols=("RELIANCE", "TCS"), market="in",
                                         timeframe="15m"), session=session)
    assert len(session.calls) == 2
    assert set(bars) == {"RELIANCE", "TCS"}


def test_one_delisted_symbol_does_not_cost_you_the_others():
    session = FakeSession({"chart": {"error": {"description": "not found"}}},
                          _chart("TCS.NS"))
    bars = fetch_bars_yahoo(AppConfig(market="in"),
                            spec(symbols=("GONE", "TCS"), market="in", timeframe="15m"),
                            session=session)
    assert set(bars) == {"TCS"}


def test_a_symbol_with_an_empty_window_is_skipped():
    session = FakeSession({"chart": {"result": [{"meta": {}, "timestamp": [],
                                                 "indicators": {"quote": [{}]}}]}})
    bars = fetch_bars_yahoo(AppConfig(market="in"),
                            spec(symbols=("TCS",), market="in", timeframe="15m"),
                            session=session)
    assert bars == {}


# ------------------------------------------------------------------- caching
def test_a_download_is_written_to_the_cache(tmp_path):
    session = FakeSession({"bars": {"AAPL": [row(0)]}})
    load_dataset(us_config(), spec(symbols=("AAPL",)), cache_dir=tmp_path, session=session)
    assert list(tmp_path.glob("bars-*.json"))


def test_the_second_load_reads_the_cache_and_makes_no_request(tmp_path):
    """A backtest that re-downloads is a backtest nobody runs twice."""
    session = FakeSession({"bars": {"AAPL": [row(0)]}})
    load_dataset(us_config(), spec(symbols=("AAPL",)), cache_dir=tmp_path, session=session)
    again = FakeSession()
    bars = load_dataset(us_config(), spec(symbols=("AAPL",)), cache_dir=tmp_path,
                        session=again)
    assert again.calls == []
    assert bars["AAPL"]


def test_refresh_forces_a_new_download(tmp_path):
    session = FakeSession({"bars": {"AAPL": [row(0)]}})
    load_dataset(us_config(), spec(symbols=("AAPL",)), cache_dir=tmp_path, session=session)
    again = FakeSession({"bars": {"AAPL": [row(0), row(15)]}})
    bars = load_dataset(us_config(), spec(symbols=("AAPL",)), cache_dir=tmp_path,
                        refresh=True, session=again)
    assert len(bars["AAPL"]) == 2


def test_an_empty_download_is_an_error_rather_than_an_empty_cache_file(tmp_path):
    """Writing an empty cache would make every later run silently reuse it."""
    with pytest.raises(MarketDataError, match="no bars"):
        load_dataset(us_config(), spec(symbols=("AAPL",)), cache_dir=tmp_path,
                     session=FakeSession({"bars": {}}))
    assert not list(tmp_path.glob("*.json"))


def test_a_us_download_requires_credentials(tmp_path, clean_env):
    config = AppConfig(market="us", alpaca_key="", alpaca_secret="")
    with pytest.raises(Exception):
        load_dataset(config, spec(symbols=("AAPL",)), cache_dir=tmp_path,
                     session=FakeSession({"bars": {"AAPL": [row(0)]}}))


def test_an_indian_download_needs_no_credentials(tmp_path, clean_env):
    """Yahoo is keyless, which is the whole reason the NSE path uses it."""
    session = FakeSession(_chart("TCS.NS"))
    bars = load_dataset(AppConfig(market="in"),
                        spec(symbols=("TCS",), market="in", timeframe="15m"),
                        cache_dir=tmp_path, session=session)
    assert bars["TCS"]


# ----------------------------------------------------------------- synthetic
def test_the_synthetic_walk_is_deterministic():
    """Otherwise a regression in the engine shows up as noise rather than as a
    changed number."""
    a = synthetic_bars(["X"], sessions=5, seed=7)
    b = synthetic_bars(["X"], sessions=5, seed=7)
    assert [bar.c for bar in a["X"]] == [bar.c for bar in b["X"]]


def test_a_different_seed_gives_a_different_walk():
    a = synthetic_bars(["X"], sessions=5, seed=7)
    b = synthetic_bars(["X"], sessions=5, seed=8)
    assert [bar.c for bar in a["X"]] != [bar.c for bar in b["X"]]


def test_weekends_are_skipped_so_the_timeline_looks_like_a_calendar():
    bars = synthetic_bars(["X"], sessions=14)
    assert all(bar.t.weekday() < 5 for bar in bars["X"])


def test_prices_stay_positive_over_a_long_walk():
    """A geometric walk cannot go negative; a bug in the exponent could."""
    bars = synthetic_bars(["X"], sessions=60, vol=0.02)
    assert all(bar.c > 0 and bar.l > 0 for bar in bars["X"])


def test_each_bar_is_internally_consistent():
    bars = synthetic_bars(["X"], sessions=10)
    assert all(bar.l <= min(bar.o, bar.c) and bar.h >= max(bar.o, bar.c)
               for bar in bars["X"])


def test_symbols_start_at_different_price_levels():
    """Identical series across the universe would make every ranking a tie."""
    bars = synthetic_bars(["A", "B", "C"], sessions=3)
    assert len({bars[s][0].o for s in bars}) == 3


def test_the_indian_profile_shapes_the_walk_like_nse():
    """Sizing Rs 2,000 stocks as though they cost Rs 100 would make every
    position-size test meaningless."""
    bars = synthetic_bars(["RELIANCE"], sessions=3, profile=INDIA_MARKET)["RELIANCE"]
    assert bars[0].o > 500
    assert INDIA_MARKET.local(bars[0].t).time() == INDIA_MARKET.open_time
    assert len([b for b in bars if b.t.date() == bars[0].t.date()]) == \
        INDIA_MARKET.bars_per_session


def test_the_us_profile_opens_at_the_us_bell():
    bars = synthetic_bars(["AAPL"], sessions=3, profile=US_MARKET)["AAPL"]
    assert US_MARKET.local(bars[0].t).time() == US_MARKET.open_time
    assert bars[0].o < 500   # fractional market, ordinary share price


def test_the_synthetic_generator_needs_no_network_or_key():
    """This is what CI and the offline smoke run depend on."""
    assert synthetic_bars(["X", "Y"], sessions=2)["Y"]


def _chart(vendor_symbol: str, count: int = 30) -> dict:
    base = int(START.timestamp())
    closes = [100.0 + i for i in range(count)]
    return {
        "chart": {
            "result": [{
                "meta": {"symbol": vendor_symbol, "regularMarketPrice": closes[-1]},
                "timestamp": [base + i * 900 for i in range(count)],
                "indicators": {"quote": [{
                    "open": closes, "high": [c * 1.01 for c in closes],
                    "low": [c * 0.99 for c in closes], "close": closes,
                    "volume": [1_000] * count,
                }]},
            }]
        }
    }
