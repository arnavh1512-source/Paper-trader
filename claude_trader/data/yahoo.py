"""Yahoo Finance chart API as a market data source.

Why this and not a broker API: every Indian broker with a REST feed (Zerodha,
Upstox, Angel One, Dhan, Fyers) requires an account, a KYC'd trading login and
in Zerodha's case a monthly fee, before you can pull a single bar. Yahoo's chart
endpoint needs no key, covers NSE via the ``.NS`` suffix, and serves the same
15-minute OHLCV the strategy consumes. That makes the whole system runnable
today, by anyone, for nothing.

What you give up, stated plainly because the risk layer depends on it:

* **No bid/ask.** The chart endpoint returns trades, not quotes. The spread gate
  therefore runs against a *modelled* spread, not an observed one. On NIFTY
  large caps the real spread is 1-5bps, so the model is close -- but it is a
  model, and ``Quote.modelled`` says so.
* **Roughly 15 minutes delayed** on NSE for the free tier, which is why the
  staleness gate compares against ``regularMarketTime`` from the payload rather
  than assuming the data is live.
* **Rate limited and unofficial.** Responses are cached per cycle and the source
  degrades to "no data for this symbol" rather than taking the cycle down.

Swap in a broker feed later by implementing ``MarketDataSource``; nothing above
this module knows where bars come from.
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import quote as urlquote

from ..errors import MarketDataError
from ..http import request_json
from ..markets import MarketProfile
from ..models import Bar, Quote

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Yahoo rejects the default python-requests agent on some edges.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# How far back each intraday interval is served. Asking for more silently
# returns less, which would look like a data gap rather than a limit.
MAX_RANGE_DAYS = {
    "1m": 7,
    "2m": 59,
    "5m": 59,
    "15m": 59,
    "30m": 59,
    "60m": 729,
    "1h": 729,
    "1d": 36_500,
}

# Modelled half-spreads in basis points, by liquidity tier. Deliberately
# pessimistic: a strategy that only works with optimistic spreads does not work.
LARGE_CAP_HALF_SPREAD_BPS = 2.5
DEFAULT_HALF_SPREAD_BPS = 5.0


def _epoch(moment: datetime) -> int:
    return int(moment.astimezone(timezone.utc).timestamp())


def parse_chart(symbol: str, payload: Mapping[str, Any]) -> tuple[Bar, ...]:
    """Convert one Yahoo chart response into bars.

    Yahoo emits ``null`` for gaps inside the OHLCV arrays (halts, illiquid
    minutes). Those rows are dropped rather than forward-filled: a fabricated
    bar is worse than a missing one, because indicators cannot tell them apart.
    """
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        raise MarketDataError(f"unexpected chart payload for {symbol}")
    if chart.get("error"):
        raise MarketDataError(f"{symbol}: {chart['error']}")

    results = chart.get("result") or []
    if not results:
        return ()
    result = results[0]

    stamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens = quotes.get("open") or []
    highs = quotes.get("high") or []
    lows = quotes.get("low") or []
    closes = quotes.get("close") or []
    volumes = quotes.get("volume") or []

    bars: list[Bar] = []
    for i, stamp in enumerate(stamps):
        try:
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        except IndexError:
            break
        if None in (o, h, l, c) or stamp is None:
            continue
        try:
            close = float(c)
            if close <= 0:
                continue
            bars.append(
                Bar(
                    symbol=symbol,
                    t=datetime.fromtimestamp(int(stamp), tz=timezone.utc),
                    o=float(o),
                    h=float(h),
                    l=float(l),
                    c=close,
                    v=float(volumes[i] or 0) if i < len(volumes) else 0.0,
                )
            )
        except (TypeError, ValueError):
            continue
    return tuple(sorted(bars, key=lambda b: b.t))


def chart_meta(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    results = (payload.get("chart") or {}).get("result") or []
    return (results[0].get("meta") or {}) if results else {}


class YahooMarketData:
    """Live NSE (or any Yahoo-covered) market data.

    ``cache_ttl`` exists because one cycle asks for the same symbol several
    times -- overview, snapshot, quote, benchmark -- and Yahoo throttles callers
    who do not notice.
    """

    def __init__(
        self,
        profile: MarketProfile,
        interval: str | None = None,
        cache_ttl: float = 60.0,
        session: object | None = None,
        pause_between_calls: float = 0.0,
    ) -> None:
        self._profile = profile
        self._interval = (interval or profile.timeframe).lower()
        if self._interval not in MAX_RANGE_DAYS:
            raise MarketDataError(
                f"interval {self._interval!r} unsupported; "
                f"choose from {', '.join(sorted(MAX_RANGE_DAYS))}"
            )
        self._ttl = cache_ttl
        self._session = session
        self._pause = pause_between_calls
        self._cache: dict[str, tuple[float, tuple[Bar, ...], Mapping[str, Any]]] = {}
        # symbol -> the vendor spelling that actually returned bars
        self._resolved: dict[str, str] = {}

    # ------------------------------------------------------------------ fetch
    def _fetch(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        use_cache: bool = True,
    ) -> tuple[tuple[Bar, ...], Mapping[str, Any]]:
        spellings = self._spellings(symbol)
        vendor = spellings[0]
        key = f"{vendor}|{interval}|{_epoch(start) if start else ''}|{_epoch(end) if end else ''}"
        now = _time.monotonic()
        if use_cache:
            hit = self._cache.get(key)
            if hit and now - hit[0] < self._ttl:
                return hit[1], hit[2]

        params: dict[str, Any] = {"interval": interval, "includePrePost": "false"}
        if start is not None and end is not None:
            params["period1"] = _epoch(start)
            params["period2"] = _epoch(end)
        else:
            params["range"] = f"{MAX_RANGE_DAYS[interval]}d"

        bars: tuple[Bar, ...] = ()
        meta: Mapping[str, Any] = {}
        for attempt, spelling in enumerate(spellings):
            if self._pause:
                _time.sleep(self._pause)
            try:
                payload = request_json(
                    "GET",
                    CHART_URL.format(symbol=urlquote(spelling, safe="")),
                    headers=HEADERS,
                    params=params,
                    timeout=20.0,
                    session=self._session,
                )
            except Exception:
                # A dead primary listing is exactly the case the fallbacks
                # exist for, so exhaust them before giving up. The last
                # spelling's failure is the real one and is allowed to raise.
                if attempt == len(spellings) - 1:
                    raise
                continue
            bars = parse_chart(symbol, payload)
            meta = chart_meta(payload)
            if bars:
                if attempt:
                    # Remember the winner: without this every cycle pays for
                    # the failed primary lookup again.
                    self._resolved[symbol] = spelling
                    log.info(
                        "%s: no data on %s, using %s",
                        symbol, spellings[0], spelling,
                    )
                break
        self._cache[key] = (now, bars, meta)
        return bars, meta

    def _spellings(self, symbol: str) -> tuple[str, ...]:
        """Vendor tickers to try, best first.

        Once a symbol has been resolved to a fallback listing it stays there for
        the life of the process. Re-probing a listing that was empty an hour ago
        costs a round trip per symbol per cycle and almost never changes answer.
        """
        resolved = self._resolved.get(symbol)
        if resolved:
            return (resolved,)
        return self._profile.data_symbols(symbol)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._resolved.clear()

    # ------------------------------------------------- MarketDataSource impl
    def bars(self, symbol: str, limit: int, as_of: datetime) -> tuple[Bar, ...]:
        series, _ = self._fetch(symbol, self._interval)
        visible = [b for b in series if b.t <= as_of]
        return tuple(visible[-limit:]) if limit > 0 else tuple(visible)

    def history(
        self, symbol: str, start: datetime, end: datetime, interval: str | None = None
    ) -> tuple[Bar, ...]:
        """Explicit window, used by the backtest dataset loader.

        Intraday history beyond the vendor's window simply does not exist, so an
        over-long request is trimmed and logged instead of returning a series
        that silently starts later than asked.
        """
        iv = (interval or self._interval).lower()
        span_cap = timedelta(days=MAX_RANGE_DAYS.get(iv, 59))
        if end - start > span_cap:
            log.warning(
                "%s: %s history is capped at %d days by the vendor; "
                "requested window trimmed.",
                symbol,
                iv,
                span_cap.days,
            )
            start = end - span_cap
        series, _ = self._fetch(symbol, iv, start=start, end=end, use_cache=False)
        return tuple(b for b in series if start <= b.t <= end)

    def quote(self, symbol: str, as_of: datetime) -> Quote | None:
        series, meta = self._fetch(symbol, self._interval)
        visible = [b for b in series if b.t <= as_of]
        price = 0.0
        stamp = None

        raw_price = meta.get("regularMarketPrice")
        raw_time = meta.get("regularMarketTime")
        try:
            if raw_price is not None and float(raw_price) > 0:
                price = float(raw_price)
            if raw_time is not None:
                stamp = datetime.fromtimestamp(int(raw_time), tz=timezone.utc)
        except (TypeError, ValueError):
            price, stamp = 0.0, None

        # The meta price is the *current* print; during a backtest-style replay
        # it would be a look-ahead, so only trust it at the leading edge.
        if visible and (stamp is None or stamp > as_of or price <= 0):
            price, stamp = visible[-1].c, visible[-1].t
        if price <= 0 or stamp is None:
            return None

        half = price * (self._half_spread_bps(symbol) / 10_000.0)
        return Quote(
            symbol=symbol,
            t=stamp,
            bid=round(price - half, 4),
            ask=round(price + half, 4),
            modelled=True,
        )

    def _half_spread_bps(self, symbol: str) -> float:
        return (
            LARGE_CAP_HALF_SPREAD_BPS
            if symbol.upper() in self._profile.sectors
            else DEFAULT_HALF_SPREAD_BPS
        )

    def latest_prices(
        self, symbols: Sequence[str], as_of: datetime
    ) -> Mapping[str, float]:
        out: dict[str, float] = {}
        for symbol in symbols:
            try:
                series = self.bars(symbol, 1, as_of)
            except MarketDataError as exc:
                log.warning("price unavailable for %s: %s", symbol, exc)
                continue
            if series:
                out[symbol] = series[-1].c
        return out

    def is_trading_now(self, as_of: datetime, reference: str | None = None) -> bool:
        """Whether the exchange is genuinely open, judged by whether the
        reference symbol has printed a bar recently.

        Asking the data rather than a hardcoded holiday list is the only way to
        stay correct through NSE's lunar-calendar holidays and unscheduled
        closures.
        """
        if not self._profile.is_session_time(as_of):
            return False
        symbol = reference or self._profile.benchmark
        try:
            series = self.bars(symbol, 1, as_of)
        except MarketDataError:
            return False
        if not series:
            return False
        # Two bar-widths of tolerance covers the vendor's publication delay.
        return (as_of - series[-1].t) <= timedelta(minutes=self._bar_minutes() * 2 + 20)

    def _bar_minutes(self) -> int:
        digits = "".join(ch for ch in self._interval if ch.isdigit())
        unit = self._interval[len(digits) :]
        value = int(digits or 1)
        return value * 60 if unit in {"h", "hr"} else value
