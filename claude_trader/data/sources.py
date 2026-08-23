"""Market data sources.

The Protocol is the whole point: the live engine and the backtester consume the
identical interface, so the decision path under test is the decision path that
trades. ``as_of`` is mandatory everywhere -- a historical source must be
physically unable to hand back a bar from the future.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Protocol, Sequence, runtime_checkable

from ..config import AppConfig
from ..errors import MarketDataError
from ..http import request_json
from ..models import Bar, Quote


@runtime_checkable
class MarketDataSource(Protocol):
    def bars(
        self, symbol: str, limit: int, as_of: datetime
    ) -> tuple[Bar, ...]: ...

    def quote(self, symbol: str, as_of: datetime) -> Quote | None: ...

    def latest_prices(
        self, symbols: Sequence[str], as_of: datetime
    ) -> Mapping[str, float]: ...


def _parse_ts(raw: str) -> datetime:
    text = raw.replace("Z", "+00:00")
    value = datetime.fromisoformat(text)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def bar_from_payload(symbol: str, payload: Mapping[str, object]) -> Bar | None:
    try:
        return Bar(
            symbol=symbol,
            t=_parse_ts(str(payload["t"])),
            o=float(payload["o"]),
            h=float(payload["h"]),
            l=float(payload["l"]),
            c=float(payload["c"]),
            v=float(payload.get("v", 0) or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _quote_from_payload(symbol: str, payload: Mapping[str, object]) -> Quote | None:
    try:
        bid = float(payload.get("bp", 0) or 0)
        ask = float(payload.get("ap", 0) or 0)
        if bid <= 0 and ask <= 0:
            return None
        return Quote(
            symbol=symbol,
            t=_parse_ts(str(payload.get("t", ""))) if payload.get("t") else datetime.now(timezone.utc),
            bid=bid,
            ask=ask,
            bid_size=float(payload.get("bs", 0) or 0),
            ask_size=float(payload.get("as", 0) or 0),
        )
    except (TypeError, ValueError):
        return None


class AlpacaMarketData:
    """Live market data from Alpaca.

    Note on ``feed``: the default 'iex' covers a small slice of consolidated
    volume and is delayed on the free tier. The engine checks quote age against
    RiskConfig.max_quote_age_seconds and refuses to trade on stale prints
    rather than pretending the data is real-time.
    """

    def __init__(self, config: AppConfig, session: object | None = None) -> None:
        self._config = config
        self._session = session

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._config.alpaca_key,
            "APCA-API-SECRET-KEY": self._config.alpaca_secret,
            "Content-Type": "application/json",
        }

    def bars(self, symbol: str, limit: int, as_of: datetime) -> tuple[Bar, ...]:
        payload = request_json(
            "GET",
            f"{self._config.alpaca_data_base}/stocks/{symbol}/bars",
            headers=self._headers,
            params={
                "timeframe": self._config.timeframe,
                "limit": limit,
                "feed": self._config.feed,
                "end": as_of.astimezone(timezone.utc).isoformat(),
                "adjustment": "raw",
            },
            timeout=15.0,
            session=self._session,
        )
        raw = payload.get("bars") or []
        bars = [b for b in (bar_from_payload(symbol, item) for item in raw) if b]
        bars = [b for b in bars if b.t <= as_of]
        return tuple(sorted(bars, key=lambda b: b.t))

    def quote(self, symbol: str, as_of: datetime) -> Quote | None:
        payload = request_json(
            "GET",
            f"{self._config.alpaca_data_base}/stocks/{symbol}/quotes/latest",
            headers=self._headers,
            params={"feed": self._config.feed},
            timeout=10.0,
            session=self._session,
        )
        raw = payload.get("quote")
        return _quote_from_payload(symbol, raw) if isinstance(raw, dict) else None

    def snapshots(self, symbols: Sequence[str]) -> Mapping[str, dict]:
        if not symbols:
            return {}
        payload = request_json(
            "GET",
            f"{self._config.alpaca_data_base}/stocks/snapshots",
            headers=self._headers,
            params={"symbols": ",".join(symbols), "feed": self._config.feed},
            timeout=20.0,
            session=self._session,
        )
        if not isinstance(payload, dict):
            raise MarketDataError("snapshots endpoint returned an unexpected shape")
        return payload

    def latest_prices(
        self, symbols: Sequence[str], as_of: datetime
    ) -> Mapping[str, float]:
        out: dict[str, float] = {}
        for symbol, data in self.snapshots(symbols).items():
            if not isinstance(data, dict):
                continue
            bar = data.get("latestTrade") or data.get("dailyBar") or {}
            price = bar.get("p") if "p" in bar else bar.get("c")
            try:
                if price is not None and float(price) > 0:
                    out[symbol] = float(price)
            except (TypeError, ValueError):
                continue
        return out


class HistoricalMarketData:
    """Replay source for the backtester.

    Holds the full bar history in memory and slices strictly at ``as_of``. The
    slice is the mechanism that makes lookahead bias structurally impossible
    rather than merely discouraged.
    """

    def __init__(self, bars_by_symbol: Mapping[str, Sequence[Bar]]) -> None:
        self._bars = {
            symbol: tuple(sorted(series, key=lambda b: b.t))
            for symbol, series in bars_by_symbol.items()
        }

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._bars))

    def all_bars(self, symbol: str) -> tuple[Bar, ...]:
        return self._bars.get(symbol, ())

    def timeline(self) -> tuple[datetime, ...]:
        stamps = {bar.t for series in self._bars.values() for bar in series}
        return tuple(sorted(stamps))

    def bars(self, symbol: str, limit: int, as_of: datetime) -> tuple[Bar, ...]:
        series = self._bars.get(symbol, ())
        visible = [b for b in series if b.t <= as_of]
        return tuple(visible[-limit:]) if limit > 0 else tuple(visible)

    def bar_at(self, symbol: str, ts: datetime) -> Bar | None:
        for bar in self._bars.get(symbol, ()):
            if bar.t == ts:
                return bar
        return None

    def next_bar_after(self, symbol: str, ts: datetime) -> Bar | None:
        for bar in self._bars.get(symbol, ()):
            if bar.t > ts:
                return bar
        return None

    def quote(self, symbol: str, as_of: datetime) -> Quote | None:
        """Synthesise a quote from the last visible close.

        A modelled half-spread is applied so the backtest cannot enjoy free
        mid-price fills that live trading never gets.
        """
        visible = self.bars(symbol, 1, as_of)
        if not visible:
            return None
        close = visible[-1].c
        if close <= 0:
            return None
        half_spread = max(close * 0.0002, 0.01)
        return Quote(
            symbol=symbol,
            t=visible[-1].t,
            bid=close - half_spread,
            ask=close + half_spread,
        )

    def latest_prices(
        self, symbols: Sequence[str], as_of: datetime
    ) -> Mapping[str, float]:
        out: dict[str, float] = {}
        for symbol in symbols:
            visible = self.bars(symbol, 1, as_of)
            if visible:
                out[symbol] = visible[-1].c
        return out

    def forward_return(
        self, symbol: str, start: datetime, horizon_bars: int
    ) -> tuple[float, float, float] | None:
        """Entry price, exit price and simple return ``horizon_bars`` after
        ``start``, or None if the horizon has not fully elapsed.

        Used by the calibration job to answer "did the 9s actually beat the 5s".
        A partially elapsed horizon returns None rather than a short window,
        because truncating would bias the sample towards recent price action.
        """
        series = self._bars.get(symbol, ())
        for index, bar in enumerate(series):
            if bar.t >= start:
                target = index + horizon_bars
                if target >= len(series) or bar.c <= 0:
                    return None
                exit_price = series[target].c
                return bar.c, exit_price, (exit_price - bar.c) / bar.c
        return None
