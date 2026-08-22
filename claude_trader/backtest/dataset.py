"""Historical bar loading.

Downloads once, caches on disk, and replays from the cache thereafter. A
backtest that re-downloads is a backtest nobody runs twice.

``synthetic_bars`` exists so the whole engine can be exercised with no
credentials and no network -- that is what CI and the smoke test use. It is a
random walk, not a market; never read a result from it as evidence about the
strategy.

Two vendors are supported because two markets are. Alpaca serves US bars and
needs credentials; Yahoo serves NSE bars and needs none, but caps intraday
history at a few weeks, which is why the cache matters more on that path.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..config import AppConfig
from ..data.sources import bar_from_payload  # one parser for live and historical
from ..errors import MarketDataError
from ..http import request_json
from ..markets import MarketProfile
from ..models import Bar

log = logging.getLogger(__name__)

MAX_PAGE_LIMIT = 10_000


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    timeframe: str
    feed: str
    market: str = "us"

    def cache_name(self) -> str:
        # The market belongs in the name: RELIANCE 15m bars and AAPL 15m bars
        # would otherwise collide in the same cache slot on symbol count alone.
        stamp = f"{self.start:%Y%m%d}-{self.end:%Y%m%d}"
        digest = f"{self.market}-{len(self.symbols)}sym-{self.timeframe}-{self.feed}-{stamp}"
        return f"bars-{digest}.json"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_bars_yahoo(
    config: AppConfig,
    spec: DatasetSpec,
    session: object | None = None,
) -> dict[str, tuple[Bar, ...]]:
    """One request per symbol; Yahoo has no batch endpoint.

    A symbol that returns nothing is logged and skipped rather than failing the
    whole download: one delisted name should not cost you the other twenty.
    """
    from ..data.yahoo import YahooMarketData

    source = YahooMarketData(
        config.profile,
        interval=spec.timeframe,
        session=session,
        pause_between_calls=0.35,  # Yahoo throttles callers who do not wait
    )
    collected: dict[str, tuple[Bar, ...]] = {}
    for symbol in spec.symbols:
        try:
            bars = source.history(symbol, spec.start, spec.end)
        except MarketDataError as exc:
            log.warning("%s: no history (%s); skipping", symbol, exc)
            continue
        if bars:
            collected[symbol] = bars
        else:
            log.warning("%s: vendor returned no bars for the window", symbol)
    return collected


def fetch_bars(
    config: AppConfig,
    spec: DatasetSpec,
    session: object | None = None,
) -> dict[str, tuple[Bar, ...]]:
    """Page through Alpaca's multi-symbol bars endpoint."""
    headers = {
        "APCA-API-KEY-ID": config.alpaca_key,
        "APCA-API-SECRET-KEY": config.alpaca_secret,
        "Content-Type": "application/json",
    }
    collected: dict[str, list[Bar]] = {s: [] for s in spec.symbols}
    page_token: str | None = None
    pages = 0

    while True:
        params: dict[str, object] = {
            "symbols": ",".join(spec.symbols),
            "timeframe": spec.timeframe,
            "start": _iso(spec.start),
            "end": _iso(spec.end),
            "limit": MAX_PAGE_LIMIT,
            "feed": spec.feed,
            "adjustment": "split",
        }
        if page_token:
            params["page_token"] = page_token

        payload = request_json(
            "GET",
            f"{config.alpaca_data_base}/stocks/bars",
            headers=headers,
            params=params,
            timeout=45.0,
            session=session,
        )
        if not isinstance(payload, dict):
            raise MarketDataError("bars endpoint returned an unexpected shape")

        chunk = payload.get("bars") or {}
        for symbol, rows in chunk.items():
            bucket = collected.setdefault(symbol, [])
            for row in rows or []:
                bar = bar_from_payload(symbol, row)
                if bar is not None:
                    bucket.append(bar)

        pages += 1
        page_token = payload.get("next_page_token")
        if not page_token:
            break
        if pages > 500:  # a runaway cursor should not download forever
            log.warning("stopping after %d pages; check the date range", pages)
            break

    return {
        symbol: tuple(sorted(bars, key=lambda b: b.t))
        for symbol, bars in collected.items()
        if bars
    }


def _timeframe_minutes(timeframe: str) -> int:
    digits = "".join(ch for ch in timeframe if ch.isdigit())
    minutes = int(digits) if digits else 15
    return minutes * 60 if timeframe.lower().endswith(("h", "hour")) else minutes


def to_json(bars_by_symbol: Mapping[str, Sequence[Bar]]) -> str:
    return json.dumps(
        {
            symbol: [
                {"t": b.t.isoformat(), "o": b.o, "h": b.h, "l": b.l, "c": b.c, "v": b.v}
                for b in bars
            ]
            for symbol, bars in bars_by_symbol.items()
        }
    )


def from_json(raw: str) -> dict[str, tuple[Bar, ...]]:
    payload = json.loads(raw)
    out: dict[str, tuple[Bar, ...]] = {}
    for symbol, rows in payload.items():
        bars = [b for b in (bar_from_payload(symbol, row) for row in rows) if b]
        if bars:
            out[symbol] = tuple(sorted(bars, key=lambda b: b.t))
    return out


def load_dataset(
    config: AppConfig,
    spec: DatasetSpec,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
    session: object | None = None,
) -> dict[str, tuple[Bar, ...]]:
    path = Path(cache_dir) / spec.cache_name()
    if path.exists() and not refresh:
        log.info("Using cached bars: %s", path)
        return from_json(path.read_text(encoding="utf-8"))

    log.info(
        "Downloading %d symbols of %s bars, %s to %s (%s)",
        len(spec.symbols),
        spec.timeframe,
        spec.start.date(),
        spec.end.date(),
        "Yahoo" if config.market != "us" else f"Alpaca {spec.feed}",
    )
    if config.market == "us":
        config.require_live_credentials(need_llm=False)
        bars = fetch_bars(config, spec, session=session)
    else:
        bars = fetch_bars_yahoo(config, spec, session=session)
    if not bars:
        raise MarketDataError("no bars returned for the requested window")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(bars), encoding="utf-8")
    log.info("Cached %d symbols to %s", len(bars), path)
    return bars


# --------------------------------------------------------------------- offline
def synthetic_bars(
    symbols: Iterable[str],
    sessions: int = 30,
    bars_per_session: int = 26,
    start: datetime | None = None,
    seed: int = 7,
    start_price: float = 100.0,
    drift: float = 0.00002,
    vol: float = 0.0025,
    step: timedelta = timedelta(minutes=15),
    profile: MarketProfile | None = None,
) -> dict[str, tuple[Bar, ...]]:
    """Deterministic geometric random walk on a synthetic session calendar.

    Used for tests and the offline smoke run. Deterministic for a given seed so
    a regression in the engine shows up as a changed result rather than noise.

    Passing a ``profile`` shapes the walk like that market: its session length,
    its opening bell, and a price level in the right order of magnitude, so an
    Indian run does not size Rs 2,000 stocks as though they cost Rs 100.
    """
    if profile is not None:
        bars_per_session = profile.bars_per_session
        step = timedelta(minutes=max(1, _timeframe_minutes(profile.timeframe)))
        start_price = 100.0 if profile.fractional_shares else 1_200.0
        if start is None:
            open_utc = (
                profile.open_time.hour * 60
                + profile.open_time.minute
                - profile.utc_offset_minutes
            ) % (24 * 60)
            start = datetime(
                2025, 1, 6, open_utc // 60, open_utc % 60, tzinfo=timezone.utc
            )

    rng = random.Random(seed)
    origin = start or datetime(2025, 1, 6, 14, 30, tzinfo=timezone.utc)  # 09:30 ET
    out: dict[str, tuple[Bar, ...]] = {}

    for index, symbol in enumerate(symbols):
        price = start_price * (1 + 0.1 * index)
        series: list[Bar] = []
        for session in range(sessions):
            day_open = origin + timedelta(days=session)
            # Skip weekends so the timeline resembles a real calendar.
            if day_open.weekday() >= 5:
                continue
            for slot in range(bars_per_session):
                shock = rng.gauss(drift, vol)
                open_price = price
                close_price = max(0.01, open_price * math.exp(shock))
                high = max(open_price, close_price) * (1 + abs(rng.gauss(0, vol / 2)))
                low = min(open_price, close_price) * (1 - abs(rng.gauss(0, vol / 2)))
                series.append(
                    Bar(
                        symbol=symbol,
                        t=day_open + step * slot,
                        o=round(open_price, 4),
                        h=round(high, 4),
                        l=round(max(0.01, low), 4),
                        c=round(close_price, 4),
                        v=float(rng.randint(10_000, 500_000)),
                    )
                )
                price = close_price
        out[symbol] = tuple(series)
    return out
