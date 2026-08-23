"""Market profiles.

A profile is everything that differs between exchanges: currency, session
hours, calendar, benchmark, whether fractional shares exist, and the tradable
universe. Isolating it here is what lets one engine trade NSE and NYSE without
a single ``if market == "in"`` anywhere in the decision path.

The two things that bite hardest when moving a US bot to India:

* NSE has **no fractional shares**. Notional orders are a US convenience; here
  size must round down to whole shares, and a name whose share price exceeds
  the per-trade cap is simply untradable at that cap.
* Costs are not zero. STT, stamp duty, exchange fees, SEBI turnover fees, GST
  and DP charges are material on small tickets -- see ``claude_trader.costs``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from datetime import timedelta, timezone
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_TZ_CACHE: dict[str, object] = {}


def _zone(name: str, fallback_minutes: int) -> object:
    """Resolve an IANA zone, falling back to a fixed offset.

    ``tzdata`` is a hard dependency on Windows, where Python ships no zone
    database. Falling back keeps a missing wheel from taking the bot down
    mid-session; the fallback is exact for IST (which has never observed DST)
    and approximate anywhere that does, so the miss is logged loudly.
    """
    cached = _TZ_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        zone: object = ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError, ImportError):
        logging.getLogger(__name__).warning(
            "tzdata missing: %s falls back to a fixed UTC%+.1fh offset. "
            "Install tzdata (pip install tzdata) for a correct calendar.",
            name,
            fallback_minutes / 60,
        )
        zone = timezone(timedelta(minutes=fallback_minutes), name)
    _TZ_CACHE[name] = zone
    return zone


@dataclass(frozen=True, slots=True)
class MarketProfile:
    key: str
    name: str
    currency: str
    currency_symbol: str
    tz_name: str
    utc_offset_minutes: int
    open_time: time
    close_time: time
    benchmark: str
    universe: tuple[str, ...]
    sectors: Mapping[str, str]
    fractional_shares: bool
    lot_size: int
    data_suffix: str
    holidays: frozenset[date]
    starting_cash: float
    max_per_trade: float
    tick_size: float
    timeframe: str
    bars_per_session: int

    # Other exchanges the same company may be listed on, as the data vendor
    # spells them. India is the reason this exists: a name absent from NSE is
    # often present on BSE under the identical ticker with a ``.BO`` suffix,
    # and a dead feed for one listing should not remove a company from the
    # universe. Tried in order, only when the primary listing returns nothing.
    data_suffix_fallbacks: tuple[str, ...] = ()

    # ------------------------------------------------------------------ time
    @property
    def tz(self):
        return _zone(self.tz_name, self.utc_offset_minutes)

    def local(self, moment: datetime) -> datetime:
        return moment.astimezone(self.tz)

    def is_session_time(self, moment: datetime) -> bool:
        """Calendar check only. Whether the exchange is *actually* trading is
        confirmed against live data by the broker, because a hardcoded holiday
        list goes stale every January."""
        here = self.local(moment)
        if here.weekday() >= 5 or here.date() in self.holidays:
            return False
        return self.open_time <= here.time() <= self.close_time

    def session_progress(self, moment: datetime) -> float:
        """0.0 at the open, 1.0 at the close. Late-session entries have less
        time to work, which the prompt is told about."""
        here = self.local(moment).time()
        start = self.open_time.hour * 60 + self.open_time.minute
        end = self.close_time.hour * 60 + self.close_time.minute
        cur = here.hour * 60 + here.minute
        span = max(1, end - start)
        return min(1.0, max(0.0, (cur - start) / span))

    # --------------------------------------------------------------- symbols
    def data_symbol(self, symbol: str) -> str:
        """Exchange ticker as the *data vendor* spells it (Yahoo wants .NS)."""
        return f"{symbol}{self.data_suffix}" if self.data_suffix else symbol

    def data_symbols(self, symbol: str) -> tuple[str, ...]:
        """Every spelling worth trying, primary listing first.

        The order is the priority order and it is not arbitrary: the primary
        exchange is where the liquidity is, so a fallback listing is a last
        resort for a name that would otherwise have no data at all, never a
        cheaper alternative to be preferred.
        """
        out = [self.data_symbol(symbol)]
        out.extend(
            f"{symbol}{suffix}"
            for suffix in self.data_suffix_fallbacks
            if f"{symbol}{suffix}" not in out
        )
        return tuple(out)

    def native_symbol(self, vendor_symbol: str) -> str:
        for suffix in (self.data_suffix, *self.data_suffix_fallbacks):
            if suffix and vendor_symbol.endswith(suffix):
                return vendor_symbol[: -len(suffix)]
        return vendor_symbol

    def sector_of(self, symbol: str) -> str:
        """Unknown tickers get a private bucket so an unmapped name is never
        silently treated as diversifying."""
        key = symbol.upper()
        return self.sectors.get(key, f"unknown:{key}")

    # ------------------------------------------------------------- quantities
    def round_qty(self, qty: float) -> float:
        """Round a desired size down to something the exchange will accept."""
        if qty <= 0:
            return 0.0
        if not self.fractional_shares:
            lot = max(1, self.lot_size)
            return float((int(qty) // lot) * lot)
        return int(max(0.0, qty) * 1_000_000) / 1_000_000

    def round_price(self, price: float) -> float:
        if self.tick_size <= 0:
            return round(price, 4)
        return round(round(price / self.tick_size) * self.tick_size, 4)

    # ----------------------------------------------------------------- money
    def money(self, value: float) -> str:
        return format_money(value, self)


def _indian_grouping(whole: str) -> str:
    """1234567 -> 12,34,567. The last three digits group normally, everything
    above them groups in twos. Rendering lakhs as '1,234,567' reads as wrong to
    an Indian user, and this bot reports numbers they have to trust."""
    if len(whole) <= 3:
        return whole
    head, tail = whole[:-3], whole[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def format_money(value: float, profile: "MarketProfile | None" = None) -> str:
    symbol = profile.currency_symbol if profile else "$"
    indian = bool(profile and profile.currency == "INR")
    sign = "-" if value < 0 else ""
    whole, _, frac = f"{abs(value):.2f}".partition(".")
    grouped = _indian_grouping(whole) if indian else f"{int(whole):,}"
    return f"{sign}{symbol}{grouped}.{frac}"


from .india import INDIA_MARKET  # noqa: E402  (registry needs the profiles)
from .us import US_MARKET  # noqa: E402

MARKETS: Mapping[str, MarketProfile] = {
    US_MARKET.key: US_MARKET,
    INDIA_MARKET.key: INDIA_MARKET,
}

DEFAULT_MARKET = "in"


def get_market(key: str = DEFAULT_MARKET) -> MarketProfile:
    """Profiles are looked up by key everywhere rather than passed around as
    booleans, so adding a third market never means editing a conditional."""
    try:
        return MARKETS[key.strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown market {key!r}; available: {', '.join(sorted(MARKETS))}"
        ) from None


__all__ = [
    "MarketProfile",
    "MARKETS",
    "DEFAULT_MARKET",
    "US_MARKET",
    "INDIA_MARKET",
    "get_market",
    "format_money",
]
