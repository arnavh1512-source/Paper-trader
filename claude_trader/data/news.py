"""Headlines, fetched from public RSS and handed to the model as context.

Three things about this module are deliberate and worth stating up front.

**It reads headlines, not articles.** A headline plus a timestamp is a weak
signal that is cheap and safe. Fetching and summarising full article bodies
would multiply the token cost, and would feed the model attacker-controlled
prose from arbitrary websites -- text that, in a system that then places orders,
is an instruction channel. Headlines from a named feed, escaped and clearly
delimited, are a much smaller surface.

**It is untrusted input.** Everything here is labelled as data in the prompt and
the model is told, in the system prompt, that headlines cannot change its
instructions. Nothing read from a feed reaches the risk layer: news can make the
model want to trade, and it still has to get past a gate that never reads it.

**It fails open.** A feed that is down, slow, or malformed produces no
headlines and a logged warning. It never blocks a cycle, and it never blocks an
exit. A trading system that cannot square off because a news site is having a
bad day is worse than one with no news at all.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, Mapping, Protocol, Sequence, runtime_checkable
from urllib.parse import quote_plus
from xml.etree import ElementTree

from ..errors import MarketDataError
from ..http import request_text

log = logging.getLogger(__name__)

__all__ = ["Headline", "NewsSource", "RssNewsSource", "NullNewsSource",
           "COMPANY_NAMES", "format_headlines"]

GOOGLE_NEWS = "https://news.google.com/rss/search"
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_DOCTYPE = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)

# A feed is a document from a stranger. Two limits apply before it is parsed at
# all: stdlib ElementTree resolves internal entities, which makes a document
# declaring nested entities ("billion laughs") a memory bomb, and a large body
# is a cheap way to stall a cycle. RSS never needs a DTD, so refusing one costs
# nothing and removes both entity-expansion attacks outright.
MAX_FEED_BYTES = 4_000_000

# Feeds that cover the market as a whole rather than one symbol.
MARKET_FEEDS: Mapping[str, tuple[str, ...]] = {
    "in": (
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.business-standard.com/rss/markets-106.rss",
    ),
    "us": (
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://finance.yahoo.com/news/rssindex",
    ),
}

# A ticker is a bad search term. "RELIANCE" returns Reliance Communications --
# a delisted, unrelated company -- and a headline about the wrong business is
# strictly worse than no headline at all. Searching the registered company name
# is the difference between context and noise. Symbols not listed here fall
# back to the ticker, which is correct for names that are already words.
COMPANY_NAMES: Mapping[str, str] = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "INFY": "Infosys",
    "BHARTIARTL": "Bharti Airtel",
    "ITC": "ITC Limited",
    "SBIN": "State Bank of India",
    "LT": "Larsen & Toubro",
    "AXISBANK": "Axis Bank",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "MARUTI": "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
    "SUNPHARMA": "Sun Pharmaceutical",
    "TITAN": "Titan Company",
    "ASIANPAINT": "Asian Paints",
    "BAJFINANCE": "Bajaj Finance",
    "HCLTECH": "HCL Technologies",
    "TATASTEEL": "Tata Steel",
    "JSWSTEEL": "JSW Steel",
    "NTPC": "NTPC Limited",
    "POWERGRID": "Power Grid Corporation",
    "ONGC": "Oil and Natural Gas Corporation",
    "M&M": "Mahindra & Mahindra",
    "WIPRO": "Wipro",
    "ADANIPORTS": "Adani Ports",
    "CIPLA": "Cipla",
    "TECHM": "Tech Mahindra",
    "ULTRACEMCO": "UltraTech Cement",
    "NIFTYBEES": "Nifty 50 index",
    # US names are mostly searchable as-is; the ambiguous ones are not.
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "AMZN": "Amazon.com",
    "META": "Meta Platforms",
    "NVDA": "Nvidia",
    "TSLA": "Tesla Inc",
}

# The bare company name matches quote-scraper pages and unrelated firms with
# similar names. Quoting it forces a phrase match, and Google News' own
# ``when:Nd`` operator filters at the source rather than after downloading a
# month of history and discarding most of it.


@dataclass(frozen=True, slots=True)
class Headline:
    """One item from a feed. Never anything longer than its own title."""

    symbol: str
    title: str
    source: str
    published: datetime | None = None

    def age(self, now: datetime) -> timedelta | None:
        return None if self.published is None else now - self.published

    def is_fresh(self, now: datetime, max_age: timedelta) -> bool:
        """Undated items count as fresh.

        Several Indian feeds omit pubDate entirely. Dropping them would
        silently reduce NSE coverage to whichever publishers happen to be
        strict about RSS, which is not a property anyone wants selecting their
        news.
        """
        age = self.age(now)
        return age is None or timedelta(0) <= age <= max_age


@runtime_checkable
class NewsSource(Protocol):
    def headlines(self, symbols: Sequence[str], now: datetime,
                  limit: int = 5) -> Mapping[str, tuple[Headline, ...]]: ...

    def market_headlines(self, now: datetime,
                         limit: int = 5) -> tuple[Headline, ...]: ...


class NullNewsSource:
    """The default. No requests, no headlines, no behaviour change.

    News is off unless it is switched on, so an existing run's decisions do not
    change because a new module was added underneath it.
    """

    name = "none"

    def headlines(self, symbols: Sequence[str], now: datetime,
                  limit: int = 5) -> Mapping[str, tuple[Headline, ...]]:
        return {}

    def market_headlines(self, now: datetime,
                         limit: int = 5) -> tuple[Headline, ...]:
        return ()


# ------------------------------------------------------------------- parsing
def _clean(text: str | None) -> str:
    """Strip markup and collapse whitespace.

    RSS titles routinely carry HTML entities and the odd stray tag. This is not
    a security control -- the prompt escaping and the risk gate are -- it just
    keeps the prompt readable.
    """
    if not text:
        return ""
    return _WS.sub(" ", _TAG.sub(" ", text)).strip()


def _published(item: ElementTree.Element) -> datetime | None:
    for tag in ("pubDate", "published", "updated"):
        raw = item.findtext(tag)
        if not raw:
            continue
        try:
            stamp = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            try:
                stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc)
    return None


def parse_rss(xml: str, symbol: str = "") -> tuple[Headline, ...]:
    """Parse an RSS or Atom document into headlines.

    A malformed document yields nothing rather than raising: the caller is a
    trading cycle, and a publisher's broken XML is not a reason to stop
    managing open positions.
    """
    if len(xml) > MAX_FEED_BYTES:
        log.warning("news feed body exceeded %d bytes; ignoring it",
                    MAX_FEED_BYTES)
        return ()
    if _DOCTYPE.search(xml):
        log.warning("news feed declared a DTD or entity; ignoring it")
        return ()

    try:
        root = ElementTree.fromstring(xml.strip())
    except ElementTree.ParseError:
        log.warning("news feed returned unparseable XML; ignoring it")
        return ()

    items: list[ElementTree.Element] = list(root.iter("item"))
    if not items:
        items = [e for e in root.iter() if e.tag.endswith("}entry")]

    out: list[Headline] = []
    for item in items:
        title = _clean(item.findtext("title")
                       or item.findtext("{http://www.w3.org/2005/Atom}title"))
        if not title:
            continue
        source = _clean(item.findtext("source")) or _clean(
            item.findtext("{http://purl.org/dc/elements/1.1/}creator")) or "rss"
        out.append(Headline(symbol=symbol, title=title[:300], source=source[:60],
                            published=_published(item)))
    return tuple(out)


# -------------------------------------------------------------------- source
class RssNewsSource:
    """Keyless headline fetch: Google News per symbol, plus market-wide feeds.

    Google News is used because it needs no key, no account and no quota
    negotiation, and because it aggregates the Indian publishers that matter
    for NSE names. It is a convenience, not an endorsement of any publisher.
    """

    name = "rss"

    def __init__(self, market: str = "in", *, max_age_hours: float = 24.0,
                 timeout: float = 8.0, session=None,
                 feeds: Sequence[str] | None = None) -> None:
        self.market = market
        self.max_age = timedelta(hours=max_age_hours)
        self._timeout = timeout
        self._session = session
        self._feeds = tuple(feeds) if feeds is not None else MARKET_FEEDS.get(
            market, ())
        self._cache: dict[str, tuple[Headline, ...]] = {}

    # ------------------------------------------------------------- fetching
    def _fetch(self, url: str, symbol: str = "") -> tuple[Headline, ...]:
        """One feed. Any failure is a warning and an empty result.

        Deliberately broad: this is decoration on a trading cycle, and there is
        no failure mode of a news site that justifies interrupting one.
        """
        if url in self._cache:
            return self._cache[url]
        try:
            xml = request_text("GET", url, timeout=self._timeout,
                               session=self._session,
                               headers={"User-Agent": "claude-trader/2"},
                               error_type=MarketDataError)
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning("news feed %s unavailable (%s); continuing without it",
                        url.split("?")[0], exc.__class__.__name__)
            self._cache[url] = ()
            return ()
        parsed = parse_rss(xml, symbol)
        self._cache[url] = parsed
        return parsed

    def _search_url(self, symbol: str) -> str:
        name = COMPANY_NAMES.get(symbol.upper(), symbol)
        days = max(1, round(self.max_age.total_seconds() / 86400))
        query = f'"{name}" when:{days}d'
        locale = ("hl=en-IN&gl=IN&ceid=IN:en" if self.market == "in"
                  else "hl=en-US&gl=US&ceid=US:en")
        return f"{GOOGLE_NEWS}?q={quote_plus(query)}&{locale}"

    def _fresh(self, items: Iterable[Headline], now: datetime,
               limit: int) -> tuple[Headline, ...]:
        keep = [h for h in items if h.is_fresh(now, self.max_age)]
        keep.sort(key=lambda h: h.published or now, reverse=True)
        return tuple(keep[:limit])

    # --------------------------------------------------------------- public
    def headlines(self, symbols: Sequence[str], now: datetime,
                  limit: int = 5) -> Mapping[str, tuple[Headline, ...]]:
        """Only ever called for the handful of symbols already under analysis.

        Fetching a feed per universe member would be dozens of requests per
        cycle for names the strategy has already passed over.
        """
        out: dict[str, tuple[Headline, ...]] = {}
        for symbol in symbols:
            found = self._fresh(self._fetch(self._search_url(symbol), symbol),
                                now, limit)
            if found:
                out[symbol] = found
        return out

    def market_headlines(self, now: datetime,
                         limit: int = 5) -> tuple[Headline, ...]:
        collected: list[Headline] = []
        for url in self._feeds:
            collected.extend(self._fetch(url))
        return self._fresh(collected, now, limit)

    def clear_cache(self) -> None:
        """Per-cycle memo. A cycle should see one consistent set of headlines;
        the next cycle should not see this one's."""
        self._cache.clear()


# ------------------------------------------------------------------ prompting
def format_headlines(items: Sequence[Headline], now: datetime,
                     indent: str = "  ") -> str:
    """Render headlines for a prompt, fenced and labelled as untrusted.

    The fence and the label are the point. Text from a public feed is going
    into a prompt that produces trade decisions, and it must be unambiguous to
    the model which part of the message is instruction and which is quoted
    third-party text.
    """
    if not items:
        return f"{indent}(no recent headlines)"

    lines = []
    for h in items:
        age = h.age(now)
        when = "undated"
        if age is not None:
            hours = age.total_seconds() / 3600
            when = f"{hours:.0f}h ago" if hours >= 1 else "just now"
        title = h.title.replace("\n", " ").replace("```", "'''")
        lines.append(f"{indent}- [{when}] {title} ({h.source})")
    return "\n".join(lines)
