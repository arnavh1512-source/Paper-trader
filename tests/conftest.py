"""Shared fixtures.

Two things every test in this suite depends on:

* **A clean environment.** ``AppConfig.from_env`` reads two dozen variables. A
  developer with ``MAX_POSITIONS=1`` exported would otherwise see failures that
  do not exist, and CI would see passes that are not real.
* **An in-memory journal.** The journal is the bot's entire state, so most
  behaviour can only be tested through one. SQLite's ``:memory:`` makes that
  free and leaves nothing behind.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import pytest

from claude_trader.journal.store import Journal
from claude_trader.markets import INDIA_MARKET, US_MARKET
from claude_trader.models import (
    Account,
    Bar,
    Indicators,
    MarketSnapshot,
    PortfolioState,
    Position,
    PositionRisk,
    Quote,
)

# Every environment variable the package reads. Kept explicit rather than
# wildcarded so adding a new one without adding it here shows up as a flaky
# test rather than as silence.
ENV_VARS = (
    "MARKET",
    "TRADE_SEGMENT",
    "BROKER",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE",
    "ALPACA_DATA_BASE",
    "ALPACA_FEED",
    "ANTHROPIC_API_KEY",
    "CLAUDE_MODEL",
    "CLAUDE_MAX_TOKENS",
    "CLAUDE_TEMPERATURE",
    "CLAUDE_TIMEOUT",
    "CLAUDE_MAX_RETRIES",
    "LLM_CACHE_ENABLED",
    "MAX_API_CALLS",
    "BAR_TIMEFRAME",
    "BAR_LOOKBACK",
    "UNIVERSE",
    "STARTING_CASH",
    "DRY_RUN",
    "VERBOSE",
    "JOURNAL_PATH",
    "STRATEGY",
    "MAX_NOTIONAL_PER_TRADE",
    "MAX_POSITION_PCT",
    "RISK_PER_TRADE_PCT",
    "MIN_TRADE_NOTIONAL",
    "MIN_CASH_RESERVE_PCT",
    "ATR_STOP_MULTIPLE",
    "ATR_TARGET_MULTIPLE",
    "TRAILING_STOP_ATR",
    "MAX_HOLDING_BARS",
    "HARD_STOP_PCT",
    "SQUARE_OFF_ENABLED",
    "SQUARE_OFF_MINUTES",
    "MAX_POSITIONS",
    "MAX_SECTOR_POSITIONS",
    "MAX_SECTOR_PCT",
    "MAX_CORRELATION",
    "DAILY_LOSS_LIMIT_PCT",
    "MAX_DRAWDOWN_PCT",
    "MAX_TRADES_PER_CYCLE",
    "MAX_TRADES_PER_DAY",
    "MIN_CONFIDENCE",
    "MAX_QUOTE_AGE_SECONDS",
    "MAX_SPREAD_BPS",
    "MAX_COST_RATIO",
    "NEWS_ENABLED",
    "NEWS_MAX_HEADLINES",
    "NEWS_MAX_AGE_HOURS",
)

IST = timezone(timedelta(minutes=330))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No test inherits the developer's shell."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def journal():
    with Journal(":memory:") as j:
        yield j


@pytest.fixture
def india():
    return INDIA_MARKET


@pytest.fixture
def us():
    return US_MARKET


# --------------------------------------------------------------- builders
def make_bars(
    symbol: str,
    closes: Sequence[float],
    start: datetime | None = None,
    step: timedelta = timedelta(minutes=15),
    volume: float = 10_000.0,
) -> tuple[Bar, ...]:
    """Bars from a list of closes. High/low straddle the close by 0.5% so ATR
    and true range are non-zero without having to spell out four prices."""
    start = start or datetime(2026, 3, 2, 4, 0, tzinfo=timezone.utc)
    bars = []
    previous = closes[0]
    for i, close in enumerate(closes):
        bars.append(
            Bar(
                symbol=symbol,
                t=start + i * step,
                o=previous,
                h=max(previous, close) * 1.005,
                l=min(previous, close) * 0.995,
                c=close,
                v=volume,
            )
        )
        previous = close
    return tuple(bars)


def ramp(n: int, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start + i * step for i in range(n)]


def make_quote(
    symbol: str,
    price: float,
    when: datetime | None = None,
    spread_bps: float = 4.0,
    modelled: bool = False,
) -> Quote:
    when = when or datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)
    half = price * spread_bps / 2 / 10_000.0
    return Quote(
        symbol=symbol,
        t=when,
        bid=price - half,
        ask=price + half,
        bid_size=100.0,
        ask_size=100.0,
        modelled=modelled,
    )


def make_snapshot(
    symbol: str = "RELIANCE",
    price: float = 1_000.0,
    atr: float | None = 10.0,
    when: datetime | None = None,
    bars: Iterable[Bar] = (),
    quote: Quote | None = None,
    position: Position | None = None,
    risk: PositionRisk | None = None,
    trend: str = "up",
) -> MarketSnapshot:
    when = when or datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)
    return MarketSnapshot(
        symbol=symbol,
        as_of=when,
        bars=tuple(bars),
        quote=quote if quote is not None else make_quote(symbol, price, when),
        indicators=Indicators(
            last_price=price,
            sma_fast=price,
            sma_slow=price * 0.98,
            rsi=55.0,
            atr=atr,
            atr_pct=(atr / price * 100) if (atr and price) else None,
            trend=trend,
        ),
        position=position,
        risk=risk,
    )


def make_state(
    cash: float = 100_000.0,
    positions: Iterable[Position] = (),
    risks: Iterable[PositionRisk] = (),
    last_equity: float = 0.0,
) -> PortfolioState:
    positions = tuple(positions)
    equity = cash + sum(p.market_value for p in positions)
    return PortfolioState.build(
        Account(
            equity=equity,
            cash=cash,
            buying_power=cash,
            last_equity=last_equity or equity,
        ),
        positions,
        risks,
    )
