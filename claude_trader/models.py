"""Immutable domain types shared by the live engine and the backtester.

Every type here is frozen. State transitions return new objects, which is what
lets a single cycle keep an accurate running view of cash and positions after
each fill instead of trading against a stale snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    TIME_STOP = "time_stop"
    SQUARE_OFF = "square_off"
    MODEL_EXIT = "model_exit"
    RISK_HALT = "risk_halt"


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    t: datetime
    o: float
    h: float
    l: float
    c: float
    v: float

    @property
    def typical_price(self) -> float:
        return (self.h + self.l + self.c) / 3.0

    @property
    def true_range_vs(self) -> float:
        return self.h - self.l


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    t: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    modelled: bool = False
    """True when bid/ask were derived from a trade price rather than observed.

    Free NSE data carries no order book, so the spread gate would otherwise be
    scoring a number nobody quoted. The flag lets the risk layer widen its
    tolerance and lets the journal record which trades were sized on a guess.
    """

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.ask or self.bid

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        if mid <= 0 or self.bid <= 0 or self.ask <= 0:
            return float("inf")
        return (self.ask - self.bid) / mid * 10_000.0

    def age_seconds(self, now: datetime) -> float:
        return max(0.0, (now - self.t).total_seconds())

    @property
    def is_tradable(self) -> bool:
        return self.mid > 0


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_entry_price

    @property
    def unrealized_pl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_plpc(self) -> float:
        basis = self.cost_basis
        return 0.0 if basis == 0 else self.unrealized_pl / basis

    def repriced(self, price: float) -> "Position":
        return replace(self, current_price=price)


@dataclass(frozen=True, slots=True)
class PositionRisk:
    """Bot-managed exit levels for an open position.

    Alpaca does not support bracket orders on fractional/notional fills, so the
    stop lives here and is enforced by the engine at the top of every cycle.
    """

    symbol: str
    entry_price: float
    entry_time: datetime
    stop_price: float
    target_price: float
    high_water: float
    atr_at_entry: float
    bars_held: int = 0

    def advanced(self, price: float, trail_atr: float) -> "PositionRisk":
        """Return a new record with the high-water mark and trailing stop moved
        up. Stops only ever ratchet upward; they never loosen."""
        high = max(self.high_water, price)
        trailed = high - trail_atr * self.atr_at_entry if self.atr_at_entry > 0 else self.stop_price
        return replace(
            self,
            high_water=high,
            stop_price=max(self.stop_price, trailed),
            bars_held=self.bars_held + 1,
        )


@dataclass(frozen=True, slots=True)
class Account:
    equity: float
    cash: float
    buying_power: float
    last_equity: float = 0.0

    @property
    def day_pl_pct(self) -> float:
        if self.last_equity <= 0:
            return 0.0
        return (self.equity - self.last_equity) / self.last_equity


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """The engine's running view of the book within a single cycle."""

    account: Account
    positions: Mapping[str, Position]
    risks: Mapping[str, PositionRisk]

    @classmethod
    def build(
        cls,
        account: Account,
        positions: Iterable[Position],
        risks: Iterable[PositionRisk] = (),
    ) -> "PortfolioState":
        return cls(
            account=account,
            positions={p.symbol: p for p in positions},
            risks={r.symbol: r for r in risks},
        )

    @property
    def open_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.positions))

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def gross_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def exposure_ratio(self) -> float:
        """Invested fraction of equity, which is what gets journalled.

        The absolute figure is currency-dependent and therefore meaningless in
        a report that compares two markets; the ratio is not.
        """
        equity = self.account.equity
        return (self.gross_exposure / equity) if equity > 0 else 0.0

    def exposure_pct(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        if pos is None or self.account.equity <= 0:
            return 0.0
        return pos.market_value / self.account.equity

    def with_fill(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        risk: PositionRisk | None = None,
    ) -> "PortfolioState":
        """Apply a fill and return a NEW state. This is what keeps cash and the
        position count honest across multiple trades in one cycle."""
        if qty <= 0 or price <= 0:
            return self

        positions = dict(self.positions)
        risks = dict(self.risks)
        existing = positions.get(symbol)
        notional = qty * price

        if side is Side.BUY:
            if existing is None:
                positions[symbol] = Position(symbol, qty, price, price)
            else:
                total_qty = existing.qty + qty
                avg = (existing.cost_basis + notional) / total_qty
                positions[symbol] = Position(symbol, total_qty, avg, price)
            if risk is not None:
                risks[symbol] = risk
            cash = self.account.cash - notional
        else:
            if existing is None:
                return self
            remaining = round(existing.qty - qty, 9)
            if remaining <= 1e-9:
                positions.pop(symbol, None)
                risks.pop(symbol, None)
            else:
                positions[symbol] = replace(
                    existing, qty=remaining, current_price=price
                )
            cash = self.account.cash + notional

        equity = cash + sum(p.market_value for p in positions.values())
        account = replace(
            self.account,
            cash=cash,
            equity=equity,
            buying_power=max(0.0, cash),
        )
        return PortfolioState(account=account, positions=positions, risks=risks)


@dataclass(frozen=True, slots=True)
class Indicators:
    """Derived features handed to the strategy. Computed once, reused by the
    prompt builder, the risk layer and the journal."""

    last_price: float
    sma_fast: float | None = None
    sma_slow: float | None = None
    ema_fast: float | None = None
    rsi: float | None = None
    atr: float | None = None
    atr_pct: float | None = None
    realized_vol: float | None = None
    ret_1: float | None = None
    ret_5: float | None = None
    ret_20: float | None = None
    volume_ratio: float | None = None
    dist_from_high_pct: float | None = None
    trend: str = "unknown"


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Everything known about one symbol at one instant."""

    symbol: str
    as_of: datetime
    bars: tuple[Bar, ...]
    quote: Quote | None
    indicators: Indicators
    position: Position | None = None
    risk: PositionRisk | None = None

    @property
    def price(self) -> float:
        if self.quote is not None and self.quote.is_tradable:
            return self.quote.mid
        return self.indicators.last_price


@dataclass(frozen=True, slots=True)
class Picks:
    symbols: tuple[str, ...]
    strategy: str
    market_mood: str = "neutral"
    abstain: bool = False
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class Decision:
    symbol: str
    action: Action
    confidence: int
    reason: str
    notional: float | None = None
    horizon_bars: int | None = None
    source: str = "model"

    @property
    def is_trade(self) -> bool:
        return self.action in (Action.BUY, Action.SELL)


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    approved: bool
    reason: str
    notional: float = 0.0
    qty: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0


@dataclass(frozen=True, slots=True)
class OrderRequest:
    symbol: str
    side: Side
    qty: float | None = None
    notional: float | None = None
    order_type: str = "market"
    time_in_force: str = "day"
    client_order_id: str | None = None
    intent: str = ""

    def __post_init__(self) -> None:
        if (self.qty is None) == (self.notional is None):
            raise ValueError("OrderRequest needs exactly one of qty or notional")


@dataclass(frozen=True, slots=True)
class OrderResult:
    symbol: str
    side: Side
    qty: float
    price: float
    order_id: str
    status: str
    submitted_at: datetime
    simulated: bool = False

    @property
    def notional(self) -> float:
        return self.qty * self.price


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
