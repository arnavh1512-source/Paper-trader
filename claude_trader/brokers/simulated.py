"""Simulated broker for backtesting.

Fills happen at the OPEN of the bar after the decision bar, plus slippage. That
one-bar delay is what stops the backtest from quietly trading on the closing
price it used to make the decision -- the most common way a backtest lies.

The second way a backtest lies is by ignoring what trading costs. On NSE a
Rs 5,000 delivery round trip surrenders roughly 0.6% to statutory charges before
the strategy has an opinion, so charges come from a ``CostModel`` and are
deducted from cash on every fill rather than modelled as a flat commission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from ..costs import Charges, CostModel, NoCosts
from ..data.sources import HistoricalMarketData
from ..markets import MarketProfile
from ..models import Account, OrderRequest, OrderResult, Position, Side
from .base import round_qty


@dataclass(frozen=True, slots=True)
class FillModel:
    """The gap between the price you saw and the price you got.

    Slippage only. Statutory and brokerage charges live in the ``CostModel``,
    because they differ by market and by segment while slippage is a property of
    the order itself.
    """

    slippage_bps: float = 5.0

    def fill_price(self, side: Side, reference: float) -> float:
        drift = reference * (self.slippage_bps / 10_000.0)
        return reference + drift if side is Side.BUY else reference - drift


class SimulatedBroker:
    """In-memory portfolio driven by historical bars."""

    def __init__(
        self,
        market: HistoricalMarketData,
        starting_cash: float = 10_000.0,
        fill_model: FillModel | None = None,
        costs: CostModel | None = None,
        profile: MarketProfile | None = None,
    ) -> None:
        self._market = market
        self._cash = float(starting_cash)
        self._starting_cash = float(starting_cash)
        self._holdings: dict[str, tuple[float, float]] = {}  # symbol -> (qty, avg)
        self._fills = fill_model or FillModel()
        self._costs: CostModel = costs or NoCosts()
        self._profile = profile
        self._now: datetime | None = None
        self._day_start_equity = float(starting_cash)
        self._current_day: str = ""
        self.orders: list[OrderResult] = []
        self.rejections: list[tuple[str, str]] = []
        self.charges_paid: float = 0.0
        self.charge_breakdown: dict[str, float] = {}

    # ----------------------------------------------------------------- clock
    def set_clock(self, now: datetime) -> None:
        self._now = now
        day = now.date().isoformat()
        if day != self._current_day:
            self._current_day = day
            self._day_start_equity = self.equity()

    @property
    def now(self) -> datetime:
        if self._now is None:
            raise RuntimeError("SimulatedBroker.set_clock must be called first")
        return self._now

    # ----------------------------------------------------------------- state
    def price_of(self, symbol: str) -> float:
        bars = self._market.bars(symbol, 1, self.now)
        return bars[-1].c if bars else 0.0

    def equity(self) -> float:
        held = sum(qty * self.price_of(sym) for sym, (qty, _) in self._holdings.items())
        return self._cash + held

    def account(self) -> Account:
        equity = self.equity()
        return Account(
            equity=equity,
            cash=self._cash,
            buying_power=max(0.0, self._cash),
            last_equity=self._day_start_equity or equity,
        )

    def positions(self) -> tuple[Position, ...]:
        out = []
        for symbol, (qty, avg) in sorted(self._holdings.items()):
            if qty <= 1e-9:
                continue
            out.append(
                Position(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=avg,
                    current_price=self.price_of(symbol) or avg,
                )
            )
        return tuple(out)

    def is_market_open(self, as_of: datetime) -> bool:
        # Bars only exist for open sessions, so their presence is the calendar.
        return any(self._market.bars(s, 1, as_of) for s in self._market.symbols[:1])

    # ---------------------------------------------------------------- orders
    def _round(self, qty: float) -> float:
        return self._profile.round_qty(qty) if self._profile else round_qty(qty)

    def _record(self, charges: Charges) -> None:
        self.charges_paid += charges.total
        for key, value in charges.as_dict().items():
            if key == "total" or not value:
                continue
            self.charge_breakdown[key] = round(
                self.charge_breakdown.get(key, 0.0) + value, 2
            )

    def _reference_price(self, symbol: str) -> float:
        """Open of the next bar: the first price actually reachable after a
        decision made on the current bar."""
        nxt = self._market.next_bar_after(symbol, self.now)
        if nxt is not None and nxt.o > 0:
            return nxt.o
        return self.price_of(symbol)

    def submit(self, order: OrderRequest) -> OrderResult | None:
        reference = self._reference_price(order.symbol)
        if reference <= 0:
            self.rejections.append((order.symbol, "no reference price"))
            return None

        price = self._fills.fill_price(order.side, reference)
        if self._profile:
            price = self._profile.round_price(price)
        if price <= 0:
            self.rejections.append((order.symbol, "non-positive fill price"))
            return None

        qty = self._round(
            order.qty if order.qty is not None else (order.notional or 0.0) / price
        )
        if qty <= 0:
            # On a whole-share market this is the ordinary outcome of a budget
            # smaller than one share, not a bug -- but it must be visible.
            self.rejections.append((order.symbol, "quantity rounds to zero"))
            return None

        if order.side is Side.BUY:
            charges = self._costs.charges(Side.BUY, qty, price)
            cost = qty * price + charges.total
            if cost > self._cash + 1e-9:
                # Shrink to what the cash can actually carry rather than
                # rejecting outright; a partial entry is a real broker outcome.
                affordable = self._round(self._cash * 0.995 / price)
                if affordable <= 0:
                    self.rejections.append((order.symbol, "insufficient cash"))
                    return None
                qty = affordable
                charges = self._costs.charges(Side.BUY, qty, price)
                cost = qty * price + charges.total
                if cost > self._cash + 1e-9:
                    self.rejections.append((order.symbol, "insufficient cash"))
                    return None
            held_qty, held_avg = self._holdings.get(order.symbol, (0.0, 0.0))
            new_qty = held_qty + qty
            new_avg = ((held_qty * held_avg) + (qty * price)) / new_qty
            self._holdings[order.symbol] = (new_qty, new_avg)
            self._cash -= cost
        else:
            held_qty, held_avg = self._holdings.get(order.symbol, (0.0, 0.0))
            if held_qty <= 0:
                self.rejections.append((order.symbol, "no position to sell"))
                return None
            qty = min(qty, held_qty)
            charges = self._costs.charges(Side.SELL, qty, price)
            proceeds = qty * price - charges.total
            remaining = round(held_qty - qty, 9)
            if remaining <= 1e-9:
                self._holdings.pop(order.symbol, None)
            else:
                self._holdings[order.symbol] = (remaining, held_avg)
            self._cash += proceeds

        self._record(charges)
        result = OrderResult(
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            order_id=f"sim-{len(self.orders) + 1}",
            status="filled",
            submitted_at=self.now,
            simulated=True,
        )
        self.orders.append(result)
        return result

    def close_position(self, symbol: str) -> OrderResult | None:
        held_qty, _ = self._holdings.get(symbol, (0.0, 0.0))
        if held_qty <= 0:
            return None
        return self.submit(
            OrderRequest(symbol=symbol, side=Side.SELL, qty=held_qty, intent="close")
        )

    # ------------------------------------------------------------- reporting
    @property
    def starting_cash(self) -> float:
        return self._starting_cash

    @property
    def cost_model_name(self) -> str:
        return self._costs.name

    def holdings(self) -> Mapping[str, tuple[float, float]]:
        return dict(self._holdings)
