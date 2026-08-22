"""Broker protocol shared by live Alpaca and the backtest simulator."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ..models import Account, OrderRequest, OrderResult, Position


@runtime_checkable
class Broker(Protocol):
    def account(self) -> Account: ...

    def positions(self) -> tuple[Position, ...]: ...

    def is_market_open(self, as_of: datetime) -> bool: ...

    def submit(self, order: OrderRequest) -> OrderResult | None: ...

    def close_position(self, symbol: str) -> OrderResult | None: ...


def round_qty(qty: float, decimals: int = 6) -> float:
    """Alpaca accepts fractional quantities to 9dp but rejects noise. Rounding
    down avoids ever trying to sell marginally more than is held."""
    factor = 10**decimals
    return int(max(0.0, qty) * factor) / factor
