"""Live Alpaca broker adapter.

Two things here are deliberate corrections of the original implementation:

1. Sells are always quantity-based. Alpaca rejects notional sells against a
   fractional position, which is why model-driven exits used to fail silently.
   Full exits go through DELETE /positions/{symbol} instead.
2. Order submission never retries. A retried market order is a duplicate
   position, not a recovered request.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from ..config import AppConfig
from ..errors import BrokerError, OrderRejected
from ..http import broker_request
from ..models import Account, OrderRequest, OrderResult, Position, Side
from .base import round_qty


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class AlpacaBroker:
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

    def _get(self, path: str) -> Any:
        return broker_request(
            "GET",
            f"{self._config.alpaca_base}{path}",
            headers=self._headers,
            timeout=15.0,
            session=self._session,
        )

    # ---------------------------------------------------------------- reads
    def account(self) -> Account:
        payload = self._get("/account")
        if not isinstance(payload, Mapping):
            raise BrokerError("account endpoint returned an unexpected shape")
        return Account(
            equity=_to_float(payload.get("equity")),
            cash=_to_float(payload.get("cash")),
            buying_power=_to_float(payload.get("buying_power")),
            last_equity=_to_float(payload.get("last_equity")),
        )

    def positions(self) -> tuple[Position, ...]:
        payload = self._get("/positions")
        if not isinstance(payload, list):
            return ()
        out: list[Position] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            qty = _to_float(item.get("qty"))
            if qty <= 0:
                continue  # long-only bot: ignore any short leg
            out.append(
                Position(
                    symbol=str(item.get("symbol", "")).upper(),
                    qty=qty,
                    avg_entry_price=_to_float(item.get("avg_entry_price")),
                    current_price=_to_float(item.get("current_price")),
                )
            )
        return tuple(out)

    def is_market_open(self, as_of: datetime) -> bool:
        payload = self._get("/clock")
        return bool(payload.get("is_open", False)) if isinstance(payload, Mapping) else False

    # --------------------------------------------------------------- writes
    def submit(self, order: OrderRequest) -> OrderResult | None:
        body: dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type,
            "time_in_force": order.time_in_force,
        }
        if order.qty is not None:
            body["qty"] = str(round_qty(order.qty))
        else:
            body["notional"] = str(round(float(order.notional or 0.0), 2))
        if order.client_order_id:
            body["client_order_id"] = order.client_order_id

        payload = broker_request(
            "POST",
            f"{self._config.alpaca_base}/orders",
            headers=self._headers,
            json_body=body,
            timeout=20.0,
            max_attempts=1,  # never retry a market order
            session=self._session,
        )
        if not isinstance(payload, Mapping):
            raise OrderRejected(f"order for {order.symbol} returned no payload")
        return self._result_from_payload(
            order.symbol, order.side, payload, requested_qty=order.qty or 0.0
        )

    def close_position(self, symbol: str) -> OrderResult | None:
        """Liquidate a whole position. The only exit that works cleanly against
        a fractional holding."""
        payload = broker_request(
            "DELETE",
            f"{self._config.alpaca_base}/positions/{symbol}",
            headers=self._headers,
            timeout=20.0,
            max_attempts=1,
            session=self._session,
        )
        if not isinstance(payload, Mapping):
            return None
        # Deliberately not an OrderRequest: that model insists on exactly one
        # of qty/notional, and a liquidation payload with neither would raise
        # inside the exit path -- leaving open the position we came to close.
        return self._result_from_payload(symbol, Side.SELL, payload)

    def _result_from_payload(
        self,
        symbol: str,
        side: Side,
        payload: Mapping[str, Any],
        requested_qty: float = 0.0,
    ) -> OrderResult:
        filled_qty = _to_float(payload.get("filled_qty"))
        qty = filled_qty or _to_float(payload.get("qty")) or requested_qty
        price = _to_float(payload.get("filled_avg_price"))
        submitted = payload.get("submitted_at") or payload.get("created_at")
        try:
            ts = (
                datetime.fromisoformat(str(submitted).replace("Z", "+00:00"))
                if submitted
                else datetime.now(timezone.utc)
            )
        except ValueError:
            ts = datetime.now(timezone.utc)
        return OrderResult(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_id=str(payload.get("id", "")),
            status=str(payload.get("status", "")),
            submitted_at=ts,
        )
