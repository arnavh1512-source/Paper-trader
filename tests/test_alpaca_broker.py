"""The live Alpaca adapter.

The rule this file exists to enforce: a market order is never retried. Every
other request in the system is safe to repeat; this one buys a second position
if you do. The original bot retried everything.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from claude_trader.brokers.alpaca import AlpacaBroker
from claude_trader.brokers.base import Broker
from claude_trader.config import AppConfig
from claude_trader.errors import BrokerError, OrderRejected
from claude_trader.models import OrderRequest, Side

NOW = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.text = "error detail"
        self.content = b"{}"

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *items):
        self.items = list(items)
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self.items.pop(0) if self.items else {}
        if isinstance(item, Exception):
            raise item
        if isinstance(item, FakeResponse):
            return item
        return FakeResponse(item)


def broker(*payloads) -> AlpacaBroker:
    config = AppConfig(market="us", alpaca_key="key", alpaca_secret="secret")
    return AlpacaBroker(config, session=FakeSession(*payloads))


def buy(**kwargs) -> OrderRequest:
    return OrderRequest(symbol="AAPL", side=Side.BUY, intent="entry", **kwargs)


def filled(**overrides) -> dict:
    return {"id": "ord-1", "status": "filled", "filled_qty": "2", "qty": "2",
            "filled_avg_price": "190.25", "submitted_at": "2026-03-02T15:00:00Z",
            **overrides}


# -------------------------------------------------------------------- reads
def test_the_account_is_parsed():
    b = broker({"equity": "10500.5", "cash": "500", "buying_power": "1000",
                "last_equity": "10000"})
    account = b.account()
    assert (account.equity, account.cash) == (10500.5, 500.0)
    assert account.last_equity == 10000.0


def test_missing_account_fields_read_as_zero_rather_than_crashing():
    """A cycle that dies on a missing field cannot even take its stops."""
    assert broker({}).account().equity == 0.0


def test_an_unexpected_account_shape_is_a_typed_error():
    with pytest.raises(BrokerError, match="unexpected shape"):
        broker([1, 2, 3]).account()


def test_positions_are_parsed_and_upper_cased():
    b = broker([{"symbol": "aapl", "qty": "3", "avg_entry_price": "180",
                 "current_price": "190"}])
    position = b.positions()[0]
    assert position.symbol == "AAPL"
    assert (position.qty, position.avg_entry_price) == (3.0, 180.0)


def test_a_short_leg_is_ignored_by_a_long_only_bot():
    """Treating a negative quantity as a holding would make the risk layer
    compute a stop on a position it cannot manage."""
    b = broker([{"symbol": "AAPL", "qty": "-3"}, {"symbol": "MSFT", "qty": "1"}])
    assert [p.symbol for p in b.positions()] == ["MSFT"]


def test_a_malformed_position_row_is_skipped():
    b = broker([{"symbol": "AAPL", "qty": "1"}, "not a position"])
    assert len(b.positions()) == 1


def test_an_unexpected_positions_shape_is_an_empty_book():
    """Reporting no positions is safe; the risk layer simply finds nothing to
    exit. Raising would abort the cycle."""
    assert broker({"error": "nope"}).positions() == ()


def test_the_clock_endpoint_is_the_calendar():
    assert broker({"is_open": True}).is_market_open(NOW) is True
    assert broker({"is_open": False}).is_market_open(NOW) is False


def test_an_unreadable_clock_is_treated_as_closed():
    """Trading blind on the calendar is the worse of the two errors."""
    assert broker("garbage").is_market_open(NOW) is False


def test_reads_carry_the_credentials():
    b = broker({})
    b.account()
    headers = b._session.calls[0]["headers"]
    assert headers["APCA-API-KEY-ID"] == "key"
    assert headers["APCA-API-SECRET-KEY"] == "secret"


# ------------------------------------------------------------------- orders
def test_a_notional_buy_is_sent_as_notional():
    b = broker(filled())
    b.submit(buy(notional=100.0))
    body = b._session.calls[0]["json"]
    assert body["notional"] == "100.0"
    assert "qty" not in body
    assert body["side"] == "buy"
    assert body["type"] == "market"


def test_a_quantity_order_is_sent_as_quantity():
    b = broker(filled())
    b.submit(OrderRequest(symbol="AAPL", side=Side.SELL, qty=1.5, intent="exit"))
    body = b._session.calls[0]["json"]
    assert body["qty"] == "1.5"
    assert "notional" not in body


def test_a_quantity_is_rounded_down_before_it_is_sent():
    """Rounding up would ask Alpaca to sell marginally more than is held, which
    it rejects -- and the exit would silently fail."""
    b = broker(filled())
    b.submit(OrderRequest(symbol="AAPL", side=Side.SELL, qty=1.9999999999,
                          intent="exit"))
    assert float(b._session.calls[0]["json"]["qty"]) < 2.0


def test_a_client_order_id_is_forwarded_for_idempotency():
    b = broker(filled())
    b.submit(buy(notional=100.0, client_order_id="cycle-42-AAPL"))
    assert b._session.calls[0]["json"]["client_order_id"] == "cycle-42-AAPL"


def test_a_market_order_is_never_retried():
    """A retried market order is a duplicate position, not a recovered
    request. This is the single most expensive bug this adapter can have."""
    b = broker(FakeResponse({}, status_code=503), filled())
    with pytest.raises(BrokerError):
        b.submit(buy(notional=100.0))
    assert len(b._session.calls) == 1


def test_a_rejected_order_raises_rather_than_returning_none():
    """Returning None would let the journal record a placed order that never
    existed."""
    with pytest.raises(OrderRejected, match="no payload"):
        broker("not an order").submit(buy(notional=100.0))


def test_the_fill_is_read_back_from_the_broker_not_assumed():
    result = broker(filled(filled_qty="1.5", filled_avg_price="191.10")).submit(
        buy(notional=300.0))
    assert (result.qty, result.price) == (1.5, 191.10)
    assert result.order_id == "ord-1"
    assert result.status == "filled"
    assert result.simulated is False


def test_an_unfilled_order_falls_back_to_the_requested_quantity():
    """A resting order has no filled quantity yet; reporting zero would make the
    journal think nothing happened."""
    result = broker(filled(filled_qty="0", qty="0", status="accepted")).submit(
        OrderRequest(symbol="AAPL", side=Side.BUY, qty=4.0, intent="entry"))
    assert result.qty == 4.0
    assert result.status == "accepted"


def test_the_submitted_timestamp_is_parsed():
    result = broker(filled()).submit(buy(notional=100.0))
    assert result.submitted_at == datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)


def test_created_at_is_used_when_submitted_at_is_absent():
    payload = filled()
    payload.pop("submitted_at")
    payload["created_at"] = "2026-03-02T14:00:00Z"
    result = broker(payload).submit(buy(notional=100.0))
    assert result.submitted_at.hour == 14


@pytest.mark.parametrize("stamp", ["", "not a date", None])
def test_an_unparseable_timestamp_falls_back_to_now(stamp):
    payload = filled(submitted_at=stamp)
    payload.pop("created_at", None)
    result = broker(payload).submit(buy(notional=100.0))
    assert result.submitted_at.tzinfo is not None


# ------------------------------------------------------------------- exits
def test_a_full_exit_goes_through_delete_positions():
    """Alpaca rejects a notional sell against a fractional position, which is
    how model-driven exits used to fail silently."""
    b = broker(filled(qty="1.5", filled_qty="1.5"))
    result = b.close_position("AAPL")
    call = b._session.calls[0]
    assert call["method"] == "DELETE"
    assert call["url"].endswith("/positions/AAPL")
    assert result.side is Side.SELL and result.qty == 1.5


def test_an_exit_is_never_retried_either():
    b = broker(FakeResponse({}, status_code=500))
    with pytest.raises(BrokerError):
        b.close_position("AAPL")
    assert len(b._session.calls) == 1


def test_closing_a_position_that_is_not_there_is_a_no_op():
    assert broker("no such position").close_position("AAPL") is None


# ---------------------------------------------------------------- protocol
def test_the_adapter_satisfies_the_broker_protocol():
    assert isinstance(broker(), Broker)
