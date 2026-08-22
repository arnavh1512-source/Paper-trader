"""The retry layer.

The original bot let a single transient 500 kill a symbol for the cycle. The
opposite failure is worse: retrying something that already happened. These
tests pin both edges, and they never sleep for real.
"""

from __future__ import annotations

import pytest
import requests

from claude_trader.errors import BrokerError, MarketDataError
from claude_trader.http import (
    DEFAULT_MAX_ATTEMPTS,
    RETRYABLE_STATUS,
    broker_request,
    request_json,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", content=b"{}"):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.content = content

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Returns each scripted item in turn; an Exception item is raised."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self.items.pop(0) if self.items else FakeResponse()
        if isinstance(item, Exception):
            raise item
        return item


def call(session, **kwargs):
    kwargs.setdefault("sleeper", lambda attempt: None)
    return request_json("GET", "https://example.test/v2/bars", session=session, **kwargs)


# ---------------------------------------------------------------- happy path
def test_a_good_response_is_decoded():
    session = FakeSession(FakeResponse(payload={"bars": [1, 2]}))
    assert call(session) == {"bars": [1, 2]}
    assert len(session.calls) == 1


def test_the_method_is_upper_cased_and_the_body_is_passed_through():
    session = FakeSession(FakeResponse())
    request_json("post", "https://example.test", json_body={"a": 1},
                 session=session, sleeper=lambda a: None)
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["json"] == {"a": 1}


def test_empty_params_are_sent_as_none_rather_than_an_empty_dict():
    """requests renders an empty mapping as a trailing '?', which some gateways
    treat as a different cache key."""
    session = FakeSession(FakeResponse())
    call(session)
    assert session.calls[0]["params"] is None


def test_params_and_headers_are_copied_not_aliased():
    """The caller's dict must not be mutated by the transport."""
    headers = {"x-api-key": "k"}
    params = {"symbols": "RELIANCE"}
    session = FakeSession(FakeResponse())
    call(session, headers=headers, params=params)
    assert session.calls[0]["headers"] == headers
    assert session.calls[0]["headers"] is not headers
    assert session.calls[0]["params"] is not params


def test_an_empty_body_is_an_empty_object_not_a_parse_error():
    """A 204 from an order cancel is success, not a malformed response."""
    session = FakeSession(FakeResponse(status_code=204, content=b""))
    assert call(session) == {}


# -------------------------------------------------------------------- retry
@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
def test_transient_statuses_are_retried(status):
    session = FakeSession(FakeResponse(status_code=status), FakeResponse(payload={"ok": True}))
    assert call(session) == {"ok": True}
    assert len(session.calls) == 2


def test_a_transport_error_is_retried():
    session = FakeSession(requests.ConnectionError("reset"), FakeResponse(payload={"ok": 1}))
    assert call(session) == {"ok": 1}


def test_a_timeout_is_retried_then_surfaces_as_the_callers_error_type():
    session = FakeSession(*[requests.Timeout("slow")] * 3)
    with pytest.raises(MarketDataError, match="after 3 attempts"):
        call(session)
    assert len(session.calls) == 3


def test_retries_are_bounded_by_max_attempts():
    session = FakeSession(*[FakeResponse(status_code=503)] * 10)
    with pytest.raises(MarketDataError):
        call(session, max_attempts=2)
    assert len(session.calls) == 2


def test_the_last_attempt_reports_the_status_rather_than_retrying_forever():
    """On the final attempt a retryable status must be raised, not swallowed."""
    session = FakeSession(FakeResponse(status_code=503, text="upstream down"))
    with pytest.raises(MarketDataError, match="503"):
        call(session, max_attempts=1)


def test_backoff_is_attempted_between_tries():
    slept: list[int] = []
    session = FakeSession(FakeResponse(status_code=500), FakeResponse(payload={}))
    request_json("GET", "https://example.test", session=session, sleeper=slept.append)
    assert slept == [1]


def test_zero_attempts_still_makes_one_call():
    """A misconfigured retry count must not silently disable the request."""
    session = FakeSession(FakeResponse(payload={"ok": 1}))
    assert call(session, max_attempts=0) == {"ok": 1}


# ------------------------------------------------------------- fatal errors
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_client_error_fails_immediately(status):
    """A 401 will never succeed on retry, and hammering it invites a ban."""
    session = FakeSession(*[FakeResponse(status_code=status, text="nope")] * 3)
    with pytest.raises(MarketDataError, match=str(status)):
        call(session)
    assert len(session.calls) == 1


def test_the_error_detail_is_truncated():
    session = FakeSession(FakeResponse(status_code=400, text="x" * 2_000))
    with pytest.raises(MarketDataError) as exc:
        call(session)
    assert len(str(exc.value)) < 600


def test_a_non_json_body_is_a_typed_error_not_a_value_error():
    session = FakeSession(FakeResponse(payload=ValueError("no json"), content=b"<html>"))
    with pytest.raises(MarketDataError, match="non-JSON"):
        call(session)


def test_the_error_type_is_chosen_by_the_caller():
    session = FakeSession(FakeResponse(status_code=400))
    with pytest.raises(BrokerError):
        call(session, error_type=BrokerError)


def test_broker_requests_default_to_broker_errors():
    """A market-data error on an order path would be routed to the wrong
    handler, and orders are the path where that matters."""
    session = FakeSession(FakeResponse(status_code=400))
    with pytest.raises(BrokerError):
        broker_request("POST", "https://example.test/orders", session=session,
                       sleeper=lambda a: None)


def test_broker_requests_keep_an_explicit_error_type():
    session = FakeSession(FakeResponse(status_code=400))
    with pytest.raises(MarketDataError):
        broker_request("GET", "https://example.test", session=session,
                       error_type=MarketDataError, sleeper=lambda a: None)


def test_the_default_attempt_count_is_conservative():
    assert DEFAULT_MAX_ATTEMPTS == 3
