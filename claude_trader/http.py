"""Thin HTTP helper with bounded retries and explicit error typing.

The original bot let a single transient 500 from Alpaca kill a whole symbol.
Retries here are deliberately conservative: idempotent reads retry, order
submissions do not (a retried market order is a duplicate position)."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Mapping

import requests

from .errors import BrokerError, MarketDataError

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 3


def _sleep_backoff(attempt: int, base: float = 0.5, cap: float = 8.0) -> None:
    delay = min(cap, base * (2 ** (attempt - 1)))
    time.sleep(delay + random.uniform(0, delay * 0.25))


def request_json(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    json_body: Any | None = None,
    timeout: float = 15.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    error_type: type[Exception] = MarketDataError,
    session: requests.Session | None = None,
    sleeper: Callable[[int], None] = _sleep_backoff,
) -> Any:
    """Perform an HTTP request and return decoded JSON.

    Retries only on transport errors and the status codes in RETRYABLE_STATUS.
    A 4xx other than 408/429 fails immediately -- it will never succeed.
    """
    caller = session or requests
    last_error: Exception | None = None

    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            response = caller.request(
                method.upper(),
                url,
                headers=dict(headers or {}),
                params=dict(params or {}) or None,
                json=json_body,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleeper(attempt)
            continue

        if response.status_code in RETRYABLE_STATUS and attempt < max_attempts:
            last_error = error_type(
                f"{method} {url} returned {response.status_code}"
            )
            sleeper(attempt)
            continue

        if response.status_code >= 400:
            detail = (response.text or "")[:400]
            raise error_type(
                f"{method} {url} failed with {response.status_code}: {detail}"
            )

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise error_type(f"{method} {url} returned non-JSON body") from exc

    raise error_type(f"{method} {url} failed after {max_attempts} attempts: {last_error}")


def broker_request(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("error_type", BrokerError)
    return request_json(*args, **kwargs)
