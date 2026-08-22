"""Anthropic Messages API client with journal-backed response caching.

The cache is what makes backtesting affordable: replaying six months of 15-minute
cycles is tens of thousands of calls, and a re-run after a code change should
cost nothing. Keys are content-addressed, so any prompt change invalidates
naturally.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from ..config import LLMConfig
from ..errors import LLMError
from ..http import request_json

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class ResponseCache(Protocol):
    def cache_get(self, key: str) -> str | None: ...

    def cache_put(self, key: str, model: str, response: str, now: datetime) -> None: ...


def prompt_fingerprint(model: str, system: str, prompt: str) -> str:
    digest = hashlib.sha256()
    for part in (model, system, prompt):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class ClaudeClient:
    """Calls the Messages API and returns raw text. Parsing lives in schemas."""

    def __init__(
        self,
        config: LLMConfig,
        cache: ResponseCache | None = None,
        session: object | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._session = session
        self.calls_made = 0
        self.cache_hits = 0

    @property
    def model(self) -> str:
        return self._config.model

    def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        key = prompt_fingerprint(self._config.model, system, prompt)

        if self._cache is not None and self._config.cache_enabled:
            cached = self._cache.cache_get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached

        if not self._config.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

        payload = request_json(
            "POST",
            ANTHROPIC_URL,
            headers={
                "x-api-key": self._config.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json_body={
                "model": self._config.model,
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": self._config.temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self._config.timeout_seconds,
            max_attempts=self._config.max_retries,
            error_type=LLMError,
            session=self._session,
        )
        self.calls_made += 1
        text = _first_text_block(payload)

        if self._cache is not None and self._config.cache_enabled:
            self._cache.cache_put(key, self._config.model, text, datetime.now(timezone.utc))
        return text

    @property
    def usage_summary(self) -> str:
        total = self.calls_made + self.cache_hits
        if total == 0:
            return "no model calls"
        return f"{self.calls_made} API calls, {self.cache_hits} cache hits"


def _first_text_block(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        raise LLMError("Anthropic response was not an object")
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise LLMError("Anthropic response had no content blocks")
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "text":
            return str(block.get("text", "")).strip()
    raise LLMError("Anthropic response had no text block")


class ScriptedClient:
    """Deterministic stand-in used by tests and offline backtests."""

    def __init__(self, responses: list[str] | None = None, default: str = "{}") -> None:
        self._responses = list(responses or [])
        self._default = default
        self.prompts: list[tuple[str, str]] = []
        self.calls_made = 0
        self.cache_hits = 0
        self.model = "scripted"

    def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        self.prompts.append((system, prompt))
        self.calls_made += 1
        if self._responses:
            return self._responses.pop(0)
        return self._default

    @property
    def usage_summary(self) -> str:
        return f"{self.calls_made} scripted calls"
