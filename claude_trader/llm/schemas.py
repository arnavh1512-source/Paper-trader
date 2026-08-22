"""Schema validation for model output.

The model is an untrusted boundary like any other. Previously its reply went
straight into json.loads and then straight into an order; a hallucinated ticker
or a confidence of 47 would have sailed through. Everything is validated and
clamped here before it reaches the risk layer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..errors import LLMResponseError

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)
_TICKER = re.compile(r"^[A-Z][A-Z.\-]{0,9}$")


def extract_json(text: str) -> Any:
    """Pull a JSON object out of a model reply.

    Handles bare JSON, fenced blocks, and prose wrapped around an object. Raises
    rather than guessing when nothing parses.
    """
    cleaned = _FENCE.sub("", (text or "").strip()).strip()
    if not cleaned:
        raise LLMResponseError("model returned an empty response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"could not parse JSON from model reply: {exc}") from exc
    raise LLMResponseError("model reply contained no JSON object")


class PickResponse(BaseModel):
    """Stock selection. An empty ``symbols`` list is a legitimate answer."""

    model_config = ConfigDict(extra="ignore")

    symbols: list[str] = Field(default_factory=list, max_length=8)
    strategy: str = Field(default="", max_length=400)
    market_mood: Literal["bullish", "bearish", "neutral"] = "neutral"
    abstain: bool = False
    rationale: str = Field(default="", max_length=600)

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence):
            raise ValueError("symbols must be a list")
        out: list[str] = []
        for item in value:
            token = str(item).strip().upper()
            if _TICKER.match(token) and token not in out:
                out.append(token)
        return out


class DecisionResponse(BaseModel):
    """Per-symbol trade decision."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["buy", "sell", "hold"] = "hold"
    confidence: int = Field(default=0, ge=0, le=10)
    notional: float | None = Field(default=None, ge=0)
    reason: str = Field(default="", max_length=500)
    invalidation: str = Field(default="", max_length=300)
    horizon_bars: int | None = Field(default=None, ge=1, le=500)

    @field_validator("action", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return str(value).strip().lower() if value is not None else "hold"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> int:
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError):
            return 0
        return max(0, min(10, number))


def parse_picks(text: str) -> PickResponse:
    try:
        return PickResponse.model_validate(extract_json(text))
    except ValidationError as exc:
        raise LLMResponseError(f"pick response failed validation: {exc}") from exc


def parse_decision(text: str) -> DecisionResponse:
    try:
        return DecisionResponse.model_validate(extract_json(text))
    except ValidationError as exc:
        raise LLMResponseError(f"decision response failed validation: {exc}") from exc
