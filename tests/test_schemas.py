"""Model output is an untrusted boundary.

In the original bot the reply went into ``json.loads`` and then straight into an
order, so a hallucinated ticker or a confidence of 47 would have been acted on.
Everything here is about refusing to let that happen.
"""

from __future__ import annotations

import pytest

from claude_trader.errors import LLMResponseError
from claude_trader.llm.schemas import (
    DecisionResponse,
    PickResponse,
    extract_json,
    parse_decision,
    parse_picks,
)


# ------------------------------------------------------------ json recovery
def test_bare_json_parses():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json_parses():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_json_is_recovered_from_surrounding_prose():
    """Models add 'Here is my analysis:' more often than anyone would like."""
    reply = 'Sure! Here is my answer:\n{"action": "buy"}\nHope that helps.'
    assert extract_json(reply) == {"action": "buy"}


@pytest.mark.parametrize("reply", ["", "   ", None, "no json here at all"])
def test_unparseable_replies_raise_rather_than_default(reply):
    with pytest.raises(LLMResponseError):
        extract_json(reply)


def test_broken_json_raises_with_the_parser_error():
    with pytest.raises(LLMResponseError, match="could not parse JSON"):
        extract_json('prose {"action": "buy", } more prose')


# -------------------------------------------------------------------- picks
def test_picks_are_upper_cased_and_de_duplicated():
    picks = PickResponse.model_validate(
        {"symbols": [" reliance ", "RELIANCE", "tcs"], "market_mood": "bullish"}
    )
    assert picks.symbols == ["RELIANCE", "TCS"]
    assert picks.market_mood == "bullish"


def test_an_empty_pick_list_is_a_valid_answer():
    """Doing nothing has to be representable, or the bot churns by construction."""
    picks = PickResponse.model_validate({"symbols": [], "abstain": True})
    assert picks.symbols == [] and picks.abstain is True
    assert PickResponse.model_validate({}).symbols == []


@pytest.mark.parametrize("value", [None, "", [], "RELIANCE"])
def test_symbols_accepts_the_shapes_models_actually_emit(value):
    result = PickResponse.model_validate({"symbols": value}).symbols
    assert result == (["RELIANCE"] if value == "RELIANCE" else [])


def test_junk_tickers_are_dropped_not_traded():
    picks = PickResponse.model_validate(
        {"symbols": ["RELIANCE", "not a ticker", "12345", "", "WAYTOOLONGSYMBOL"]}
    )
    assert picks.symbols == ["RELIANCE"]


def test_an_unknown_mood_is_rejected():
    with pytest.raises(LLMResponseError, match="pick response failed validation"):
        parse_picks('{"market_mood": "euphoric"}')


def test_extra_keys_are_ignored_rather_than_fatal():
    picks = parse_picks('{"symbols": ["TCS"], "confidence_in_universe": 9}')
    assert picks.symbols == ["TCS"]


def test_pick_prose_is_bounded():
    with pytest.raises(LLMResponseError):
        parse_picks('{"strategy": "%s"}' % ("x" * 500))


# ---------------------------------------------------------------- decisions
def test_decision_defaults_to_hold():
    decision = DecisionResponse.model_validate({})
    assert decision.action == "hold"
    assert decision.confidence == 0
    assert decision.notional is None


def test_action_is_case_insensitive():
    assert DecisionResponse.model_validate({"action": " BUY "}).action == "buy"


def test_an_invented_action_is_rejected():
    with pytest.raises(LLMResponseError):
        parse_decision('{"action": "short"}')


@pytest.mark.parametrize(
    "raw,expected",
    [(47, 10), (-3, 0), ("8", 8), (7.6, 8), (None, 0), ("very high", 0), (True, 1)],
)
def test_confidence_is_clamped_to_the_scale(raw, expected):
    """A confidence of 47 sailing through the >= 7 gate is exactly the class of
    bug this layer exists to stop."""
    import json

    assert parse_decision(json.dumps({"confidence": raw})).confidence == expected


def test_a_negative_notional_is_rejected():
    with pytest.raises(LLMResponseError):
        parse_decision('{"notional": -500}')


def test_horizon_must_be_a_sane_number_of_bars():
    assert parse_decision('{"horizon_bars": 12}').horizon_bars == 12
    with pytest.raises(LLMResponseError):
        parse_decision('{"horizon_bars": 0}')
    with pytest.raises(LLMResponseError):
        parse_decision('{"horizon_bars": 5000}')


def test_a_full_decision_round_trips():
    decision = parse_decision(
        """```json
        {"action": "buy", "confidence": 8, "notional": 9500,
         "reason": "breakout on volume", "invalidation": "below 1,240",
         "horizon_bars": 6}
        ```"""
    )
    assert (decision.action, decision.confidence, decision.notional) == ("buy", 8, 9500)
    assert decision.invalidation == "below 1,240"
