"""The Anthropic client and its cache.

The cache is what makes backtesting affordable -- six months of 15-minute cycles
is tens of thousands of calls -- so the thing that must never break is the key:
if two different prompts collide, the backtest replays somebody else's answer.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from claude_trader.config import LLMConfig
from claude_trader.errors import ConfigError, LLMBudgetExceeded, LLMError
from claude_trader.llm.client import (
    ANTHROPIC_URL,
    ANTHROPIC_VERSION,
    ClaudeClient,
    ScriptedClient,
    prompt_fingerprint,
)

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.text = ""
        self.content = b"{}"

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        payload = self.payloads.pop(0) if self.payloads else text_reply("{}")
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload)


def text_reply(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def client(journal=None, session=None, **cfg) -> ClaudeClient:
    config = LLMConfig(api_key="sk-test", model="claude-sonnet-5", **cfg)
    return ClaudeClient(config, cache=journal, session=session or FakeSession())


# --------------------------------------------------------------- fingerprint
def test_the_fingerprint_is_stable_for_identical_input():
    assert prompt_fingerprint("m", "sys", "p") == prompt_fingerprint("m", "sys", "p")


@pytest.mark.parametrize(
    "a,b",
    [
        (("m1", "sys", "p"), ("m2", "sys", "p")),
        (("m", "sys1", "p"), ("m", "sys2", "p")),
        (("m", "sys", "p1"), ("m", "sys", "p2")),
    ],
)
def test_every_component_changes_the_fingerprint(a, b):
    assert prompt_fingerprint(*a) != prompt_fingerprint(*b)


def test_the_parts_cannot_be_shifted_across_the_boundary():
    """Without a separator, ('ab','c') and ('a','bc') would hash identically and
    a backtest would replay the wrong cached answer."""
    assert prompt_fingerprint("m", "ab", "c") != prompt_fingerprint("m", "a", "bc")


def test_a_prompt_change_invalidates_the_cache_naturally(journal):
    session = FakeSession(text_reply("first"), text_reply("second"))
    c = client(journal, session)
    assert c.complete("sys", "prompt A") == "first"
    assert c.complete("sys", "prompt B") == "second"
    assert c.calls_made == 2 and c.cache_hits == 0


# --------------------------------------------------------------------- calls
def test_a_completion_returns_the_first_text_block():
    session = FakeSession({
        "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "  {\"action\": \"buy\"}  "},
        ]
    })
    assert client(session=session).complete("sys", "p") == '{"action": "buy"}'


def test_the_request_carries_the_api_key_and_version_headers():
    session = FakeSession(text_reply("ok"))
    client(session=session).complete("sys", "p")
    headers = session.calls[0]["headers"]
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"] == ANTHROPIC_VERSION
    assert session.calls[0]["url"] == ANTHROPIC_URL
    assert session.calls[0]["method"] == "POST"


def test_the_payload_carries_the_configured_model_and_temperature():
    session = FakeSession(text_reply("ok"))
    client(session=session, temperature=0.0).complete("sys", "prompt text")
    body = session.calls[0]["json"]
    assert body["model"] == "claude-sonnet-5"
    assert body["temperature"] == 0.0
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "prompt text"}]


def test_the_call_site_can_override_max_tokens():
    session = FakeSession(text_reply("ok"))
    client(session=session, max_tokens=1024).complete("sys", "p", max_tokens=400)
    assert session.calls[0]["json"]["max_tokens"] == 400


def test_without_an_override_the_configured_budget_is_used():
    session = FakeSession(text_reply("ok"))
    c = ClaudeClient(LLMConfig(api_key="k", max_tokens=777), session=session)
    c.complete("sys", "p")
    assert session.calls[0]["json"]["max_tokens"] == 777


def test_a_missing_api_key_fails_loudly_before_any_network_call():
    session = FakeSession(text_reply("ok"))
    c = ClaudeClient(LLMConfig(api_key=""), session=session)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        c.complete("sys", "p")
    assert session.calls == []


# --------------------------------------------------------------- bad replies
@pytest.mark.parametrize(
    "payload,match",
    [
        ("not an object", "not an object"),
        ({"content": []}, "no content blocks"),
        ({"content": "text"}, "no content blocks"),
        ({}, "no content blocks"),
        ({"content": [{"type": "tool_use", "id": "x"}]}, "no text block"),
    ],
)
def test_a_malformed_response_is_a_typed_error(payload, match):
    """Degrading to hold on a typed error is safe. Returning None here would
    surface as a TypeError three layers away, mid-cycle."""
    with pytest.raises(LLMError, match=match):
        client(session=FakeSession(payload)).complete("sys", "p")


def test_an_empty_text_block_is_returned_rather_than_raising():
    """The schema layer rejects it with a better message than the transport can."""
    assert client(session=FakeSession(text_reply(""))).complete("sys", "p") == ""


# --------------------------------------------------------------------- cache
def test_a_repeated_prompt_is_served_from_the_journal(journal):
    session = FakeSession(text_reply("cached me"))
    c = client(journal, session)
    assert c.complete("sys", "p") == "cached me"
    assert c.complete("sys", "p") == "cached me"
    assert len(session.calls) == 1
    assert (c.calls_made, c.cache_hits) == (1, 1)


def test_the_cache_survives_a_new_client(journal):
    """On GitHub Actions every cycle is a new process."""
    client(journal, FakeSession(text_reply("stored"))).complete("sys", "p")
    fresh = client(journal, FakeSession(text_reply("should not be used")))
    assert fresh.complete("sys", "p") == "stored"
    assert fresh.calls_made == 0


def test_the_cache_can_be_switched_off(journal):
    session = FakeSession(text_reply("a"), text_reply("b"))
    c = client(journal, session, cache_enabled=False)
    assert c.complete("sys", "p") == "a"
    assert c.complete("sys", "p") == "b"


def test_without_a_cache_every_call_goes_out():
    session = FakeSession(text_reply("a"), text_reply("b"))
    c = ClaudeClient(LLMConfig(api_key="k"), cache=None, session=session)
    c.complete("sys", "p")
    c.complete("sys", "p")
    assert len(session.calls) == 2


def test_a_failed_call_is_not_cached(journal):
    session = FakeSession({"content": []}, text_reply("good"))
    c = client(journal, session)
    with pytest.raises(LLMError):
        c.complete("sys", "p")
    assert c.complete("sys", "p") == "good"


def test_different_models_do_not_share_cache_entries(journal):
    """Two models answering the same prompt are two different experiments."""
    session_a = FakeSession(text_reply("sonnet says"))
    session_b = FakeSession(text_reply("haiku says"))
    ClaudeClient(LLMConfig(api_key="k", model="claude-sonnet-5"),
                 cache=journal, session=session_a).complete("sys", "p")
    other = ClaudeClient(LLMConfig(api_key="k", model="claude-haiku-4-5"),
                         cache=journal, session=session_b)
    assert other.complete("sys", "p") == "haiku says"


# ------------------------------------------------------------------- usage
def test_usage_is_reported_honestly(journal):
    c = client(journal, FakeSession(text_reply("x")))
    assert c.usage_summary == "no model calls"
    c.complete("sys", "p")
    c.complete("sys", "p")
    assert c.usage_summary == "1 API calls, 1 cache hits"


def test_the_model_name_is_exposed_for_the_prompt_fingerprint():
    assert client().model == "claude-sonnet-5"


# --------------------------------------------------------------- scripted
def test_the_scripted_client_replays_in_order():
    c = ScriptedClient(["one", "two"])
    assert (c.complete("s", "a"), c.complete("s", "b")) == ("one", "two")
    assert c.calls_made == 2


def test_the_scripted_client_falls_back_to_its_default():
    c = ScriptedClient(["only"], default='{"action": "hold"}')
    c.complete("s", "p")
    assert c.complete("s", "p") == '{"action": "hold"}'


def test_the_scripted_client_records_what_it_was_asked():
    """Prompt content is the thing under test in the strategy suite."""
    c = ScriptedClient()
    c.complete("SYSTEM", "PROMPT")
    assert c.prompts == [("SYSTEM", "PROMPT")]
    assert c.usage_summary == "1 scripted calls"
    assert c.model == "scripted"


# ------------------------------------------------------------- spend ceiling
def test_calls_are_unlimited_unless_a_ceiling_is_configured():
    """Live trading runs one hourly cycle and cannot run away, so the default
    must not be a limit someone has to remember to raise."""
    session = FakeSession(*[text_reply("ok")] * 5)
    c = client(session=session)
    for _ in range(5):
        c.complete("sys", str(_))
    assert c.calls_made == 5


def test_the_ceiling_stops_billable_calls_once_it_is_reached():
    session = FakeSession(*[text_reply("ok")] * 5)
    c = client(session=session, max_api_calls=2)
    c.complete("sys", "one")
    c.complete("sys", "two")
    with pytest.raises(LLMBudgetExceeded, match="ceiling of 2"):
        c.complete("sys", "three")
    assert c.calls_made == 2
    assert len(session.calls) == 2, "the third request must never leave the process"


def test_exceeding_the_budget_is_an_llm_error_so_the_bot_degrades_to_hold():
    """The strategy catches LLMError and holds; the risk engine's stops and
    square-off never consult the model. A budget that raised something the
    strategy did not catch would turn a spending limit into a crash that
    stranded open positions."""
    assert issubclass(LLMBudgetExceeded, LLMError)


def test_a_cached_answer_is_free_and_does_not_count_against_the_ceiling(journal):
    session = FakeSession(text_reply("first"))
    c = client(journal=journal, session=session, max_api_calls=1)
    assert c.complete("sys", "p") == "first"
    # Same prompt: served from cache, so it must not raise even though the
    # ceiling is already spent.
    assert c.complete("sys", "p") == "first"
    assert (c.calls_made, c.cache_hits) == (1, 1)


def test_a_negative_ceiling_is_refused_rather_than_silently_blocking_everything():
    with pytest.raises(ConfigError):
        LLMConfig(max_api_calls=-1)


# --------------------------------------------------------------- token usage
def test_billed_tokens_are_read_back_from_the_response_not_estimated():
    session = FakeSession(
        {"content": [{"type": "text", "text": "a"}], "usage": {"input_tokens": 1800, "output_tokens": 240}},
        {"content": [{"type": "text", "text": "b"}], "usage": {"input_tokens": 900, "output_tokens": 110}},
    )
    c = client(session=session)
    c.complete("sys", "one")
    c.complete("sys", "two")
    assert (c.input_tokens, c.output_tokens) == (2700, 350)
    assert "2,700 in / 350 out tokens" in c.usage_summary


def test_a_response_without_usage_accounting_is_still_a_usable_answer():
    """Accounting is a convenience. Discarding a tradeable decision because the
    usage block was missing would let a reporting detail block a trade."""
    session = FakeSession(text_reply("still fine"))
    c = client(session=session)
    assert c.complete("sys", "p") == "still fine"
    assert (c.input_tokens, c.output_tokens) == (0, 0)


@pytest.mark.parametrize("usage", [
    {"input_tokens": "1800"},      # string
    {"input_tokens": -5},          # negative
    {"input_tokens": True},        # bool is an int subclass
    {"input_tokens": None},
    "not a mapping",
])
def test_malformed_usage_accounting_is_ignored_rather_than_trusted(usage):
    session = FakeSession({"content": [{"type": "text", "text": "a"}], "usage": usage})
    c = client(session=session)
    c.complete("sys", "p")
    assert c.input_tokens == 0


def test_the_summary_names_the_ceiling_so_a_truncated_run_is_explicable():
    c = client(session=FakeSession(text_reply("ok")), max_api_calls=50)
    c.complete("sys", "p")
    assert "(ceiling 50)" in c.usage_summary


def test_the_scripted_stand_in_carries_the_same_counters_as_the_real_client():
    """Tests and offline backtests read these off whichever client they were
    handed; an attribute the real one has and the fake lacks is a test that
    passes offline and breaks in production."""
    scripted = ScriptedClient()
    for attr in ("calls_made", "cache_hits", "input_tokens", "output_tokens", "model"):
        assert hasattr(scripted, attr)
