"""The two strategies.

The momentum rule is the control group: if the expensive model cannot beat
twenty lines of moving-average arithmetic on the same engine, costs and period,
the model is not earning its keep. Both must satisfy the same Protocol, so the
engine cannot tell them apart.

The Claude strategy's job is not to be clever -- it is to refuse to pass a
plausible sentence through to an order.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from claude_trader.costs import IndianEquityCosts
from claude_trader.errors import LLMError
from claude_trader.llm.client import ScriptedClient
from claude_trader.markets import INDIA_MARKET, US_MARKET
from claude_trader.models import Action, Indicators, Position
from claude_trader.strategies.claude_strategy import ClaudeStrategy
from claude_trader.strategies.momentum import MomentumStrategy
from tests.conftest import make_snapshot, make_state

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)
UNIVERSE = ("RELIANCE", "TCS", "INFY")


def ind(price=1_000.0, fast=1_010.0, slow=1_000.0, rsi=55.0, ret_5=0.01) -> Indicators:
    return Indicators(
        last_price=price, sma_fast=fast, sma_slow=slow, rsi=rsi,
        atr=price * 0.01, atr_pct=1.0, ret_5=ret_5, trend="up",
    )


def position(symbol="RELIANCE", qty=10, price=1_000.0) -> Position:
    return Position(symbol=symbol, qty=qty, avg_entry_price=price, current_price=price)


def reply(**fields) -> str:
    return json.dumps(fields)


def claude(client, **kwargs) -> ClaudeStrategy:
    kwargs.setdefault("profile", INDIA_MARKET)
    return ClaudeStrategy(client, UNIVERSE, **kwargs)


# ============================================================== momentum
def test_momentum_ranks_the_cleanest_uptrend_first():
    strategy = MomentumStrategy(UNIVERSE)
    picks = strategy.pick(NOW, make_state(), {
        "RELIANCE": ind(fast=1_005.0, slow=1_000.0, ret_5=0.001),
        "TCS": ind(fast=1_100.0, slow=1_000.0, ret_5=0.05),
        "INFY": ind(fast=1_002.0, slow=1_000.0, ret_5=0.0),
    }, max_new_positions=2)
    assert picks.symbols[0] == "TCS"
    assert len(picks.symbols) == 2


def test_momentum_refuses_to_fill_the_slate_with_weak_candidates():
    """Picking the best of a bad bunch is how a trend follower bleeds sideways."""
    strategy = MomentumStrategy(UNIVERSE)
    picks = strategy.pick(NOW, make_state(), {
        s: ind(fast=1_000.0, slow=1_000.0, ret_5=0.0) for s in UNIVERSE
    }, max_new_positions=3)
    assert picks.symbols == ()
    assert picks.abstain is True
    assert picks.market_mood == "neutral"


def test_momentum_penalises_an_overbought_name():
    strategy = MomentumStrategy(UNIVERSE)
    picks = strategy.pick(NOW, make_state(), {
        "RELIANCE": ind(fast=1_100.0, slow=1_000.0, ret_5=0.05, rsi=95.0),
        "TCS": ind(fast=1_080.0, slow=1_000.0, ret_5=0.05, rsi=55.0),
    }, max_new_positions=1)
    assert picks.symbols == ("TCS",)


def test_momentum_never_picks_something_it_already_holds():
    strategy = MomentumStrategy(UNIVERSE)
    picks = strategy.pick(NOW, make_state(positions=[position("TCS", price=3_000.0)]), {
        "TCS": ind(fast=1_100.0, slow=1_000.0, ret_5=0.05),
        "RELIANCE": ind(fast=1_050.0, slow=1_000.0, ret_5=0.02),
    }, max_new_positions=2)
    assert picks.symbols == ("RELIANCE",)


def test_momentum_ignores_symbols_outside_its_universe():
    strategy = MomentumStrategy(("RELIANCE",))
    picks = strategy.pick(NOW, make_state(), {
        "NOTMINE": ind(fast=2_000.0, slow=1_000.0, ret_5=0.5),
    }, max_new_positions=2)
    assert picks.symbols == ()


def test_momentum_abstains_when_there_is_no_room():
    picks = MomentumStrategy(UNIVERSE).pick(NOW, make_state(), {}, max_new_positions=0)
    assert picks.abstain is True and picks.symbols == ()


def test_momentum_scores_nothing_without_the_moving_averages():
    """A freshly listed name has too few bars for the slow SMA. Scoring it as
    zero would rank it above a genuine downtrend."""
    picks = MomentumStrategy(UNIVERSE).pick(NOW, make_state(), {
        "RELIANCE": Indicators(last_price=100.0, sma_fast=None, sma_slow=None),
    }, max_new_positions=1)
    assert picks.symbols == ()


def test_momentum_buys_a_clean_trend():
    decision = MomentumStrategy(UNIVERSE).decide(
        make_snapshot("RELIANCE"), "note", make_state()
    )
    assert decision.action is Action.BUY
    assert decision.confidence >= 7
    assert decision.source == "rule"


def test_momentum_scales_confidence_with_the_strength_of_the_trend():
    strategy = MomentumStrategy(UNIVERSE)
    weak = make_snapshot("RELIANCE")
    weak = replace(weak, indicators=ind(fast=1_005.0, slow=1_000.0, ret_5=0.0))
    strong = make_snapshot("RELIANCE")
    strong = replace(strong, indicators=ind(fast=1_100.0, slow=1_000.0, ret_5=0.05))
    assert strategy.decide(weak, "n", make_state()).confidence < \
        strategy.decide(strong, "n", make_state()).confidence


def test_momentum_will_not_buy_an_overbought_name():
    snapshot = make_snapshot("RELIANCE")
    snapshot = replace(snapshot, indicators=ind(rsi=85.0))
    assert MomentumStrategy(UNIVERSE).decide(snapshot, "n", make_state()).action is Action.HOLD


def test_momentum_holds_a_position_whose_trend_is_intact():
    snapshot = make_snapshot("RELIANCE", position=position())
    decision = MomentumStrategy(UNIVERSE).decide(snapshot, "n", make_state())
    assert decision.action is Action.HOLD
    assert decision.reason == "trend intact"


def test_momentum_sells_when_the_trend_breaks():
    snapshot = make_snapshot("RELIANCE", position=position())
    snapshot = replace(snapshot, indicators=ind(fast=990.0, slow=1_000.0))
    decision = MomentumStrategy(UNIVERSE).decide(snapshot, "n", make_state())
    assert decision.action is Action.SELL
    assert decision.confidence == 8


def test_momentum_takes_profit_when_a_held_name_runs_hot():
    snapshot = make_snapshot("RELIANCE", position=position())
    snapshot = replace(snapshot, indicators=ind(rsi=85.0))
    decision = MomentumStrategy(UNIVERSE).decide(snapshot, "n", make_state())
    assert decision.action is Action.SELL
    assert "overbought" in decision.reason


def test_momentum_needs_no_network_and_no_key():
    """This is why it is the default in CI."""
    strategy = MomentumStrategy(UNIVERSE)
    assert strategy.name == "momentum"
    assert strategy.decide(make_snapshot(), "n", make_state()).source == "rule"


# ================================================================ claude
def test_claude_returns_the_models_picks_verbatim_when_they_are_legal():
    client = ScriptedClient([reply(symbols=["RELIANCE", "TCS"], strategy="buy dips",
                                   market_mood="bullish")])
    picks = claude(client).pick(NOW, make_state(), {"RELIANCE": ind()}, 3)
    assert picks.symbols == ("RELIANCE", "TCS")
    assert picks.strategy == "buy dips"
    assert picks.market_mood == "bullish"


def test_claude_drops_a_hallucinated_ticker():
    """The model will confidently name a stock that is not in the universe, and
    on NSE that symbol may not even be tradable."""
    client = ScriptedClient([reply(symbols=["RELIANCE", "TESLA"])])
    picks = claude(client).pick(NOW, make_state(), {}, 3)
    assert picks.symbols == ("RELIANCE",)


def test_claude_abstains_when_every_pick_was_illegal():
    client = ScriptedClient([reply(symbols=["TESLA", "GME"], strategy="yolo")])
    assert claude(client).pick(NOW, make_state(), {}, 3).abstain is True


def test_claude_respects_the_room_the_risk_layer_allows():
    client = ScriptedClient([reply(symbols=["RELIANCE", "TCS", "INFY"])])
    assert len(claude(client).pick(NOW, make_state(), {}, 2).symbols) == 2


def test_claude_does_not_call_the_api_when_there_is_no_room():
    """Position limits are free to enforce. Paying for a pick that cannot be
    acted on is pure waste over thousands of cycles."""
    client = ScriptedClient([reply(symbols=["RELIANCE"])])
    picks = claude(client).pick(NOW, make_state(), {}, 0)
    assert picks.abstain is True
    assert client.calls_made == 0


def test_claude_abstains_rather_than_guessing_when_the_api_fails():
    class Broken(ScriptedClient):
        def complete(self, system, prompt, max_tokens=None):
            raise LLMError("529 overloaded")

    picks = claude(Broken()).pick(NOW, make_state(), {}, 3)
    assert picks.symbols == () and picks.abstain is True
    assert "529" in picks.rationale


def test_claude_abstains_on_an_unparseable_reply():
    picks = claude(ScriptedClient(["I think you should buy Reliance!"])).pick(
        NOW, make_state(), {}, 3
    )
    assert picks.abstain is True


def test_the_picker_prompt_carries_the_universe_and_the_market():
    client = ScriptedClient([reply(symbols=[])])
    claude(client).pick(NOW, make_state(), {"RELIANCE": ind()}, 3)
    _, prompt = client.prompts[0]
    assert "RELIANCE" in prompt
    assert "NSE" in prompt or "India" in prompt


# ------------------------------------------------------------------ decide
def test_claude_passes_a_well_formed_buy_through():
    client = ScriptedClient([reply(action="buy", confidence=8, reason="breakout",
                                   invalidation="below 980")])
    decision = claude(client).decide(make_snapshot(), "note", make_state())
    assert decision.action is Action.BUY
    assert decision.confidence == 8
    assert "invalidation: below 980" in decision.reason
    assert decision.source == "model"


def test_a_buy_without_an_invalidation_level_is_downgraded_to_hold():
    """A trade you cannot describe the failure of is not a trade, it is a hope."""
    client = ScriptedClient([reply(action="buy", confidence=9, reason="feels strong",
                                   invalidation="  ")])
    decision = claude(client).decide(make_snapshot(), "note", make_state())
    assert decision.action is Action.HOLD
    assert decision.reason == "no invalidation level supplied"


def test_the_invalidation_requirement_can_be_relaxed():
    client = ScriptedClient([reply(action="buy", confidence=9, reason="strong")])
    decision = claude(client, require_invalidation=False).decide(
        make_snapshot(), "note", make_state()
    )
    assert decision.action is Action.BUY


def test_a_sell_with_no_position_is_downgraded_to_hold():
    """The NSE cash segment has no shorting. Passing this through would have the
    executor try to sell something that does not exist."""
    client = ScriptedClient([reply(action="sell", confidence=9, reason="topping")])
    decision = claude(client).decide(make_snapshot(), "note", make_state())
    assert decision.action is Action.HOLD
    assert decision.reason == "no position to sell"


def test_a_sell_on_a_held_name_is_allowed():
    client = ScriptedClient([reply(action="sell", confidence=8, reason="reversal")])
    decision = claude(client).decide(
        make_snapshot(position=position()), "note", make_state()
    )
    assert decision.action is Action.SELL


def test_a_transport_failure_degrades_to_hold_not_to_a_guess():
    class Broken(ScriptedClient):
        def complete(self, system, prompt, max_tokens=None):
            raise LLMError("timeout")

    decision = claude(Broken()).decide(make_snapshot(), "note", make_state())
    assert decision.action is Action.HOLD
    assert decision.confidence == 0
    assert decision.source == "error"


def test_a_schema_failure_degrades_to_hold():
    decision = claude(ScriptedClient(["buy it now"])).decide(
        make_snapshot(), "note", make_state()
    )
    assert decision.action is Action.HOLD
    assert decision.source == "error"


def test_a_hold_never_gets_the_invalidation_suffix():
    client = ScriptedClient([reply(action="hold", confidence=5, reason="unclear",
                                   invalidation="below 900")])
    decision = claude(client).decide(make_snapshot(), "note", make_state())
    assert decision.reason == "unclear"


def test_the_reason_is_truncated_before_it_reaches_the_journal():
    client = ScriptedClient([reply(action="hold", confidence=5, reason="x" * 900)])
    assert len(claude(client).decide(make_snapshot(), "n", make_state()).reason) <= 400


def test_the_prompt_fingerprint_is_recorded_for_every_decision():
    """Without it, a calibration report cannot tell which prompt produced which
    outcome, and prompt changes become unmeasurable."""
    strategy = claude(ScriptedClient([reply(action="hold", confidence=5, reason="x")]))
    assert strategy.last_prompt_sha == ""
    strategy.decide(make_snapshot(), "note", make_state())
    first = strategy.last_prompt_sha
    assert len(first) == 64
    strategy.decide(make_snapshot("TCS", price=3_000.0), "note", make_state())
    assert strategy.last_prompt_sha != first


def test_the_decision_prompt_states_what_a_round_trip_costs():
    """On NSE the fee is a material fraction of a 15-minute move; a model that
    cannot see it will propose trades that lose money by construction."""
    client = ScriptedClient([reply(action="hold", confidence=5, reason="x")])
    strategy = ClaudeStrategy(client, UNIVERSE, profile=INDIA_MARKET,
                              segment="intraday", costs=IndianEquityCosts("intraday"))
    strategy.decide(make_snapshot(), "note", make_state())
    _, prompt = client.prompts[0]
    assert "cost" in prompt.lower()


def test_the_cost_floor_is_zero_without_a_cost_model():
    strategy = ClaudeStrategy(ScriptedClient(), UNIVERSE, profile=US_MARKET)
    assert strategy._cost_floor(make_snapshot()) == 0.0


def test_the_cost_floor_survives_a_zero_price():
    strategy = ClaudeStrategy(ScriptedClient(), UNIVERSE, profile=INDIA_MARKET,
                              costs=IndianEquityCosts("intraday"))
    assert strategy._cost_floor(make_snapshot(price=0.0)) == 0.0


def test_the_client_is_exposed_so_usage_can_be_reported():
    client = ScriptedClient()
    assert claude(client).client is client


def test_both_strategies_satisfy_the_same_contract():
    """The engine must not be able to tell them apart."""
    for strategy in (MomentumStrategy(UNIVERSE), claude(ScriptedClient())):
        assert isinstance(strategy.name, str)
        assert callable(strategy.pick) and callable(strategy.decide)
