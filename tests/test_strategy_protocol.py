"""The strategy interface itself.

The comparison this whole project exists to make -- model against a
deterministic rule -- is only honest if both sides are held to one interface. A
strategy that could size its own position or set its own stop would not be
comparable to one that cannot, so the surface is deliberately two methods wide
and this file is what stops it quietly growing a third.
"""

from __future__ import annotations

from claude_trader.strategies.base import Strategy
from claude_trader.strategies.claude_strategy import ClaudeStrategy
from claude_trader.strategies.momentum import MomentumStrategy


def test_the_momentum_rule_satisfies_the_protocol():
    assert isinstance(MomentumStrategy(universe=("RELIANCE",)), Strategy)


def test_the_claude_strategy_satisfies_the_same_protocol():
    """Both sides of the experiment must be substitutable, or the comparison is
    between two different systems rather than two decision sources."""
    assert isinstance(ClaudeStrategy(client=None, universe=("RELIANCE",)),
                      Strategy)


def test_an_object_missing_decide_is_not_a_strategy():
    class HalfStrategy:
        name = "half"

        def pick(self, now, state, overview, max_new_positions):  # pragma: no cover
            ...

    assert not isinstance(HalfStrategy(), Strategy)


def test_the_proposal_surface_is_exactly_two_methods():
    """Sizing, stops and limits belong to the risk layer. A third method here
    would be the first step in a strategy overruling the gate that exists to
    hold it in check."""
    surface = {name for name in vars(Strategy)
               if not name.startswith("_") and callable(getattr(Strategy, name))}
    assert surface == {"pick", "decide"}
