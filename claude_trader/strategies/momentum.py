"""Deterministic momentum baseline.

This exists to answer the only question that matters: does the model add
anything? Run the same engine, same risk limits, same costs, same period, and
compare. If Claude cannot beat a twenty-line moving-average rule, the expensive
part of the system is not earning its keep.

It also runs free, which makes it the default for CI and smoke tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from ..models import Action, Decision, Indicators, MarketSnapshot, Picks, PortfolioState


def _score(ind: Indicators) -> float:
    """Crude trend-quality score; higher is a cleaner uptrend."""
    if ind.sma_fast is None or ind.sma_slow is None or ind.sma_slow <= 0:
        return float("-inf")
    separation = (ind.sma_fast - ind.sma_slow) / ind.sma_slow
    momentum = ind.ret_5 or 0.0
    crowding = 0.0
    if ind.rsi is not None and ind.rsi > 70:
        crowding = -(ind.rsi - 70) / 100.0
    return separation + momentum + crowding


class MomentumStrategy:
    name = "momentum"

    def __init__(self, universe: Sequence[str], min_score: float = 0.002) -> None:
        self._universe = tuple(s.upper() for s in universe)
        self._min_score = min_score

    def pick(
        self,
        now: datetime,
        state: PortfolioState,
        overview: Mapping[str, Indicators],
        max_new_positions: int,
    ) -> Picks:
        if max_new_positions <= 0:
            return Picks((), "at position limit", abstain=True)

        ranked = sorted(
            (
                (symbol, _score(ind))
                for symbol, ind in overview.items()
                if symbol in self._universe and symbol not in state.positions
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        chosen = tuple(sym for sym, score in ranked[:max_new_positions] if score >= self._min_score)
        return Picks(
            symbols=chosen,
            strategy="trend-following: fast SMA above slow SMA, avoiding overbought",
            market_mood="bullish" if chosen else "neutral",
            abstain=not chosen,
            rationale=f"{len(chosen)} of {len(ranked)} candidates cleared the score threshold",
        )

    def decide(
        self,
        snapshot: MarketSnapshot,
        strategy_note: str,
        state: PortfolioState,
    ) -> Decision:
        ind = snapshot.indicators
        held = snapshot.position is not None
        score = _score(ind)

        if held:
            broken = (
                ind.sma_fast is not None
                and ind.sma_slow is not None
                and ind.sma_fast < ind.sma_slow
            )
            overbought = ind.rsi is not None and ind.rsi > 78
            if broken or overbought:
                return Decision(
                    symbol=snapshot.symbol,
                    action=Action.SELL,
                    confidence=8 if broken else 7,
                    reason="trend broken" if broken else "overbought, taking profit",
                    source="rule",
                )
            return Decision(
                symbol=snapshot.symbol,
                action=Action.HOLD,
                confidence=5,
                reason="trend intact",
                source="rule",
            )

        if score >= self._min_score and (ind.rsi is None or ind.rsi < 70):
            confidence = 7 if score < 0.01 else 8
            return Decision(
                symbol=snapshot.symbol,
                action=Action.BUY,
                confidence=confidence,
                reason=f"trend score {score:.4f} above threshold",
                source="rule",
            )

        return Decision(
            symbol=snapshot.symbol,
            action=Action.HOLD,
            confidence=4,
            reason=f"trend score {score:.4f} below threshold",
            source="rule",
        )
