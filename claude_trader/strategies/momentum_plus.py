"""Momentum with a fixed set of textbook filters on top.

The control (``momentum``) is deliberately crude and must stay frozen while the
experiment scores Claude against it. This arm asks a narrower question: do a
handful of standard, untuned filters change the control's outcome at all?
Each filter is a hypothesis stated before any data was looked at, not a claim
that it makes money:

* **Breadth.** Abstain when fewer than half the names in view are in an
  uptrend. One rising stock in a falling market is the trade most likely to be
  dragged down with it, and breadth needs no benchmark feed.
* **Opening window.** Sit out the first fifteen minutes of the session, when
  the newest bar is a stub and the overnight gap dominates every short return.
* **Confirmation.** The 20-bar return must agree with the short trend, and
  volume must be at least close to normal.
* **Cost floor.** One ATR of movement must at least pay for a round trip at the
  ticket size this book would actually send.
* **Volatility-scaled ranking.** Candidates are ranked by trend per unit of
  ATR, so a quiet clean trend beats a noisy big one.

Exits are simpler than the control's: sell when the trend breaks, and leave
winners to the risk layer's trailing stop and target instead of capping them
with an RSI take-profit. Every buy is a subset of what the control's own entry
rule would accept, which keeps the two arms comparable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from ..costs import CostModel, round_trip_cost_pct
from ..markets import MarketProfile
from ..models import Action, Decision, Indicators, MarketSnapshot, Picks, PortfolioState
from .momentum import _score

MIN_BREADTH = 0.5
OPENING_WINDOW_MINUTES = 15
MIN_VOLUME_RATIO = 0.8
OVERBOUGHT_RSI = 70
STRONG_TREND_ATRS = 2.0


def _breadth(overview: Mapping[str, Indicators]) -> float | None:
    """Share of names with both averages defined whose fast is above slow."""
    trends = [
        ind.sma_fast > ind.sma_slow
        for ind in overview.values()
        if ind.sma_fast is not None and ind.sma_slow is not None
    ]
    return sum(trends) / len(trends) if trends else None


def _trend_in_atrs(ind: Indicators) -> float:
    return _score(ind) / (ind.atr_pct / 100)


class MomentumPlusStrategy:
    name = "momentum_plus"

    def __init__(
        self,
        universe: Sequence[str],
        profile: MarketProfile,
        costs: CostModel,
        position_pct: float,
        min_score: float = 0.002,
    ) -> None:
        self._universe = tuple(s.upper() for s in universe)
        self._profile = profile
        self._costs = costs
        self._position_pct = position_pct
        self._min_score = min_score

    def _rejection(self, ind: Indicators, state: PortfolioState) -> str | None:
        """Why this setup is not an entry, or None if it is one."""
        if ind.rsi is not None and ind.rsi >= OVERBOUGHT_RSI:
            return f"overbought (RSI {ind.rsi:.0f})"
        score = _score(ind)
        if score < self._min_score:
            return f"trend score {score:.4f} below threshold"
        if ind.ret_20 is None or ind.ret_20 <= 0:
            return "20-bar return does not confirm the trend"
        if ind.volume_ratio is None or ind.volume_ratio < MIN_VOLUME_RATIO:
            return "volume below normal"
        if ind.atr_pct is None or ind.atr_pct <= 0:
            return "volatility unknown"
        ticket = state.account.equity * self._position_pct
        cost = round_trip_cost_pct(self._costs, ticket, ind.last_price)
        if ind.atr_pct / 100 < cost:
            return f"one ATR ({ind.atr_pct:.2f}%) is below the round-trip cost ({cost:.2%})"
        return None

    def _in_opening_window(self, now: datetime) -> bool:
        here = self._profile.local(now)
        opened = self._profile.open_time
        minutes = (here.hour - opened.hour) * 60 + here.minute - opened.minute
        return minutes < OPENING_WINDOW_MINUTES

    def pick(
        self,
        now: datetime,
        state: PortfolioState,
        overview: Mapping[str, Indicators],
        max_new_positions: int,
    ) -> Picks:
        strategy = "filtered trend-following: breadth, confirmation, cost floor"
        if max_new_positions <= 0:
            return Picks((), "at position limit", abstain=True)
        if self._in_opening_window(now):
            return Picks((), strategy, abstain=True, rationale="opening window: gap still settling")
        breadth = _breadth(overview)
        if breadth is None or breadth < MIN_BREADTH:
            shown = "unknown" if breadth is None else f"{breadth:.0%}"
            return Picks((), strategy, market_mood="bearish", abstain=True,
                         rationale=f"breadth {shown} below {MIN_BREADTH:.0%}")

        candidates = [
            (symbol, ind) for symbol, ind in overview.items()
            if symbol in self._universe and symbol not in state.positions
        ]
        passing = sorted(
            (pair for pair in candidates if self._rejection(pair[1], state) is None),
            key=lambda pair: _trend_in_atrs(pair[1]),
            reverse=True,
        )
        chosen = tuple(symbol for symbol, _ in passing[:max_new_positions])
        return Picks(
            symbols=chosen,
            strategy=strategy,
            market_mood="bullish" if chosen else "neutral",
            abstain=not chosen,
            rationale=f"{len(passing)} of {len(candidates)} candidates cleared every filter "
                      f"(breadth {breadth:.0%})",
        )

    def decide(
        self,
        snapshot: MarketSnapshot,
        strategy_note: str,
        state: PortfolioState,
    ) -> Decision:
        ind = snapshot.indicators

        if snapshot.position is not None:
            broken = (
                ind.sma_fast is not None
                and ind.sma_slow is not None
                and ind.sma_fast < ind.sma_slow
            )
            return Decision(
                symbol=snapshot.symbol,
                action=Action.SELL if broken else Action.HOLD,
                confidence=8 if broken else 5,
                reason="trend broken" if broken else "trend intact; trailing stop manages the exit",
                source="rule",
            )

        rejection = self._rejection(ind, state)
        if rejection is not None:
            return Decision(snapshot.symbol, Action.HOLD, 4, rejection, source="rule")

        atrs = _trend_in_atrs(ind)
        return Decision(
            symbol=snapshot.symbol,
            action=Action.BUY,
            confidence=8 if atrs >= STRONG_TREND_ATRS else 7,
            reason=f"trend {atrs:.1f} ATRs, every filter cleared",
            source="rule",
        )
