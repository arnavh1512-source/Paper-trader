"""Claude-driven strategy.

Guards applied to every model response, because a plausible sentence is not
evidence:

* symbols outside the configured universe are dropped
* a sell on a symbol that is not held is downgraded to hold
* a buy with no invalidation level is downgraded to hold
* transport or schema failures degrade to hold, never to a guess
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Mapping, Sequence

from ..costs import CostModel, round_trip_cost_pct
from ..errors import LLMError, LLMResponseError
from ..markets import MarketProfile, get_market
from ..models import Action, Decision, Indicators, MarketSnapshot, Picks, PortfolioState
from ..llm.client import prompt_fingerprint
from ..llm.prompts import (
    DECIDER_SYSTEM,
    PICKER_SYSTEM,
    build_decision_prompt,
    build_picker_prompt,
)
from ..llm.schemas import parse_decision, parse_picks

log = logging.getLogger(__name__)


class ClaudeStrategy:
    name = "claude"

    def __init__(
        self,
        client,
        universe: Sequence[str],
        require_invalidation: bool = True,
        profile: MarketProfile | None = None,
        segment: str = "",
        costs: CostModel | None = None,
    ) -> None:
        self._client = client
        self._universe = tuple(s.upper() for s in universe)
        self._require_invalidation = require_invalidation
        self._profile = profile or get_market()
        self._segment = segment
        self._costs = costs
        self.last_prompt_sha = ""

    def _cost_floor(self, snapshot: MarketSnapshot) -> float:
        """What a round trip on a typical ticket costs, as a fraction.

        Shown to the model so it stops proposing moves smaller than the fee, and
        computed at the profile's per-trade cap because that is the ticket size
        the sizing layer is most likely to land on.
        """
        if self._costs is None or snapshot.price <= 0:
            return 0.0
        return round_trip_cost_pct(
            self._costs, self._profile.max_per_trade, snapshot.price
        )

    @property
    def client(self):
        """Exposed so callers can report token usage and cache hit rates."""
        return self._client

    # ------------------------------------------------------------------ pick
    def pick(
        self,
        now: datetime,
        state: PortfolioState,
        overview: Mapping[str, Indicators],
        max_new_positions: int,
    ) -> Picks:
        if max_new_positions <= 0:
            return Picks(
                symbols=(),
                strategy="at position limit; managing existing book only",
                abstain=True,
                rationale="no room for new positions",
            )

        prompt = build_picker_prompt(
            now,
            state,
            overview,
            self._universe,
            max_new_positions,
            profile=self._profile,
            segment=self._segment,
        )
        try:
            raw = self._client.complete(PICKER_SYSTEM, prompt, max_tokens=400)
            parsed = parse_picks(raw)
        except (LLMError, LLMResponseError) as exc:
            log.warning("stock picking failed (%s); abstaining this cycle", exc)
            return Picks(
                symbols=(),
                strategy="model unavailable",
                abstain=True,
                rationale=str(exc)[:200],
            )

        allowed = [s for s in parsed.symbols if s in self._universe]
        dropped = [s for s in parsed.symbols if s not in self._universe]
        if dropped:
            log.warning("dropping out-of-universe picks: %s", ", ".join(dropped))

        return Picks(
            symbols=tuple(allowed[:max_new_positions]),
            strategy=parsed.strategy,
            market_mood=parsed.market_mood,
            abstain=parsed.abstain or not allowed,
            rationale=parsed.rationale,
        )

    # ---------------------------------------------------------------- decide
    def decide(
        self,
        snapshot: MarketSnapshot,
        strategy_note: str,
        state: PortfolioState,
    ) -> Decision:
        prompt = build_decision_prompt(
            snapshot,
            strategy_note,
            state,
            profile=self._profile,
            segment=self._segment,
            round_trip_cost_pct=self._cost_floor(snapshot),
        )
        self.last_prompt_sha = prompt_fingerprint(
            getattr(self._client, "model", "unknown"), DECIDER_SYSTEM, prompt
        )
        try:
            raw = self._client.complete(DECIDER_SYSTEM, prompt, max_tokens=400)
            parsed = parse_decision(raw)
        except (LLMError, LLMResponseError) as exc:
            log.warning("decision failed for %s (%s); holding", snapshot.symbol, exc)
            return Decision(
                symbol=snapshot.symbol,
                action=Action.HOLD,
                confidence=0,
                reason=f"model error: {exc}"[:200],
                source="error",
            )

        action = Action(parsed.action)
        reason = parsed.reason

        if action is Action.SELL and snapshot.position is None:
            log.info("%s: model proposed a sell with no position; holding", snapshot.symbol)
            action, reason = Action.HOLD, "no position to sell"

        if (
            action is Action.BUY
            and self._require_invalidation
            and not parsed.invalidation.strip()
        ):
            log.info("%s: buy proposed without an invalidation level; holding", snapshot.symbol)
            action, reason = Action.HOLD, "no invalidation level supplied"

        if parsed.invalidation.strip() and action is not Action.HOLD:
            reason = f"{reason} [invalidation: {parsed.invalidation.strip()}]"

        return Decision(
            symbol=snapshot.symbol,
            action=action,
            confidence=parsed.confidence,
            reason=reason[:400],
            notional=parsed.notional,
            horizon_bars=parsed.horizon_bars,
            source="model",
        )
