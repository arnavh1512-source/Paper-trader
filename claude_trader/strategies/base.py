"""Strategy protocol.

A strategy only ever *proposes*. It cannot size a position, set a stop, or
bypass a limit -- that authority belongs to the risk layer. Keeping the
interface this narrow is what allows the Claude strategy and a dumb momentum
rule to be swapped and compared on identical footing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Protocol, runtime_checkable

from ..models import Decision, Indicators, MarketSnapshot, Picks, PortfolioState


@runtime_checkable
class Strategy(Protocol):
    name: str

    def pick(
        self,
        now: datetime,
        state: PortfolioState,
        overview: Mapping[str, Indicators],
        max_new_positions: int,
    ) -> Picks: ...

    def decide(
        self,
        snapshot: MarketSnapshot,
        strategy_note: str,
        state: PortfolioState,
    ) -> Decision: ...
