"""Order execution and journalling.

Two corrections to the original behaviour live here:

* every fill produces a NEW PortfolioState, so cash and the position count stay
  correct across multiple trades within one cycle instead of being read once
  before the loop and then trusted
* exits are quantity-based (or full liquidations), because Alpaca rejects a
  notional sell against a fractional position

Entries are sent as a quantity on markets without fractional shares (NSE) and as
a notional amount where they exist (US), because a notional order on a whole-share
market silently becomes a rounding decision made by someone else.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

from ..errors import BrokerError, OrderRejected
from ..markets import MarketProfile, get_market
from ..models import (
    Decision,
    ExitReason,
    MarketSnapshot,
    OrderRequest,
    OrderResult,
    PortfolioState,
    PositionRisk,
    RiskVerdict,
    Side,
)

log = logging.getLogger(__name__)


class _Broker(Protocol):
    def submit(self, order: OrderRequest) -> OrderResult | None: ...
    def close_position(self, symbol: str) -> OrderResult | None: ...


class Executor:
    """Places orders, records them, and returns the updated portfolio state."""

    def __init__(
        self,
        broker: _Broker,
        journal,
        run_id: int,
        dry_run: bool = False,
        profile: MarketProfile | None = None,
    ) -> None:
        self._broker = broker
        self._journal = journal
        self._run_id = run_id
        self._dry_run = dry_run
        self._profile = profile or get_market()
        self.trades_this_cycle = 0

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def _money(self, amount: float) -> str:
        return self._profile.money(amount)

    def open_position(
        self,
        state: PortfolioState,
        decision: Decision,
        verdict: RiskVerdict,
        snapshot: MarketSnapshot,
        risk: PositionRisk,
        cycle_id: int,
        decision_id: int | None,
        now: datetime,
    ) -> tuple[PortfolioState, OrderResult | None]:
        if self._dry_run:
            log.info(
                "[DRY RUN] would BUY %s of %s (%g sh, stop %s, target %s)",
                self._money(verdict.notional),
                decision.symbol,
                verdict.qty,
                self._money(verdict.stop_price),
                self._money(verdict.target_price),
            )
            return state, None

        fractional = self._profile.fractional_shares
        order = OrderRequest(
            symbol=decision.symbol,
            side=Side.BUY,
            qty=None if fractional else verdict.qty,
            notional=round(verdict.notional, 2) if fractional else None,
            intent="entry",
            client_order_id=f"ct-{cycle_id}-{decision.symbol}-buy",
        )
        try:
            result = self._broker.submit(order)
        except (BrokerError, OrderRejected) as exc:
            log.error("BUY %s rejected: %s", decision.symbol, exc)
            return state, None

        if result is None:
            return state, None

        result = self._with_price_fallback(result, snapshot.price)
        self.trades_this_cycle += 1

        entry_risk = (
            risk
            if result.price <= 0
            else PositionRisk(
                symbol=risk.symbol,
                entry_price=result.price,
                entry_time=now,
                stop_price=verdict.stop_price,
                target_price=verdict.target_price,
                high_water=result.price,
                atr_at_entry=risk.atr_at_entry,
                bars_held=0,
            )
        )

        new_state = state.with_fill(
            result.symbol, Side.BUY, result.qty, result.price, entry_risk
        )
        self._journal.record_order(self._run_id, cycle_id, result, decision_id, "entry")
        self._journal.upsert_position_risk(self._run_id, entry_risk)
        if decision_id is not None:
            self._journal.mark_decision_executed(decision_id)

        log.info(
            "BUY %s: %g @ %s (%s) stop %s target %s",
            result.symbol,
            result.qty,
            self._money(result.price),
            self._money(result.notional),
            self._money(entry_risk.stop_price),
            self._money(entry_risk.target_price),
        )
        return new_state, result

    def close_position(
        self,
        state: PortfolioState,
        symbol: str,
        snapshot: MarketSnapshot | None,
        reason: ExitReason,
        detail: str,
        cycle_id: int,
        decision_id: int | None = None,
    ) -> tuple[PortfolioState, OrderResult | None]:
        position = state.positions.get(symbol)
        if position is None:
            return state, None

        if self._dry_run:
            log.info("[DRY RUN] would EXIT %s (%s: %s)", symbol, reason.value, detail)
            return state, None

        try:
            result = self._broker.close_position(symbol)
        except (BrokerError, OrderRejected) as exc:
            log.error("EXIT %s failed: %s", symbol, exc)
            return state, None

        if result is None:
            log.warning("EXIT %s returned no fill", symbol)
            return state, None

        reference = snapshot.price if snapshot else position.current_price
        result = self._with_price_fallback(result, reference)
        qty = result.qty or position.qty
        self.trades_this_cycle += 1

        new_state = state.with_fill(symbol, Side.SELL, qty, result.price)
        self._journal.record_order(
            self._run_id, cycle_id, result, decision_id, reason.value
        )
        self._journal.close_position_risk(self._run_id, symbol)
        if decision_id is not None:
            self._journal.mark_decision_executed(decision_id)

        pl = (result.price - position.avg_entry_price) * qty
        log.info(
            "EXIT %s (%s): %g @ %s, realised %s -- %s",
            symbol,
            reason.value,
            qty,
            self._money(result.price),
            self._money(pl),
            detail,
        )
        return new_state, result

    @staticmethod
    def _with_price_fallback(result: OrderResult, reference: float) -> OrderResult:
        """A freshly accepted market order often has no fill price yet. Fall back
        to the reference price so the journal is not polluted with zeros."""
        if result.price > 0 or reference <= 0:
            return result
        qty = result.qty or 0.0
        return OrderResult(
            symbol=result.symbol,
            side=result.side,
            qty=qty,
            price=reference,
            order_id=result.order_id,
            status=result.status or "accepted",
            submitted_at=result.submitted_at,
            simulated=result.simulated,
        )
