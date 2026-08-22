"""Deterministic risk layer.

Nothing in this module consults the model. It decides three things:

1. whether new exposure is permitted at all this cycle (circuit breakers)
2. which open positions must be closed regardless of any opinion (stops)
3. how large an approved entry may be, and where its stop sits

Sizing is volatility-based rather than a flat cap: risking a fixed fraction of
equity between entry and stop means a quiet name gets more capital than a wild
one for the same risk, which a flat notional cannot express.

It is also market-aware. On NSE a position is a whole number of shares and a
round trip surrenders real money to STT, stamp duty and GST before the idea is
tested, so an entry whose expected move cannot clear its own transaction costs
is rejected here rather than discovered later in the P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Sequence

from ..config import RiskConfig
from ..costs import CostModel, NoCosts, round_trip_cost_pct
from ..data.indicators import correlation
from ..markets import MarketProfile, get_market
from ..models import (
    Action,
    Decision,
    ExitReason,
    MarketSnapshot,
    PortfolioState,
    PositionRisk,
    RiskVerdict,
)


@dataclass(frozen=True, slots=True)
class RiskState:
    """Whether the book may grow this cycle. Exits are always permitted."""

    halted: bool = False
    reason: str = ""

    @property
    def may_open(self) -> bool:
        return not self.halted


@dataclass(frozen=True, slots=True)
class ForcedExit:
    symbol: str
    reason: ExitReason
    detail: str


class RiskEngine:
    def __init__(
        self,
        config: RiskConfig,
        profile: MarketProfile | None = None,
        costs: CostModel | None = None,
        sector_lookup: Callable[[str], str] | None = None,
    ) -> None:
        self._cfg = config
        self._profile = profile or get_market()
        self._costs: CostModel = costs or NoCosts()
        self._sector = sector_lookup or self._profile.sector_of

    @property
    def profile(self) -> MarketProfile:
        return self._profile

    def _money(self, amount: float) -> str:
        return self._profile.money(amount)

    # ------------------------------------------------------- circuit breakers
    def assess_portfolio(
        self,
        state: PortfolioState,
        peak_equity: float,
        trades_today: int,
        trades_this_cycle: int = 0,
    ) -> RiskState:
        cfg = self._cfg
        equity = state.account.equity

        day_pl = state.account.day_pl_pct
        if day_pl <= -cfg.daily_loss_limit_pct:
            return RiskState(
                True,
                f"daily loss limit hit ({day_pl * 100:.2f}% vs "
                f"-{cfg.daily_loss_limit_pct * 100:.2f}%)",
            )

        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown >= cfg.max_drawdown_pct:
                return RiskState(
                    True,
                    f"max drawdown breached ({drawdown * 100:.2f}% from peak "
                    f"{self._money(peak_equity)})",
                )

        if trades_today >= cfg.max_trades_per_day:
            return RiskState(True, f"daily trade cap reached ({trades_today})")

        if trades_this_cycle >= cfg.max_trades_per_cycle:
            return RiskState(True, f"per-cycle trade cap reached ({trades_this_cycle})")

        if state.position_count >= cfg.max_positions:
            return RiskState(True, f"at max positions ({state.position_count})")

        return RiskState(False, "")

    # ------------------------------------------------------------- hard exits
    def square_off_due(self, now: datetime | None) -> bool:
        """True once the intraday cut-off has passed.

        NSE brokers close open intraday positions themselves at around 15:20 and
        bill for the privilege. Exiting first is both cheaper and the only way
        the intraday cost model stays honest about what was actually paid.
        """
        if not self._cfg.square_off_enabled or now is None:
            return False
        local = self._profile.local(now)
        close = self._profile.close_time
        cutoff_minutes = (
            close.hour * 60 + close.minute - self._cfg.square_off_minutes_before_close
        )
        return local.hour * 60 + local.minute >= cutoff_minutes

    def forced_exits(
        self,
        state: PortfolioState,
        snapshots: Mapping[str, MarketSnapshot],
        now: datetime | None = None,
    ) -> tuple[ForcedExit, ...]:
        """Stops, targets and time stops. Checked before the model is consulted
        so a confident narrative can never talk the bot out of an exit."""
        out: list[ForcedExit] = []

        if self.square_off_due(now) and state.positions:
            cutoff = self._cfg.square_off_minutes_before_close
            return tuple(
                ForcedExit(
                    symbol,
                    ExitReason.SQUARE_OFF,
                    f"intraday square-off, {cutoff}m before close",
                )
                for symbol in sorted(state.positions)
            )

        for symbol, position in state.positions.items():
            risk = state.risks.get(symbol)
            snapshot = snapshots.get(symbol)
            price = snapshot.price if snapshot else position.current_price
            if price <= 0:
                continue

            if risk is None:
                # Untracked position (e.g. opened before this run): enforce the
                # blunt percentage backstop rather than leaving it unguarded.
                loss = position.unrealized_plpc
                if loss <= -self._cfg.hard_stop_pct:
                    out.append(
                        ForcedExit(
                            symbol,
                            ExitReason.STOP_LOSS,
                            f"untracked position down {loss * 100:.2f}%",
                        )
                    )
                continue

            if price <= risk.stop_price:
                reason = (
                    ExitReason.TRAILING_STOP
                    if risk.stop_price > risk.entry_price
                    else ExitReason.STOP_LOSS
                )
                out.append(
                    ForcedExit(
                        symbol,
                        reason,
                        f"price {self._money(price)} at or below stop {self._money(risk.stop_price)}",
                    )
                )
            elif risk.target_price > 0 and price >= risk.target_price:
                out.append(
                    ForcedExit(
                        symbol,
                        ExitReason.TAKE_PROFIT,
                        f"price {self._money(price)} reached target {self._money(risk.target_price)}",
                    )
                )
            elif risk.bars_held >= self._cfg.max_holding_bars:
                out.append(
                    ForcedExit(
                        symbol,
                        ExitReason.TIME_STOP,
                        f"held {risk.bars_held} bars without resolution",
                    )
                )
        return tuple(out)

    # ---------------------------------------------------------- entry gating
    def approve_entry(
        self,
        decision: Decision,
        snapshot: MarketSnapshot,
        state: PortfolioState,
        returns: Mapping[str, Sequence[float]] | None = None,
        now: datetime | None = None,
    ) -> RiskVerdict:
        cfg = self._cfg
        symbol = decision.symbol

        if decision.action is not Action.BUY:
            return RiskVerdict(False, "not a buy")

        if decision.confidence < cfg.min_confidence:
            return RiskVerdict(
                False, f"confidence {decision.confidence} below {cfg.min_confidence}"
            )

        price = snapshot.price
        if price <= 0:
            return RiskVerdict(False, "no usable price")

        quote = snapshot.quote
        if quote is None:
            # No quote means the feed gave us nothing to check freshness or
            # spread against. Opening on that is trading on an assumption; the
            # exit path is deliberately unaffected, because being unable to
            # price a position is not a reason to keep holding it.
            return RiskVerdict(False, "no quote to price the entry against")
        if now is not None and quote.age_seconds(now) > cfg.max_quote_age_seconds:
            return RiskVerdict(
                False,
                f"quote is {quote.age_seconds(now):.0f}s old "
                f"(limit {cfg.max_quote_age_seconds}s)",
            )
        # A modelled spread is an assumption of ours, not an observation of
        # the book, so rejecting on it would only ever reject our own guess.
        if not quote.modelled and quote.spread_bps > cfg.max_spread_bps:
            return RiskVerdict(
                False,
                f"spread {quote.spread_bps:.1f}bps exceeds {cfg.max_spread_bps}bps",
            )

        if symbol in state.positions:
            if state.exposure_pct(symbol) >= cfg.max_position_pct:
                return RiskVerdict(False, "position already at size cap")
        elif state.position_count >= cfg.max_positions:
            return RiskVerdict(False, f"at max positions ({cfg.max_positions})")

        sector_verdict = self._check_sector(symbol, state)
        if sector_verdict is not None:
            return sector_verdict

        corr_verdict = self._check_correlation(symbol, state, returns or {})
        if corr_verdict is not None:
            return corr_verdict

        return self._size(decision, snapshot, state, price)

    # ------------------------------------------------------------- internals
    def _check_sector(self, symbol: str, state: PortfolioState) -> RiskVerdict | None:
        cfg = self._cfg
        sector = self._sector(symbol)
        peers = [s for s in state.positions if self._sector(s) == sector and s != symbol]
        if len(peers) >= cfg.max_sector_positions:
            return RiskVerdict(
                False,
                f"sector {sector} already holds {len(peers)} positions "
                f"(limit {cfg.max_sector_positions})",
            )
        equity = state.account.equity
        if equity > 0:
            sector_value = sum(state.positions[s].market_value for s in peers)
            if sector_value / equity >= cfg.max_sector_pct:
                return RiskVerdict(
                    False,
                    f"sector {sector} at {sector_value / equity * 100:.1f}% of equity "
                    f"(limit {cfg.max_sector_pct * 100:.0f}%)",
                )
        return None

    def _check_correlation(
        self,
        symbol: str,
        state: PortfolioState,
        returns: Mapping[str, Sequence[float]],
    ) -> RiskVerdict | None:
        candidate = returns.get(symbol)
        if not candidate:
            return None
        for held in state.positions:
            other = returns.get(held)
            if not other:
                continue
            rho = correlation(candidate, other)
            if rho is not None and rho >= self._cfg.max_correlation:
                return RiskVerdict(
                    False,
                    f"correlation {rho:.2f} with open position {held} "
                    f"exceeds {self._cfg.max_correlation:.2f}",
                )
        return None

    def stop_distance(self, snapshot: MarketSnapshot, price: float) -> float:
        atr = snapshot.indicators.atr
        if atr and atr > 0:
            return atr * self._cfg.atr_stop_multiple
        return price * self._cfg.hard_stop_pct

    def _size(
        self,
        decision: Decision,
        snapshot: MarketSnapshot,
        state: PortfolioState,
        price: float,
    ) -> RiskVerdict:
        cfg = self._cfg
        equity = state.account.equity
        distance = self.stop_distance(snapshot, price)
        if distance <= 0:
            return RiskVerdict(False, "cannot determine a stop distance")

        risk_budget = equity * cfg.risk_per_trade_pct
        by_risk = (risk_budget / distance) * price if distance > 0 else 0.0

        deployable = max(0.0, state.account.cash - equity * cfg.min_cash_reserve_pct)

        existing = state.positions.get(decision.symbol)
        existing_value = existing.market_value if existing is not None else 0.0
        headroom = max(0.0, equity * cfg.max_position_pct - existing_value)

        caps = {
            "volatility budget": by_risk,
            "per-trade cap": cfg.max_notional_per_trade,
            "position size cap": headroom,
            "available cash": deployable,
        }
        if decision.notional and decision.notional > 0:
            caps["model suggestion"] = float(decision.notional)

        binding, budget = min(caps.items(), key=lambda kv: kv[1])
        budget = max(0.0, budget)

        # Round to what the market will actually accept before judging the size.
        # On NSE that is a whole number of shares, so a budget of Rs 900 against a
        # Rs 500 share is a Rs 500 trade, not a Rs 900 one, and every downstream
        # check has to be told the truth about that.
        qty = self._profile.round_qty(budget / price)
        if qty <= 0:
            return RiskVerdict(
                False,
                f"{self._money(budget)} does not cover one tradeable lot at "
                f"{self._money(price)} (binding: {binding})",
            )

        notional = round(qty * price, 2)
        if notional < cfg.min_trade_notional:
            return RiskVerdict(
                False,
                f"sized below minimum trade ({self._money(notional)}, binding: {binding})",
            )

        stop = round(price - distance, 4)
        target = round(
            price + distance / cfg.atr_stop_multiple * cfg.atr_target_multiple, 4
        )

        # The cost hurdle. An idea that has to clear its own STT, stamp duty and
        # GST before it earns anything is not an edge, and this is the only gate
        # that can see that -- the model is not given the fee schedule.
        expected_move = (target - price) / price
        if expected_move > 0:
            cost_pct = round_trip_cost_pct(self._costs, notional, price)
            if cost_pct > expected_move * cfg.max_cost_ratio:
                return RiskVerdict(
                    False,
                    f"round-trip cost {cost_pct * 100:.3f}% eats "
                    f"{cost_pct / expected_move * 100:.0f}% of the {expected_move * 100:.2f}% "
                    f"target (limit {cfg.max_cost_ratio * 100:.0f}%)",
                )

        return RiskVerdict(
            approved=True,
            reason=(
                f"sized {qty:g} @ {self._money(price)} = {self._money(notional)} "
                f"(binding constraint: {binding})"
            ),
            notional=notional,
            qty=qty,
            stop_price=max(0.0, stop),
            target_price=target,
        )

    def initial_risk(
        self,
        symbol: str,
        entry_price: float,
        snapshot: MarketSnapshot,
        now: datetime,
        verdict: RiskVerdict,
    ) -> PositionRisk:
        return PositionRisk(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=now,
            stop_price=verdict.stop_price,
            target_price=verdict.target_price,
            high_water=entry_price,
            atr_at_entry=snapshot.indicators.atr or 0.0,
            bars_held=0,
        )

    def trail(self, risk: PositionRisk, price: float) -> PositionRisk:
        return risk.advanced(price, self._cfg.trailing_stop_atr)
