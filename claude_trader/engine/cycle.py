"""The trading cycle.

This is the single decision path. Live trading and backtesting both call
``run_cycle``; only the injected broker and market data differ. If the two ever
diverge, the backtest stops being evidence about the live system, so nothing
here is allowed to branch on which mode it is running in.

Order of operations matters and is deliberate:

    risk assessment -> forced exits -> selection -> per-symbol decisions

Exits run before the model is consulted, so a stop can never be argued away.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from ..config import AppConfig
from ..data import indicators as ind_mod
from ..models import (
    Action,
    Bar,
    Decision,
    ExitReason,
    Indicators,
    MarketSnapshot,
    Picks,
    PortfolioState,
    RiskVerdict,
)
from ..risk.engine import RiskEngine, RiskState
from .executor import Executor

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CycleDeps:
    config: AppConfig
    market: object          # MarketDataSource
    broker: object          # Broker
    strategy: object        # Strategy
    risk: RiskEngine
    journal: object         # Journal
    executor: Executor
    run_id: int


@dataclass(frozen=True, slots=True)
class CycleReport:
    ts: datetime
    market_open: bool
    equity: float
    cash: float
    position_count: int
    picks: Picks | None = None
    risk_state: RiskState = field(default_factory=RiskState)
    decisions: tuple[Decision, ...] = ()
    exits: tuple[str, ...] = ()
    entries: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def traded(self) -> int:
        return len(self.exits) + len(self.entries)


def _load_bars(
    market, symbols: Sequence[str], limit: int, now: datetime
) -> dict[str, tuple[Bar, ...]]:
    out: dict[str, tuple[Bar, ...]] = {}
    for symbol in symbols:
        try:
            series = market.bars(symbol, limit, now)
        except Exception as exc:  # one bad symbol must not kill the cycle
            log.warning("bars unavailable for %s: %s", symbol, exc)
            continue
        if series:
            out[symbol] = series
    return out


def _quote_for(market, symbol: str, now: datetime):
    try:
        return market.quote(symbol, now)
    except Exception as exc:
        log.warning("quote unavailable for %s: %s", symbol, exc)
        return None


def _build_snapshot(
    symbol: str,
    bars: tuple[Bar, ...],
    market,
    state: PortfolioState,
    now: datetime,
) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        as_of=now,
        bars=bars,
        quote=_quote_for(market, symbol, now),
        indicators=ind_mod.compute(bars),
        position=state.positions.get(symbol),
        risk=state.risks.get(symbol),
    )


def _peak_equity(journal, run_id: int, current: float) -> float:
    try:
        rows = journal.query(
            "SELECT MAX(equity) AS peak FROM equity_curve WHERE run_id = ?", (run_id,)
        )
    except Exception:
        return current
    peak = rows[0]["peak"] if rows and rows[0]["peak"] is not None else 0.0
    return max(float(peak), current)


def run_cycle(deps: CycleDeps, now: datetime) -> CycleReport:
    cfg = deps.config
    journal = deps.journal
    market = deps.market
    broker = deps.broker

    if not broker.is_market_open(now):
        account = broker.account()
        state = PortfolioState.build(account, broker.positions())
        log.info("Market closed at %s; nothing to do.", now.isoformat())
        return CycleReport(
            ts=now,
            market_open=False,
            equity=account.equity,
            cash=account.cash,
            position_count=state.position_count,
        )

    # --- current book, including bot-managed stop levels ---------------------
    state = PortfolioState.build(
        broker.account(),
        broker.positions(),
        journal.open_position_risks(deps.run_id),
    )
    money = cfg.money
    log.info(
        "Equity %s | cash %s | %d open: %s",
        money(state.account.equity),
        money(state.account.cash),
        state.position_count,
        ", ".join(state.open_symbols) or "none",
    )

    # --- market data ---------------------------------------------------------
    universe = tuple(dict.fromkeys((*cfg.universe, *state.open_symbols)))
    bars_by_symbol = _load_bars(market, universe, cfg.bar_lookback, now)
    if not bars_by_symbol:
        log.error("No market data available this cycle; standing down.")

    overview: dict[str, Indicators] = {
        symbol: ind_mod.compute(series) for symbol, series in bars_by_symbol.items()
    }
    returns = {
        symbol: ind_mod.simple_returns([b.c for b in series][-cfg.risk.correlation_lookback :])
        for symbol, series in bars_by_symbol.items()
    }

    held_snapshots = {
        symbol: _build_snapshot(symbol, bars_by_symbol.get(symbol, ()), market, state, now)
        for symbol in state.open_symbols
    }

    cycle_id = journal.record_cycle(
        deps.run_id, now, state, None, market_open=True
    )

    # --- 1. trail stops upward, then take any forced exit --------------------
    for symbol, snapshot in held_snapshots.items():
        risk = state.risks.get(symbol)
        if risk is not None and snapshot.price > 0:
            journal.upsert_position_risk(deps.run_id, deps.risk.trail(risk, snapshot.price))
    state = PortfolioState.build(
        state.account, state.positions.values(), journal.open_position_risks(deps.run_id)
    )

    exits: list[str] = []
    for forced in deps.risk.forced_exits(state, held_snapshots, now):
        state, result = deps.executor.close_position(
            state,
            forced.symbol,
            held_snapshots.get(forced.symbol),
            forced.reason,
            forced.detail,
            cycle_id,
        )
        if result is not None:
            exits.append(forced.symbol)

    # --- 2. circuit breakers -------------------------------------------------
    trades_today = journal.trades_today(deps.run_id, now)
    risk_state = deps.risk.assess_portfolio(
        state,
        _peak_equity(journal, deps.run_id, state.account.equity),
        trades_today,
        deps.executor.trades_this_cycle,
    )
    if deps.risk.square_off_due(now) and not risk_state.halted:
        # Past the intraday cut-off a new entry could only be carried overnight,
        # which is a different segment with different costs and a different
        # margin. Refusing is cheaper than discovering that at 09:15 tomorrow.
        risk_state = RiskState(True, "past the intraday square-off cut-off")
    if risk_state.halted:
        log.warning("New entries blocked: %s", risk_state.reason)

    # --- 3. selection --------------------------------------------------------
    room = max(0, cfg.risk.max_positions - state.position_count)
    picks = Picks((), "entries blocked", abstain=True, rationale=risk_state.reason)
    if risk_state.may_open and room > 0 and overview:
        picks = deps.strategy.pick(now, state, overview, room)
        log.info(
            "Selection: %s | mood=%s | %s",
            ", ".join(picks.symbols) or "none (abstained)",
            picks.market_mood,
            picks.strategy,
        )

    # --- 4. per-symbol decisions --------------------------------------------
    review: list[str] = [s for s in state.open_symbols if s not in exits]
    review += [s for s in picks.symbols if s not in review]

    decisions: list[Decision] = []
    entries: list[str] = []
    skipped: list[tuple[str, str]] = []

    for symbol in review:
        series = bars_by_symbol.get(symbol, ())
        if not series:
            skipped.append((symbol, "no bars"))
            continue

        snapshot = _build_snapshot(symbol, series, market, state, now)
        try:
            decision = deps.strategy.decide(snapshot, picks.strategy, state)
        except Exception as exc:
            log.error("decision failed for %s: %s", symbol, exc)
            skipped.append((symbol, f"decision error: {exc}"))
            continue

        decisions.append(decision)
        is_held = symbol in state.positions

        verdict = None
        if decision.action is Action.BUY:
            verdict = (
                deps.risk.approve_entry(decision, snapshot, state, returns, now)
                if risk_state.may_open
                else RiskVerdict(False, risk_state.reason)
            )

        decision_id = journal.record_decision(
            deps.run_id,
            cycle_id,
            now,
            decision,
            snapshot.price,
            indicators=_indicator_dict(snapshot.indicators),
            verdict=verdict,
            prompt_sha=getattr(deps.strategy, "last_prompt_sha", ""),
            # What the model was shown, stored beside what it decided. Without
            # this, "why did it buy that" is unanswerable after the fact.
            news=[h.title for h in
                  getattr(deps.strategy, "last_headlines", ())],
        )

        if decision.action is Action.SELL and is_held:
            state, result = deps.executor.close_position(
                state,
                symbol,
                snapshot,
                ExitReason.MODEL_EXIT,
                decision.reason,
                cycle_id,
                decision_id,
            )
            if result is not None:
                exits.append(symbol)
            continue

        if decision.action is Action.BUY and verdict is not None:
            if not verdict.approved:
                log.info("%s: entry declined -- %s", symbol, verdict.reason)
                skipped.append((symbol, verdict.reason))
                continue
            risk_record = deps.risk.initial_risk(
                symbol, snapshot.price, snapshot, now, verdict
            )
            state, result = deps.executor.open_position(
                state, decision, verdict, snapshot, risk_record, cycle_id, decision_id, now
            )
            if result is not None:
                entries.append(symbol)
            continue

        log.info(
            "%s: HOLD (confidence %d) -- %s", symbol, decision.confidence, decision.reason
        )

    # --- 5. record where we ended up ----------------------------------------
    benchmark = _benchmark_price(market, now, cfg.benchmark)
    journal.record_equity(
        deps.run_id,
        now,
        state.account.equity,
        state.account.cash,
        state.exposure_ratio,
        benchmark,
    )
    journal.query(
        "UPDATE cycles SET picks_json = ?, strategy_note = ?, market_mood = ?,"
        " halted = ?, halt_reason = ?, abstained = ?, equity = ?, cash = ?,"
        " position_count = ? WHERE id = ?",
        (
            json.dumps(list(picks.symbols)),
            picks.strategy,
            picks.market_mood,
            int(risk_state.halted),
            risk_state.reason,
            int(picks.abstain),
            state.account.equity,
            state.account.cash,
            state.position_count,
            cycle_id,
        ),
    )

    return CycleReport(
        ts=now,
        market_open=True,
        equity=state.account.equity,
        cash=state.account.cash,
        position_count=state.position_count,
        picks=picks,
        risk_state=risk_state,
        decisions=tuple(decisions),
        exits=tuple(exits),
        entries=tuple(entries),
        skipped=tuple(skipped),
    )


def _indicator_dict(ind: Indicators) -> dict[str, object]:
    return {
        "last_price": ind.last_price,
        "sma_fast": ind.sma_fast,
        "sma_slow": ind.sma_slow,
        "rsi": ind.rsi,
        "atr": ind.atr,
        "atr_pct": ind.atr_pct,
        "ret_5": ind.ret_5,
        "ret_20": ind.ret_20,
        "volume_ratio": ind.volume_ratio,
        "trend": ind.trend,
    }


def _benchmark_price(market, now: datetime, symbol: str) -> float | None:
    """Whatever index the market profile nominates -- SPY on US, NIFTYBEES on
    NSE. Without it the equity curve has nothing to be measured against, which
    is the difference between a track record and a number."""
    if not symbol:
        return None
    try:
        prices = market.latest_prices([symbol], now)
    except Exception:
        return None
    value = prices.get(symbol)
    return float(value) if value else None
