"""Backtest driver.

This module owns the clock and nothing else. Every decision is made by the same
``run_cycle`` the live bot uses, against the same risk engine, with the same
journal writes. The backtester's only privileges are that it advances time and
that its broker fills from bars.

That constraint is deliberate and worth defending: the moment the backtester
gets its own decision logic, its results stop being evidence about the thing
that trades real money.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

from ..analytics.calibration import DEFAULT_HORIZONS, Calibration, calibrate, resolve_outcomes
from ..analytics.metrics import Performance, compute_performance
from ..config import AppConfig
from ..costs import build_cost_model
from ..data.sources import HistoricalMarketData
from ..engine.cycle import CycleDeps, CycleReport, run_cycle
from ..engine.executor import Executor
from ..journal.store import Journal
from ..risk.engine import RiskEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    run_id: int
    label: str
    cycles: int
    orders: int
    performance: Performance
    calibration: Calibration | None
    final_equity: float
    llm_calls: int = 0
    llm_cache_hits: int = 0


def _orders_for(journal: Journal, run_id: int) -> list[dict]:
    return [
        {
            "symbol": r["symbol"],
            "side": r["side"],
            "qty": r["qty"],
            "price": r["price"],
            "ts": r["ts"],
        }
        for r in journal.query(
            "SELECT symbol, side, qty, price, ts FROM orders WHERE run_id = ? ORDER BY ts, id",
            (run_id,),
        )
    ]


def run_backtest(
    config: AppConfig,
    market: HistoricalMarketData,
    strategy,
    journal: Journal,
    *,
    label: str = "backtest",
    starting_cash: float | None = None,
    warmup_bars: int = 40,
    fill_model=None,
    calibrate_horizon: int | None = 26,
    progress: Callable[[int, int, CycleReport], None] | None = None,
) -> BacktestResult:
    from ..brokers.simulated import SimulatedBroker  # local: keeps live path lean

    timeline = market.timeline()
    if len(timeline) <= warmup_bars:
        raise ValueError(
            f"timeline has {len(timeline)} bars; need more than the {warmup_bars}-bar warmup"
        )

    starting_cash = config.starting_cash if starting_cash is None else starting_cash
    # The broker charges what the market charges. Without this the Indian
    # backtest reports profits that the statutory charges would have eaten.
    costs = build_cost_model(config.market, config.segment)
    broker = SimulatedBroker(
        market,
        starting_cash=starting_cash,
        fill_model=fill_model,
        costs=costs,
        profile=config.profile,
    )
    run_id = journal.start_run(
        kind="backtest",
        strategy=getattr(strategy, "name", label),
        started_at=timeline[warmup_bars],
        config={
            "label": label,
            "starting_cash": starting_cash,
            "warmup_bars": warmup_bars,
            "symbols": list(market.symbols),
            "timeframe": config.timeframe,
            "market": config.market,
            "segment": config.segment,
            "currency": config.profile.currency,
            "risk": config.risk.as_dict(),
        },
        notes=label,
    )

    deps = CycleDeps(
        config=config,
        market=market,
        broker=broker,
        strategy=strategy,
        risk=RiskEngine(config.risk, profile=config.profile, costs=costs),
        journal=journal,
        executor=Executor(
            broker, journal, run_id, dry_run=False, profile=config.profile
        ),
        run_id=run_id,
    )

    steps = timeline[warmup_bars:]
    total = len(steps)
    for index, now in enumerate(steps, start=1):
        broker.set_clock(now)
        deps.executor.trades_this_cycle = 0
        report = run_cycle(deps, now)
        if progress is not None:
            progress(index, total, report)
        elif index % 200 == 0 or index == total:
            log.info(
                "[%s] %d/%d %s equity %s (%d open)",
                label,
                index,
                total,
                now.date(),
                config.money(report.equity),
                report.position_count,
            )

    journal.finish_run(run_id, steps[-1])
    journal.commit()

    calibration = None
    if calibrate_horizon:
        horizons = tuple({*DEFAULT_HORIZONS, calibrate_horizon})
        resolved = resolve_outcomes(
            journal, run_id, market, horizons, benchmark_symbol=config.benchmark
        )
        journal.commit()
        log.info("[%s] resolved %d decision outcomes", label, resolved)
        calibration = calibrate(journal, run_id, calibrate_horizon)

    performance = compute_performance(
        journal.equity_curve(run_id),
        _orders_for(journal, run_id),
        periods_per_year=config.periods_per_year,
    )
    client = getattr(strategy, "client", None)

    return BacktestResult(
        run_id=run_id,
        label=label,
        cycles=total,
        orders=len(broker.orders),
        performance=performance,
        calibration=calibration,
        final_equity=broker.equity(),
        llm_calls=getattr(client, "calls_made", 0),
        llm_cache_hits=getattr(client, "cache_hits", 0),
    )


def run_buy_and_hold(
    config: AppConfig,
    market: HistoricalMarketData,
    journal: Journal,
    *,
    symbol: str | None = None,
    starting_cash: float | None = None,
    warmup_bars: int = 40,
    fill_model=None,
) -> BacktestResult:
    """The line every strategy has to beat.

    Buys once at the first tradeable bar and holds, paying the same slippage and
    the same statutory charges the strategy pays. No stops, no opinions.
    """
    from ..brokers.simulated import SimulatedBroker
    from ..models import OrderRequest, Side

    symbol = symbol or config.benchmark
    starting_cash = config.starting_cash if starting_cash is None else starting_cash
    if symbol not in market.symbols:
        raise ValueError(f"{symbol} is not in the dataset; cannot compute buy-and-hold")

    timeline = market.timeline()
    steps = timeline[warmup_bars:]
    if not steps:
        raise ValueError("not enough bars for a buy-and-hold run")

    broker = SimulatedBroker(
        market,
        starting_cash=starting_cash,
        fill_model=fill_model,
        costs=build_cost_model(config.market, config.segment),
        profile=config.profile,
    )
    run_id = journal.start_run(
        kind="backtest",
        strategy="buy_and_hold",
        started_at=steps[0],
        config={
            "symbol": symbol,
            "starting_cash": starting_cash,
            "market": config.market,
        },
        notes=f"buy and hold {symbol}",
    )

    bought = False
    budget = round(starting_cash * 0.999, 2)
    fractional = config.profile.fractional_shares
    for now in steps:
        broker.set_clock(now)
        if not bought:
            # Where shares are indivisible the benchmark has to buy whole ones
            # too, or it gets a fill the strategy could never have had.
            qty = None
            if not fractional:
                price = broker.price_of(symbol)
                qty = config.profile.round_qty(budget / price) if price > 0 else 0.0
                if qty <= 0:
                    continue
            result = broker.submit(
                OrderRequest(
                    symbol=symbol,
                    side=Side.BUY,
                    qty=qty,
                    notional=None if qty else budget,
                    intent="benchmark entry",
                )
            )
            if result is not None:
                bought = True
                cycle_id = journal.record_cycle(
                    run_id, now, _flat_state(broker), None, market_open=True
                )
                journal.record_order(run_id, cycle_id, result, None, "benchmark entry")

        price = broker.price_of(symbol)
        journal.record_equity(
            run_id,
            now,
            broker.equity(),
            broker.account().cash,
            1.0 if bought else 0.0,
            price or None,
        )

    journal.finish_run(run_id, steps[-1])
    journal.commit()

    performance = compute_performance(
        journal.equity_curve(run_id),
        _orders_for(journal, run_id),
        periods_per_year=config.periods_per_year,
    )
    return BacktestResult(
        run_id=run_id,
        label=f"buy and hold {symbol}",
        cycles=len(steps),
        orders=len(broker.orders),
        performance=performance,
        calibration=None,
        final_equity=broker.equity(),
    )


def _flat_state(broker):
    from ..models import PortfolioState

    return PortfolioState.build(broker.account(), broker.positions())


def compare(results: Sequence[BacktestResult]) -> list[tuple[str, Performance]]:
    return [(r.label, r.performance) for r in results]
