"""Composition root.

Everything that decides *which* implementation to use lives here, so the rest
of the package can depend on Protocols and stay testable. Nothing below this
module reads the environment or constructs a network client.

Two markets are wired up. ``market=us`` uses Alpaca for both data and the paper
book; ``market=in`` uses Yahoo for data and a journal-backed paper book, because
NSE has no free broker sandbox. The cycle below cannot tell the difference.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from typing import Any

from .config import AppConfig
from .costs import build_cost_model
from .engine.cycle import CycleDeps, CycleReport, run_cycle
from .engine.executor import Executor
from .journal.store import Journal
from .llm.client import ClaudeClient
from .risk.engine import RiskEngine
from .strategies.claude_strategy import ClaudeStrategy
from .strategies.momentum import MomentumStrategy

log = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(verbose: bool = False) -> None:
    # Rupee amounts are logged on every cycle and the default Windows console
    # codepage cannot encode the symbol, so a run would die on a log line rather
    # than on anything that matters. Force UTF-8 and degrade instead of raising.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # already-detached or exotic stream
                pass

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def build_news(config: AppConfig, *, live: bool):
    """Headlines, and only when they can honestly be used.

    News is deliberately unavailable to a backtest. The feeds return today's
    headlines, and a 2024 bar priced against a 2026 headline is not a backtest,
    it is a machine for producing encouraging numbers. Live cycles only.
    """
    from .data.news import NullNewsSource, RssNewsSource

    if not (live and config.news_enabled):
        return NullNewsSource()
    return RssNewsSource(config.market,
                         max_age_hours=config.news_max_age_hours)


def build_strategy(config: AppConfig, journal: Journal | None = None, *,
                   live: bool = False):
    """Pick the decision maker. ``momentum`` costs nothing and is the control
    group; ``claude`` is the thing being evaluated against it."""
    if config.strategy == "momentum":
        return MomentumStrategy(config.universe)
    if config.strategy != "claude":
        raise ValueError(
            f"unknown strategy {config.strategy!r}; expected 'claude' or 'momentum'"
        )
    # The journal doubles as the response cache: a re-run of the same prompt
    # is free, which is what makes iterating on a backtest affordable.
    return ClaudeStrategy(
        ClaudeClient(config.llm, cache=journal),
        config.universe,
        profile=config.profile,
        segment=config.segment,
        costs=build_cost_model(config.market, config.segment),
        news=build_news(config, live=live),
        max_headlines=config.news_max_headlines,
    )


def build_market_data(config: AppConfig):
    """The data feed for the configured market."""
    if config.market == "us":
        from .data.sources import AlpacaMarketData

        return AlpacaMarketData(config)

    from .data.yahoo import YahooMarketData

    return YahooMarketData(config.profile, interval=config.timeframe)


def build_broker(config: AppConfig, market, journal: Journal, now: datetime):
    """Alpaca keeps paper state server-side; everywhere else the journal is the
    account, which is why the journal file has to survive between runs."""
    if config.broker == "alpaca":
        from .brokers.alpaca import AlpacaBroker

        return AlpacaBroker(config)

    from .brokers.paper import PaperBroker

    broker = PaperBroker(
        journal=journal,
        market=market,
        profile=config.profile,
        costs=build_cost_model(config.market, config.segment),
        account=config.strategy,
        starting_cash=config.starting_cash,
        clock=now,
    )
    return broker


def summarise(report: CycleReport, config: AppConfig) -> str:
    if not report.market_open:
        return "market closed"
    bits = [
        f"equity {config.money(report.equity)}",
        f"cash {config.money(report.cash)}",
        f"{report.position_count} open",
    ]
    if report.entries:
        bits.append("bought " + ", ".join(report.entries))
    if report.exits:
        bits.append("sold " + ", ".join(report.exits))
    if not report.entries and not report.exits:
        bits.append("no trades")
    if report.risk_state.halted:
        bits.append(f"halted: {report.risk_state.reason}")
    return " | ".join(bits)


def run_live_cycle(
    config: AppConfig,
    now: datetime | None = None,
    journal: Journal | None = None,
) -> CycleReport:
    """One live (paper) cycle. This is what GitHub Actions invokes."""
    config.require_live_credentials()
    owns_journal = journal is None
    journal = journal or Journal(config.journal_path)
    now = now or datetime.now(timezone.utc)

    try:
        strategy = build_strategy(config, journal, live=True)
        run_id = journal.resolve_live_run(
            strategy=config.strategy,
            now=now,
            config={
                "market": config.market,
                "segment": config.segment,
                "broker": config.broker,
                "currency": config.currency,
                "universe": list(config.universe),
                "timeframe": config.timeframe,
                "feed": config.feed,
                "dry_run": config.dry_run,
                "paper": config.is_paper,
                "risk": config.risk.as_dict(),
            },
        )
        market = build_market_data(config)
        broker = build_broker(config, market, journal, now)
        costs = build_cost_model(config.market, config.segment)
        deps = CycleDeps(
            config=config,
            market=market,
            broker=broker,
            strategy=strategy,
            risk=RiskEngine(config.risk, profile=config.profile, costs=costs),
            journal=journal,
            executor=Executor(
                broker, journal, run_id, dry_run=config.dry_run, profile=config.profile
            ),
            run_id=run_id,
        )

        if config.dry_run:
            log.warning("DRY RUN: decisions will be journalled, no orders sent.")
        if not config.is_paper:
            log.warning(
                "ALPACA_BASE is not the paper endpoint. Orders would be real; "
                "verify this is intended."
            )
        log.info(
            "Market %s (%s, %s) | costs %s | broker %s",
            config.market.upper(),
            config.currency,
            config.segment,
            costs.name,
            config.broker,
        )

        report = run_cycle(deps, now)
        journal.commit()
        log.info("Cycle complete: %s", summarise(report, config))

        client: Any = getattr(strategy, "client", None)
        if client is not None and getattr(client, "calls_made", 0):
            log.info("Model usage: %s", client.usage_summary())
        return report
    finally:
        if owns_journal:
            journal.close()
