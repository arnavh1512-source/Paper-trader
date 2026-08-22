"""The composition root.

Every decision about *which* implementation gets used lives in one module, so
this file is where the two markets are held apart: an NSE run must never end up
holding an Alpaca client, and a US run must never end up on the Yahoo feed. It
also pins the two warnings a live cycle owes its operator -- dry run, and a base
URL that is not the paper endpoint.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from claude_trader import app
from claude_trader.brokers.alpaca import AlpacaBroker
from claude_trader.brokers.paper import PaperBroker
from claude_trader.config import AppConfig, LLMConfig
from claude_trader.data.sources import AlpacaMarketData
from claude_trader.data.yahoo import YahooMarketData
from claude_trader.engine.cycle import CycleReport
from claude_trader.models import Account, Position
from claude_trader.risk.engine import RiskState
from claude_trader.strategies.claude_strategy import ClaudeStrategy
from claude_trader.strategies.momentum import MomentumStrategy

NOW = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)


class StubMarket:
    """Enough of a feed for the composition root; the cycle is tested elsewhere."""

    def bars(self, symbol, limit=100, as_of=None):
        return ()

    def quote(self, symbol, as_of=None):
        return None

    def snapshots(self, symbols, as_of=None):
        return {}

    def latest_prices(self, symbols, as_of=None):
        return {}


class ClosedBroker:
    def __init__(self, equity=100_000.0, positions=()):
        self._equity = equity
        self._positions = tuple(positions)

    def account(self):
        return Account(equity=self._equity, cash=self._equity,
                       buying_power=self._equity, last_equity=self._equity)

    def positions(self):
        return self._positions

    def is_market_open(self, now):
        return False

    def submit(self, order):  # pragma: no cover - the market is shut
        raise AssertionError("no order may be sent while the market is closed")

    def close_position(self, symbol):  # pragma: no cover
        raise AssertionError("no exit may be sent while the market is closed")


def report(**overrides) -> CycleReport:
    base = dict(ts=NOW, market_open=True, equity=100_500.0, cash=40_000.0,
                position_count=2)
    return CycleReport(**{**base, **overrides})


# ---------------------------------------------------------------- strategy
def test_momentum_is_the_free_control_group():
    """It must construct with no API key at all, or the control group costs the
    same as the thing it is controlling for."""
    strategy = app.build_strategy(AppConfig(market="in", strategy="momentum"))
    assert isinstance(strategy, MomentumStrategy)


def test_the_claude_strategy_is_wired_to_the_model(journal):
    strategy = app.build_strategy(
        AppConfig(market="in", strategy="claude", llm=LLMConfig(api_key="key")),
        journal)
    assert isinstance(strategy, ClaudeStrategy)
    assert strategy.client is not None


def test_the_journal_doubles_as_the_prompt_cache(journal):
    """Re-running a backtest is only affordable because an identical prompt is
    answered from the journal instead of from the API."""
    strategy = app.build_strategy(
        AppConfig(market="in", strategy="claude", llm=LLMConfig(api_key="key")),
        journal)
    assert strategy.client._cache is journal


def test_an_unknown_strategy_names_the_ones_that_exist():
    """AppConfig rejects this too, but the composition root is reachable from
    code that never went through validation."""
    with pytest.raises(ValueError, match="expected 'claude' or 'momentum'"):
        app.build_strategy(SimpleNamespace(strategy="astrology"))


def test_the_strategy_is_given_the_universe_it_is_allowed_to_trade():
    strategy = app.build_strategy(
        AppConfig(market="in", strategy="momentum", universe=("RELIANCE", "TCS")))
    assert strategy._universe == ("RELIANCE", "TCS")


# -------------------------------------------------------------------- data
def test_a_us_run_reads_from_alpaca():
    source = app.build_market_data(
        AppConfig(market="us", alpaca_key="k", alpaca_secret="s"))
    assert isinstance(source, AlpacaMarketData)


def test_an_nse_run_reads_from_yahoo():
    """NSE has no free broker sandbox, so the data feed and the book come from
    different places -- and neither needs a key."""
    assert isinstance(app.build_market_data(AppConfig(market="in")),
                      YahooMarketData)


def test_the_feed_is_asked_for_the_configured_timeframe():
    source = app.build_market_data(AppConfig(market="in", timeframe="5m"))
    assert source._interval == "5m"


# ------------------------------------------------------------------ broker
def test_the_alpaca_broker_is_used_when_it_is_configured(journal):
    config = AppConfig(market="us", broker="alpaca", alpaca_key="k",
                       alpaca_secret="s")
    broker = app.build_broker(config, StubMarket(), journal, NOW)
    assert isinstance(broker, AlpacaBroker)


def test_everywhere_else_the_journal_is_the_account(journal):
    """There is no NSE paper API, so the book only survives between runs
    because it is written to the journal file."""
    broker = app.build_broker(AppConfig(market="in"), StubMarket(), journal, NOW)
    assert isinstance(broker, PaperBroker)


def test_the_paper_book_is_scoped_to_the_strategy(journal):
    """A momentum run and a claude run share one journal; without separate
    accounts each would spend the other's cash."""
    broker = app.build_broker(AppConfig(market="in", strategy="momentum"),
                              StubMarket(), journal, NOW)
    assert broker._account.startswith("momentum:")


def test_the_paper_book_starts_with_the_market_s_own_cash(journal):
    broker = app.build_broker(AppConfig(market="in"), StubMarket(), journal, NOW)
    assert broker.account().equity == AppConfig(market="in").starting_cash


# --------------------------------------------------------------- summarise
def test_a_closed_market_summarises_in_two_words():
    assert app.summarise(report(market_open=False), AppConfig()) == "market closed"


def test_a_quiet_cycle_says_no_trades_rather_than_nothing():
    """An empty summary reads as a crash; 'no trades' reads as a decision."""
    text = app.summarise(report(), AppConfig(market="in"))
    assert "no trades" in text
    assert "2 open" in text


def test_entries_and_exits_are_both_named():
    text = app.summarise(report(entries=("TCS",), exits=("RELIANCE",)),
                         AppConfig(market="in"))
    assert "bought TCS" in text
    assert "sold RELIANCE" in text
    assert "no trades" not in text


def test_the_summary_is_denominated_in_the_market_s_currency():
    assert "₹" in app.summarise(report(), AppConfig(market="in"))
    assert "$" in app.summarise(report(), AppConfig(market="us"))


def test_a_halt_and_its_reason_reach_the_summary():
    text = app.summarise(
        report(risk_state=RiskState(halted=True, reason="drawdown breaker")),
        AppConfig(market="in"))
    assert "halted: drawdown breaker" in text


# ----------------------------------------------------------------- logging
def test_logging_configures_without_raising():
    app.configure_logging(verbose=True)
    assert logging.getLogger("urllib3").level == logging.WARNING


def test_logging_survives_a_stream_that_cannot_be_reconfigured(monkeypatch):
    """A detached or exotic stdout must not take the whole run down before it
    has done anything."""
    class Stubborn:
        def reconfigure(self, **kwargs):
            raise ValueError("detached")

    monkeypatch.setattr(app.sys, "stdout", Stubborn())
    app.configure_logging()


def test_rupee_amounts_do_not_kill_a_windows_console(capsys):
    """The default Windows codepage cannot encode the symbol; a run that dies on
    a log line has failed for no reason at all."""
    app.configure_logging()
    logging.getLogger("test").info("equity ₹1,00,000")


# ------------------------------------------------------------- live cycle
def live(config: AppConfig, journal, monkeypatch, broker=None, market=None):
    monkeypatch.setattr(app, "build_market_data", lambda cfg: market or StubMarket())
    monkeypatch.setattr(app, "build_broker",
                        lambda cfg, mkt, jrn, now: broker or ClosedBroker())
    return app.run_live_cycle(config, now=NOW, journal=journal)


def test_a_live_cycle_runs_and_reports(journal, monkeypatch):
    result = live(AppConfig(market="in", strategy="momentum"), journal, monkeypatch)
    assert isinstance(result, CycleReport)
    assert result.market_open is False


def test_a_live_cycle_records_its_run_before_it_trades(journal, monkeypatch):
    """A cycle that trades without a run row leaves orders no report can find."""
    live(AppConfig(market="in", strategy="momentum"), journal, monkeypatch)
    rows = journal.query("SELECT strategy, kind, config_json FROM runs")
    assert rows[0]["strategy"] == "momentum"
    assert "INR" in rows[0]["config_json"] or "in" in rows[0]["config_json"]


def test_the_recorded_config_carries_the_risk_limits(journal, monkeypatch):
    """Two runs with different limits are different experiments; without the
    limits on the run row they are indistinguishable afterwards."""
    live(AppConfig(market="in", strategy="momentum"), journal, monkeypatch)
    config_json = journal.query("SELECT config_json FROM runs")[0]["config_json"]
    assert "max_positions" in config_json
    assert "universe" in config_json


def test_a_dry_run_says_so_loudly(journal, monkeypatch, caplog):
    """Silence here means someone watches a log full of decisions and believes
    orders were placed."""
    with caplog.at_level(logging.WARNING):
        live(AppConfig(market="in", strategy="momentum", dry_run=True), journal,
             monkeypatch)
    assert "DRY RUN" in caplog.text


def test_a_non_paper_endpoint_is_a_warning_not_a_silent_default(journal,
                                                                monkeypatch,
                                                                caplog):
    """This is the single line standing between a research tool and real money
    leaving an account."""
    config = AppConfig(market="us", broker="alpaca", strategy="momentum",
                       alpaca_key="k", alpaca_secret="s",
                       alpaca_base="https://api.alpaca.markets")
    with caplog.at_level(logging.WARNING):
        live(config, journal, monkeypatch)
    assert "Orders would be real" in caplog.text


def test_a_paper_endpoint_raises_no_such_alarm(journal, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        live(AppConfig(market="in", strategy="momentum"), journal, monkeypatch)
    assert "Orders would be real" not in caplog.text


def test_the_market_and_cost_model_are_logged(journal, monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        live(AppConfig(market="in", segment="intraday", strategy="momentum"),
             journal, monkeypatch)
    assert "Market IN" in caplog.text
    assert "intraday" in caplog.text


def test_a_borrowed_journal_is_left_open_for_its_owner(journal, monkeypatch):
    """Closing a journal the caller passed in would break the backtester, which
    runs many cycles against one connection."""
    live(AppConfig(market="in", strategy="momentum"), journal, monkeypatch)
    assert journal.query("SELECT 1 AS ok")[0]["ok"] == 1


def test_a_cycle_that_owns_its_journal_closes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "build_market_data", lambda cfg: StubMarket())
    monkeypatch.setattr(app, "build_broker",
                        lambda cfg, mkt, jrn, now: ClosedBroker())
    config = AppConfig(market="in", strategy="momentum",
                       journal_path=str(tmp_path / "j.db"))
    app.run_live_cycle(config, now=NOW)
    assert (tmp_path / "j.db").exists()


def test_missing_credentials_stop_the_cycle_before_any_network_call(journal,
                                                                    monkeypatch):
    """Failing at the config check is a one-line error; failing later is a
    stack trace halfway through a run."""
    def explode(cfg):  # pragma: no cover - must not be reached
        raise AssertionError("the feed was built before credentials were checked")

    monkeypatch.setattr(app, "build_market_data", explode)
    with pytest.raises(Exception):
        app.run_live_cycle(AppConfig(market="us", broker="alpaca"), now=NOW,
                           journal=journal)


def test_the_position_count_survives_a_closed_market(journal, monkeypatch):
    """The book still exists when the exchange is shut, and a report that says
    zero would look like everything was liquidated overnight."""
    broker = ClosedBroker(positions=(
        Position(symbol="RELIANCE", qty=10, avg_entry_price=1400.0,
                 current_price=1420.0),))
    result = live(AppConfig(market="in", strategy="momentum"), journal,
                  monkeypatch, broker=broker)
    assert result.position_count == 1
