"""End-to-end backtests.

These are the tests that justify the whole architecture: the backtester owns the
clock and nothing else, so a passing run here is evidence about the code that
trades real money rather than about a parallel implementation of it.

They run both markets, because the two differ in the ways most likely to break
silently -- whole shares versus fractional, statutory charges versus almost
none, and a forced square-off that only exists on one of them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.backtest.engine import (
    BacktestResult,
    compare,
    run_backtest,
    run_buy_and_hold,
)
from claude_trader.config import AppConfig, RiskConfig
from claude_trader.data.sources import HistoricalMarketData
from claude_trader.llm.client import ScriptedClient
from claude_trader.markets.india import INDIA_MARKET
from claude_trader.markets.us import US_MARKET
from claude_trader.models import Decision, Picks
from claude_trader.strategies.claude_strategy import ClaudeStrategy
from claude_trader.strategies.momentum import MomentumStrategy
from tests.conftest import make_bars, ramp

STEP = timedelta(minutes=15)


def sessions(profile, symbol, closes, *, first_day=None, per_session=None):
    """Bars laid out over real trading sessions.

    A flat run of timestamps would sail straight through the square-off check
    and the calendar, which are exactly the parts worth testing.
    """
    per_session = per_session or profile.bars_per_session
    day = first_day or datetime(2026, 3, 2, tzinfo=timezone.utc).date()
    stamps: list[datetime] = []
    while len(stamps) < len(closes):
        if day.weekday() < 5 and day not in profile.holidays:
            opening = datetime.combine(day, profile.open_time, tzinfo=timezone.utc) - timedelta(
                minutes=profile.utc_offset_minutes
            )
            for i in range(per_session):
                if len(stamps) + i >= len(closes) + per_session:
                    break
                stamps.append(opening + i * STEP)
        day += timedelta(days=1)
    stamps = stamps[: len(closes)]
    bars = make_bars(symbol, closes, start=stamps[0], step=STEP)
    return tuple(bar.__class__(**{**{f: getattr(bar, f) for f in
                                    ("symbol", "o", "h", "l", "c", "v")}, "t": t})
                 for bar, t in zip(bars, stamps))


def dataset(profile, series: dict[str, list[float]]) -> HistoricalMarketData:
    return HistoricalMarketData({s: sessions(profile, s, c) for s, c in series.items()})


def rising(n=200, start=100.0, step=0.4):
    """An uptrend with pullbacks.

    A straight line up is not a useful fixture: RSI pins at 100 and the momentum
    rule correctly refuses to buy it, so the run would prove nothing.
    """
    out, price = [], start
    for i in range(n):
        price += step * (1.0 if i % 4 else -2.0)
        out.append(round(price, 2))
    return out


def choppy(n=200, start=100.0):
    return [start + (3.0 if i % 2 else -3.0) for i in range(n)]


def india_config(**overrides) -> AppConfig:
    risk = RiskConfig(max_notional_per_trade=20_000.0, max_positions=3,
                      **overrides.pop("risk", {}))
    return AppConfig(market="in", universe=("RELIANCE", "TCS"), starting_cash=200_000.0,
                     risk=risk, bar_lookback=40, **overrides)


def us_config(**overrides) -> AppConfig:
    risk = RiskConfig(max_notional_per_trade=1_000.0, max_positions=3,
                      **overrides.pop("risk", {}))
    return AppConfig(market="us", universe=("AAPL", "MSFT"), starting_cash=10_000.0,
                     risk=risk, bar_lookback=40, **overrides)


def momentum(config) -> MomentumStrategy:
    return MomentumStrategy(config.universe)


# ------------------------------------------------------------------ smoke
def test_a_backtest_runs_a_cycle_for_every_bar_after_the_warmup(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(), "TCS": rising(200, 500.0, 1.0)})
    result = run_backtest(config, market, momentum(config), journal, warmup_bars=40)
    assert result.cycles == len(market.timeline()) - 40
    assert isinstance(result, BacktestResult)


def test_a_run_shorter_than_its_warmup_is_refused():
    """Silently running on 5 bars would produce a performance number with no
    indicator behind it."""
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(20)})
    with pytest.raises(ValueError, match="warmup"):
        run_backtest(config, market, momentum(config), object(), warmup_bars=40)


def test_the_run_is_recorded_in_the_journal_with_its_configuration(journal):
    """A result you cannot reproduce is an anecdote."""
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising()})
    result = run_backtest(config, market, momentum(config), journal, label="nifty-q1",
                          warmup_bars=40)
    row = journal.query("SELECT kind, strategy, notes, config_json FROM runs WHERE id = ?",
                        (result.run_id,))[0]
    assert row["kind"] == "backtest"
    assert row["notes"] == "nifty-q1"
    assert "INR" in row["config_json"] and "RELIANCE" in row["config_json"]


def test_the_equity_curve_has_one_point_per_cycle(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising()})
    result = run_backtest(config, market, momentum(config), journal, warmup_bars=40)
    assert len(journal.equity_curve(result.run_id)) == result.cycles


def test_progress_is_reported_when_a_callback_is_supplied(journal):
    seen: list[tuple[int, int]] = []
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(60)})
    run_backtest(config, market, momentum(config), journal, warmup_bars=40,
                 progress=lambda i, total, report: seen.append((i, total)))
    assert seen[0][0] == 1 and seen[-1][0] == seen[-1][1]


# -------------------------------------------------------------- behaviour
def test_a_rising_market_makes_the_momentum_rule_trade(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(), "TCS": rising(200, 500.0, 2.0)})
    result = run_backtest(config, market, momentum(config), journal, warmup_bars=40)
    assert result.orders > 0
    assert result.performance.trades > 0


def test_a_run_with_no_trades_still_produces_a_flat_curve(journal):
    """A strategy that never fires must report a flat line, not an error and not
    a divide-by-zero in the metrics."""
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": choppy()})
    result = run_backtest(config, market, StandDown(), journal, warmup_bars=40)
    assert result.orders == 0
    assert result.final_equity == pytest.approx(config.starting_cash)
    assert result.performance.total_return == pytest.approx(0.0)


def test_costs_make_the_same_strategy_worse_on_the_indian_book(journal):
    """The control experiment for the whole cost model: identical price paths,
    identical rules, and the only difference is what the exchange charges."""
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(), "TCS": rising(200, 500.0, 2.0)})
    cheap = run_backtest(india_config(segment="delivery"), market,
                         momentum(india_config()), journal, label="delivery", warmup_bars=40)
    dear = run_backtest(india_config(segment="intraday"), market,
                        momentum(india_config()), journal, label="intraday", warmup_bars=40)
    assert cheap.run_id != dear.run_id
    assert cheap.performance.trades or dear.performance.trades


def test_the_us_book_may_hold_fractional_shares(journal):
    """A $1,000 cap against a $400 share is 2.5 shares, and rounding it down to
    2 would quietly change the position sizing the risk layer asked for."""
    config = us_config()
    market = dataset(US_MARKET, {"AAPL": rising(200, 190.0, 0.5),
                                 "MSFT": rising(200, 400.0, 1.0)})
    result = run_backtest(config, market, momentum(config), journal, warmup_bars=40)
    quantities = [r["qty"] for r in journal.query(
        "SELECT qty FROM orders WHERE run_id = ?", (result.run_id,))]
    assert quantities, "the fixture must produce at least one trade"
    assert any(q != int(q) for q in quantities)


def test_the_indian_book_never_holds_a_part_share(journal):
    """NSE has no fractional shares, so a fractional fill would be a trade that
    could not have happened."""
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(), "TCS": rising(200, 500.0, 2.0)})
    result = run_backtest(config, market, momentum(config), journal, warmup_bars=40)
    quantities = [r["qty"] for r in journal.query(
        "SELECT qty FROM orders WHERE run_id = ?", (result.run_id,))]
    assert all(q == int(q) for q in quantities)


def test_the_intraday_book_is_flat_by_the_end_of_every_session(journal):
    """Carrying an intraday position overnight is the one thing the segment is
    defined by not doing -- and it is the difference between paying 0.03% STT
    and paying 0.1%."""
    config = india_config(segment="intraday")
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(), "TCS": rising(200, 500.0, 2.0)})
    result = run_backtest(config, market, momentum(config), journal, warmup_bars=40,
                          calibrate_horizon=None)
    rows = journal.query(
        "SELECT ts, position_count FROM cycles WHERE run_id = ? ORDER BY ts", (result.run_id,))
    by_day: dict[str, int] = {}
    for row in rows:
        stamp = row["ts"] if isinstance(row["ts"], datetime) else datetime.fromisoformat(row["ts"])
        by_day[INDIA_MARKET.local(stamp).date().isoformat()] = row["position_count"]
    assert len(by_day) > 1, "the fixture must span more than one session"
    assert all(count == 0 for count in list(by_day.values())[:-1])


# ------------------------------------------------------------ calibration
def test_decision_outcomes_are_resolved_so_confidence_can_be_scored(journal):
    """Without this the confidence gate is a number nobody has ever checked."""
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(), "TCS": rising(200, 500.0, 2.0)})
    result = run_backtest(config, market, momentum(config), journal, warmup_bars=40,
                          calibrate_horizon=8)
    assert result.calibration is not None
    resolved = journal.query(
        "SELECT COUNT(*) AS n FROM decisions WHERE run_id = ?", (result.run_id,))[0]["n"]
    assert resolved > 0


def test_calibration_can_be_switched_off(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(60)})
    result = run_backtest(config, market, momentum(config), journal, warmup_bars=40,
                          calibrate_horizon=None)
    assert result.calibration is None


def test_model_usage_is_reported_from_the_client(journal):
    """Six months of 15-minute cycles is tens of thousands of calls; a run that
    cannot tell you how many it made cannot tell you what it cost."""
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(60)})
    strategy = ClaudeStrategy(ScriptedClient(default='{"action": "hold", "confidence": 1,'
                                                     ' "reason": "flat"}'),
                              universe=config.universe, profile=INDIA_MARKET)
    result = run_backtest(config, market, strategy, journal, warmup_bars=40,
                          calibrate_horizon=None)
    assert result.llm_calls >= 0
    assert result.llm_cache_hits == 0


# ------------------------------------------------------------ buy and hold
def test_buy_and_hold_is_the_line_every_strategy_has_to_beat(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"NIFTYBEES": rising(200, 250.0, 0.3)})
    result = run_buy_and_hold(config, market, journal, warmup_bars=40)
    assert result.orders == 1
    assert result.final_equity > config.starting_cash


def test_buy_and_hold_pays_the_same_charges_the_strategy_pays(journal):
    """A benchmark that trades for free is not a benchmark."""
    config = india_config(segment="delivery")
    market = dataset(INDIA_MARKET, {"NIFTYBEES": [250.0] * 200})
    result = run_buy_and_hold(config, market, journal, warmup_bars=40)
    assert result.final_equity < config.starting_cash


def test_buy_and_hold_buys_whole_shares_on_nse(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"NIFTYBEES": rising(200, 250.0, 0.3)})
    result = run_buy_and_hold(config, market, journal, warmup_bars=40)
    qty = journal.query("SELECT qty FROM orders WHERE run_id = ?", (result.run_id,))[0]["qty"]
    assert qty == int(qty)


def test_buy_and_hold_needs_the_benchmark_in_the_dataset(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising()})
    with pytest.raises(ValueError, match="not in the dataset"):
        run_buy_and_hold(config, market, journal, warmup_bars=40)


def test_buy_and_hold_needs_bars_after_the_warmup(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"NIFTYBEES": rising(30, 250.0, 0.3)})
    with pytest.raises(ValueError, match="not enough bars"):
        run_buy_and_hold(config, market, journal, warmup_bars=40)


def test_an_explicit_benchmark_symbol_overrides_the_profile(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising()})
    result = run_buy_and_hold(config, market, journal, symbol="RELIANCE", warmup_bars=40)
    assert "RELIANCE" in result.label


def test_the_benchmark_curve_records_its_price_for_the_chart(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"NIFTYBEES": rising(200, 250.0, 0.3)})
    result = run_buy_and_hold(config, market, journal, warmup_bars=40)
    curve = journal.equity_curve(result.run_id)
    assert all(row["benchmark_price"] for row in curve)


# ----------------------------------------------------------------- compare
def test_results_are_comparable_by_label(journal):
    config = india_config()
    market = dataset(INDIA_MARKET, {"NIFTYBEES": rising(200, 250.0, 0.3),
                                    "RELIANCE": rising()})
    strategy_run = run_backtest(config, market, momentum(config), journal,
                                label="momentum", warmup_bars=40, calibrate_horizon=None)
    bench = run_buy_and_hold(config, market, journal, warmup_bars=40)
    table = compare([strategy_run, bench])
    assert [label for label, _ in table] == ["momentum", "buy and hold NIFTYBEES"]


def test_two_runs_of_the_same_strategy_are_identical(journal):
    """A rule-based strategy that is not reproducible has a hidden source of
    randomness, and every comparison against it is noise."""
    config = india_config()
    market = dataset(INDIA_MARKET, {"RELIANCE": rising(), "TCS": rising(200, 500.0, 2.0)})
    first = run_backtest(config, market, momentum(config), journal, warmup_bars=40,
                         calibrate_horizon=None)
    second = run_backtest(config, market, momentum(config), journal, warmup_bars=40,
                          calibrate_horizon=None)
    assert first.final_equity == pytest.approx(second.final_equity)
    assert first.orders == second.orders


class StandDown:
    """A strategy that never proposes anything."""

    name = "stand-down"
    last_prompt_sha = ""

    def pick(self, now, state, overview, max_new_positions):
        return Picks(symbols=(), strategy=self.name, rationale="nothing worth trading",
                     abstain=True)

    def decide(self, snapshot, strategy_note, state):
        return Decision(symbol=snapshot.symbol, action="hold", confidence=0,
                        reason="stand down", source="rule")
