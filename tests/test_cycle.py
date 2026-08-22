"""The trading cycle -- the single decision path.

These are integration tests on purpose. The components are wired the way the
live engine wires them (real journal, real risk engine, real executor, real
paper broker) and only the market feed and the strategy are stubbed, because
those are the two things a test cannot honestly supply.

The invariant that matters most: the deterministic risk layer runs *before* the
strategy is consulted, so a stop can never be argued away by a confident model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.brokers.paper import PaperBroker, SlippageModel
from claude_trader.config import AppConfig, RiskConfig
from claude_trader.costs import IndianEquityCosts
from claude_trader.engine.cycle import CycleDeps, run_cycle
from claude_trader.engine.executor import Executor
from claude_trader.markets import INDIA_MARKET
from claude_trader.models import Action, Decision, Picks, Quote
from claude_trader.risk.engine import RiskEngine
from tests.conftest import make_bars, make_quote, ramp

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)        # 10:30 IST
LATE = datetime(2026, 3, 2, 9, 50, tzinfo=timezone.utc)      # 15:20 IST
CLOSED = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)    # 19:30 IST


class StubMarket:
    """Bars, quotes and last prices. Everything the cycle asks a feed for."""

    def __init__(self, prices=None, *, bar_error=(), quote_error=(), price_error=False):
        self.prices = dict(prices or {"RELIANCE": 1_000.0, "TCS": 3_000.0})
        self.series = {}
        self.bar_error = set(bar_error)
        self.quote_error = set(quote_error)
        self.price_error = price_error
        self.bars_asked: list[str] = []

    def set_series(self, symbol, closes):
        self.series[symbol] = tuple(make_bars(symbol, closes, start=NOW - timedelta(hours=6)))
        self.prices[symbol] = closes[-1]

    def bars(self, symbol, limit, now):
        self.bars_asked.append(symbol)
        if symbol in self.bar_error:
            raise RuntimeError("feed timed out")
        if symbol in self.series:
            return self.series[symbol][-limit:]
        price = self.prices.get(symbol)
        if price is None:
            return ()
        return tuple(make_bars(symbol, ramp(40, price * 0.96, price * 0.001),
                               start=NOW - timedelta(hours=6)))[-limit:]

    def quote(self, symbol, now) -> Quote | None:
        if symbol in self.quote_error:
            raise RuntimeError("no order book")
        price = self.prices.get(symbol)
        return None if price is None else make_quote(symbol, price, now)

    def latest_prices(self, symbols, now):
        if self.price_error:
            raise RuntimeError("feed down")
        return {s: self.prices[s] for s in symbols if s in self.prices}


class StubStrategy:
    """Proposes exactly what the test tells it to propose."""

    name = "stub"
    last_prompt_sha = "deadbeef"

    def __init__(self, picks=("RELIANCE",), decisions=None, decide_error=None,
                 mood="bullish", note="momentum", abstain=False):
        self._picks = tuple(picks)
        self._decisions = decisions or {}
        self._decide_error = decide_error
        self._mood = mood
        self._note = note
        self._abstain = abstain
        self.picked = 0
        self.decided: list[str] = []

    def pick(self, now, state, overview, max_new_positions):
        self.picked += 1
        return Picks(self._picks[:max_new_positions], self._note,
                     market_mood=self._mood, abstain=self._abstain)

    def decide(self, snapshot, strategy_note, state):
        self.decided.append(snapshot.symbol)
        if self._decide_error:
            raise self._decide_error
        return self._decisions.get(
            snapshot.symbol, Decision(snapshot.symbol, Action.HOLD, 5, "nothing here")
        )


def build(journal, market, strategy, *, risk_overrides=None, dry_run=False,
          universe=("RELIANCE", "TCS"), cash=100_000.0):
    risk_cfg = RiskConfig(**{
        "max_notional_per_trade": 20_000.0,
        "max_positions": 3,
        **(risk_overrides or {}),
    })
    cfg = AppConfig(
        market="in", universe=universe, starting_cash=cash,
        risk=risk_cfg, dry_run=dry_run, bar_lookback=40,
    )
    costs = IndianEquityCosts(cfg.segment)
    broker = PaperBroker(
        journal=journal, market=market, profile=INDIA_MARKET, costs=costs,
        starting_cash=cash, slippage=SlippageModel(5.0), clock=NOW,
    )
    run_id = journal.start_run("live", "stub", NOW, {"market": cfg.market})
    return CycleDeps(
        config=cfg,
        market=market,
        broker=broker,
        strategy=strategy,
        risk=RiskEngine(risk_cfg, profile=INDIA_MARKET, costs=costs),
        journal=journal,
        executor=Executor(broker, journal, run_id, dry_run=dry_run, profile=INDIA_MARKET),
        run_id=run_id,
    )


def buy(symbol="RELIANCE", confidence=8, notional=None):
    return {symbol: Decision(symbol, Action.BUY, confidence, "breakout", notional)}


def hold_a_position(deps, market, symbol="RELIANCE", qty=10, stop=None, target=None):
    """Put a real position on the real paper book, with a real stop row."""
    from claude_trader.models import OrderRequest, PositionRisk, Side

    result = deps.broker.submit(OrderRequest(symbol, Side.BUY, qty=qty))
    deps.journal.upsert_position_risk(deps.run_id, PositionRisk(
        symbol=symbol,
        entry_price=result.price,
        entry_time=NOW - timedelta(hours=1),
        stop_price=stop if stop is not None else result.price * 0.95,
        target_price=target if target is not None else result.price * 1.10,
        high_water=result.price,
        atr_at_entry=result.price * 0.01,
        bars_held=2,
    ))
    return result


# --------------------------------------------------------------- closed market
def test_a_closed_market_does_nothing_at_all(journal):
    market = StubMarket()
    strategy = StubStrategy()
    deps = build(journal, market, strategy)
    deps.broker.set_clock(CLOSED)

    report = run_cycle(deps, CLOSED)

    assert report.market_open is False
    assert report.entries == () and report.exits == ()
    assert strategy.picked == 0 and strategy.decided == []
    assert journal.query("SELECT * FROM cycles") == []
    assert journal.query("SELECT * FROM orders") == []


def test_a_closed_market_still_reports_the_book(journal):
    """Someone reading the log needs to know the bot ran and saw the book."""
    deps = build(journal, StubMarket(), StubStrategy())
    deps.broker.set_clock(CLOSED)
    report = run_cycle(deps, CLOSED)
    assert report.equity == pytest.approx(100_000.0)
    assert report.cash == pytest.approx(100_000.0)
    assert report.position_count == 0


# ---------------------------------------------------------------- happy path
def test_an_approved_buy_is_placed_sized_and_written_down(journal):
    market = StubMarket()
    deps = build(journal, market, StubStrategy(decisions=buy()))
    report = run_cycle(deps, NOW)

    assert report.entries == ("RELIANCE",)
    assert report.traded == 1
    assert deps.broker.positions()[0].symbol == "RELIANCE"

    order = journal.query("SELECT * FROM orders")[0]
    assert order["intent"] == "entry" and order["side"] == "buy"
    decision = journal.query("SELECT * FROM decisions")[0]
    assert decision["action"] == "buy" and decision["executed"] == 1
    assert decision["prompt_sha"] == "deadbeef"
    assert journal.open_position_risks(deps.run_id)[0].symbol == "RELIANCE"


def test_the_cycle_row_is_completed_with_what_actually_happened(journal):
    deps = build(journal, StubMarket(), StubStrategy(decisions=buy()))
    run_cycle(deps, NOW)
    row = journal.query("SELECT * FROM cycles")[0]
    assert row["market_open"] == 1
    assert row["halted"] == 0
    assert row["market_mood"] == "bullish"
    assert row["strategy_note"] == "momentum"
    assert "RELIANCE" in row["picks_json"]
    assert row["position_count"] == 1


def test_the_equity_curve_is_stamped_every_cycle(journal):
    deps = build(journal, StubMarket(), StubStrategy(decisions=buy()))
    run_cycle(deps, NOW)
    row = journal.query("SELECT * FROM equity_curve")[0]
    assert row["equity"] > 0
    assert row["exposure"] > 0


def test_the_benchmark_is_captured_alongside_the_equity(journal):
    """Without it the curve is a number, not a track record."""
    market = StubMarket({"RELIANCE": 1_000.0, INDIA_MARKET.benchmark: 250.0})
    deps = build(journal, market, StubStrategy(decisions=buy()))
    run_cycle(deps, NOW)
    assert journal.query("SELECT * FROM equity_curve")[0]["benchmark_price"] == 250.0


def test_a_missing_benchmark_is_recorded_as_absent_not_as_zero(journal):
    deps = build(journal, StubMarket(price_error=True), StubStrategy(decisions=buy()))
    run_cycle(deps, NOW)
    assert journal.query("SELECT * FROM equity_curve")[0]["benchmark_price"] is None


def test_a_low_confidence_buy_never_reaches_the_broker(journal):
    deps = build(journal, StubMarket(), StubStrategy(decisions=buy(confidence=6)))
    report = run_cycle(deps, NOW)
    assert report.entries == ()
    assert any("below 7" in reason for _, reason in report.skipped)
    assert journal.query("SELECT * FROM orders") == []


def test_a_declined_entry_is_still_recorded_as_a_decision(journal):
    """Rejected decisions are the counterfactual for every rule in the risk
    layer. Discarding them makes the rules unmeasurable."""
    deps = build(journal, StubMarket(), StubStrategy(decisions=buy(confidence=6)))
    run_cycle(deps, NOW)
    row = journal.query("SELECT * FROM decisions")[0]
    assert row["action"] == "buy" and row["executed"] == 0
    assert row["confidence"] == 6


def test_a_hold_is_recorded_and_nothing_is_traded(journal):
    deps = build(journal, StubMarket(), StubStrategy())
    report = run_cycle(deps, NOW)
    assert report.entries == () and report.exits == ()
    assert [d.action for d in report.decisions] == [Action.HOLD]
    assert journal.query("SELECT * FROM orders") == []


# ------------------------------------------------------- exits come first
def test_a_breached_stop_exits_before_the_model_is_asked(journal):
    """The whole ordering of the cycle exists for this test."""
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(picks=()))
    hold_a_position(deps, market, stop=1_050.0)      # already below the stop

    report = run_cycle(deps, NOW)

    assert report.exits == ("RELIANCE",)
    assert "RELIANCE" not in deps.strategy.decided
    assert deps.broker.positions() == ()
    assert journal.query("SELECT intent FROM orders ORDER BY id")[-1]["intent"] \
        in {"stop_loss", "trailing_stop"}


def test_a_forced_exit_retires_the_stop_row(journal):
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(picks=()))
    hold_a_position(deps, market, stop=1_050.0)
    run_cycle(deps, NOW)
    assert journal.open_position_risks(deps.run_id) == ()


def test_a_healthy_position_is_left_alone_and_reviewed_by_the_model(journal):
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(picks=()))
    hold_a_position(deps, market, stop=900.0, target=1_500.0)

    report = run_cycle(deps, NOW)

    assert report.exits == ()
    assert "RELIANCE" in deps.strategy.decided     # held names are always reviewed
    assert deps.broker.positions()[0].qty == 10


def test_the_trailing_stop_is_ratcheted_up_before_exits_are_judged(journal):
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(picks=()))
    hold_a_position(deps, market, stop=900.0, target=1_500.0)
    before = journal.open_position_risks(deps.run_id)[0].stop_price

    market.prices["RELIANCE"] = 1_200.0
    run_cycle(deps, NOW)

    after = journal.open_position_risks(deps.run_id)[0]
    assert after.stop_price > before
    assert after.high_water >= 1_200.0


def test_a_model_sell_on_a_held_name_closes_it(journal):
    market = StubMarket({"RELIANCE": 1_000.0})
    strategy = StubStrategy(
        picks=(), decisions={"RELIANCE": Decision("RELIANCE", Action.SELL, 9, "reversal")}
    )
    deps = build(journal, market, strategy)
    hold_a_position(deps, market, stop=900.0, target=1_500.0)

    report = run_cycle(deps, NOW)

    assert report.exits == ("RELIANCE",)
    assert journal.query("SELECT intent FROM orders ORDER BY id")[-1]["intent"] \
        == "model_exit"


def test_a_model_sell_on_something_not_held_does_nothing(journal):
    """The cash segment has no shorting. A sell with no position is just noise."""
    strategy = StubStrategy(
        decisions={"RELIANCE": Decision("RELIANCE", Action.SELL, 9, "reversal")}
    )
    deps = build(journal, StubMarket(), strategy)
    report = run_cycle(deps, NOW)
    assert report.exits == () and report.entries == ()
    assert journal.query("SELECT * FROM orders") == []
    assert len(journal.query("SELECT * FROM decisions")) == 1


# ----------------------------------------------------------- circuit breakers
def test_a_halted_book_opens_nothing_and_says_why(journal):
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(decisions=buy()),
                 risk_overrides={"max_trades_per_day": 0})
    report = run_cycle(deps, NOW)

    assert report.risk_state.halted is True
    assert report.entries == ()
    assert report.picks.abstain is True
    assert report.picks.rationale == report.risk_state.reason
    assert deps.strategy.picked == 0          # the model is not even consulted
    assert journal.query("SELECT * FROM cycles")[0]["halted"] == 1


def test_a_halted_book_still_takes_its_exits(journal):
    """Circuit breakers stop the book growing. They never trap a losing position."""
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(picks=()),
                 risk_overrides={"max_trades_per_day": 0})
    hold_a_position(deps, market, stop=1_050.0)
    report = run_cycle(deps, NOW)
    assert report.risk_state.halted is True
    assert report.exits == ("RELIANCE",)


def test_past_the_square_off_cut_off_no_new_position_is_opened(journal):
    """After the cut-off a new entry could only be carried overnight, which is a
    different segment with different costs."""
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(decisions=buy()),
                 risk_overrides={"square_off_enabled": True})
    deps.broker.set_clock(LATE)
    report = run_cycle(deps, LATE)
    assert report.risk_state.halted is True
    assert "square-off" in report.risk_state.reason
    assert report.entries == ()


def test_square_off_flattens_the_book_at_the_cut_off(journal):
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(picks=()),
                 risk_overrides={"square_off_enabled": True})
    hold_a_position(deps, market, stop=900.0, target=1_500.0)
    deps.broker.set_clock(LATE)

    report = run_cycle(deps, LATE)

    assert report.exits == ("RELIANCE",)
    assert deps.broker.positions() == ()


def test_a_full_book_asks_for_nothing_new(journal):
    market = StubMarket({"RELIANCE": 1_000.0, "TCS": 3_000.0, "INFY": 1_500.0})
    deps = build(journal, market, StubStrategy(picks=()),
                 universe=("RELIANCE", "TCS", "INFY"), risk_overrides={"max_positions": 1})
    hold_a_position(deps, market, stop=900.0, target=1_500.0)
    report = run_cycle(deps, NOW)
    assert deps.strategy.picked == 0
    assert report.picks.symbols == ()


def test_the_peak_for_the_drawdown_breaker_comes_off_the_recorded_curve(journal):
    """Peak equity has to survive the process exiting between cycles, or the
    drawdown breaker can never fire on a bot that restarts every 15 minutes."""
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(decisions=buy()),
                 risk_overrides={"max_drawdown_pct": 0.05})
    journal.record_equity(deps.run_id, NOW - timedelta(days=1), 200_000.0, 200_000.0, 0.0, None)

    report = run_cycle(deps, NOW)

    assert report.risk_state.halted is True
    assert "drawdown" in report.risk_state.reason


# ------------------------------------------------------------- degraded data
def test_one_broken_symbol_does_not_kill_the_cycle(journal):
    market = StubMarket(bar_error={"TCS"})
    deps = build(journal, market, StubStrategy(decisions=buy()))
    report = run_cycle(deps, NOW)
    assert report.entries == ("RELIANCE",)


def test_a_symbol_with_no_bars_is_skipped_with_a_reason(journal):
    market = StubMarket({"RELIANCE": 1_000.0})
    deps = build(journal, market, StubStrategy(picks=("GHOST",)),
                 universe=("RELIANCE", "GHOST"))
    report = run_cycle(deps, NOW)
    assert ("GHOST", "no bars") in report.skipped
    assert "GHOST" not in deps.strategy.decided


def test_no_data_at_all_stands_down_without_crashing(journal, caplog):
    caplog.set_level("ERROR")
    market = StubMarket(bar_error={"RELIANCE", "TCS"})
    deps = build(journal, market, StubStrategy(decisions=buy()))
    report = run_cycle(deps, NOW)
    assert report.entries == () and report.exits == ()
    assert deps.strategy.picked == 0          # no overview, so nothing to pick from
    assert "No market data" in caplog.text
    assert len(journal.query("SELECT * FROM equity_curve")) == 1


def test_a_missing_quote_stops_the_entry_but_not_the_cycle(journal):
    """A dead quote endpoint must not take the cycle down -- the decision is
    still made, journalled and available for calibration. It must also not open
    a position: with no quote there is nothing to check staleness or spread
    against, so the entry would rest entirely on a bar close of unknown age."""
    market = StubMarket(quote_error={"RELIANCE"})
    deps = build(journal, market, StubStrategy(decisions=buy()))
    report = run_cycle(deps, NOW)
    assert "RELIANCE" in deps.strategy.decided
    assert report.decisions[0].symbol == "RELIANCE"
    assert report.entries == ()
    assert any("no quote" in reason for _, reason in report.skipped)


def test_a_strategy_that_raises_costs_one_symbol_not_the_cycle(journal):
    market = StubMarket()
    deps = build(journal, market, StubStrategy(decide_error=RuntimeError("api 529")))
    report = run_cycle(deps, NOW)
    assert report.decisions == ()
    assert any("decision error" in reason for _, reason in report.skipped)
    assert len(journal.query("SELECT * FROM equity_curve")) == 1


# ------------------------------------------------------------------ plumbing
def test_held_names_are_reviewed_even_when_they_are_not_picked(journal):
    market = StubMarket({"RELIANCE": 1_000.0, "TCS": 3_000.0})
    deps = build(journal, market, StubStrategy(picks=("TCS",),
                                               decisions={"TCS": Decision("TCS", Action.HOLD, 4, "no")}))
    hold_a_position(deps, market, stop=900.0, target=1_500.0)
    run_cycle(deps, NOW)
    assert set(deps.strategy.decided) == {"RELIANCE", "TCS"}


def test_the_universe_includes_held_names_that_fell_out_of_it(journal):
    """A position must never become invisible because the config changed."""
    market = StubMarket({"RELIANCE": 1_000.0, "TCS": 3_000.0})
    deps = build(journal, market, StubStrategy(picks=()), universe=("TCS",))
    hold_a_position(deps, market, stop=900.0, target=1_500.0)
    run_cycle(deps, NOW)
    assert "RELIANCE" in market.bars_asked


def test_dry_run_decides_everything_and_places_nothing(journal):
    deps = build(journal, StubMarket(), StubStrategy(decisions=buy()), dry_run=True)
    report = run_cycle(deps, NOW)
    assert report.entries == ()
    assert len(journal.query("SELECT * FROM decisions")) == 1
    assert journal.query("SELECT * FROM orders") == []
    assert deps.broker.positions() == ()


def test_two_cycles_accumulate_rather_than_reset(journal):
    market = StubMarket()
    deps = build(journal, market, StubStrategy(decisions=buy()))
    run_cycle(deps, NOW)
    deps.broker.set_clock(NOW + timedelta(minutes=15))
    run_cycle(deps, NOW + timedelta(minutes=15))
    assert len(journal.query("SELECT * FROM cycles")) == 2
    assert len(journal.query("SELECT * FROM equity_curve")) == 2


def test_the_per_cycle_trade_cap_is_enforced_across_the_loop(journal):
    market = StubMarket({"RELIANCE": 1_000.0, "TCS": 3_000.0, "INFY": 1_500.0})
    strategy = StubStrategy(
        picks=("RELIANCE", "TCS", "INFY"),
        decisions={s: Decision(s, Action.BUY, 9, "breakout") for s in ("RELIANCE", "TCS", "INFY")},
    )
    deps = build(journal, market, strategy,
                 universe=("RELIANCE", "TCS", "INFY"),
                 risk_overrides={"max_positions": 5, "max_trades_per_cycle": 2})
    report = run_cycle(deps, NOW)
    assert len(report.entries) <= 2


def test_the_report_counts_both_sides_of_the_book(journal):
    market = StubMarket({"RELIANCE": 1_000.0, "TCS": 3_000.0})
    strategy = StubStrategy(picks=("TCS",), decisions=buy("TCS"))
    deps = build(journal, market, strategy)
    hold_a_position(deps, market, stop=1_050.0)       # will be stopped out
    report = run_cycle(deps, NOW)
    assert report.exits == ("RELIANCE",)
    assert report.entries == ("TCS",)
    assert report.traded == 2
