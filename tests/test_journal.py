"""The journal.

This is the component whose absence makes everything else unmeasurable. The
original bot had none: it printed a decision, placed an order, and forgot. These
tests care mostly about continuity -- a live run spans hundreds of GitHub Actions
invocations, and each one has to pick the same thread back up.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.journal.schema import SCHEMA_VERSION
from claude_trader.journal.store import Journal
from claude_trader.models import (
    Action,
    Decision,
    OrderResult,
    Picks,
    PortfolioState,
    Position,
    PositionRisk,
    RiskVerdict,
    Side,
)
from tests.conftest import make_state

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)


def _order(symbol="RELIANCE", side=Side.BUY, qty=10.0, price=1_000.0, when=NOW):
    return OrderResult(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        order_id="sim-1",
        status="filled",
        submitted_at=when,
        simulated=True,
    )


def _cycle(journal: Journal, run_id: int, state: PortfolioState | None = None) -> int:
    return journal.record_cycle(
        run_id, NOW, state or make_state(), None, market_open=True
    )


# ------------------------------------------------------------------- schema
def test_a_new_journal_stamps_its_schema_version(journal):
    rows = journal.query("SELECT value FROM meta WHERE key = 'schema_version'")
    assert rows[0]["value"] == str(SCHEMA_VERSION)


def test_opening_an_existing_journal_is_not_destructive(tmp_path):
    path = tmp_path / "nested" / "journal.sqlite3"
    with Journal(path) as first:
        run_id = first.start_run("live", "momentum", NOW)
        first.record_equity(run_id, NOW, 100_000.0, 100_000.0)
    with Journal(path) as second:
        assert second.latest_run("live")["id"] == run_id
        assert len(second.equity_curve(run_id)) == 1


def test_the_parent_directory_is_created(tmp_path):
    """GitHub Actions restores an empty workspace; data/ will not exist."""
    path = tmp_path / "data" / "journal.sqlite3"
    Journal(path).close()
    assert path.exists()


# --------------------------------------------------------------------- runs
def test_a_live_run_is_resumed_rather_than_restarted(journal):
    """A new run every 15 minutes would reset the equity curve, the peak, and
    the daily trade count -- every circuit breaker depends on this."""
    first = journal.resolve_live_run("momentum", NOW, {})
    second = journal.resolve_live_run("momentum", NOW + timedelta(minutes=15), {})
    assert first == second


def test_a_different_strategy_gets_its_own_run(journal):
    momentum = journal.resolve_live_run("momentum", NOW, {})
    claude = journal.resolve_live_run("claude", NOW, {})
    assert momentum != claude


def test_backtest_runs_do_not_capture_the_live_thread(journal):
    live = journal.resolve_live_run("momentum", NOW, {})
    journal.start_run("backtest", "momentum", NOW)
    assert journal.resolve_live_run("momentum", NOW, {}) == live
    assert journal.latest_run("backtest")["kind"] == "backtest"


def test_the_configuration_is_stored_with_the_run(journal):
    """A result nobody can reproduce is an anecdote."""
    run_id = journal.start_run("backtest", "momentum", NOW, {"min_confidence": 7})
    row = journal.query("SELECT config_json FROM runs WHERE id = ?", (run_id,))[0]
    assert '"min_confidence": 7' in row["config_json"]


def test_finishing_a_run_is_recorded(journal):
    run_id = journal.start_run("backtest", "momentum", NOW)
    journal.finish_run(run_id, NOW + timedelta(hours=1))
    assert journal.latest_run()["finished_at"] is not None


def test_latest_run_is_none_on_an_empty_journal(journal):
    assert journal.latest_run() is None


# ------------------------------------------------------------------- cycles
def test_a_cycle_records_the_book_and_the_picks(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    state = make_state(cash=90_000.0, positions=[Position("TCS", 3, 3_000.0, 3_100.0)])
    picks = Picks(("TCS", "INFY"), "buy strength", "bullish", abstain=False)
    cycle_id = journal.record_cycle(run_id, NOW, state, picks, market_open=True)
    row = journal.query("SELECT * FROM cycles WHERE id = ?", (cycle_id,))[0]
    assert row["position_count"] == 1
    assert row["market_mood"] == "bullish"
    assert "INFY" in row["picks_json"]
    assert row["abstained"] == 0


def test_a_halted_cycle_records_why(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    cycle_id = journal.record_cycle(
        run_id, NOW, make_state(), None, market_open=True,
        halted=True, halt_reason="daily loss limit",
    )
    row = journal.query("SELECT * FROM cycles WHERE id = ?", (cycle_id,))[0]
    assert row["halted"] == 1 and row["halt_reason"] == "daily loss limit"


def test_an_abstaining_cycle_is_distinguishable_from_a_quiet_one(journal):
    """'The model looked and declined' and 'nothing ran' produce the same empty
    order list, and only one of them is a working bot."""
    run_id = journal.resolve_live_run("claude", NOW, {})
    picks = Picks((), "no edge", "neutral", abstain=True)
    cycle_id = journal.record_cycle(run_id, NOW, make_state(), picks, True)
    assert journal.query("SELECT abstained FROM cycles WHERE id = ?", (cycle_id,))[0][0] == 1


# ---------------------------------------------------------------- decisions
def test_a_rejected_decision_is_still_recorded(journal):
    """The trades the risk layer refused are the most informative rows in the
    table -- they are the counterfactual for every rule."""
    run_id = journal.resolve_live_run("claude", NOW, {})
    cycle_id = _cycle(journal, run_id)
    decision = Decision("RELIANCE", Action.BUY, 9, "breakout", notional=9_500)
    decision_id = journal.record_decision(
        run_id, cycle_id, NOW, decision, price=1_000.0,
        verdict=RiskVerdict(False, "spread 210 bps too wide"),
        prompt_sha="abc123",
    )
    row = journal.query("SELECT * FROM decisions WHERE id = ?", (decision_id,))[0]
    assert row["risk_approved"] == 0
    assert row["risk_reason"] == "spread 210 bps too wide"
    assert row["executed"] == 0
    assert row["prompt_sha"] == "abc123"
    assert row["confidence"] == 9


def test_execution_is_marked_separately_from_the_decision(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    cycle_id = _cycle(journal, run_id)
    decision_id = journal.record_decision(
        run_id, cycle_id, NOW, Decision("TCS", Action.BUY, 8, "x"), price=3_000.0
    )
    journal.mark_decision_executed(decision_id)
    assert journal.query(
        "SELECT executed FROM decisions WHERE id = ?", (decision_id,)
    )[0][0] == 1


def test_indicators_are_stored_alongside_the_decision(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    cycle_id = _cycle(journal, run_id)
    decision_id = journal.record_decision(
        run_id, cycle_id, NOW, Decision("TCS", Action.HOLD, 4, "flat"),
        price=3_000.0, indicators={"rsi": 51.2, "trend": "sideways"},
    )
    row = journal.query("SELECT indicators_json FROM decisions WHERE id = ?", (decision_id,))[0]
    assert "51.2" in row["indicators_json"]


# ------------------------------------------------------------------- orders
def test_orders_are_linked_to_the_decision_that_caused_them(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    cycle_id = _cycle(journal, run_id)
    decision_id = journal.record_decision(
        run_id, cycle_id, NOW, Decision("RELIANCE", Action.BUY, 8, "x"), 1_000.0
    )
    order_id = journal.record_order(
        run_id, cycle_id, _order(), decision_id=decision_id, intent="entry"
    )
    row = journal.query("SELECT * FROM orders WHERE id = ?", (order_id,))[0]
    assert row["decision_id"] == decision_id
    assert row["intent"] == "entry"
    assert row["notional"] == pytest.approx(10_000.0)
    assert row["simulated"] == 1


def test_the_daily_trade_count_is_per_day_not_per_run(journal):
    """The per-day cap has to survive the process exiting after every cycle."""
    run_id = journal.resolve_live_run("claude", NOW, {})
    cycle_id = _cycle(journal, run_id)
    journal.record_order(run_id, cycle_id, _order(when=NOW))
    journal.record_order(run_id, cycle_id, _order(when=NOW + timedelta(hours=1)))
    journal.record_order(run_id, cycle_id, _order(when=NOW + timedelta(days=1)))
    assert journal.trades_today(run_id, NOW) == 2
    assert journal.trades_today(run_id, NOW + timedelta(days=1)) == 1


def test_another_runs_orders_do_not_count(journal):
    a = journal.resolve_live_run("claude", NOW, {})
    b = journal.start_run("backtest", "momentum", NOW)
    journal.record_order(b, journal.record_cycle(b, NOW, make_state(), None, True), _order())
    assert journal.trades_today(a, NOW) == 0


# ------------------------------------------------------------ position risk
def test_stops_survive_the_process_exiting(journal):
    """Between two GitHub Actions runs there is nothing in memory. If the stop
    is not in the database, the position is unprotected."""
    run_id = journal.resolve_live_run("claude", NOW, {})
    risk = PositionRisk("RELIANCE", 1_000.0, NOW, 980.0, 1_030.0, 1_000.0, 10.0)
    journal.upsert_position_risk(run_id, risk)
    restored = journal.open_position_risks(run_id)
    assert len(restored) == 1
    assert restored[0].stop_price == 980.0
    assert restored[0].entry_time == NOW


def test_a_trailed_stop_overwrites_rather_than_duplicating(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    journal.upsert_position_risk(
        run_id, PositionRisk("RELIANCE", 1_000.0, NOW, 980.0, 1_030.0, 1_000.0, 10.0)
    )
    journal.upsert_position_risk(
        run_id,
        PositionRisk("RELIANCE", 1_000.0, NOW, 1_020.0, 1_030.0, 1_040.0, 10.0, bars_held=3),
    )
    risks = journal.open_position_risks(run_id)
    assert len(risks) == 1
    assert risks[0].stop_price == 1_020.0
    assert risks[0].bars_held == 3


def test_closing_a_position_retires_its_risk_record(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    journal.upsert_position_risk(
        run_id, PositionRisk("RELIANCE", 1_000.0, NOW, 980.0, 1_030.0, 1_000.0, 10.0)
    )
    journal.close_position_risk(run_id, "RELIANCE")
    assert journal.open_position_risks(run_id) == ()


def test_reopening_a_closed_symbol_makes_it_open_again(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    risk = PositionRisk("RELIANCE", 1_000.0, NOW, 980.0, 1_030.0, 1_000.0, 10.0)
    journal.upsert_position_risk(run_id, risk)
    journal.close_position_risk(run_id, "RELIANCE")
    journal.upsert_position_risk(run_id, risk)
    assert len(journal.open_position_risks(run_id)) == 1


# ------------------------------------------------------------ equity curve
def test_the_equity_curve_is_ordered_and_de_duplicated(journal):
    """A retried Actions run must not write the same timestamp twice, or the
    return series double-counts a bar."""
    run_id = journal.resolve_live_run("claude", NOW, {})
    journal.record_equity(run_id, NOW + timedelta(minutes=15), 101_000.0, 90_000.0, 0.11)
    journal.record_equity(run_id, NOW, 100_000.0, 100_000.0, 0.0)
    journal.record_equity(run_id, NOW, 100_500.0, 100_000.0, 0.0)   # same ts, corrected
    curve = journal.equity_curve(run_id)
    assert [row["equity"] for row in curve] == [100_500.0, 101_000.0]


def test_the_benchmark_is_stored_next_to_equity(journal):
    """Beating a number requires storing the number at the time, not looking it
    up later and hoping the alignment is right."""
    run_id = journal.resolve_live_run("claude", NOW, {})
    journal.record_equity(run_id, NOW, 100_000.0, 100_000.0, benchmark_price=250.5)
    assert journal.equity_curve(run_id)[0]["benchmark_price"] == 250.5


# ---------------------------------------------------------------- outcomes
def test_outcomes_attach_to_decisions_and_are_idempotent(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    cycle_id = _cycle(journal, run_id)
    decision_id = journal.record_decision(
        run_id, cycle_id, NOW, Decision("TCS", Action.BUY, 8, "x"), 3_000.0
    )
    journal.record_outcome(decision_id, 6, 3_000.0, 3_030.0, 0.01, 0.002, NOW)
    journal.record_outcome(decision_id, 6, 3_000.0, 3_060.0, 0.02, 0.002, NOW)
    rows = journal.query("SELECT * FROM outcomes WHERE decision_id = ?", (decision_id,))
    assert len(rows) == 1
    assert rows[0]["forward_return"] == pytest.approx(0.02)


def test_unresolved_decisions_are_the_ones_without_an_outcome(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    cycle_id = _cycle(journal, run_id)
    first = journal.record_decision(
        run_id, cycle_id, NOW, Decision("TCS", Action.BUY, 8, "x"), 3_000.0
    )
    journal.record_decision(
        run_id, cycle_id, NOW, Decision("INFY", Action.BUY, 8, "x"), 1_500.0
    )
    journal.record_outcome(first, 6, 3_000.0, 3_030.0, 0.01, None, NOW)
    pending = journal.unresolved_decisions(run_id, 6)
    assert [row["symbol"] for row in pending] == ["INFY"]


def test_an_outcome_at_another_horizon_leaves_the_decision_unresolved(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    cycle_id = _cycle(journal, run_id)
    decision_id = journal.record_decision(
        run_id, cycle_id, NOW, Decision("TCS", Action.BUY, 8, "x"), 3_000.0
    )
    journal.record_outcome(decision_id, 6, 3_000.0, 3_030.0, 0.01, None, NOW)
    assert len(journal.unresolved_decisions(run_id, 24)) == 1


# --------------------------------------------------------------- llm cache
def test_cache_returns_what_was_stored_and_counts_hits(journal):
    journal.cache_put("k1", "claude-test", '{"action": "hold"}', NOW)
    assert journal.cache_get("k1") == '{"action": "hold"}'
    journal.cache_get("k1")
    hits = journal.query("SELECT hits FROM llm_cache WHERE key = 'k1'")[0]["hits"]
    assert hits == 2


def test_a_cache_miss_is_none_not_an_error(journal):
    assert journal.cache_get("never-seen") is None


def test_re_caching_a_key_keeps_the_hit_count(journal):
    journal.cache_put("k1", "m", "a", NOW)
    journal.cache_get("k1")
    journal.cache_put("k1", "m", "b", NOW)
    row = journal.query("SELECT response, hits FROM llm_cache WHERE key = 'k1'")[0]
    assert row["response"] == "b" and row["hits"] == 1


# ------------------------------------------------------------ transactions
def test_a_failed_transaction_leaves_nothing_behind(journal):
    run_id = journal.resolve_live_run("claude", NOW, {})
    with pytest.raises(RuntimeError):
        with journal.transaction() as conn:
            conn.execute(
                "INSERT INTO equity_curve(run_id, ts, equity, cash, exposure)"
                " VALUES (?, '2026-03-02T05:00:00+00:00', 1, 1, 0)",
                (run_id,),
            )
            raise RuntimeError("broker call failed")
    assert journal.equity_curve(run_id) == []


def test_naive_timestamps_are_stored_as_utc(journal):
    """Mixing naive and aware timestamps in one column makes every ORDER BY and
    every date prefix comparison silently wrong."""
    run_id = journal.start_run("backtest", "momentum", datetime(2026, 3, 2, 5, 0))
    stored = journal.latest_run()["started_at"]
    assert stored.endswith("+00:00")
