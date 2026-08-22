"""Order execution and journalling.

The original bot read its account once at the top of the cycle and then placed
several orders against that stale snapshot, so the second and third trades of a
cycle were sized against money already spent. The property being defended here
is that every fill produces a *new* state, and that everything which happened is
written down whether or not it worked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.engine.executor import Executor
from claude_trader.errors import BrokerError, OrderRejected
from claude_trader.markets import INDIA_MARKET, US_MARKET
from claude_trader.models import (
    Action,
    Decision,
    ExitReason,
    OrderResult,
    Position,
    PositionRisk,
    RiskVerdict,
    Side,
)
from tests.conftest import make_bars, make_snapshot, make_state, ramp

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)


class StubBroker:
    """Records what it was asked to do and fills whatever it is told to fill."""

    def __init__(self, fill=None, raises=None, close_fill=None, close_raises=None):
        self.fill = fill
        self.raises = raises
        self.close_fill = close_fill
        self.close_raises = close_raises
        self.orders = []
        self.closed = []

    def submit(self, order):
        self.orders.append(order)
        if self.raises:
            raise self.raises
        return self.fill

    def close_position(self, symbol):
        self.closed.append(symbol)
        if self.close_raises:
            raise self.close_raises
        return self.close_fill


def filled(symbol="RELIANCE", side=Side.BUY, qty=9.0, price=1_000.0, **kwargs):
    return OrderResult(
        symbol=symbol, side=side, qty=qty, price=price,
        order_id="paper-1", status="filled", submitted_at=NOW, **kwargs
    )


def snapshot(symbol="RELIANCE", price=1_000.0):
    return make_snapshot(
        symbol, price=price, bars=make_bars(symbol, ramp(20, price)), when=NOW
    )


def verdict(**kwargs):
    base = dict(approved=True, reason="ok", notional=9_000.0, qty=9.0,
                stop_price=980.0, target_price=1_040.0)
    return RiskVerdict(**{**base, **kwargs})


def risk(symbol="RELIANCE", **kwargs):
    base = dict(symbol=symbol, entry_price=1_000.0, entry_time=NOW, stop_price=980.0,
                target_price=1_040.0, high_water=1_000.0, atr_at_entry=10.0)
    return PositionRisk(**{**base, **kwargs})


@pytest.fixture
def run(journal):
    run_id = journal.start_run("live", "claude", NOW)
    cycle_id = journal.record_cycle(run_id, NOW, make_state(), None, market_open=True)
    return run_id, cycle_id


def executor(journal, run, broker, *, dry_run=False, profile=INDIA_MARKET):
    return Executor(broker, journal, run[0], dry_run=dry_run, profile=profile)


def enter(ex, run, state, *, decision=None, v=None, snap=None, r=None, decision_id=None):
    return ex.open_position(
        state=state,
        decision=decision or Decision("RELIANCE", Action.BUY, 8, "breakout"),
        verdict=v or verdict(),
        snapshot=snap or snapshot(),
        risk=r or risk(),
        cycle_id=run[1],
        decision_id=decision_id,
        now=NOW,
    )


# ------------------------------------------------------------------ entries
def test_a_fill_produces_a_new_state_rather_than_mutating_the_old_one(journal, run):
    """This is the bug that made multi-trade cycles overspend: the old state must
    still read as untouched so nothing can accidentally keep using it."""
    ex = executor(journal, run, StubBroker(fill=filled(qty=9, price=1_000.0)))
    before = make_state(cash=100_000.0)
    after, result = enter(ex, run, before)

    assert result is not None
    assert before.positions == {}
    assert before.account.cash == 100_000.0
    assert after.account.cash == pytest.approx(91_000.0)
    assert after.positions["RELIANCE"].qty == 9


def test_the_second_trade_of_a_cycle_sees_the_first_ones_cash(journal, run):
    ex = executor(journal, run, StubBroker(fill=filled(qty=9, price=1_000.0)))
    state = make_state(cash=100_000.0)
    state, _ = enter(ex, run, state)
    state, _ = enter(ex, run, state)
    assert state.account.cash == pytest.approx(82_000.0)
    assert ex.trades_this_cycle == 2


def test_nse_entries_are_sent_as_whole_shares(journal, run):
    """A notional order on a whole-share market delegates the rounding decision
    to the broker, who will round somewhere the risk engine did not agree to."""
    broker = StubBroker(fill=filled())
    enter(executor(journal, run, broker), run, make_state())
    order = broker.orders[0]
    assert order.qty == 9.0 and order.notional is None
    assert order.side is Side.BUY
    assert order.intent == "entry"


def test_us_entries_are_sent_as_notional(journal, run):
    broker = StubBroker(fill=filled(price=100.0, qty=95.0))
    enter(executor(journal, run, broker, profile=US_MARKET), run, make_state())
    order = broker.orders[0]
    assert order.notional == 9_000.0 and order.qty is None


def test_the_client_order_id_makes_a_retry_idempotent(journal, run):
    broker = StubBroker(fill=filled())
    enter(executor(journal, run, broker), run, make_state())
    assert broker.orders[0].client_order_id == f"ct-{run[1]}-RELIANCE-buy"


def test_the_stop_is_re_anchored_to_the_price_actually_paid(journal, run):
    """Anchoring to the intended price would leave the real risk on the trade
    different from the risk the sizing was computed against."""
    ex = executor(journal, run, StubBroker(fill=filled(price=1_012.0)))
    state, _ = enter(ex, run, make_state(), r=risk(entry_price=1_000.0))
    entry = state.risks["RELIANCE"]
    assert entry.entry_price == 1_012.0
    assert entry.high_water == 1_012.0
    assert entry.entry_time == NOW
    assert entry.bars_held == 0
    assert entry.stop_price == 980.0          # the risk engine's level, unchanged


def test_the_entry_is_written_to_the_journal(journal, run):
    ex = executor(journal, run, StubBroker(fill=filled()))
    decision_id = journal.record_decision(
        run[0], run[1], NOW, Decision("RELIANCE", Action.BUY, 8, "x"), 1_000.0
    )
    enter(ex, run, make_state(), decision_id=decision_id)

    order_row = journal.query("SELECT * FROM orders")[0]
    assert order_row["intent"] == "entry"
    assert order_row["decision_id"] == decision_id
    assert journal.query("SELECT executed FROM decisions")[0]["executed"] == 1
    assert journal.query("SELECT * FROM position_risk")[0]["symbol"] == "RELIANCE"


def test_an_entry_without_a_decision_row_still_records_the_order(journal, run):
    """Forced re-entries and manual orders have no model decision behind them."""
    enter(executor(journal, run, StubBroker(fill=filled())), run, make_state())
    assert len(journal.query("SELECT * FROM orders")) == 1


# ------------------------------------------------------------------- exits
def test_an_exit_flattens_the_position_and_returns_the_cash(journal, run):
    broker = StubBroker(close_fill=filled(side=Side.SELL, qty=9, price=1_100.0))
    state = make_state(
        cash=10_000.0,
        positions=[Position("RELIANCE", 9, 1_000.0, 1_100.0)],
        risks=[risk()],
    )
    after, result = executor(journal, run, broker).close_position(
        state, "RELIANCE", snapshot(price=1_100.0), ExitReason.TAKE_PROFIT,
        "target hit", run[1],
    )
    assert broker.closed == ["RELIANCE"]
    assert result.qty == 9
    assert "RELIANCE" not in after.positions
    assert after.account.cash == pytest.approx(19_900.0)


def test_the_exit_reason_is_recorded_as_the_order_intent(journal, run):
    """Six months later, 'why did it sell' is answerable only if this is stored."""
    broker = StubBroker(close_fill=filled(side=Side.SELL, qty=9, price=970.0))
    state = make_state(positions=[Position("RELIANCE", 9, 1_000.0, 970.0)], risks=[risk()])
    executor(journal, run, broker).close_position(
        state, "RELIANCE", snapshot(price=970.0), ExitReason.STOP_LOSS,
        "stop breached", run[1],
    )
    assert journal.query("SELECT * FROM orders")[0]["intent"] == "stop_loss"


def test_the_stop_row_is_retired_when_the_position_closes(journal, run):
    """A stale stop row would fire against a position that no longer exists."""
    broker = StubBroker(close_fill=filled(side=Side.SELL, qty=9, price=970.0))
    journal.upsert_position_risk(run[0], risk())
    state = make_state(positions=[Position("RELIANCE", 9, 1_000.0, 970.0)], risks=[risk()])
    executor(journal, run, broker).close_position(
        state, "RELIANCE", None, ExitReason.STOP_LOSS, "x", run[1]
    )
    assert journal.open_position_risks(run[0]) == ()


def test_closing_something_that_is_not_held_is_a_no_op(journal, run):
    broker = StubBroker(close_fill=filled(side=Side.SELL))
    state = make_state()
    after, result = executor(journal, run, broker).close_position(
        state, "TCS", None, ExitReason.TIME_STOP, "x", run[1]
    )
    assert after is state and result is None
    assert broker.closed == []


def test_an_exit_without_a_snapshot_marks_against_the_position(journal, run):
    """Square-off runs after the data fetch has already been thrown away."""
    broker = StubBroker(close_fill=filled(side=Side.SELL, qty=9, price=0.0))
    state = make_state(positions=[Position("RELIANCE", 9, 1_000.0, 1_050.0)], risks=[risk()])
    after, result = executor(journal, run, broker).close_position(
        state, "RELIANCE", None, ExitReason.SQUARE_OFF, "session end", run[1]
    )
    assert result.price == 1_050.0
    assert after.account.cash == pytest.approx(100_000.0 + 9 * 1_050.0)


# --------------------------------------------------------------- robustness
@pytest.mark.parametrize("error", [BrokerError("down"), OrderRejected("no shorting")])
def test_a_broker_error_on_entry_leaves_the_book_untouched(journal, run, error):
    ex = executor(journal, run, StubBroker(raises=error))
    state = make_state(cash=100_000.0)
    after, result = enter(ex, run, state)
    assert after is state and result is None
    assert ex.trades_this_cycle == 0
    assert journal.query("SELECT * FROM orders") == []


@pytest.mark.parametrize("error", [BrokerError("down"), OrderRejected("nothing held")])
def test_a_broker_error_on_exit_leaves_the_position_open(journal, run, error):
    """Pretending the exit worked would drop the stop on a position that is
    still live -- strictly worse than failing loudly."""
    ex = executor(journal, run, StubBroker(close_raises=error))
    state = make_state(positions=[Position("RELIANCE", 9, 1_000.0, 990.0)], risks=[risk()])
    after, result = ex.close_position(
        state, "RELIANCE", None, ExitReason.STOP_LOSS, "x", run[1]
    )
    assert after is state and result is None
    assert "RELIANCE" in after.positions


def test_a_rejected_order_that_returns_nothing_is_not_counted_as_a_trade(journal, run):
    ex = executor(journal, run, StubBroker(fill=None))
    state, result = enter(ex, run, make_state())
    assert result is None and ex.trades_this_cycle == 0


def test_an_exit_that_returns_no_fill_keeps_the_position(journal, run):
    ex = executor(journal, run, StubBroker(close_fill=None))
    state = make_state(positions=[Position("RELIANCE", 9, 1_000.0, 990.0)], risks=[risk()])
    after, result = ex.close_position(
        state, "RELIANCE", None, ExitReason.TIME_STOP, "x", run[1]
    )
    assert result is None and "RELIANCE" in after.positions


def test_an_accepted_order_with_no_fill_price_yet_uses_the_reference(journal, run):
    """A market order accepted a millisecond ago has no average price. Writing
    the zero would poison every downstream P&L number."""
    pending = OrderResult("RELIANCE", Side.BUY, 9.0, 0.0, "id", "", NOW)
    ex = executor(journal, run, StubBroker(fill=pending))
    state, result = enter(ex, run, make_state(), snap=snapshot(price=1_007.0))
    assert result.price == 1_007.0
    assert result.status == "accepted"
    assert state.positions["RELIANCE"].avg_entry_price == 1_007.0


def test_a_zero_price_with_no_reference_is_left_alone(journal, run):
    """Inventing a price is worse than recording that there wasn't one."""
    pending = OrderResult("RELIANCE", Side.BUY, 9.0, 0.0, "id", "new", NOW)
    ex = executor(journal, run, StubBroker(fill=pending))
    state, result = enter(ex, run, make_state(), snap=snapshot(price=0.0))
    assert result.price == 0.0
    assert state.positions == {}          # with_fill refuses a zero-price fill


def test_a_zero_price_entry_keeps_the_planned_risk_levels(journal, run):
    pending = OrderResult("RELIANCE", Side.BUY, 9.0, 0.0, "id", "new", NOW)
    ex = executor(journal, run, StubBroker(fill=pending))
    planned = risk(entry_price=999.0)
    _, result = enter(ex, run, make_state(), snap=snapshot(price=0.0), r=planned)
    assert journal.open_position_risks(run[0])[0].entry_price == 999.0


# ----------------------------------------------------------------- dry run
def test_dry_run_places_nothing_and_records_nothing(journal, run):
    broker = StubBroker(fill=filled())
    ex = executor(journal, run, broker, dry_run=True)
    state = make_state(cash=100_000.0)
    after, result = enter(ex, run, state)

    assert ex.dry_run is True
    assert after is state and result is None
    assert broker.orders == []
    assert journal.query("SELECT * FROM orders") == []
    assert ex.trades_this_cycle == 0


def test_dry_run_also_refuses_to_exit(journal, run):
    """A dry run that skips entries but performs exits would liquidate a real
    book on the first accidental run."""
    broker = StubBroker(close_fill=filled(side=Side.SELL))
    ex = executor(journal, run, broker, dry_run=True)
    state = make_state(positions=[Position("RELIANCE", 9, 1_000.0, 990.0)], risks=[risk()])
    after, result = ex.close_position(
        state, "RELIANCE", None, ExitReason.STOP_LOSS, "x", run[1]
    )
    assert after is state and result is None
    assert broker.closed == []


# ---------------------------------------------------------------- currency
def test_log_amounts_follow_the_market_profile(journal, run, caplog):
    caplog.set_level("INFO")
    ex = executor(journal, run, StubBroker(fill=filled()))
    enter(ex, run, make_state())
    assert "₹" in caplog.text and "$" not in caplog.text


def test_the_us_profile_logs_dollars(journal, run, caplog):
    caplog.set_level("INFO")
    ex = executor(journal, run, StubBroker(fill=filled(price=100.0)), profile=US_MARKET)
    enter(ex, run, make_state())
    assert "$" in caplog.text and "₹" not in caplog.text


def test_the_trade_counter_spans_entries_and_exits(journal, run):
    broker = StubBroker(
        fill=filled(), close_fill=filled(side=Side.SELL, qty=9, price=1_050.0)
    )
    ex = executor(journal, run, broker)
    state, _ = enter(ex, run, make_state())
    ex.close_position(state, "RELIANCE", None, ExitReason.TAKE_PROFIT, "x", run[1])
    assert ex.trades_this_cycle == 2
