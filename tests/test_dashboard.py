"""The HTML dashboard.

The page exists to answer "what did it just do, and what did it refuse to do".
These tests defend the second half of that sentence -- the part a broker screen
never shows you -- and the property that makes the file safe to open at all:
it renders model-authored text, and it must render it as text.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from claude_trader.analytics.dashboard import DashboardData, collect, render_dashboard
from claude_trader.analytics.metrics import Performance, RoundTrip

IST = timezone.utc


def _performance(**overrides) -> Performance:
    base = dict(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 3, 1, tzinfo=timezone.utc),
        samples=100, starting_equity=100_000.0, ending_equity=104_000.0,
        total_return=0.04, annualised_return=0.31, annualised_vol=0.18,
        sharpe=1.2, sortino=1.5, max_drawdown=0.06, avg_exposure=0.4,
        benchmark_return=0.01, excess_return=0.03, trades=2, win_rate=0.5,
        profit_factor=1.4, avg_win=200.0, avg_loss=-140.0, expectancy=30.0,
    )
    base.update(overrides)
    return Performance(**base)


def _data(**overrides) -> DashboardData:
    base = dict(
        run_id=1, strategy="momentum", kind="backtest",
        started="2025-01-01T03:45:00+00:00", finished="2025-03-01T10:00:00+00:00",
        market="in", currency="INR", benchmark="NIFTYBEES",
        performance=_performance(), equity=(), positions=(), round_trips=(),
        orders=(), decisions=(), cycles=(),
    )
    base.update(overrides)
    return DashboardData(**base)


def _money(amount: float) -> str:
    return f"Rs {amount:,.2f}"


def _render(data: DashboardData) -> str:
    return render_dashboard(data, _money)


# ------------------------------------------------------------------- safety
def test_a_symbol_carrying_markup_is_rendered_as_text():
    """Symbols and reasons arrive from a language model. A dashboard that
    renders them as markup is a dashboard the model can write to."""
    hostile = {"symbol": "<img src=x onerror=alert(1)>", "action": "buy",
               "confidence": 8, "executed": 0, "risk_approved": 0,
               "risk_reason": "</table><script>steal()</script>",
               "reason": "", "ts": "2025-01-02T04:00:00+00:00"}
    html = _render(_data(decisions=(hostile,)))

    assert "<script>" not in html
    assert "<img" not in html
    assert "&lt;img" in html
    assert "&lt;script&gt;" in html


def test_the_page_makes_no_external_requests():
    """One file, opened from disk, with the network unplugged. Anything that
    reaches out is a dependency that can break or watch."""
    html = _render(_data())
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)
    assert "<script" not in html


# ------------------------------------------------- the decisions that matter
def test_holds_are_counted_rather_than_listed():
    """A momentum run journals thousands of holds. Listing them buries the
    handful of rows that carry information."""
    html = _render(_data(holds=3312))
    assert "3312" in html
    assert "hold" in html.lower()


def test_a_blocked_decision_shows_the_reason_the_gate_gave():
    """Without this the log cannot distinguish a strategy that found nothing
    from a risk layer that refused everything."""
    blocked = {"symbol": "SBIN", "action": "buy", "confidence": 9,
               "executed": 0, "risk_approved": 0,
               "risk_reason": "daily loss limit reached", "reason": "breakout",
               "ts": "2025-01-02T04:00:00+00:00"}
    html = _render(_data(decisions=(blocked,)))

    assert "daily loss limit reached" in html
    assert "blocked" in html


def test_an_executed_decision_is_not_reported_as_blocked():
    executed = {"symbol": "INFY", "action": "buy", "confidence": 8,
                "executed": 1, "risk_approved": 1, "risk_reason": "",
                "reason": "trend intact", "ts": "2025-01-02T04:00:00+00:00"}
    html = _render(_data(decisions=(executed,)))

    assert "executed" in html
    assert "blocked" not in html


# --------------------------------------------------------------- the numbers
def test_a_run_with_no_activity_still_renders():
    """The first thing a new user does is render an empty journal. It must
    explain itself rather than raise."""
    html = _render(_data(performance=_performance(trades=0)))

    assert "No equity history yet" in html
    assert "Nothing held right now" in html
    assert "No orders sent yet" in html


def test_a_closed_trade_reports_its_own_profit_and_loss():
    trip = RoundTrip(symbol="TCS", entry_time="2025-01-02T04:00:00+00:00",
                     exit_time="2025-01-02T09:00:00+00:00", qty=3,
                     entry_price=100.0, exit_price=110.0)
    html = _render(_data(round_trips=(trip,)))

    assert "TCS" in html
    assert "Rs 30.00" in html


def test_an_extrapolated_annual_figure_says_so():
    """A six-week sample annualised to a headline number is the single easiest
    way for this page to mislead its own author."""
    html = _render(_data(performance=_performance(annualised_extrapolated=True)))
    assert "extrapolated" in html


def test_the_page_names_the_run_and_the_market():
    html = _render(_data())
    assert "Run 1" in html
    assert "momentum" in html
    assert "NIFTYBEES" in html


def test_a_dry_run_is_flagged_on_the_page():
    assert "dry run" in _render(_data(dry_run=True))


def test_the_footer_refuses_to_be_read_as_advice():
    html = _render(_data())
    assert "not financial advice" in html.lower()
    assert "paper trading" in html.lower()


# ------------------------------------------------------------------ collect
def test_collecting_an_unknown_run_is_an_error_not_an_empty_page(journal):
    """An empty page for a run id that does not exist would read as 'this run
    did nothing', which is a different and much worse claim."""
    class _Config:
        market = "in"
        currency = "INR"
        benchmark = "NIFTYBEES"

        class profile:
            tz = timezone.utc

    with pytest.raises(ValueError, match="not in the journal"):
        collect(journal, 999, _Config(), _performance())


# ------------------------------------------------------------------- tables
def _equity(n: int = 5, bench: bool = True):
    return tuple(
        {"ts": f"2025-01-0{i + 1}T04:00:00+00:00", "equity": 100_000.0 + i * 500,
         "cash": 50_000.0, "exposure": 0.5,
         "benchmark_price": (275.0 + i) if bench else None}
        for i in range(n)
    )


def test_the_equity_curve_draws_both_lines_when_a_benchmark_is_present():
    """Shape against shape. Without the benchmark line a rising account looks
    like skill even when the whole index rose further."""
    html = _render(_data(equity=_equity()))

    assert html.count("<polyline") == 2
    assert "NIFTYBEES" in html
    assert "5 samples" in html


def test_a_run_without_benchmark_prices_still_draws_the_account():
    html = _render(_data(equity=_equity(bench=False)))
    assert html.count("<polyline") == 1


def test_a_single_equity_sample_is_not_a_curve():
    """One point is a number, not a line, and drawing it as one implies a
    history that does not exist."""
    assert "No equity history yet" in _render(_data(equity=_equity(1)))


def test_an_open_position_shows_its_stop_and_target():
    """The two prices that decide what happens next, on the same row as the
    entry that is being judged against them."""
    pos = {"symbol": "RELIANCE", "entry_price": 1400.0, "stop_price": 1350.0,
           "target_price": 1500.0, "bars_held": 4,
           "entry_time": "2025-01-02T04:00:00+00:00"}
    html = _render(_data(positions=(pos,)))

    assert "RELIANCE" in html
    assert "Rs 1,350.00" in html
    assert "Rs 1,500.00" in html


def test_a_simulated_order_is_labelled_as_simulated():
    """A backtest fill and a paper fill look identical in a table. Only one of
    them ever went near a broker."""
    order = {"ts": "2025-01-02T04:00:00+00:00", "symbol": "TCS", "side": "buy",
             "qty": 3, "price": 100.0, "notional": 300.0, "intent": "entry",
             "simulated": 1}
    html = _render(_data(orders=(order,)))

    assert "TCS" in html
    assert ">sim<" in html
    assert "BUY" in html


def test_only_the_hundred_most_recent_closed_trades_are_listed():
    """A three-hundred-row table is a scroll, not a report -- but the headline
    totals must still cover every trade."""
    trips = tuple(
        RoundTrip(symbol=f"S{i}", entry_time=f"2025-01-{i % 28 + 1:02d}T04:00:00+00:00",
                  exit_time=f"2025-02-{i % 28 + 1:02d}T04:00:00+00:00", qty=1,
                  entry_price=100.0, exit_price=101.0)
        for i in range(150)
    )
    html = _render(_data(round_trips=trips))

    assert html.count('<td class="sym">') == 100
    assert "150 closed trades" in html


def test_the_calibration_table_prints_the_verdict_it_was_given():
    """The verdict is the one line that says whether the confidence gate is
    doing anything, and it is the line most worth not burying."""
    from claude_trader.analytics.calibration import Bucket, Calibration

    cal = Calibration(
        horizon_bars=26, resolved=120,
        buckets=(
            Bucket(label="0-4", low=0, high=4, count=0, directional=0, hits=0,
                   avg_return=0.0, median_return=0.0, avg_benchmark=0.0),
            Bucket(label="7 (at gate)", low=7, high=7, count=80, directional=80,
                   hits=44, avg_return=0.004, median_return=0.003,
                   avg_benchmark=0.001),
        ),
        rank_correlation=0.31, monotonic=True,
        verdict="Higher confidence did track higher forward returns.",
    )
    html = _render(_data(calibration=cal))

    assert "Higher confidence did track higher forward returns." in html
    assert "rho +0.31" in html
    assert "ordered correctly: yes" in html


def test_calibration_with_nothing_resolved_says_what_to_run():
    from claude_trader.analytics.calibration import Calibration

    cal = Calibration(horizon_bars=26, resolved=0, buckets=(),
                      rank_correlation=None, monotonic=False, verdict="")
    assert "calibrate" in _render(_data(calibration=cal))


# --------------------------------------------------------- reading the journal
def _seed(journal, run_id: int = 1) -> None:
    """Enough of a run to exercise every read `collect` performs."""
    journal.query(
        "INSERT INTO runs (id, kind, strategy, started_at, finished_at, "
        "config_json, notes) VALUES (?, 'backtest', 'momentum', "
        "'2025-01-01T04:00:00+00:00', '2025-01-05T10:00:00+00:00', ?, '')",
        (run_id, '{"dry_run": true}'))
    journal.query(
        "INSERT INTO cycles (id, run_id, ts, market_open, equity, cash, "
        "position_count) VALUES (1, ?, '2025-01-02T04:00:00+00:00', 1, "
        "100000.0, 90000.0, 0)", (run_id,))
    journal.query(
        "INSERT INTO orders (run_id, cycle_id, ts, symbol, side, qty, price, "
        "notional, broker_id, status, simulated, intent) VALUES "
        "(?, 1, '2025-01-02T04:00:00+00:00', 'TCS', 'buy', 2, 100.0, 200.0, "
        "'x', 'filled', 1, 'entry')", (run_id,))
    journal.query(
        "INSERT INTO orders (run_id, cycle_id, ts, symbol, side, qty, price, "
        "notional, broker_id, status, simulated, intent) VALUES "
        "(?, 1, '2025-01-03T04:00:00+00:00', 'TCS', 'sell', 2, 110.0, 220.0, "
        "'y', 'filled', 1, 'stop_loss')", (run_id,))
    for action in ("buy", "hold", "hold"):
        journal.query(
            "INSERT INTO decisions (run_id, cycle_id, ts, symbol, action, "
            "confidence, reason, dollars, source, price, risk_approved, "
            "executed) VALUES (?, 1, '2025-01-02T04:00:00+00:00', 'TCS', ?, "
            "8, 'r', 200.0, 'momentum', 100.0, 1, 1)", (run_id, action))
    journal.query(
        "INSERT INTO position_risk (run_id, symbol, entry_price, entry_time, "
        "stop_price, target_price, high_water, atr_at_entry, bars_held, "
        "is_open) VALUES (?, 'INFY', 100.0, '2025-01-02T04:00:00+00:00', 95.0, "
        "115.0, 100.0, 2.0, 3, 1)", (run_id,))
    journal.query(
        "INSERT INTO equity_curve (run_id, ts, equity, cash, exposure, "
        "benchmark_price) VALUES (?, '2025-01-02T04:00:00+00:00', 100000.0, "
        "90000.0, 0.1, 275.0)", (run_id,))


class _Config:
    market = "in"
    currency = "INR"
    benchmark = "NIFTYBEES"

    class profile:
        tz = timezone.utc


def test_collect_pairs_orders_into_round_trips(journal):
    """The orders table stores legs. A trade is two of them, and the page is
    about trades."""
    _seed(journal)
    data = collect(journal, 1, _Config(), _performance())

    assert len(data.round_trips) == 1
    assert data.round_trips[0].symbol == "TCS"
    assert data.round_trips[0].pnl == pytest.approx(20.0)


def test_collect_counts_holds_without_listing_them(journal):
    _seed(journal)
    data = collect(journal, 1, _Config(), _performance())

    assert data.holds == 2
    assert [d["action"] for d in data.decisions] == ["buy"]


def test_collect_carries_the_dry_run_flag_out_of_the_stored_config(journal):
    """Whether orders were real is a property of the run, recorded at the time.
    Re-deriving it from today's settings would relabel history."""
    _seed(journal)
    assert collect(journal, 1, _Config(), _performance()).dry_run is True


def test_collect_survives_a_run_whose_config_is_unreadable(journal):
    """Old rows, hand-edited rows, and rows from a future version all exist.
    None of them should stop the page rendering."""
    _seed(journal)
    journal.query("UPDATE runs SET config_json = 'not json' WHERE id = 1")

    assert collect(journal, 1, _Config(), _performance()).dry_run is False


def test_collect_reads_the_open_book_and_the_equity_curve(journal):
    _seed(journal)
    data = collect(journal, 1, _Config(), _performance())

    assert [p["symbol"] for p in data.positions] == ["INFY"]
    assert len(data.equity) == 1
    assert data.strategy == "momentum"


# --------------------------------------------------- equity before first cycle
def test_an_untouched_account_reports_its_cash_not_a_balance_of_zero():
    """Every performance figure is derived from equity samples, and a cycle that
    finds the market closed records none. The first live run of a fresh journal
    is therefore an all-zeros ``Performance`` -- and rendering that verbatim
    tells the owner of a Rs 2,000 account that they have nothing, which is both
    wrong and alarming. The cash balance is knowable regardless.
    """
    html = _render(_data(
        equity=(),
        book={"cash": 2000.0, "starting_cash": 2000.0},
        performance=_performance(samples=0, starting_equity=0.0,
                                 ending_equity=0.0, total_return=0.0),
    ))
    assert "Rs 2,000.00" in html
    assert "no cycle has run while the market was open" in html


def test_the_stated_balance_is_the_measured_one_once_a_cycle_has_run():
    """The fallback is for the gap before the first sample, not a second source
    of truth. The moment the curve exists it wins, even though the account row
    is still there and still says something different."""
    html = _render(_data(
        equity=({"ts": "2025-01-01T04:00:00+00:00", "equity": 104_000.0,
                 "benchmark_price": 100.0},),
        book={"cash": 2000.0, "starting_cash": 2000.0},
    ))
    assert "Rs 104,000.00" in html
    assert "Rs 2,000.00" not in html


def test_an_external_broker_leaves_the_figures_alone():
    """Alpaca keeps its paper state server-side, so there is no account row to
    read. The dashboard must show what it measured rather than invent a
    balance it has no way to know."""
    html = _render(_data(equity=(), book=None,
                         performance=_performance(samples=0,
                                                  starting_equity=0.0,
                                                  ending_equity=0.0)))
    assert "no cycle has run while the market was open" not in html


def test_the_page_refuses_to_be_served_from_cache():
    """One URL, rewritten every cycle. A cached copy shows an old balance under
    a fresh-looking header, and the reader has no way to tell -- which is worse
    than showing nothing."""
    html = _render(_data())
    assert 'http-equiv="cache-control"' in html
    assert "no-cache" in html
