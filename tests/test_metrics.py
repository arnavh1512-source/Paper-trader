"""Performance measurement.

The original bot printed a running P&L and nothing else, which cannot answer
"is this better than doing nothing". These are the numbers that can -- so they
have to be right, including the boring ones nobody looks at until the strategy
looks good and somebody asks how.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.analytics.metrics import (
    MIN_YEARS_TO_ANNUALISE,
    RoundTrip,
    build_round_trips,
    compute_performance,
    periods_per_year_for,
)
from claude_trader.data.indicators import simple_returns, stdev

START = datetime(2026, 3, 2, 4, 0, tzinfo=timezone.utc)


def order(symbol, side, qty, price, minute):
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "ts": START + timedelta(minutes=minute),
    }


def curve(values, *, exposure=0.5, benchmark=None, step=15):
    rows = []
    for i, equity in enumerate(values):
        row = {
            "ts": START + timedelta(minutes=step * i),
            "equity": equity,
            "exposure": exposure,
        }
        if benchmark is not None:
            row["benchmark_price"] = benchmark[i]
        rows.append(row)
    return rows


# ------------------------------------------------------------- round trips
def test_a_round_trip_knows_its_own_arithmetic():
    trip = RoundTrip("RELIANCE", START, START, 10, 1_000.0, 1_050.0)
    assert trip.pnl == pytest.approx(500.0)
    assert trip.return_pct == pytest.approx(0.05)
    assert trip.is_win is True


def test_a_flat_round_trip_is_not_a_win():
    """Breaking even after paying charges is a loss in every way that matters,
    and it must never be counted toward the win rate."""
    assert RoundTrip("X", START, START, 1, 100.0, 100.0).is_win is False


def test_a_worthless_basis_does_not_divide_by_zero():
    assert RoundTrip("X", START, START, 0, 0.0, 0.0).return_pct == 0.0


def test_lots_are_matched_first_in_first_out():
    trips = build_round_trips([
        order("RELIANCE", "buy", 10, 1_000.0, 0),
        order("RELIANCE", "buy", 10, 1_100.0, 15),
        order("RELIANCE", "sell", 10, 1_200.0, 30),
    ])
    assert len(trips) == 1
    assert trips[0].entry_price == 1_000.0      # the older lot, not the cheaper one
    assert trips[0].pnl == pytest.approx(2_000.0)


def test_one_sell_can_close_several_lots():
    trips = build_round_trips([
        order("TCS", "buy", 5, 3_000.0, 0),
        order("TCS", "buy", 5, 3_100.0, 15),
        order("TCS", "sell", 10, 3_200.0, 30),
    ])
    assert [t.qty for t in trips] == [5, 5]
    assert [t.entry_price for t in trips] == [3_000.0, 3_100.0]


def test_a_partial_sell_leaves_the_rest_open():
    trips = build_round_trips([
        order("TCS", "buy", 10, 3_000.0, 0),
        order("TCS", "sell", 4, 3_100.0, 15),
    ])
    assert len(trips) == 1 and trips[0].qty == 4


def test_selling_more_than_was_bought_only_counts_what_was_bought():
    """Defensive: the broker refuses this, but the reconstruction must not
    invent a short leg if a hand-edited journal ever contains one."""
    trips = build_round_trips([
        order("TCS", "buy", 3, 3_000.0, 0),
        order("TCS", "sell", 10, 3_100.0, 15),
    ])
    assert sum(t.qty for t in trips) == 3


def test_symbols_do_not_borrow_each_others_lots():
    trips = build_round_trips([
        order("RELIANCE", "buy", 10, 1_000.0, 0),
        order("TCS", "sell", 10, 3_000.0, 15),
    ])
    assert trips == []


def test_string_timestamps_are_parsed():
    """Rows come back from SQLite as ISO text, not datetimes."""
    rows = [order("X", "buy", 1, 100.0, 0), order("X", "sell", 1, 110.0, 15)]
    for row in rows:
        row["ts"] = row["ts"].isoformat()
    assert build_round_trips(rows)[0].exit_time - build_round_trips(rows)[0].entry_time \
        == timedelta(minutes=15)


def test_zero_quantity_rows_are_ignored():
    assert build_round_trips([order("X", "buy", 0, 100.0, 0)]) == []


# ------------------------------------------------------------ trade stats
def test_trade_stats_summarise_the_round_trips():
    perf = compute_performance(
        curve([100_000.0, 101_000.0]),
        orders=[
            order("A", "buy", 10, 100.0, 0), order("A", "sell", 10, 120.0, 15),
            order("B", "buy", 10, 100.0, 0), order("B", "sell", 10, 90.0, 15),
        ],
    )
    assert perf.trades == 2
    assert perf.win_rate == pytest.approx(0.5)
    assert perf.avg_win == pytest.approx(200.0)
    assert perf.avg_loss == pytest.approx(-100.0)
    assert perf.profit_factor == pytest.approx(2.0)
    assert perf.expectancy == pytest.approx(50.0)


def test_profit_factor_without_a_single_loss_is_infinite_not_a_crash():
    perf = compute_performance(
        curve([100_000.0, 101_000.0]),
        orders=[order("A", "buy", 1, 100.0, 0), order("A", "sell", 1, 110.0, 15)],
    )
    assert perf.profit_factor == math.inf


def test_no_trades_reports_zeros_rather_than_none():
    perf = compute_performance(curve([100_000.0, 101_000.0]))
    assert (perf.trades, perf.win_rate, perf.profit_factor) == (0, 0.0, 0.0)


# ------------------------------------------------------------- performance
def test_an_empty_curve_is_reported_not_crashed():
    """A run that halted before its first cycle still has to produce a report."""
    perf = compute_performance([])
    assert perf.samples == 0
    assert perf.start is None and perf.end is None
    assert perf.total_return == 0.0


def test_a_single_sample_cannot_have_a_return():
    perf = compute_performance(curve([100_000.0]))
    assert perf.samples == 1
    assert perf.total_return == 0.0
    assert perf.max_drawdown == 0.0
    assert perf.avg_exposure == pytest.approx(0.5)


def test_total_return_and_drawdown_come_off_the_same_curve():
    perf = compute_performance(curve([100_000.0, 101_000.0, 100_500.0, 102_000.0]))
    assert perf.total_return == pytest.approx(0.02)
    assert perf.max_drawdown == pytest.approx(500 / 101_000)
    assert perf.starting_equity == 100_000.0
    assert perf.ending_equity == 102_000.0
    assert perf.samples == 4


def test_sharpe_uses_the_supplied_sampling_frequency():
    values = [100_000.0, 101_000.0, 100_500.0, 102_000.0]
    perf = compute_performance(curve(values), periods_per_year=6_250)
    rets = simple_returns(values)
    expected = (sum(rets) / len(rets)) / stdev(rets) * math.sqrt(6_250)
    assert perf.sharpe == pytest.approx(expected)
    assert perf.periods_per_year == 6_250


def test_a_flat_curve_has_no_sharpe_instead_of_an_infinite_one():
    perf = compute_performance(curve([100_000.0] * 6))
    assert perf.sharpe == 0.0
    assert perf.sortino == 0.0
    assert perf.annualised_vol == 0.0


def test_sortino_only_punishes_downside():
    values = [100.0, 101.0, 100.0, 102.0, 101.0, 104.0]
    perf = compute_performance(curve(values), periods_per_year=6_250)
    assert perf.sortino > perf.sharpe > 0


def test_a_risk_free_rate_lowers_the_sharpe():
    values = [100.0, 101.0, 100.5, 102.0]
    plain = compute_performance(curve(values), periods_per_year=6_250)
    charged = compute_performance(
        curve(values), risk_free_rate=0.07, periods_per_year=6_250
    )
    assert charged.sharpe < plain.sharpe


def test_a_short_sample_is_flagged_rather_than_annualised_silently():
    """Twenty bars annualised to 6,250 a year produces a number in the millions
    of percent. Printing it without the flag would be a lie."""
    perf = compute_performance(curve([100_000.0] * 19 + [101_000.0]), periods_per_year=6_250)
    assert perf.annualised_extrapolated is True


def test_a_long_enough_sample_is_not_flagged():
    n = int(MIN_YEARS_TO_ANNUALISE * 6_250) + 5
    perf = compute_performance(
        curve([100_000.0 + i for i in range(n)]), periods_per_year=6_250
    )
    assert perf.annualised_extrapolated is False
    assert perf.annualised_return > 0


def test_the_sampling_frequency_is_inferred_from_the_median_gap():
    """Overnight and weekend gaps must not be read as the sampling interval."""
    rows = curve([100.0, 101.0, 102.0, 103.0], step=15)
    rows[3]["ts"] = rows[2]["ts"] + timedelta(hours=18)   # the overnight jump
    perf = compute_performance(rows)
    assert perf.periods_per_year == pytest.approx(365.25 * 24 * 3600 / 900)


def test_two_samples_fall_back_to_a_daily_assumption():
    assert compute_performance(curve([100.0, 101.0])).periods_per_year == 252.0


def test_exposure_is_averaged_over_the_run():
    rows = curve([100.0, 101.0])
    rows[0]["exposure"] = 0.0
    rows[1]["exposure"] = 1.0
    assert compute_performance(rows).avg_exposure == pytest.approx(0.5)


def test_a_missing_exposure_reads_as_flat():
    rows = curve([100.0, 101.0])
    rows[0]["exposure"] = None
    assert compute_performance(rows).avg_exposure == pytest.approx(0.25)


# --------------------------------------------------------------- benchmark
def test_the_benchmark_is_measured_over_the_identical_window():
    perf = compute_performance(
        curve([100_000.0, 110_000.0], benchmark=[20_000.0, 21_000.0])
    )
    assert perf.benchmark_return == pytest.approx(0.05)
    assert perf.excess_return == pytest.approx(0.05)
    assert perf.beats_benchmark is True


def test_losing_to_the_index_is_reported_as_losing():
    """Making money in a market that made more is the failure mode this whole
    module exists to catch."""
    perf = compute_performance(
        curve([100_000.0, 102_000.0], benchmark=[20_000.0, 21_000.0])
    )
    assert perf.excess_return == pytest.approx(-0.03)
    assert perf.beats_benchmark is False


def test_without_a_benchmark_the_comparison_is_unknown_not_favourable():
    perf = compute_performance(curve([100_000.0, 102_000.0]))
    assert perf.benchmark_return is None
    assert perf.excess_return is None
    assert perf.beats_benchmark is None


def test_a_benchmark_that_starts_at_zero_is_discarded():
    perf = compute_performance(
        curve([100_000.0, 102_000.0], benchmark=[0.0, 21_000.0])
    )
    assert perf.benchmark_return is None


# ------------------------------------------------------------- periodicity
@pytest.mark.parametrize(
    "bars_per_session,expected",
    [(25, 6_250.0), (26, 6_500.0), (1, 250.0), (0, 250.0), (-5, 250.0)],
)
def test_periods_per_year_follows_the_session_not_the_clock(bars_per_session, expected):
    assert periods_per_year_for(bars_per_session) == expected
