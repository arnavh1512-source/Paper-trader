"""Confidence calibration.

The whole point of this module is to be able to say "the confidence number is
decoration" out loud if that is what the data shows. So these tests care as much
about the sceptical verdicts as the flattering ones -- a calibration report that
cannot report failure is worse than none.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.analytics.calibration import (
    BANDS,
    Bucket,
    calibrate,
    resolve_outcomes,
    spearman,
)
from claude_trader.markets import INDIA_MARKET
from claude_trader.models import Action, Decision
from tests.conftest import make_state

START = datetime(2026, 3, 2, 4, 0, tzinfo=timezone.utc)


class StubMarket:
    """Only the one method calibration reaches for."""

    def __init__(self, returns=None, benchmark=0.001, blow_up=False, bench_blow_up=False):
        self.returns = returns if returns is not None else {}
        self.benchmark = benchmark
        self.blow_up = blow_up
        self.bench_blow_up = bench_blow_up
        self.asked: list[tuple[str, int]] = []

    def forward_return(self, symbol, when, horizon):
        self.asked.append((symbol, horizon))
        if symbol == "NIFTYBEES":
            if self.bench_blow_up:
                raise RuntimeError("benchmark history unavailable")
            return (100.0, 100.0 * (1 + self.benchmark), self.benchmark)
        if self.blow_up:
            raise RuntimeError("yahoo said no")
        ret = self.returns.get(symbol)
        if ret is None:
            return None
        return (1_000.0, 1_000.0 * (1 + ret), ret)


@pytest.fixture
def run(journal):
    run_id = journal.start_run("backtest", "claude", START)
    cycle_id = journal.record_cycle(run_id, START, make_state(), None, market_open=True)
    return run_id, cycle_id


def add(journal, run, confidence, action, symbol="RELIANCE", minute=0):
    run_id, cycle_id = run
    return journal.record_decision(
        run_id=run_id,
        cycle_id=cycle_id,
        ts=START + timedelta(minutes=minute),
        decision=Decision(symbol, Action(action), confidence, "because"),
        price=1_000.0,
    )


def resolve(journal, decision_id, forward_return, benchmark=None, horizon=26):
    journal.record_outcome(
        decision_id=decision_id,
        horizon_bars=horizon,
        entry_price=1_000.0,
        exit_price=1_000.0 * (1 + forward_return),
        forward_return=forward_return,
        benchmark_return=benchmark,
        resolved_at=START,
    )


# ------------------------------------------------------------ rank maths
def test_spearman_of_a_perfectly_ordered_pair_is_one():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_ignores_the_shape_of_the_relationship():
    """Rank, not level: a curved but strictly increasing relationship is still a
    perfect ordering, and ordering is all the gate needs."""
    assert spearman([1, 2, 3, 4], [1, 4, 900, 10_000]) == pytest.approx(1.0)


def test_spearman_shares_ranks_across_ties():
    assert spearman([1, 1, 2, 2], [5, 5, 9, 9]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "xs,ys",
    [
        ([1, 2], [1, 2]),                    # too few points
        ([1, 2, 3], [1, 2]),                 # mismatched lengths
        ([1, 1, 1, 1], [1, 2, 3, 4]),        # no variation to correlate
        ([1, 2, 3, 4], [7, 7, 7, 7]),
    ],
)
def test_spearman_returns_none_rather_than_a_meaningless_number(xs, ys):
    assert spearman(xs, ys) is None


# --------------------------------------------------------------- buckets
def test_a_bucket_scores_hits_only_on_directional_calls():
    """A hold cannot be right or wrong about direction, so counting it would
    quietly inflate the hit rate of whichever way the market drifted."""
    bucket = Bucket("x", 7, 7, count=3, directional=2, hits=1,
                    avg_return=0.01, median_return=0.01, avg_benchmark=None)
    assert bucket.hit_rate == pytest.approx(0.5)
    assert bucket.edge is None


def test_an_empty_bucket_has_no_hit_rate_instead_of_dividing_by_zero():
    bucket = Bucket("x", 0, 4, 0, 0, 0, 0.0, 0.0, None)
    assert bucket.hit_rate == 0.0


def test_edge_is_the_return_over_the_benchmark():
    bucket = Bucket("x", 9, 10, 5, 5, 4, avg_return=0.02, median_return=0.02,
                    avg_benchmark=0.015)
    assert bucket.edge == pytest.approx(0.005)


def test_decisions_land_in_the_band_that_matches_their_confidence(journal, run):
    for confidence in range(11):
        resolve(journal, add(journal, run, confidence, "buy"), 0.01)
    counts = {b.label: b.count for b in calibrate(journal, run[0]).buckets}
    assert counts == {
        "0-4 (no conviction)": 5,
        "5-6 (below gate)": 2,
        "7 (at gate)": 1,
        "8": 1,
        "9-10 (high conviction)": 2,
    }
    assert [b.label for b in calibrate(journal, run[0]).buckets] == [b[0] for b in BANDS]


def test_a_buy_that_rose_and_a_sell_that_fell_both_count_as_hits(journal, run):
    resolve(journal, add(journal, run, 8, "buy"), 0.02)
    resolve(journal, add(journal, run, 8, "sell"), -0.02)
    resolve(journal, add(journal, run, 8, "buy"), -0.01)
    bucket = next(b for b in calibrate(journal, run[0]).buckets if b.label == "8")
    assert (bucket.count, bucket.directional, bucket.hits) == (3, 3, 2)


def test_holds_are_counted_but_never_scored(journal, run):
    resolve(journal, add(journal, run, 8, "hold"), 0.05)
    resolve(journal, add(journal, run, 8, "buy"), 0.05)
    bucket = next(b for b in calibrate(journal, run[0]).buckets if b.label == "8")
    assert bucket.count == 2
    assert bucket.directional == 1
    assert bucket.avg_return == pytest.approx(0.05)   # holds still move the mean


def test_the_median_is_reported_next_to_the_mean(journal, run):
    """One 40% winner can carry a bucket's mean on its own. The median says
    whether the bucket is actually working."""
    for ret in (0.001, 0.002, 0.400):
        resolve(journal, add(journal, run, 9, "buy"), ret)
    bucket = next(b for b in calibrate(journal, run[0]).buckets if b.label.startswith("9"))
    assert bucket.median_return == pytest.approx(0.002)
    assert bucket.avg_return > 0.13


def test_a_bucket_without_benchmark_rows_reports_no_edge(journal, run):
    resolve(journal, add(journal, run, 8, "buy"), 0.02, benchmark=None)
    bucket = next(b for b in calibrate(journal, run[0]).buckets if b.label == "8")
    assert bucket.avg_benchmark is None and bucket.edge is None


def test_the_benchmark_average_ignores_rows_that_have_none(journal, run):
    resolve(journal, add(journal, run, 8, "buy"), 0.02, benchmark=0.01)
    resolve(journal, add(journal, run, 8, "buy"), 0.02, benchmark=None)
    bucket = next(b for b in calibrate(journal, run[0]).buckets if b.label == "8")
    assert bucket.avg_benchmark == pytest.approx(0.01)


# --------------------------------------------------------------- verdicts
def test_a_thin_sample_refuses_to_draw_a_conclusion(journal, run):
    for i in range(10):
        resolve(journal, add(journal, run, 9, "buy", minute=i), 0.01 * i)
    report = calibrate(journal, run[0])
    assert report.resolved == 10
    assert "Not enough resolved decisions (10)" in report.verdict
    assert "unvalidated" in report.verdict


def test_a_gate_that_orders_outcomes_correctly_is_reported_as_working(journal, run):
    for i in range(33):
        confidence = i % 11
        resolve(journal, add(journal, run, confidence, "buy", minute=i), confidence * 0.001)
    report = calibrate(journal, run[0])
    assert report.rank_correlation == pytest.approx(1.0)
    assert report.monotonic is True
    assert "carries signal" in report.verdict


def test_an_inverted_gate_is_called_out_as_dangerous(journal, run):
    """This is the outcome the original bot could not detect, and the one that
    costs the most money."""
    for i in range(33):
        confidence = i % 11
        resolve(journal, add(journal, run, confidence, "buy", minute=i), -confidence * 0.001)
    report = calibrate(journal, run[0])
    assert report.rank_correlation < -0.05
    assert "inverted" in report.verdict
    assert "Do not trade on this number" in report.verdict


def test_no_relationship_is_stated_plainly(journal, run):
    # Every confidence sees the identical spread of outcomes, so rank
    # correlation is exactly zero by construction.
    minute = 0
    for confidence in (5, 6, 7, 8, 9):
        for step in range(6):
            resolve(journal, add(journal, run, confidence, "buy", minute=minute), step * 0.001)
            minute += 1
    report = calibrate(journal, run[0])
    assert report.rank_correlation == pytest.approx(0.0)
    assert "No relationship" in report.verdict
    assert "filtering noise" in report.verdict


def test_a_constant_confidence_means_the_gate_does_nothing(journal, run):
    for i in range(33):
        resolve(journal, add(journal, run, 8, "buy", minute=i), 0.001 * (i % 7))
    report = calibrate(journal, run[0])
    assert report.rank_correlation is None
    assert "doing no work" in report.verdict


def test_a_marginal_gate_is_described_as_marginal(journal, run):
    """Mostly noise with a faint tilt towards higher confidence -- the ambiguous
    middle, where the honest answer is 'maybe' rather than a recommendation."""
    rng = random.Random(1)      # seeded so the verdict band is not flaky
    for i in range(60):
        confidence = i % 10
        ret = 0.0004 * confidence + rng.uniform(-0.01, 0.01)
        resolve(journal, add(journal, run, confidence, "buy", minute=i), ret)
    report = calibrate(journal, run[0])
    assert 0.05 <= report.rank_correlation < 0.15
    assert "marginal" in report.verdict


def test_monotonic_needs_at_least_two_populated_buckets(journal, run):
    resolve(journal, add(journal, run, 8, "buy"), 0.01)
    assert calibrate(journal, run[0]).monotonic is False


# -------------------------------------------------------------- filtering
def test_holds_can_be_excluded_from_the_report(journal, run):
    resolve(journal, add(journal, run, 8, "hold"), 0.05)
    resolve(journal, add(journal, run, 8, "buy"), 0.05)
    assert calibrate(journal, run[0], actions=("buy", "sell")).resolved == 1
    assert calibrate(journal, run[0]).resolved == 2


def test_each_horizon_is_calibrated_separately(journal, run):
    """An edge that shows up over three days and vanishes over one hour is a
    different strategy, not the same one measured twice."""
    decision_id = add(journal, run, 9, "buy")
    resolve(journal, decision_id, 0.01, horizon=4)
    resolve(journal, decision_id, 0.05, horizon=78)
    assert calibrate(journal, run[0], horizon_bars=4).resolved == 1
    assert calibrate(journal, run[0], horizon_bars=26).resolved == 0
    hi = calibrate(journal, run[0], horizon_bars=78)
    assert hi.buckets[-1].avg_return == pytest.approx(0.05)


def test_runs_do_not_contaminate_each_other(journal, run):
    resolve(journal, add(journal, run, 8, "buy"), 0.05)
    other = journal.start_run("backtest", "claude", START)
    assert calibrate(journal, other).resolved == 0


# -------------------------------------------------------------- resolution
def test_resolution_backfills_outcomes_from_the_market(journal, run):
    ids = [add(journal, run, 8, "buy", minute=i) for i in range(3)]
    written = resolve_outcomes(
        journal, run[0], StubMarket({"RELIANCE": 0.02}),
        horizons=(26,), benchmark_symbol="NIFTYBEES",
    )
    assert written == 3
    rows = journal.query("SELECT * FROM outcomes ORDER BY decision_id")
    assert [r["decision_id"] for r in rows] == ids
    assert rows[0]["forward_return"] == pytest.approx(0.02)
    assert rows[0]["benchmark_return"] == pytest.approx(0.001)


def test_every_horizon_is_resolved_independently(journal, run):
    add(journal, run, 8, "buy")
    written = resolve_outcomes(
        journal, run[0], StubMarket({"RELIANCE": 0.02}),
        horizons=(4, 26, 78), benchmark_symbol="NIFTYBEES",
    )
    assert written == 3
    assert {int(r["horizon_bars"]) for r in journal.query("SELECT * FROM outcomes")} \
        == {4, 26, 78}


def test_a_decision_whose_horizon_has_not_elapsed_is_left_alone(journal, run):
    """Resolving a half-elapsed horizon biases the whole sample towards
    whatever the market did most recently."""
    add(journal, run, 8, "buy")
    assert resolve_outcomes(journal, run[0], StubMarket({}), horizons=(26,)) == 0
    assert journal.query("SELECT * FROM outcomes") == []


def test_a_data_failure_skips_the_row_rather_than_killing_the_backfill(journal, run):
    add(journal, run, 8, "buy")
    assert resolve_outcomes(journal, run[0], StubMarket(blow_up=True), horizons=(26,)) == 0


def test_a_benchmark_failure_still_records_the_outcome(journal, run):
    add(journal, run, 8, "buy")
    written = resolve_outcomes(
        journal, run[0], StubMarket({"RELIANCE": 0.02}, bench_blow_up=True),
        horizons=(26,), benchmark_symbol="NIFTYBEES",
    )
    assert written == 1
    assert journal.query("SELECT * FROM outcomes")[0]["benchmark_return"] is None


def test_resolution_is_resumable_and_does_not_redo_work(journal, run):
    add(journal, run, 8, "buy")
    market = StubMarket({"RELIANCE": 0.02})
    assert resolve_outcomes(journal, run[0], market, horizons=(26,)) == 1
    assert resolve_outcomes(journal, run[0], market, horizons=(26,)) == 0


def test_the_benchmark_symbol_is_whatever_the_market_says_it_is(journal, run):
    add(journal, run, 8, "buy")
    market = StubMarket({"RELIANCE": 0.02})
    resolve_outcomes(
        journal, run[0], market, horizons=(26,),
        benchmark_symbol=INDIA_MARKET.benchmark,
    )
    assert INDIA_MARKET.benchmark in {sym for sym, _ in market.asked}
