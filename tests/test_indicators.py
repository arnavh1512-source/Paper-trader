"""Indicator maths.

The rule these tests enforce everywhere: when there is not enough history, the
answer is ``None``. A fabricated indicator is worse than a missing one, because
the model reasons over it without knowing it was invented.
"""

from __future__ import annotations

import math

import pytest

from claude_trader.data.indicators import (
    atr,
    classify_trend,
    compute,
    correlation,
    ema,
    max_drawdown,
    pct_change,
    realized_vol,
    rsi,
    simple_returns,
    sma,
    stdev,
    true_ranges,
)
from claude_trader.models import Bar
from tests.conftest import make_bars, ramp


# ------------------------------------------------------------------ averages
def test_sma_is_the_mean_of_the_window():
    assert sma([1, 2, 3, 4, 5], 5) == 3.0
    assert sma([1, 2, 3, 4, 5], 2) == 4.5


@pytest.mark.parametrize("values,period", [([1, 2], 5), ([1, 2, 3], 0), ([], 3)])
def test_sma_refuses_to_guess(values, period):
    assert sma(values, period) is None


def test_ema_weights_the_recent_bar_more():
    # Seed is the mean of the first two (1.5); k = 2/3; then 3 arrives.
    assert ema([1, 2, 3], 2) == pytest.approx(2.5)
    assert ema([1, 2, 3, 4, 5], 5) == pytest.approx(3.0)
    assert ema([1, 2], 5) is None


def test_ema_reacts_faster_than_sma():
    values = [10.0] * 10 + [20.0] * 3
    assert ema(values, 5) > sma(values, 5)


# ----------------------------------------------------------------------- rsi
def test_rsi_needs_one_more_bar_than_its_period():
    assert rsi(list(range(14)), 14) is None
    assert rsi(list(range(15)), 14) is not None


def test_rsi_pins_at_100_when_nothing_ever_falls():
    assert rsi([float(i) for i in range(1, 30)], 14) == 100.0


def test_flat_prices_are_neutral_not_overbought():
    """gains == losses == 0 is a division by zero waiting to happen. Neutral is
    the only honest answer."""
    assert rsi([100.0] * 30, 14) == 50.0


def test_rsi_stays_inside_its_bounds():
    values = [100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 104.0] * 5
    value = rsi(values, 14)
    assert 0.0 <= value <= 100.0


# ----------------------------------------------------------------------- atr
def test_true_range_uses_the_previous_close():
    bars = [
        Bar("X", None, o=100, h=110, l=90, c=105, v=1),
        Bar("X", None, o=105, h=112, l=108, c=110, v=1),
    ]
    # First bar has no predecessor, so it is just the range.
    assert true_ranges(bars)[0] == 20
    # Second: high-low is 4, but the gap from the prior close of 105 is 7.
    assert true_ranges(bars)[1] == 7
    assert true_ranges([]) == []


def test_atr_of_a_constant_range_is_that_range():
    bars = make_bars("X", [100.0] * 20)
    assert atr(bars, 14) == pytest.approx(bars[0].h - bars[0].l)


def test_atr_without_enough_bars_is_none():
    assert atr(make_bars("X", [100.0] * 5), 14) is None


# ---------------------------------------------------------------- dispersion
def test_stdev_is_the_sample_deviation():
    assert stdev([1, 2, 3, 4]) == pytest.approx(math.sqrt(5 / 3))
    assert stdev([5.0]) is None


def test_returns_skip_a_zero_denominator():
    assert simple_returns([100.0, 110.0, 99.0]) == pytest.approx([0.1, -0.1])
    assert simple_returns([0.0, 100.0]) == []
    assert simple_returns([100.0]) == []


def test_pct_change_looks_back_exactly_n_bars():
    assert pct_change([100.0, 110.0], 1) == pytest.approx(0.1)
    assert pct_change([100.0, 105.0, 110.0], 2) == pytest.approx(0.1)
    assert pct_change([100.0, 110.0], 5) is None
    assert pct_change([0.0, 110.0], 1) is None


def test_realized_vol_annualises_on_trading_periods():
    values = [100.0, 101.0, 100.0, 101.0, 100.0]
    expected = stdev(simple_returns(values)) * math.sqrt(6_250)
    assert realized_vol(values, periods_per_year=6_250) == pytest.approx(expected)


def test_realized_vol_of_a_flat_series_is_zero_not_none():
    assert realized_vol([100.0] * 10) == 0.0


# --------------------------------------------------------------- correlation
def test_correlation_of_a_series_with_itself_is_one():
    series = [0.01, -0.02, 0.015, 0.004, -0.008]
    assert correlation(series, series) == pytest.approx(1.0)
    assert correlation(series, [-v for v in series]) == pytest.approx(-1.0)


def test_correlation_uses_the_overlapping_tail():
    long = [0.0, 0.0, 0.01, -0.02, 0.03]
    short = [0.01, -0.02, 0.03]
    assert correlation(long, short) == pytest.approx(1.0)


def test_correlation_is_none_when_it_would_be_meaningless():
    assert correlation([0.01, 0.02], [0.01, 0.02]) is None   # too short
    assert correlation([0.0] * 5, [0.01, 0.02, 0.03, 0.04, 0.05]) is None  # flat


# ------------------------------------------------------------------ drawdown
def test_max_drawdown_is_peak_to_trough():
    assert max_drawdown([100, 120, 60, 90]) == pytest.approx(0.5)
    assert max_drawdown([100, 110, 120]) == 0.0
    assert max_drawdown([]) == 0.0


def test_drawdown_measures_from_the_peak_not_the_start():
    """A run-up followed by a collapse back to the starting value is still a
    drawdown, even though the account never lost money against day one."""
    assert max_drawdown([100, 200, 100]) == pytest.approx(0.5)


# --------------------------------------------------------------------- trend
@pytest.mark.parametrize(
    "fast,slow,rsi_value,expected",
    [
        (110.0, 100.0, 50.0, "up"),
        (110.0, 100.0, 80.0, "overbought_up"),
        (90.0, 100.0, 50.0, "down"),
        (90.0, 100.0, 20.0, "oversold_down"),
        (100.0, 100.0, 50.0, "sideways"),
        (None, 100.0, 50.0, "unknown"),
        (100.0, None, 50.0, "unknown"),
    ],
)
def test_trend_classification(fast, slow, rsi_value, expected):
    assert classify_trend(fast, slow, rsi_value) == expected


# ------------------------------------------------------------------- bundle
def test_compute_fills_the_bundle_from_a_full_history():
    bars = make_bars("RELIANCE", ramp(40, start=1_000.0, step=2.0))
    ind = compute(bars)
    assert ind.last_price == bars[-1].c
    assert ind.trend in {"up", "overbought_up"}   # a pure ramp prints RSI 100
    assert ind.atr is not None and ind.atr > 0
    assert ind.rsi is not None
    assert ind.sma_fast > ind.sma_slow
    assert ind.volume_ratio == pytest.approx(1.0)     # constant volume
    assert ind.dist_from_high_pct == pytest.approx(0.0)  # a ramp ends at its high


def test_compute_degrades_field_by_field_on_thin_history():
    ind = compute(make_bars("RELIANCE", [1_000.0, 1_002.0]))
    assert ind.last_price == 1_002.0
    assert ind.sma_slow is None
    assert ind.atr is None
    assert ind.rsi is None
    assert ind.trend == "unknown"


def test_compute_survives_an_empty_series():
    """Yahoo returns nothing for a halted scrip. That must not crash the run."""
    ind = compute(())
    assert ind.last_price == 0.0
    assert ind.trend == "unknown"
    assert ind.dist_from_high_pct is None


def test_atr_pct_is_scaled_to_the_price():
    bars = make_bars("RELIANCE", ramp(40, start=1_000.0, step=2.0))
    ind = compute(bars)
    assert ind.atr_pct == pytest.approx(ind.atr / ind.last_price * 100.0)


def test_a_falling_series_is_classified_as_down():
    bars = make_bars("RELIANCE", ramp(40, start=1_100.0, step=-2.0))
    assert compute(bars).trend in {"down", "oversold_down"}
