"""The markdown report.

A report is the only artefact a person actually reads, so the things worth
pinning here are the ones that mislead silently: a benchmark comparison that
goes missing, an annualised number printed without the warning that six weeks
of samples were extrapolated into a year, and caveats that describe a market
the run was not conducted in.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from claude_trader.analytics.calibration import Bucket, Calibration
from claude_trader.analytics.metrics import Performance
from claude_trader.analytics.report import (
    DEFAULT_CAVEATS,
    render_calibration,
    render_comparison,
    render_performance,
    render_report,
)

START = datetime(2026, 3, 2, 9, 15, tzinfo=timezone.utc)
END = datetime(2026, 4, 30, 15, 30, tzinfo=timezone.utc)


def perf(**overrides) -> Performance:
    base = dict(
        start=START,
        end=END,
        samples=1_000,
        starting_equity=100_000.0,
        ending_equity=112_500.0,
        total_return=0.125,
        annualised_return=0.31,
        annualised_vol=0.18,
        sharpe=1.42,
        sortino=2.05,
        max_drawdown=0.084,
        avg_exposure=0.62,
        benchmark_return=0.09,
        excess_return=0.035,
        trades=24,
        win_rate=0.54,
        profit_factor=1.65,
        avg_win=820.5,
        avg_loss=-510.25,
        expectancy=138.0,
    )
    return Performance(**{**base, **overrides})


def bucket(**overrides) -> Bucket:
    base = dict(
        label="7-8",
        low=7,
        high=8,
        count=40,
        directional=30,
        hits=18,
        avg_return=0.0031,
        median_return=0.0024,
        avg_benchmark=0.0009,
    )
    return Bucket(**{**base, **overrides})


def calibration(**overrides) -> Calibration:
    base = dict(
        horizon_bars=26,
        resolved=120,
        buckets=(bucket(),),
        rank_correlation=0.41,
        monotonic=True,
        verdict="confidence separates outcomes",
    )
    return Calibration(**{**base, **overrides})


def rupees(value: float) -> str:
    return f"Rs {value:,.2f}"


# ------------------------------------------------------------- performance
def test_the_headline_numbers_appear_in_the_table():
    text = render_performance(perf())
    assert "| Total return | +12.50% | +9.00% |" in text
    assert "| Sharpe | 1.42 | — |" in text
    assert "| Excess over benchmark | +3.50% | — |" in text


def test_returns_carry_a_sign_but_rates_do_not():
    """A volatility of -18% is not a thing; printing one would look like a bug
    in the metrics rather than in the formatter."""
    text = render_performance(perf())
    assert "| Annualised volatility | 18.00% |" in text
    assert "| Average exposure | 62.00% |" in text
    assert "| Win rate | 54.00% |" in text


def test_the_drawdown_is_printed_as_a_loss():
    """Metrics store drawdown as a positive magnitude. Printing it unchanged
    would read as a 8.4% gain."""
    assert "| Max drawdown | -8.40% |" in render_performance(perf())


def test_beating_the_benchmark_is_stated_in_words():
    text = render_performance(perf(), benchmark="NIFTYBEES")
    assert "**beat** buy-and-hold NIFTYBEES" in text


def test_trailing_the_benchmark_is_stated_just_as_plainly():
    """The report is worthless if it only reads well when the strategy won."""
    text = render_performance(perf(excess_return=-0.02))
    assert "**trailed** buy-and-hold" in text


def test_a_run_with_no_benchmark_says_so_rather_than_claiming_a_win():
    text = render_performance(perf(benchmark_return=None, excess_return=None))
    assert "no benchmark recorded" in text
    assert "beat" not in text


def test_a_missing_benchmark_return_prints_as_not_available():
    text = render_performance(perf(benchmark_return=None, excess_return=None))
    assert "| Total return | +12.50% | n/a |" in text


def test_an_extrapolated_annualised_figure_is_flagged_and_footnoted():
    """Six weeks annualised is an extrapolation, and someone will quote it as a
    forecast unless the table says otherwise."""
    text = render_performance(perf(samples=60, annualised_extrapolated=True))
    assert "| Annualised return * |" in text
    assert "| Annualised volatility * |" in text
    assert "annualised from 60 samples" in text
    assert "too short a window to treat as a rate" in text


def test_a_well_sampled_run_carries_no_footnote():
    text = render_performance(perf())
    assert "Annualised return |" in text          # no " *" marker on the label
    assert "too short a window" not in text


def test_the_period_and_sample_count_are_recorded():
    text = render_performance(perf())
    assert "2026-03-02T09:15" in text
    assert "2026-04-30T15:30" in text
    assert "(1000 samples)" in text


def test_an_empty_run_reports_a_period_of_not_available():
    """An equity curve with no points must still render; crashing here would
    lose the rest of the report as well."""
    text = render_performance(perf(start=None, end=None, samples=0))
    assert "Period: n/a to n/a (0 samples)" in text


def test_an_infinite_profit_factor_is_readable():
    """A run with no losing trade divides by zero; 'inf' is honest, a Python
    float repr in a markdown table is not."""
    assert "| Profit factor | inf |" in render_performance(
        perf(profit_factor=float("inf")))


def test_money_is_formatted_by_the_injected_currency_formatter():
    """The same report renders for two markets; hardcoding a dollar sign would
    quietly relabel every rupee figure."""
    text = render_performance(perf(), money=rupees)
    assert "| Starting equity | Rs 100,000.00 | — |" in text
    assert "| Average loss | Rs -510.25 |" in text


def test_dollars_are_the_default():
    assert "| Ending equity | $112,500.00 | — |" in render_performance(perf())


def test_the_title_is_configurable():
    assert render_performance(perf(), title="Momentum rule").startswith(
        "## Momentum rule")


def test_the_trade_table_is_present():
    text = render_performance(perf())
    assert "### Trades" in text
    assert "| Closed round trips | 24 |" in text
    assert "| Expectancy per trade | $138.00 |" in text


def test_a_run_that_never_traded_still_renders_its_trade_table():
    text = render_performance(perf(trades=0, win_rate=0.0, profit_factor=0.0,
                                   avg_win=0.0, avg_loss=0.0, expectancy=0.0))
    assert "| Closed round trips | 0 |" in text
    assert "| Win rate | 0.00% |" in text


# ------------------------------------------------------------ calibration
def test_the_calibration_table_reports_every_bucket():
    text = render_calibration(calibration(
        buckets=(bucket(label="1-3"), bucket(label="9-10"))))
    assert "| 1-3 |" in text and "| 9-10 |" in text


def test_a_bucket_reports_hit_rate_edge_and_sample_size():
    text = render_calibration(calibration())
    assert "| 7-8 | 40 | 30 | +60.0% | +0.310% | +0.240% | +0.090% | +0.220% |" in text


def test_an_empty_bucket_is_shown_rather_than_dropped():
    """A confidence band the model never used is itself a finding -- dropping
    the row would hide that the gate only ever fires at one level."""
    text = render_calibration(calibration(buckets=(bucket(count=0),)))
    assert "| 7-8 | 0 | — | — | — | — | — | — |" in text


def test_a_bucket_with_no_benchmark_prints_no_edge():
    text = render_calibration(calibration(
        buckets=(bucket(avg_benchmark=None),)))
    assert "| +0.090% |" not in text
    assert "| n/a | n/a |" in text


def test_a_bucket_with_no_directional_calls_has_a_zero_hit_rate():
    """Every decision in the band was a hold; the control group is not a
    failure to predict."""
    text = render_calibration(calibration(
        buckets=(bucket(directional=0, hits=0),)))
    assert "+0.0%" in text


def test_the_horizon_and_resolved_count_are_stated():
    text = render_calibration(calibration())
    assert "Horizon: 26 bars | resolved decisions: 120" in text


def test_the_rank_correlation_and_ordering_are_reported():
    text = render_calibration(calibration())
    assert "**0.41**" in text
    assert "Buckets ordered correctly: **yes**" in text


def test_buckets_out_of_order_are_reported_as_such():
    text = render_calibration(calibration(monotonic=False))
    assert "Buckets ordered correctly: **no**" in text


def test_an_uncomputable_rank_correlation_is_not_available():
    """Too few resolved decisions is a real state, and printing 0.00 would
    read as 'no relationship' rather than 'no evidence'."""
    text = render_calibration(calibration(rank_correlation=None, resolved=1))
    assert "**n/a**" in text


def test_the_verdict_is_carried_through_verbatim():
    text = render_calibration(calibration(verdict="confidence is decoration"))
    assert "**Verdict:** confidence is decoration" in text


def test_a_calibration_with_no_buckets_still_renders_a_header():
    text = render_calibration(calibration(buckets=()))
    assert "## Confidence calibration" in text


# ------------------------------------------------------------- comparison
def test_the_comparison_table_lists_every_run():
    text = render_comparison([("rule", perf()), ("llm", perf(total_return=0.02))])
    assert "| rule |" in text and "| llm |" in text
    assert "+12.50%" in text and "+2.00%" in text


def test_the_comparison_names_the_benchmark_column():
    assert "| vs NIFTYBEES |" in render_comparison(
        [("rule", perf())], benchmark="NIFTYBEES")


def test_the_comparison_states_what_is_being_held_constant():
    """Without that sentence the table looks like two strategies; it is in fact
    one engine with the decision maker swapped."""
    text = render_comparison([("rule", perf())])
    assert "The only thing that differs is the decision maker." in text


def test_nothing_to_compare_renders_nothing():
    assert render_comparison([]) == ""


# ----------------------------------------------------------------- report
def test_the_report_assembles_every_section():
    text = render_report("Backtest", perf(), calibration(),
                         [("rule", perf())])
    assert text.startswith("# Backtest")
    assert "## Performance" in text
    assert "## Strategy comparison" in text
    assert "## Confidence calibration" in text
    assert "## Caveats" in text


def test_the_optional_sections_are_omitted_when_absent():
    text = render_report("Backtest", perf())
    assert "## Strategy comparison" not in text
    assert "## Confidence calibration" not in text


def test_metadata_is_rendered_as_a_bullet_list():
    text = render_report("Backtest", perf(),
                         meta={"market": "in", "symbols": "RELIANCE, TCS"})
    assert "- **market**: in" in text
    assert "- **symbols**: RELIANCE, TCS" in text


def test_the_currency_and_benchmark_reach_the_nested_sections():
    text = render_report("Backtest", perf(), comparison=[("rule", perf())],
                         money=rupees, benchmark="NIFTYBEES")
    assert "Rs 100,000.00" in text
    assert "Benchmark (NIFTYBEES)" in text
    assert "| vs NIFTYBEES |" in text


def test_an_indian_run_gets_indian_caveats():
    """Warning about the IEX feed on an NSE run would be noise, and worse, it
    would hide that the spreads in that run were modelled rather than quoted."""
    text = render_report("Backtest", perf(), market="in")
    assert "modelled" in text
    assert "STT and stamp duty" in text
    assert "IEX" not in text


def test_a_us_run_gets_the_iex_warning():
    text = render_report("Backtest", perf(), market="us")
    assert "IEX feed" in text
    assert "STT" not in text


def test_every_market_is_told_that_fills_are_assumed():
    for market in ("us", "in"):
        text = render_report("Backtest", perf(), market=market)
        assert "open of the bar following the decision" in text


@pytest.mark.parametrize("market", ["us", "in"])
def test_every_report_says_this_is_not_financial_advice(market):
    """The line is not decoration. Someone will act on these numbers."""
    text = render_report("Backtest", perf(), market=market)
    assert "not financial advice" in text
    assert "do not predict future results" in text


def test_explicit_caveats_replace_the_defaults():
    text = render_report("Backtest", perf(), caveats=["synthetic prices"])
    assert "- synthetic prices" in text
    assert "IEX" not in text


def test_the_default_caveats_are_the_us_set():
    assert DEFAULT_CAVEATS == tuple(
        line.lstrip("- ")
        for line in render_report("x", perf(), market="us").split("## Caveats")[1]
        .strip()
        .splitlines()
    )


def test_the_report_is_markdown_a_human_can_read():
    """Rendered output should not carry stray Nones or dataclass reprs."""
    text = render_report("Backtest", perf(), calibration(), [("rule", perf())],
                         meta={"run": 7})
    assert "None" not in text
    assert "Performance(" not in text


def test_performance_is_not_mutated_by_rendering():
    original = perf()
    render_report("Backtest", original, calibration())
    assert original == replace(original)
