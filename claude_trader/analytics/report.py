"""Markdown reporting.

The report exists so the answer to "is this working" is a document, not a
feeling. It leads with the comparison against buy-and-hold, because a strategy
that trails the benchmark is not a strategy, it is an expensive index fund.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Mapping, Sequence

from .calibration import Calibration
from .metrics import Performance

Money = Callable[[float], str]


def _pct(value: float | None, places: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.{places}f}%"


def _rate(value: float | None, places: int = 2) -> str:
    """Unsigned percentage, for quantities that cannot be negative."""
    if value is None:
        return "n/a"
    return f"{value * 100:.{places}f}%"


def _num(value: float | None, places: int = 2) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    return f"{value:.{places}f}"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _caveats(market: str) -> tuple[str, ...]:
    """Caveats that are true of the market actually being reported on."""
    shared = (
        "Backtested results assume fills at the open of the bar following the "
        "decision, plus modelled slippage. Real fills will differ.",
        "Past results, simulated or live, do not predict future results.",
        "This is a paper-trading research tool, not financial advice.",
    )
    if market == "in":
        return (
            shared[0],
            "NSE bars come from a free vendor feed with no order book: quoted "
            "spreads are modelled, so liquidity is assumed rather than observed.",
            "Statutory charges, STT and stamp duty are modelled at published "
            "rates and will differ from any particular broker's contract note.",
            *shared[1:],
        )
    return (
        shared[0],
        "The IEX feed used for free Alpaca accounts is a thin slice of "
        "consolidated volume and can lag; live prices may not match the "
        "backtest's bars.",
        *shared[1:],
    )


def _when(value: datetime | None) -> str:
    return value.isoformat(timespec="minutes") if value else "n/a"


def render_performance(
    perf: Performance,
    title: str = "Performance",
    money: Money = _money,
    benchmark: str = "SPY",
) -> str:
    verdict = {
        None: "no benchmark recorded",
        True: f"**beat** buy-and-hold {benchmark}",
        False: f"**trailed** buy-and-hold {benchmark}",
    }[perf.beats_benchmark]

    # Annualising six weeks of data is extrapolation. Saying so in the table is
    # cheaper than watching someone quote the number as if it were a forecast.
    extrapolated = " *" if perf.annualised_extrapolated else ""

    return "\n".join(
        [
            f"## {title}",
            "",
            f"Period: {_when(perf.start)} to {_when(perf.end)} ({perf.samples} samples)",
            "",
            f"| Metric | Strategy | Benchmark ({benchmark}) |",
            "| --- | ---: | ---: |",
            f"| Total return | {_pct(perf.total_return)} | {_pct(perf.benchmark_return)} |",
            f"| Excess over benchmark | {_pct(perf.excess_return)} | — |",
            f"| Annualised return{extrapolated} | {_pct(perf.annualised_return)} | — |",
            f"| Annualised volatility{extrapolated} | {_rate(perf.annualised_vol)} | — |",
            f"| Sharpe | {_num(perf.sharpe)} | — |",
            f"| Sortino | {_num(perf.sortino)} | — |",
            f"| Max drawdown | {_pct(-perf.max_drawdown)} | — |",
            f"| Average exposure | {_rate(perf.avg_exposure)} | — |",
            f"| Starting equity | {money(perf.starting_equity)} | — |",
            f"| Ending equity | {money(perf.ending_equity)} | — |",
            "",
            f"**Verdict:** the strategy {verdict}.",
            "",
            *(
                [
                    f"\\* annualised from {perf.samples} samples at "
                    f"{perf.periods_per_year:,.0f} periods/year -- too short a"
                    " window to treat as a rate.",
                    "",
                ]
                if perf.annualised_extrapolated
                else []
            ),
            "### Trades",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Closed round trips | {perf.trades} |",
            f"| Win rate | {_rate(perf.win_rate)} |",
            f"| Profit factor | {_num(perf.profit_factor)} |",
            f"| Average win | {money(perf.avg_win)} |",
            f"| Average loss | {money(perf.avg_loss)} |",
            f"| Expectancy per trade | {money(perf.expectancy)} |",
            "",
        ]
    )


def render_calibration(cal: Calibration) -> str:
    lines = [
        "## Confidence calibration",
        "",
        f"Horizon: {cal.horizon_bars} bars | resolved decisions: {cal.resolved}",
        "",
        "| Confidence | Decisions | Directional | Hit rate | Avg forward return | Median | Benchmark | Edge |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bucket in cal.buckets:
        if bucket.count == 0:
            lines.append(f"| {bucket.label} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {bucket.label} | {bucket.count} | {bucket.directional} | "
            f"{_pct(bucket.hit_rate, 1)} | {_pct(bucket.avg_return, 3)} | "
            f"{_pct(bucket.median_return, 3)} | {_pct(bucket.avg_benchmark, 3)} | "
            f"{_pct(bucket.edge, 3)} |"
        )

    lines += [
        "",
        f"Rank correlation (confidence vs forward return): **{_num(cal.rank_correlation)}**",
        f"Buckets ordered correctly: **{'yes' if cal.monotonic else 'no'}**",
        "",
        f"**Verdict:** {cal.verdict}",
        "",
    ]
    return "\n".join(lines)


def render_comparison(
    runs: Sequence[tuple[str, Performance]], benchmark: str = "SPY"
) -> str:
    if not runs:
        return ""
    lines = [
        "## Strategy comparison",
        "",
        "Same engine, same risk limits, same period, same costs. The only thing"
        " that differs is the decision maker.",
        "",
        f"| Run | Total return | vs {benchmark} | Sharpe | Max DD | Trades | Win rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, perf in runs:
        lines.append(
            f"| {label} | {_pct(perf.total_return)} | {_pct(perf.excess_return)} | "
            f"{_num(perf.sharpe)} | {_pct(-perf.max_drawdown)} | {perf.trades} | "
            f"{_rate(perf.win_rate, 1)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_report(
    header: str,
    performance: Performance,
    calibration: Calibration | None = None,
    comparison: Sequence[tuple[str, Performance]] = (),
    meta: Mapping[str, object] | None = None,
    caveats: Sequence[str] = (),
    money: Money = _money,
    benchmark: str = "SPY",
    market: str = "us",
) -> str:
    parts = [f"# {header}", ""]
    if meta:
        parts += [f"- **{key}**: {value}" for key, value in meta.items()]
        parts.append("")
    parts.append(render_performance(performance, money=money, benchmark=benchmark))
    if comparison:
        parts.append(render_comparison(comparison, benchmark=benchmark))
    if calibration is not None:
        parts.append(render_calibration(calibration))

    parts += ["## Caveats", ""]
    parts += [f"- {line}" for line in (caveats or _caveats(market))]
    parts.append("")
    return "\n".join(parts)


DEFAULT_CAVEATS: tuple[str, ...] = _caveats("us")
