"""Performance measurement.

Answers the question the original bot could not: is this better than doing
nothing? Every figure is computed against the same equity curve the engine
wrote, and the benchmark is carried through the identical period so the
comparison is apples to apples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Iterable, Mapping, Sequence

from ..data.indicators import max_drawdown, simple_returns, stdev

SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Both NSE and NYSE clear roughly this many sessions a year after holidays.
TRADING_SESSIONS_PER_YEAR = 250.0

# Below this, annualising is extrapolation from a sample too short to carry it,
# and the report says so rather than printing a confident number.
MIN_YEARS_TO_ANNUALISE = 0.25


@dataclass(frozen=True, slots=True)
class RoundTrip:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    qty: float
    entry_price: float
    exit_price: float

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def return_pct(self) -> float:
        basis = self.entry_price * self.qty
        return 0.0 if basis == 0 else self.pnl / basis

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass(frozen=True, slots=True)
class Performance:
    start: datetime | None
    end: datetime | None
    samples: int
    starting_equity: float
    ending_equity: float
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    avg_exposure: float
    benchmark_return: float | None
    excess_return: float | None
    trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    periods_per_year: float = 252.0
    annualised_extrapolated: bool = False

    @property
    def beats_benchmark(self) -> bool | None:
        if self.excess_return is None:
            return None
        return self.excess_return > 0


def periods_per_year_for(bars_per_session: int) -> float:
    """Sampling frequency implied by a market's session, not by wall clock.

    The inferred version below cannot tell a 15-minute bar inside a 6.25-hour
    session from one that runs all night, so it annualises as though the market
    never closed. Callers that know the session should pass this in.
    """
    return max(1.0, float(bars_per_session)) * TRADING_SESSIONS_PER_YEAR


def _periods_per_year(stamps: Sequence[datetime]) -> float:
    if len(stamps) < 3:
        return 252.0
    gaps = [
        (b - a).total_seconds()
        for a, b in zip(stamps, stamps[1:])
        if (b - a).total_seconds() > 0
    ]
    if not gaps:
        return 252.0
    # Median gap ignores overnight and weekend jumps that would otherwise
    # understate the sampling frequency.
    return SECONDS_PER_YEAR / median(gaps)


def build_round_trips(orders: Iterable[Mapping]) -> list[RoundTrip]:
    """Reconstruct closed trades from the order log, FIFO per symbol."""
    open_lots: dict[str, list[list]] = {}
    trips: list[RoundTrip] = []

    for order in orders:
        symbol = str(order["symbol"])
        side = str(order["side"]).lower()
        qty = float(order["qty"])
        price = float(order["price"])
        ts = order["ts"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if qty <= 0:
            continue

        if side == "buy":
            open_lots.setdefault(symbol, []).append([qty, price, ts])
            continue

        remaining = qty
        lots = open_lots.get(symbol, [])
        while remaining > 1e-9 and lots:
            lot = lots[0]
            matched = min(remaining, lot[0])
            trips.append(
                RoundTrip(
                    symbol=symbol,
                    entry_time=lot[2],
                    exit_time=ts,
                    qty=matched,
                    entry_price=lot[1],
                    exit_price=price,
                )
            )
            lot[0] -= matched
            remaining -= matched
            if lot[0] <= 1e-9:
                lots.pop(0)
    return trips


def _trade_stats(trips: Sequence[RoundTrip]) -> dict[str, float]:
    if not trips:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
        }
    wins = [t.pnl for t in trips if t.is_win]
    losses = [t.pnl for t in trips if not t.is_win]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(trips),
        "win_rate": len(wins) / len(trips),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_win > 0 else 0.0
        ),
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (-gross_loss / len(losses)) if losses else 0.0,
        "expectancy": sum(t.pnl for t in trips) / len(trips),
    }


def compute_performance(
    equity_rows: Sequence[Mapping],
    orders: Sequence[Mapping] = (),
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
) -> Performance:
    stamps: list[datetime] = []
    equity: list[float] = []
    exposure: list[float] = []
    benchmark: list[float] = []

    for row in equity_rows:
        ts = row["ts"]
        stamps.append(datetime.fromisoformat(ts) if isinstance(ts, str) else ts)
        equity.append(float(row["equity"]))
        exposure.append(float(row["exposure"] or 0.0))
        bench = row["benchmark_price"] if "benchmark_price" in row.keys() else None
        if bench is not None:
            benchmark.append(float(bench))

    stats = _trade_stats(build_round_trips(orders))

    if len(equity) < 2:
        return Performance(
            start=stamps[0] if stamps else None,
            end=stamps[-1] if stamps else None,
            samples=len(equity),
            starting_equity=equity[0] if equity else 0.0,
            ending_equity=equity[-1] if equity else 0.0,
            total_return=0.0,
            annualised_return=0.0,
            annualised_vol=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            avg_exposure=(sum(exposure) / len(exposure)) if exposure else 0.0,
            benchmark_return=None,
            excess_return=None,
            periods_per_year=periods_per_year or 252.0,
            **stats,
        )

    rets = simple_returns(equity)
    ppy = periods_per_year or _periods_per_year(stamps)
    total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] else 0.0

    # Elapsed *trading* time, not wall-clock time: a strategy that only holds
    # during the session should not be charged for the weekend it sat flat.
    years = len(rets) / ppy if ppy > 0 else 0.0
    annualised = ((1 + total_return) ** (1 / years) - 1) if years > 0 and total_return > -1 else 0.0

    sd = stdev(rets) or 0.0
    ann_vol = sd * math.sqrt(ppy)
    mean_ret = (sum(rets) / len(rets)) if rets else 0.0
    excess_per_period = mean_ret - risk_free_rate / ppy
    sharpe = (excess_per_period / sd * math.sqrt(ppy)) if sd > 0 else 0.0

    downside = [r for r in rets if r < 0]
    dsd = stdev(downside) if len(downside) > 1 else None
    sortino = (excess_per_period / dsd * math.sqrt(ppy)) if dsd else 0.0

    bench_return = None
    if len(benchmark) >= 2 and benchmark[0] > 0:
        bench_return = (benchmark[-1] - benchmark[0]) / benchmark[0]

    return Performance(
        start=stamps[0],
        end=stamps[-1],
        samples=len(equity),
        starting_equity=equity[0],
        ending_equity=equity[-1],
        total_return=total_return,
        annualised_return=annualised,
        annualised_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown(equity),
        avg_exposure=(sum(exposure) / len(exposure)) if exposure else 0.0,
        benchmark_return=bench_return,
        excess_return=(total_return - bench_return) if bench_return is not None else None,
        periods_per_year=ppy,
        annualised_extrapolated=years < MIN_YEARS_TO_ANNUALISE,
        **stats,
    )
