"""Confidence calibration.

The original bot traded whenever the model said 7 or higher, and never checked
whether a 9 was actually better than a 7. This module answers that.

Every decision -- including holds, which are the control group -- is resolved
against what the price actually did over a fixed horizon. If the buckets do not
separate, the confidence number is decoration and the gate should be replaced
with something that is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Mapping, Sequence

DEFAULT_HORIZONS = (4, 26, 78)  # ~1 hour, ~1 day, ~3 days on 15-minute bars


@dataclass(frozen=True, slots=True)
class Bucket:
    label: str
    low: int
    high: int
    count: int
    directional: int
    hits: int
    avg_return: float
    median_return: float
    avg_benchmark: float | None

    @property
    def hit_rate(self) -> float:
        return self.hits / self.directional if self.directional else 0.0

    @property
    def edge(self) -> float | None:
        if self.avg_benchmark is None:
            return None
        return self.avg_return - self.avg_benchmark


@dataclass(frozen=True, slots=True)
class Calibration:
    horizon_bars: int
    resolved: int
    buckets: tuple[Bucket, ...]
    rank_correlation: float | None
    monotonic: bool
    verdict: str


def resolve_outcomes(
    journal,
    run_id: int,
    market,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    benchmark_symbol: str = "SPY",
) -> int:
    """Backfill the outcomes table. Returns the number of rows written.

    Only decisions whose horizon has fully elapsed are resolved; a partially
    elapsed horizon would bias the sample towards whatever the market did most
    recently.
    """
    written = 0
    now = datetime.now().astimezone()

    for horizon in horizons:
        for row in journal.unresolved_decisions(run_id, horizon):
            ts = row["ts"]
            when = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
            try:
                forward = market.forward_return(row["symbol"], when, horizon)
            except Exception:
                forward = None
            if forward is None:
                continue

            entry, exit_price, ret = forward
            try:
                bench = market.forward_return(benchmark_symbol, when, horizon)
            except Exception:
                bench = None

            journal.record_outcome(
                decision_id=row["id"],
                horizon_bars=horizon,
                entry_price=entry,
                exit_price=exit_price,
                forward_return=ret,
                benchmark_return=bench[2] if bench else None,
                resolved_at=now,
            )
            written += 1
    return written


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation. Used instead of Pearson because we care whether higher
    confidence orders outcomes correctly, not whether the relationship is linear."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _rank(xs), _rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _is_hit(action: str, ret: float) -> bool | None:
    if action == "buy":
        return ret > 0
    if action == "sell":
        return ret < 0
    return None  # holds have no direction to be right about


def _bucket(label: str, low: int, high: int, rows: Sequence[Mapping]) -> Bucket:
    rets = [float(r["forward_return"]) for r in rows]
    benches = [
        float(r["benchmark_return"])
        for r in rows
        if r["benchmark_return"] is not None
    ]
    hits = 0
    directional = 0
    for r in rows:
        verdict = _is_hit(str(r["action"]).lower(), float(r["forward_return"]))
        if verdict is None:
            continue
        directional += 1
        hits += int(verdict)

    return Bucket(
        label=label,
        low=low,
        high=high,
        count=len(rows),
        directional=directional,
        hits=hits,
        avg_return=(sum(rets) / len(rets)) if rets else 0.0,
        median_return=median(rets) if rets else 0.0,
        avg_benchmark=(sum(benches) / len(benches)) if benches else None,
    )


BANDS: tuple[tuple[str, int, int], ...] = (
    ("0-4 (no conviction)", 0, 4),
    ("5-6 (below gate)", 5, 6),
    ("7 (at gate)", 7, 7),
    ("8", 8, 8),
    ("9-10 (high conviction)", 9, 10),
)


def calibrate(
    journal,
    run_id: int,
    horizon_bars: int = 26,
    actions: Sequence[str] = ("buy", "sell", "hold"),
) -> Calibration:
    placeholders = ",".join("?" for _ in actions)
    rows = journal.query(
        f"""
        SELECT d.confidence, d.action, o.forward_return, o.benchmark_return
        FROM decisions d
        JOIN outcomes o ON o.decision_id = d.id
        WHERE d.run_id = ? AND o.horizon_bars = ?
          AND LOWER(d.action) IN ({placeholders})
        """,
        (run_id, horizon_bars, *[a.lower() for a in actions]),
    )

    buckets = tuple(
        _bucket(label, low, high, [r for r in rows if low <= int(r["confidence"]) <= high])
        for label, low, high in BANDS
    )
    populated = [b for b in buckets if b.count > 0]

    rho = spearman(
        [float(r["confidence"]) for r in rows],
        [float(r["forward_return"]) for r in rows],
    )

    monotonic = all(
        a.avg_return <= b.avg_return for a, b in zip(populated, populated[1:])
    ) and len(populated) >= 2

    return Calibration(
        horizon_bars=horizon_bars,
        resolved=len(rows),
        buckets=buckets,
        rank_correlation=rho,
        monotonic=monotonic,
        verdict=_verdict(len(rows), rho, monotonic),
    )


def _verdict(n: int, rho: float | None, monotonic: bool) -> str:
    if n < 30:
        return (
            f"Not enough resolved decisions ({n}) to say anything. "
            "Treat the confidence gate as unvalidated."
        )
    if rho is None:
        return "Confidence never varied; the gate is doing no work."
    if rho >= 0.15 and monotonic:
        return f"Confidence carries signal (rho={rho:+.2f}) and buckets are ordered correctly."
    if rho >= 0.05:
        return f"Weak but positive relationship (rho={rho:+.2f}); the gate is marginal."
    if rho <= -0.05:
        return (
            f"Confidence is inverted (rho={rho:+.2f}) -- high-conviction calls did "
            "worse. Do not trade on this number."
        )
    return (
        f"No relationship between confidence and outcome (rho={rho:+.2f}). "
        "The gate is filtering noise, not selecting edge."
    )
