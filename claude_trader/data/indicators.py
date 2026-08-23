"""Pure indicator maths. No I/O, no config, no side effects.

Every function returns None rather than guessing when there is not enough
history. Callers must handle that -- a fabricated indicator is worse than a
missing one, because the model will happily reason over it.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..models import Bar, Indicators


def sma(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    k = 2.0 / (period + 1.0)
    out = sum(values[:period]) / period
    for value in values[period:]:
        out = value * k + out * (1.0 - k)
    return out


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    """Wilder's RSI. Needs period+1 closes to produce the first value."""
    if len(values) < period + 1:
        return None
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    seed = deltas[:period]
    gains = sum(d for d in seed if d > 0) / period
    losses = -sum(d for d in seed if d < 0) / period
    for delta in deltas[period:]:
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        gains = (gains * (period - 1) + gain) / period
        losses = (losses * (period - 1) + loss) / period
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def true_ranges(bars: Sequence[Bar]) -> list[float]:
    if not bars:
        return []
    out = [bars[0].h - bars[0].l]
    for prev, cur in zip(bars, bars[1:]):
        out.append(
            max(
                cur.h - cur.l,
                abs(cur.h - prev.c),
                abs(cur.l - prev.c),
            )
        )
    return out


def atr(bars: Sequence[Bar], period: int = 14) -> float | None:
    """Average True Range -- the volatility unit used for stops and sizing."""
    trs = true_ranges(bars)
    if len(trs) < period:
        return None
    value = sum(trs[:period]) / period
    for tr in trs[period:]:
        value = (value * (period - 1) + tr) / period
    return value


def stdev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def simple_returns(values: Sequence[float]) -> list[float]:
    return [
        (cur - prev) / prev
        for prev, cur in zip(values, values[1:])
        if prev not in (0, None)
    ]


def pct_change(values: Sequence[float], lookback: int) -> float | None:
    if lookback <= 0 or len(values) <= lookback:
        return None
    past = values[-1 - lookback]
    if past == 0:
        return None
    return (values[-1] - past) / past


def realized_vol(values: Sequence[float], periods_per_year: int = 6_552) -> float | None:
    """Annualised realised volatility from bar returns.

    The default assumes 15-minute bars: 26 bars/session * 252 sessions.
    """
    rets = simple_returns(values)
    sd = stdev(rets)
    if sd is None:
        return None
    return sd * math.sqrt(periods_per_year)


def correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Pearson correlation over the overlapping tail of two return series."""
    n = min(len(a), len(b))
    if n < 3:
        return None
    x = list(a[-n:])
    y = list(b[-n:])
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def max_drawdown(equity: Sequence[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction."""
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def classify_trend(fast: float | None, slow: float | None, rsi_value: float | None) -> str:
    if fast is None or slow is None:
        return "unknown"
    if fast > slow * 1.001:
        return "overbought_up" if (rsi_value or 50) > 70 else "up"
    if fast < slow * 0.999:
        return "oversold_down" if (rsi_value or 50) < 30 else "down"
    return "sideways"


def compute(bars: Sequence[Bar], fast: int = 5, slow: int = 20) -> Indicators:
    """Build the full indicator bundle from a bar series.

    Missing history degrades individual fields to None instead of failing --
    but last_price is always present, because everything else depends on it.
    """
    closes = [b.c for b in bars]
    volumes = [b.v for b in bars]
    last = closes[-1] if closes else 0.0

    atr_value = atr(bars)
    rsi_value = rsi(closes)
    fast_ma = sma(closes, fast)
    slow_ma = sma(closes, slow)

    avg_vol = sma(volumes, min(20, len(volumes))) if volumes else None
    vol_ratio = (
        volumes[-1] / avg_vol if avg_vol not in (None, 0) and volumes else None
    )

    window = closes[-min(len(closes), 40) :]
    high = max(window) if window else 0.0
    dist_high = ((last - high) / high * 100.0) if high > 0 else None

    return Indicators(
        last_price=last,
        sma_fast=fast_ma,
        sma_slow=slow_ma,
        ema_fast=ema(closes, fast),
        rsi=rsi_value,
        atr=atr_value,
        atr_pct=(atr_value / last * 100.0) if atr_value and last > 0 else None,
        realized_vol=realized_vol(closes),
        ret_1=pct_change(closes, 1),
        ret_5=pct_change(closes, 5),
        ret_20=pct_change(closes, 20),
        volume_ratio=vol_ratio,
        dist_from_high_pct=dist_high,
        trend=classify_trend(fast_ma, slow_ma, rsi_value),
    )
