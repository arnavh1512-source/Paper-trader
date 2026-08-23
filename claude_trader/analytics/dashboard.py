"""A single self-contained HTML page showing what the bot actually did.

A markdown report answers "how did the run go". This answers a different and
more immediate question: "what is it holding right now, what did it just trade,
and what did it decide *not* to trade". The last of those is the one no
broker screen will ever show you, and it is the reason the decision log here
includes every rejection alongside the reason the risk layer gave.

The page has no external requests of any kind -- no CDN, no fonts, no scripts
fetched from anywhere. It is one file you can open from disk, mail to yourself,
or publish, and it will render identically with the network unplugged. Charts
are hand-drawn SVG for the same reason.
"""

from __future__ import annotations

import html
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

from .calibration import Calibration
from .metrics import Performance, RoundTrip

__all__ = ["DashboardData", "collect", "render_dashboard"]

_POSITIVE = "#3fb950"
_NEGATIVE = "#f85149"
_MUTED = "#8b949e"


@dataclass(frozen=True, slots=True)
class DashboardData:
    """Everything the page draws, already read out of the journal."""

    run_id: int
    strategy: str
    kind: str
    started: str
    finished: str
    market: str
    currency: str
    benchmark: str
    performance: Performance
    equity: tuple[Mapping, ...]
    positions: tuple[Mapping, ...]
    round_trips: tuple[RoundTrip, ...]
    orders: tuple[Mapping, ...]
    decisions: tuple[Mapping, ...]
    cycles: tuple[Mapping, ...]
    calibration: Calibration | None = None
    dry_run: bool = False
    holds: int = 0
    tz: object = timezone.utc


# --------------------------------------------------------------- collection
def collect(journal, run_id: int, config, performance: Performance,
            calibration: Calibration | None = None) -> DashboardData:
    """Read one run out of the journal.

    Deliberately a plain read: the page must never be able to change the
    account it is describing.
    """
    run = journal.query("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not run:
        raise ValueError(f"run {run_id} is not in the journal")
    row = run[0]

    orders = journal.query(
        "SELECT * FROM orders WHERE run_id = ? ORDER BY ts DESC, id DESC",
        (run_id,))
    # Only the decisions that proposed something. Holds are counted rather
    # than listed: there are thousands of them and they all say the same thing.
    decisions = journal.query(
        "SELECT * FROM decisions WHERE run_id = ? AND lower(action) != 'hold' "
        "ORDER BY ts DESC, id DESC LIMIT 200", (run_id,))
    holds = journal.query(
        "SELECT count(*) AS n FROM decisions WHERE run_id = ? "
        "AND lower(action) = 'hold'", (run_id,))[0]["n"]
    cycles = journal.query(
        "SELECT * FROM cycles WHERE run_id = ? ORDER BY ts DESC LIMIT 50",
        (run_id,))
    positions = journal.query(
        "SELECT * FROM position_risk WHERE run_id = ? AND is_open = 1 "
        "ORDER BY symbol", (run_id,))

    from .metrics import build_round_trips
    ascending = list(reversed([dict(o) for o in orders]))

    dry = False
    try:
        dry = bool(json.loads(row["config_json"]).get("dry_run", False))
    except (ValueError, TypeError):
        pass

    return DashboardData(
        run_id=run_id,
        strategy=row["strategy"],
        kind=row["kind"],
        started=row["started_at"],
        finished=row["finished_at"] or "still running",
        market=config.market,
        currency=config.currency,
        tz=config.profile.tz,
        benchmark=config.benchmark,
        performance=performance,
        equity=tuple(dict(r) for r in journal.equity_curve(run_id)),
        positions=tuple(dict(r) for r in positions),
        round_trips=tuple(build_round_trips(ascending)),
        orders=tuple(dict(o) for o in orders),
        decisions=tuple(dict(d) for d in decisions),
        holds=int(holds),
        cycles=tuple(dict(c) for c in cycles),
        calibration=calibration,
        dry_run=dry,
    )


# ------------------------------------------------------------------ helpers
def _e(value: object) -> str:
    """Escape everything. Symbols and reasons can carry model-authored text,
    and a dashboard that renders that as markup is a dashboard that can be
    written to by whatever the model said."""
    return html.escape(str(value), quote=True)


def _pct(value: float | None, places: int = 2) -> str:
    return "n/a" if value is None else f"{value * 100:+.{places}f}%"


def _rate(value: float | None, places: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{places}f}%"


def _short(stamp: object, tz: object = None) -> str:
    """A timestamp in the market's own clock.

    The journal stores UTC, which is correct and unreadable: an NSE close shows
    as 10:00 and nothing in the row says why.
    """
    text = str(stamp)
    if tz is not None:
        try:
            return datetime.fromisoformat(text).astimezone(tz).strftime(
                "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            pass
    return text[:16].replace("T", " ") if len(text) >= 16 else text


def _tone(value: float | None) -> str:
    if value is None or value == 0:
        return "flat"
    return "up" if value > 0 else "down"


def _sparkline(values: Sequence[float], width: int, height: int,
               colour: str, fill: bool = False) -> str:
    """A polyline scaled to its own range.

    A shared y-axis with the benchmark would be more honest but unreadable when
    an index trades at 276 and the account at 100,000, so each line is scaled
    independently and the axis is deliberately unlabelled -- shape only, with
    the real numbers in the table above it.
    """
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = high - low or 1.0
    step = width / (len(values) - 1)
    points = [
        (i * step, height - ((v - low) / span) * (height - 6) - 3)
        for i, v in enumerate(values)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    out = ""
    if fill:
        area = f"0,{height} {line} {width:.1f},{height}"
        out += f'<polygon points="{area}" fill="{colour}" opacity="0.10"/>'
    out += (f'<polyline points="{line}" fill="none" stroke="{colour}" '
            f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>')
    return out


def _card(label: str, value: str, tone: str = "flat", note: str = "") -> str:
    sub = f'<div class="note">{_e(note)}</div>' if note else ""
    return (f'<div class="card"><div class="label">{_e(label)}</div>'
            f'<div class="value {tone}">{_e(value)}</div>{sub}</div>')


def _empty(message: str) -> str:
    return f'<p class="empty">{_e(message)}</p>'


# ------------------------------------------------------------------ sections
def _chart(data: DashboardData) -> str:
    equity = [float(r["equity"]) for r in data.equity]
    if len(equity) < 2:
        return _empty("No equity history yet — the curve appears after the "
                      "first cycle that runs while the market is open.")

    bench = [r["benchmark_price"] for r in data.equity]
    bench = [float(b) for b in bench if b is not None]

    w, h = 900, 220
    colour = _POSITIVE if equity[-1] >= equity[0] else _NEGATIVE
    svg = _sparkline(equity, w, h, colour, fill=True)
    legend = f'<span class="key" style="--c:{colour}"></span>account'
    if len(bench) == len(equity) and len(bench) >= 2:
        svg += _sparkline(bench, w, h, _MUTED)
        legend += (f'<span class="key" style="--c:{_MUTED}"></span>'
                   f'{_e(data.benchmark)}')

    return (
        f'<div class="legend">{legend}</div>'
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
        f'class="chart" role="img" aria-label="Equity curve">{svg}</svg>'
        f'<div class="axis"><span>{_e(_short(data.equity[0]["ts"], data.tz))}</span>'
        f'<span>{len(equity)} samples</span>'
        f'<span>{_e(_short(data.equity[-1]["ts"], data.tz))}</span></div>'
    )


def _headline(data: DashboardData, money: Callable[[float], str]) -> str:
    p = data.performance
    star = " *" if p.annualised_extrapolated else ""
    cards = [
        _card("Equity", money(p.ending_equity), _tone(p.total_return),
              f"from {money(p.starting_equity)}"),
        _card("Total return", _pct(p.total_return), _tone(p.total_return)),
        _card(f"vs {data.benchmark}", _pct(p.excess_return),
              _tone(p.excess_return),
              "benchmark " + _pct(p.benchmark_return)),
        _card("Max drawdown", _pct(-abs(p.max_drawdown)),
              "down" if p.max_drawdown else "flat"),
        _card("Trades", str(p.trades),
              "flat", f"win rate {_rate(p.win_rate)}"),
        _card("Annualised" + star, _pct(p.annualised_return),
              _tone(p.annualised_return),
              "extrapolated from a short sample" if p.annualised_extrapolated
              else f"Sharpe {p.sharpe:.2f}"),
    ]
    return f'<div class="cards">{"".join(cards)}</div>'


def _positions(data: DashboardData, money: Callable[[float], str]) -> str:
    if not data.positions:
        return _empty("Nothing held right now.")
    rows = []
    for p in data.positions:
        rows.append(
            "<tr>"
            f'<td class="sym">{_e(p["symbol"])}</td>'
            f"<td>{money(float(p['entry_price']))}</td>"
            f'<td class="down">{money(float(p["stop_price"]))}</td>'
            f'<td class="up">{money(float(p["target_price"]))}</td>'
            f"<td>{_e(p['bars_held'])}</td>"
            f'<td class="dim">{_e(_short(p["entry_time"], data.tz))}</td>'
            "</tr>")
    return (
        '<table><thead><tr><th>Symbol</th><th>Entry</th><th>Stop</th>'
        '<th>Target</th><th>Bars held</th><th>Opened</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>')


def _trades(data: DashboardData, money: Callable[[float], str]) -> str:
    if not data.round_trips:
        return _empty("No completed round trips yet — a trade appears here "
                      "once it has been both opened and closed.")
    rows = []
    shown = sorted(data.round_trips, key=lambda x: x.exit_time, reverse=True)[:100]
    for t in shown:
        tone = "up" if t.is_win else "down"
        rows.append(
            "<tr>"
            f'<td class="sym">{_e(t.symbol)}</td>'
            f"<td>{t.qty:g}</td>"
            f"<td>{money(t.entry_price)}</td>"
            f"<td>{money(t.exit_price)}</td>"
            f'<td class="{tone}">{money(t.pnl)}</td>'
            f'<td class="{tone}">{_pct(t.return_pct)}</td>'
            f'<td class="dim">{_e(_short(t.exit_time, data.tz))}</td>'
            "</tr>")
    return (
        '<table><thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th>'
        '<th>Exit</th><th>P&amp;L</th><th>Return</th><th>Closed</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        + (f'<p class="verdict">Showing the 100 most recent of '
           f'{len(data.round_trips)} closed trades. The totals above cover all '
           f'of them.</p>' if len(data.round_trips) > 100 else ''))


def _orders(data: DashboardData, money: Callable[[float], str]) -> str:
    if not data.orders:
        return _empty("No orders sent yet.")
    rows = []
    for o in data.orders[:60]:
        side = str(o["side"]).lower()
        tone = "up" if side == "buy" else "down"
        sim = ' <span class="tag">sim</span>' if o["simulated"] else ""
        rows.append(
            "<tr>"
            f'<td class="dim">{_e(_short(o["ts"], data.tz))}</td>'
            f'<td class="sym">{_e(o["symbol"])}</td>'
            f'<td class="{tone}">{_e(side.upper())}</td>'
            f"<td>{float(o['qty']):g}</td>"
            f"<td>{money(float(o['price']))}</td>"
            f"<td>{money(float(o['notional']))}</td>"
            f'<td class="dim">{_e(o["intent"])}{sim}</td>'
            "</tr>")
    return (
        '<table><thead><tr><th>Time</th><th>Symbol</th><th>Side</th>'
        '<th>Qty</th><th>Price</th><th>Notional</th><th>Intent</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _headlines_cell(row: Mapping[str, object]) -> str:
    """The news the model saw when it made this call.

    Escaped like everything else on this page. A headline is text written by a
    stranger; it is displayed because it explains a decision, not because it is
    trusted.
    """
    try:
        titles = json.loads(str(row["news_json"] or "[]"))
    except (ValueError, TypeError, KeyError, IndexError):
        return "—"
    if not isinstance(titles, list) or not titles:
        return "—"
    shown = " · ".join(_e(str(t)[:90]) for t in titles[:3])
    extra = f" (+{len(titles) - 3} more)" if len(titles) > 3 else ""
    return shown + _e(extra)


def _decisions(data: DashboardData) -> str:
    """Every decision, including the ones that never became a trade.

    This is the section the whole page exists for. A rejected buy with the
    reason attached is the only way to tell a strategy that found nothing from
    a risk layer that blocked everything."""
    if not data.decisions and not data.holds:
        return _empty("No decisions journalled yet.")

    # A hold is not a rejection. Listing every one of them as "blocked" buries
    # the handful of rows that carry information -- the trades, and the trades
    # the gate refused -- under hundreds of identical do-nothings.
    acted, held = list(data.decisions), data.holds
    if not acted:
        return _empty(f"Every decision so far was a hold ({held} of them). "
                      "Nothing was proposed for the gate to rule on.")

    rows = []
    for d in acted[:120]:
        action = str(d["action"]).lower()
        tone = {"buy": "up", "sell": "down"}.get(action, "dim")
        if d["executed"]:
            status, cls, why = "executed", "up", d["reason"] or ""
        elif not d["risk_approved"]:
            status, cls = "blocked", "down"
            why = d["risk_reason"] or ""
        else:
            status, cls = "no order", "dim"
            why = d["risk_reason"] or d["reason"] or ""
        rows.append(
            "<tr>"
            f'<td class="dim">{_e(_short(d["ts"], data.tz))}</td>'
            f'<td class="sym">{_e(d["symbol"])}</td>'
            f'<td class="{tone}">{_e(action.upper())}</td>'
            f'<td>{_e(d["confidence"])}</td>'
            f'<td class="{cls}">{_e(status)}</td>'
            f'<td class="dim reason">{_e(why or "—")}</td>'
            f'<td class="dim reason">{_headlines_cell(d)}</td>'
            "</tr>")
    return (
        '<table><thead><tr><th>Time</th><th>Symbol</th><th>Action</th>'
        '<th>Conf</th><th>Outcome</th><th>Reason</th>'
        '<th>Headlines shown</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'<p class="verdict">Showing the {len(rows)} most recent buy/sell '
        f'decisions. A further {held} decisions were holds and are counted '
        f'rather than listed.</p>')


def _calibration(cal: Calibration | None) -> str:
    if cal is None or not cal.resolved:
        return _empty("Not enough resolved decisions yet. Run `calibrate` "
                      "once decisions are old enough for their horizon to "
                      "have elapsed.")
    rows = []
    for b in cal.buckets:
        if not b.count:
            rows.append(f'<tr><td>{_e(b.label)}</td><td>0</td>'
                        '<td class="dim">—</td><td class="dim">—</td>'
                        '<td class="dim">—</td></tr>')
            continue
        rows.append(
            "<tr>"
            f"<td>{_e(b.label)}</td>"
            f"<td>{b.count}</td>"
            f"<td>{_rate(b.hit_rate) if b.directional else '—'}</td>"
            f'<td class="{_tone(b.avg_return)}">{_pct(b.avg_return, 3)}</td>'
            f"<td>{_pct(b.avg_benchmark, 3)}</td>"
            "</tr>")
    rho = "n/a" if cal.rank_correlation is None else f"{cal.rank_correlation:+.2f}"
    return (
        '<table><thead><tr><th>Confidence</th><th>Decisions</th>'
        '<th>Hit rate</th><th>Avg forward return</th><th>Benchmark</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        f'<p class="verdict"><strong>rho {rho}</strong> · '
        f'ordered correctly: {"yes" if cal.monotonic else "no"}<br>'
        f'{_e(cal.verdict)}</p>')


# -------------------------------------------------------------------- styles
_CSS = """
:root {
  --bg:#0d1117; --panel:#161b22; --line:#232a35; --text:#e6edf3;
  --dim:#8b949e; --up:#3fb950; --down:#f85149; --accent:#58a6ff;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --bg:#ffffff; --panel:#f6f8fa; --line:#d8dee4; --text:#1f2328;
    --dim:#636c76; --up:#1a7f37; --down:#cf222e; --accent:#0969da;
  }
}
:root[data-theme="light"] {
  --bg:#ffffff; --panel:#f6f8fa; --line:#d8dee4; --text:#1f2328;
  --dim:#636c76; --up:#1a7f37; --down:#cf222e; --accent:#0969da;
}
* { box-sizing:border-box; }
body {
  margin:0; padding:28px 20px 64px; background:var(--bg); color:var(--text);
  font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif;
}
.wrap { max-width:1080px; margin:0 auto; }
header { margin-bottom:22px; }
h1 { font-size:21px; margin:0 0 6px; letter-spacing:-.2px; }
.sub { color:var(--dim); font-size:13px; }
.badges { margin-top:10px; display:flex; gap:6px; flex-wrap:wrap; }
.badge {
  font-size:11px; letter-spacing:.4px; text-transform:uppercase;
  border:1px solid var(--line); border-radius:999px; padding:3px 9px;
  color:var(--dim);
}
.badge.warn { color:#d29922; border-color:#d29922; }
h2 {
  font-size:13px; text-transform:uppercase; letter-spacing:1px;
  color:var(--dim); margin:34px 0 10px; font-weight:600;
}
section { border-top:1px solid var(--line); }
section:first-of-type { border-top:none; }
.cards { display:grid; gap:10px; grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); }
.card { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:12px 14px; }
.label { font-size:11px; text-transform:uppercase; letter-spacing:.7px; color:var(--dim); }
.value { font-size:20px; font-weight:600; margin-top:3px; letter-spacing:-.3px; }
.note { font-size:11px; color:var(--dim); margin-top:2px; }
.up { color:var(--up); } .down { color:var(--down); }
.dim, .flat { color:var(--dim); }
.value.flat { color:var(--text); }
.chart { width:100%; height:220px; display:block; background:var(--panel);
         border:1px solid var(--line); border-radius:9px; }
.legend { display:flex; gap:16px; font-size:12px; color:var(--dim); margin-bottom:7px; }
.key { display:inline-block; width:10px; height:3px; border-radius:2px;
       background:var(--c); margin-right:6px; vertical-align:middle; }
.axis { display:flex; justify-content:space-between; font-size:11px;
        color:var(--dim); margin-top:6px; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:9px; }
table { width:100%; border-collapse:collapse; font-size:13px;
        font-variant-numeric:tabular-nums; }
th { text-align:left; font-size:11px; text-transform:uppercase;
     letter-spacing:.6px; color:var(--dim); font-weight:600;
     padding:9px 12px; background:var(--panel);
     border-bottom:1px solid var(--line); white-space:nowrap; }
td { padding:8px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }
tr:last-child td { border-bottom:none; }
.sym { font-weight:600; }
.reason { white-space:normal; min-width:220px; color:var(--dim); font-size:12px; }
.tag { font-size:10px; border:1px solid var(--line); border-radius:4px;
       padding:1px 4px; color:var(--dim); }
.empty { color:var(--dim); font-size:13px; background:var(--panel);
         border:1px solid var(--line); border-radius:9px; padding:16px; margin:0; }
.verdict { font-size:13px; color:var(--dim); margin:10px 0 0; }
footer { margin-top:44px; padding-top:16px; border-top:1px solid var(--line);
         color:var(--dim); font-size:12px; }
footer strong { color:var(--text); }
"""


def render_dashboard(data: DashboardData,
                     money: Callable[[float], str],
                     generated_at: datetime | None = None) -> str:
    """One HTML string. No external requests, no scripts, no state."""
    stamp = (generated_at or datetime.now().astimezone()).strftime(
        "%Y-%m-%d %H:%M %Z").strip()

    badges = [f'<span class="badge">{_e(data.market.upper())}</span>',
              f'<span class="badge">{_e(data.strategy)}</span>',
              f'<span class="badge">{_e(data.kind)}</span>']
    if data.dry_run:
        badges.append('<span class="badge warn">dry run</span>')
    if data.kind == "backtest":
        badges.append('<span class="badge warn">simulated fills</span>')

    def section(title: str, body: str, scroll: bool = True) -> str:
        inner = f'<div class="scroll">{body}</div>' if scroll and "<table" in body else body
        return f"<section><h2>{_e(title)}</h2>{inner}</section>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run {data.run_id} — {_e(data.strategy)} — claude-trader</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>Run {data.run_id} · {_e(data.strategy)}</h1>
  <div class="sub">{_e(_short(data.started, data.tz))} → {_e(_short(data.finished, data.tz))}
    · generated {_e(stamp)}</div>
  <div class="badges">{"".join(badges)}</div>
</header>

<section>{_headline(data, money)}</section>
{section("Equity", _chart(data), scroll=False)}
{section("Open positions", _positions(data, money))}
{section("Closed trades", _trades(data, money))}
{section("Order log", _orders(data, money))}
{section("Decisions — including the ones that never traded", _decisions(data))}
{section("Confidence calibration", _calibration(data.calibration))}

<footer>
<strong>Paper trading only.</strong> This page describes simulated activity in a
research tool. It is not financial advice, not a recommendation, and no figure
on it predicts future results. Backtested fills assume the open of the bar after
the decision plus modelled slippage; real fills differ.
</footer>
</div></body></html>
"""
