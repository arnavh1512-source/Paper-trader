"""Command line interface.

    python -m claude_trader trade                       # one live paper cycle
    python -m claude_trader --market in trade
    python -m claude_trader backtest --days 30
    python -m claude_trader backtest --synthetic --strategy momentum
    python -m claude_trader calibrate --run 3
    python -m claude_trader report --run 3 --out report.md
    python -m claude_trader dashboard --open
    python -m claude_trader doctor

``--market`` and ``--segment`` sit on the top-level parser rather than on each
subcommand because they change what every subcommand means: which universe,
which currency, which benchmark, which costs, and which broker.
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .analytics.calibration import DEFAULT_HORIZONS, calibrate, resolve_outcomes
from .analytics.dashboard import collect, render_dashboard
from .analytics.metrics import compute_performance
from .analytics.report import render_calibration, render_report
from .app import build_strategy, configure_logging, run_live_cycle, summarise
from .backtest.dataset import DatasetSpec, load_dataset, synthetic_bars
from .backtest.engine import compare, run_backtest, run_buy_and_hold
from .config import AppConfig, load_dotenv
from .data.sources import HistoricalMarketData
from .errors import TraderError
from .journal.store import Journal

log = logging.getLogger(__name__)


def market_label(config: AppConfig) -> str:
    """``IN intraday`` / ``US``.

    The cash/delivery split is an NSE concept. Printing "US intraday" would
    imply a square-off that never happens on the US path.
    """
    if config.market == "in":
        return f"{config.market.upper()} {config.segment}"
    return config.market.upper()


def make_config(args: argparse.Namespace, **overrides) -> AppConfig:
    """One place where command line flags become configuration.

    ``None`` means "not specified", and ``from_env`` drops those, so the
    environment (and then the profile defaults) still decide.
    """
    return AppConfig.from_env(
        market=getattr(args, "market", None),
        segment=getattr(args, "segment", None),
        journal_path=getattr(args, "journal", None),
        **overrides,
    )


# ----------------------------------------------------------------------- trade
def cmd_trade(args: argparse.Namespace) -> int:
    config = make_config(args, dry_run=args.dry_run, strategy=args.strategy)
    report = run_live_cycle(config)
    print(summarise(report, config))
    return 0


# -------------------------------------------------------------------- backtest
def _resolve_market(args: argparse.Namespace, config: AppConfig) -> HistoricalMarketData:
    symbols = tuple(
        dict.fromkeys(
            [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
            or list(config.universe)
        )
    )
    if config.benchmark not in symbols:
        symbols = symbols + (config.benchmark,)

    if args.synthetic:
        log.warning(
            "Synthetic data: a random walk, not a market. Useful for proving the "
            "engine runs; useless as evidence about the strategy."
        )
        return HistoricalMarketData(
            synthetic_bars(
                symbols, sessions=args.days, seed=args.seed, profile=config.profile
            )
        )

    end = datetime.now(timezone.utc)
    spec = DatasetSpec(
        symbols=symbols,
        start=end - timedelta(days=args.days),
        end=end,
        timeframe=config.timeframe,
        feed=config.feed,
        market=config.market,
    )
    return HistoricalMarketData(
        load_dataset(config, spec, cache_dir=args.cache_dir, refresh=args.refresh)
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    config = make_config(args, strategy=args.strategy)
    cash = config.starting_cash if args.cash is None else args.cash
    market = _resolve_market(args, config)
    log.info(
        "Dataset: %d symbols, %d bars on the timeline (%s)",
        len(market.symbols),
        len(market.timeline()),
        market_label(config),
    )

    with Journal(config.journal_path) as journal:
        strategy = build_strategy(config, journal)
        primary = run_backtest(
            config,
            market,
            strategy,
            journal,
            label=config.strategy,
            starting_cash=cash,
            warmup_bars=args.warmup,
            calibrate_horizon=args.horizon,
        )

        results = [primary]
        if args.baseline and config.strategy != "momentum":
            baseline_config = make_config(args, strategy="momentum")
            results.append(
                run_backtest(
                    baseline_config,
                    market,
                    build_strategy(baseline_config, journal),
                    journal,
                    label="momentum (control)",
                    starting_cash=cash,
                    warmup_bars=args.warmup,
                    calibrate_horizon=None,
                )
            )

        if config.benchmark in market.symbols:
            results.append(
                run_buy_and_hold(
                    config,
                    market,
                    journal,
                    starting_cash=cash,
                    warmup_bars=args.warmup,
                )
            )

        source = (
            "synthetic random walk"
            if args.synthetic
            else f"{'Yahoo' if config.market != 'us' else 'Alpaca ' + config.feed} "
                 f"{config.timeframe}, last {args.days} days"
        )
        markdown = render_report(
            header=f"Backtest: {config.strategy} ({market_label(config)})",
            performance=primary.performance,
            calibration=primary.calibration,
            comparison=compare(results),
            meta={
                "run id": primary.run_id,
                "market": f"{config.profile.name} ({config.profile.currency})",
                "symbols": len(market.symbols),
                "cycles": primary.cycles,
                "orders": primary.orders,
                "starting cash": config.money(cash),
                "data": source,
                "model calls": f"{primary.llm_calls} ({primary.llm_cache_hits} cache hits)",
            },
            money=config.money,
            benchmark=config.benchmark,
            market=config.market,
        )

    _emit(markdown, args.out)
    return 0


# ------------------------------------------------------------------- calibrate
def cmd_calibrate(args: argparse.Namespace) -> int:
    config = make_config(args)
    with Journal(config.journal_path) as journal:
        run_id = args.run or _latest_run_id(journal)
        if run_id is None:
            print("No runs in the journal yet.", file=sys.stderr)
            return 1

        rows = journal.query(
            "SELECT DISTINCT symbol FROM decisions WHERE run_id = ?", (run_id,)
        )
        if not rows:
            print(f"Run {run_id} has no decisions to resolve.", file=sys.stderr)
            return 1
        symbols = tuple(sorted({r["symbol"] for r in rows} | {config.benchmark}))

        span = journal.query(
            "SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM decisions WHERE run_id = ?",
            (run_id,),
        )[0]
        start = datetime.fromisoformat(span["lo"]) - timedelta(days=1)
        end = datetime.now(timezone.utc)

        spec = DatasetSpec(
            symbols=symbols,
            start=start,
            end=end,
            timeframe=config.timeframe,
            feed=config.feed,
            market=config.market,
        )
        market = HistoricalMarketData(
            load_dataset(config, spec, cache_dir=args.cache_dir, refresh=args.refresh)
        )
        written = resolve_outcomes(
            journal, run_id, market, DEFAULT_HORIZONS, benchmark_symbol=config.benchmark
        )
        journal.commit()
        log.info("Resolved %d new outcomes for run %d", written, run_id)

        result = calibrate(journal, run_id, args.horizon)

    _emit(render_calibration(result), args.out)
    return 0


# ---------------------------------------------------------------------- report
def cmd_report(args: argparse.Namespace) -> int:
    config = make_config(args)
    with Journal(config.journal_path) as journal:
        run_id = args.run or _latest_run_id(journal)
        if run_id is None:
            print("No runs in the journal yet.", file=sys.stderr)
            return 1
        run = journal.query("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not run:
            print(f"Run {run_id} not found.", file=sys.stderr)
            return 1

        orders = [dict(r) for r in journal.query(
            "SELECT symbol, side, qty, price, ts FROM orders WHERE run_id = ? ORDER BY ts, id",
            (run_id,),
        )]
        performance = compute_performance(
            journal.equity_curve(run_id), orders, periods_per_year=config.periods_per_year
        )
        calibration = calibrate(journal, run_id, args.horizon)
        markdown = render_report(
            header=f"Run {run_id}: {run[0]['strategy']} ({run[0]['kind']})",
            performance=performance,
            calibration=calibration if calibration.resolved else None,
            meta={
                "started": run[0]["started_at"],
                "finished": run[0]["finished_at"] or "still running",
                "notes": run[0]["notes"] or "-",
            },
            money=config.money,
            benchmark=config.benchmark,
            market=config.market,
        )

    _emit(markdown, args.out)
    return 0


# ------------------------------------------------------------------- dashboard
def cmd_dashboard(args: argparse.Namespace) -> int:
    """Render one run as a self-contained HTML page.

    Writes a file rather than serving one. A long-lived server next to a
    trading account is an attack surface with no upside, and the whole point of
    inlining everything is that the file needs nothing to render.
    """
    config = make_config(args)
    with Journal(config.journal_path) as journal:
        run_id = args.run or _latest_run_id(journal)
        if run_id is None:
            print("No runs in the journal yet. Run `trade` or `backtest` first.",
                  file=sys.stderr)
            return 1
        orders = [dict(r) for r in journal.query(
            "SELECT symbol, side, qty, price, ts FROM orders WHERE run_id = ? "
            "ORDER BY ts, id", (run_id,),
        )]
        performance = compute_performance(
            journal.equity_curve(run_id), orders,
            periods_per_year=config.periods_per_year,
        )
        calibration = calibrate(journal, run_id, args.horizon)
        try:
            data = collect(journal, run_id, config, performance,
                           calibration if calibration.resolved else None)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard(data, config.money), encoding="utf-8")
    print(f"Wrote {target.resolve()}")
    if args.open:
        webbrowser.open(target.resolve().as_uri())
    return 0


# ---------------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    """Answer one question: would a live cycle work right now, and with what?

    Checks are scoped to the configured market. Demanding Alpaca credentials
    from someone trading NSE would report a failure that does not exist.
    """
    config = make_config(args)
    profile = config.profile
    checks: list[tuple[str, bool, str]] = [
        (
            "market profile",
            True,
            f"{profile.name} | {profile.currency} | {profile.open_time:%H:%M}-"
            f"{profile.close_time:%H:%M} {profile.tz_name} | {market_label(config)}",
        ),
        (
            "timezone database",
            _tz_ok(profile.tz_name),
            f"{profile.tz_name} resolves"
            if _tz_ok(profile.tz_name)
            else "tzdata missing -- run: pip install tzdata (fixed-offset fallback in use)",
        ),
        (
            "broker",
            config.is_paper,
            f"{config.broker} ({'paper' if config.is_paper else 'LIVE MONEY'})",
        ),
        (
            "journal writable",
            _journal_ok(config.journal_path),
            f"{config.journal_path} -- this file IS the account on the paper broker",
        ),
        (
            "ANTHROPIC_API_KEY set",
            bool(config.llm.api_key),
            "only needed for the claude strategy",
        ),
    ]
    if config.market == "us":
        checks[3:3] = [
            ("ALPACA_API_KEY set", bool(config.alpaca_key), "required for US market data"),
            ("ALPACA_SECRET_KEY set", bool(config.alpaca_secret), "required for US market data"),
        ]

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")

    data_ok, detail = _data_source_ok(config)
    print(f"{'PASS' if data_ok else 'FAIL'}  {'market data':<{width}}  {detail}")

    if data_ok:
        reach_ok, detail = _affordable_ok(config)
        print(f"{'PASS' if reach_ok else 'WARN'}  {'universe reach':<{width}}  {detail}")

    # The Anthropic key is advisory unless the claude strategy is selected, and
    # a warm timezone fallback is a warning, not a stoppage.
    blocking = [
        ok
        for name, ok, _ in checks
        if name not in {"ANTHROPIC_API_KEY set", "timezone database"}
    ]
    if config.strategy == "claude":
        blocking.append(bool(config.llm.api_key))
    return 0 if all(blocking) and data_ok else 1


def _tz_ok(name: str) -> bool:
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(name)
        return True
    except Exception:
        return False


def _affordable_ok(config: AppConfig) -> tuple[bool, str]:
    """How much of the universe can this book actually buy?

    On a whole-share market the position cap and the share price interact in a
    way no amount of config validation catches: a Rs 800 ceiling simply cannot
    reach one share of a Rs 2,300 stock, so that name is silently unreachable.
    Nothing errors, nothing is logged, it is just never picked -- and the run
    quietly measures a smaller universe than the one it reports.

    A warning, never a failure: a reduced universe is still a valid experiment.
    """
    from .app import build_market_data

    profile = config.profile
    if profile.fractional_shares:
        return True, "fractional shares -- every name is reachable at any size"

    budget = min(
        config.starting_cash * config.risk.max_position_pct,
        config.risk.max_notional_per_trade,
    )
    try:
        prices = build_market_data(config).latest_prices(
            list(config.universe), datetime.now(timezone.utc)
        )
    except TraderError as exc:
        return True, f"could not price the universe ({exc}); skipped"

    priced = {s: p for s, p in prices.items() if p and p > 0}
    lot = profile.lot_size or 1
    unreachable = sorted(s for s, p in priced.items() if p * lot > budget)
    sym = profile.currency_symbol
    if not priced:
        return True, "no prices returned; skipped"
    if not unreachable:
        return True, (
            f"all {len(priced)} priced names reachable "
            f"within the {sym}{budget:,.0f} position cap"
        )
    shown = ", ".join(unreachable[:6]) + (
        f" (+{len(unreachable) - 6} more)" if len(unreachable) > 6 else ""
    )
    return False, (
        f"{len(unreachable)}/{len(priced)} names cost more than the "
        f"{sym}{budget:,.0f} position cap and can never be bought: {shown}"
    )


def _data_source_ok(config: AppConfig) -> tuple[bool, str]:
    from .app import build_market_data

    if config.market == "us" and not (config.alpaca_key and config.alpaca_secret):
        return False, "no Alpaca credentials to test with"
    try:
        source = build_market_data(config)
        prices = source.latest_prices([config.benchmark], datetime.now(timezone.utc))
    except TraderError as exc:
        return False, f"unreachable: {exc}"
    if not prices:
        return False, f"reachable, but returned no price for {config.benchmark}"
    return True, ", ".join(f"{k} {config.money(v)}" for k, v in prices.items())


def _journal_ok(path: str) -> bool:
    try:
        Journal(path).close()
        return True
    except Exception:
        return False


def _latest_run_id(journal: Journal) -> int | None:
    """The most recent run worth looking at.

    A backtest writes two runs: the strategy and the buy-and-hold control that
    exists to be compared against it. The control is always written second, so
    a naive "latest" resolves to the baseline and reports an empty decision log
    for a run that made hundreds of decisions.
    """
    rows = journal.query(
        "SELECT id FROM runs WHERE strategy != 'buy_and_hold' "
        "ORDER BY id DESC LIMIT 1"
    )
    if rows:
        return int(rows[0]["id"])
    row = journal.latest_run()
    return int(row["id"]) if row else None


def _emit(text: str, out: str | None) -> None:
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"Wrote {target}")
    else:
        print(text)


# ------------------------------------------------------------------ arg parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-trader",
        description="Paper-trading research bot with a shared live/backtest decision path.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--journal", default=None, help="path to the SQLite journal")
    parser.add_argument(
        "--market",
        default=None,
        choices=["in", "us"],
        help="which market to trade; changes universe, currency, costs and broker",
    )
    parser.add_argument(
        "--segment",
        default=None,
        choices=["intraday", "delivery"],
        help="NSE cash segment; intraday squares off before the close and costs less",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    trade = sub.add_parser("trade", help="run one live paper-trading cycle")
    trade.add_argument("--dry-run", action="store_true", help="journal decisions, send no orders")
    trade.add_argument("--strategy", default=None, choices=["claude", "momentum"])
    trade.set_defaults(func=cmd_trade)

    back = sub.add_parser("backtest", help="replay the decision path over history")
    back.add_argument("--days", type=int, default=30)
    back.add_argument("--symbols", default="", help="comma separated; defaults to the universe")
    back.add_argument("--strategy", default=None, choices=["claude", "momentum"])
    back.add_argument("--cash", type=float, default=None, help="defaults to the market's starting cash")
    back.add_argument("--warmup", type=int, default=40, help="bars reserved for indicators")
    back.add_argument("--horizon", type=int, default=26, help="calibration horizon in bars")
    back.add_argument("--synthetic", action="store_true", help="offline random-walk data")
    back.add_argument("--seed", type=int, default=7)
    back.add_argument("--no-baseline", dest="baseline", action="store_false")
    back.add_argument("--refresh", action="store_true", help="ignore the dataset cache")
    back.add_argument("--cache-dir", default="data/cache")
    back.add_argument("--out", default=None, help="write the markdown report here")
    back.set_defaults(func=cmd_backtest, baseline=True)

    cal = sub.add_parser("calibrate", help="resolve outcomes and score the confidence gate")
    cal.add_argument("--run", type=int, default=None)
    cal.add_argument("--horizon", type=int, default=26)
    cal.add_argument("--refresh", action="store_true")
    cal.add_argument("--cache-dir", default="data/cache")
    cal.add_argument("--out", default=None)
    cal.set_defaults(func=cmd_calibrate)

    rep = sub.add_parser("report", help="render a markdown report for a journalled run")
    rep.add_argument("--run", type=int, default=None)
    rep.add_argument("--horizon", type=int, default=26)
    rep.add_argument("--out", default=None)
    rep.set_defaults(func=cmd_report)

    dash = sub.add_parser("dashboard", help="render a run as a self-contained HTML page")
    dash.add_argument("--run", type=int, default=None)
    dash.add_argument("--horizon", type=int, default=26)
    dash.add_argument("--out", default="data/dashboard.html")
    dash.add_argument("--open", action="store_true", help="open it in your browser")
    dash.set_defaults(func=cmd_dashboard)

    doc = sub.add_parser("doctor", help="check configuration, credentials and connectivity")
    doc.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    if load_dotenv():
        log.debug("loaded settings from .env")

    try:
        return int(args.func(args))
    except TraderError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
