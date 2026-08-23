"""The command line.

The CLI is the only surface a person touches, so the failures worth pinning are
the ones that lose money quietly rather than crash: a flag that silently does
not reach the config, a missing run that prints a traceback instead of a
sentence, and ``doctor`` reporting PASS on a setup that cannot trade.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pytest

from claude_trader import cli
from claude_trader.config import AppConfig, RiskConfig
from claude_trader.engine.cycle import CycleReport
from claude_trader.errors import MarketDataError, TraderError
from claude_trader.journal.store import Journal
from claude_trader.risk.engine import RiskState

NOW = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)


def parse(*argv: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(list(argv))


def report(**overrides) -> CycleReport:
    base = dict(ts=NOW, market_open=True, equity=101_000.0, cash=51_000.0,
                position_count=2)
    return CycleReport(**{**base, **overrides})


# ------------------------------------------------------------------ labels
def test_an_nse_run_is_labelled_with_its_segment():
    """intraday and delivery are different products with different costs and a
    forced square-off; the label is how a reader tells two reports apart."""
    assert cli.market_label(AppConfig(market="in", segment="intraday")) == "IN intraday"
    assert cli.market_label(AppConfig(market="in", segment="delivery")) == "IN delivery"


def test_a_us_run_is_not_labelled_with_a_segment_it_does_not_have():
    """'US intraday' would imply a square-off that never happens."""
    assert cli.market_label(AppConfig(market="us", segment="intraday")) == "US"


# ------------------------------------------------------------------ config
def test_flags_reach_the_configuration():
    config = cli.make_config(parse("--market", "in", "--segment", "delivery", "trade"))
    assert (config.market, config.segment) == ("in", "delivery")


def test_an_unspecified_flag_leaves_the_environment_in_charge(monkeypatch):
    """``None`` must mean 'not specified', not 'override with None' -- otherwise
    the profile defaults are unreachable from the CLI."""
    monkeypatch.setenv("MARKET", "us")
    assert cli.make_config(parse("trade")).market == "us"


def test_an_explicit_flag_beats_the_environment(monkeypatch):
    monkeypatch.setenv("MARKET", "us")
    assert cli.make_config(parse("--market", "in", "trade")).market == "in"


def test_the_journal_path_is_overridable(tmp_path):
    path = str(tmp_path / "run.db")
    assert cli.make_config(parse("--journal", path, "trade")).journal_path == path


# ------------------------------------------------------------------ parser
def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_an_unknown_market_is_rejected_at_the_parser():
    """Better a usage error than a run that quietly falls back to US."""
    with pytest.raises(SystemExit):
        parse("--market", "uk", "trade")


@pytest.mark.parametrize("strategy", ["claude", "momentum"])
def test_both_strategies_are_selectable(strategy):
    assert parse("backtest", "--strategy", strategy).strategy == strategy


def test_an_unknown_strategy_is_rejected():
    with pytest.raises(SystemExit):
        parse("backtest", "--strategy", "hunch")


def test_the_control_baseline_runs_unless_it_is_turned_off():
    """A strategy with no control group cannot be shown to be worth its cost."""
    assert parse("backtest").baseline is True
    assert parse("backtest", "--no-baseline").baseline is False


def test_backtest_defaults_are_conservative():
    args = parse("backtest")
    assert args.days == 30
    assert args.warmup == 40
    assert args.synthetic is False
    assert args.refresh is False
    assert args.cash is None          # so the market profile decides


def test_dry_run_is_off_unless_asked_for():
    assert parse("trade").dry_run is False
    assert parse("trade", "--dry-run").dry_run is True


def test_each_command_binds_its_handler():
    for name, func in [("trade", cli.cmd_trade), ("backtest", cli.cmd_backtest),
                       ("calibrate", cli.cmd_calibrate), ("report", cli.cmd_report),
                       ("doctor", cli.cmd_doctor)]:
        assert parse(name).func is func


# ------------------------------------------------------------------- trade
def test_trade_runs_one_cycle_and_prints_the_summary(monkeypatch, capsys):
    seen: list[AppConfig] = []
    monkeypatch.setattr(cli, "run_live_cycle",
                        lambda config: seen.append(config) or report())
    assert cli.cmd_trade(parse("--market", "in", "trade", "--dry-run")) == 0
    assert seen[0].dry_run is True
    assert "2 open" in capsys.readouterr().out


def test_a_halted_cycle_says_so_on_the_command_line(monkeypatch, capsys):
    """A silent halt looks identical to a quiet market."""
    monkeypatch.setattr(cli, "run_live_cycle", lambda config: report(
        risk_state=RiskState(halted=True, reason="daily loss limit")))
    cli.cmd_trade(parse("trade"))
    assert "halted: daily loss limit" in capsys.readouterr().out


# ------------------------------------------------------------------ output
def test_output_goes_to_stdout_by_default(capsys):
    cli._emit("# Report", None)
    assert capsys.readouterr().out.strip() == "# Report"


def test_output_can_be_written_to_a_file(tmp_path, capsys):
    target = tmp_path / "nested" / "report.md"
    cli._emit("# Report", str(target))
    assert target.read_text(encoding="utf-8") == "# Report"
    assert "Wrote" in capsys.readouterr().out


def test_a_report_with_rupee_symbols_survives_the_write(tmp_path):
    """The default Windows codepage cannot encode the symbol; writing without an
    explicit encoding would kill the run at the last step."""
    target = tmp_path / "report.md"
    cli._emit("Starting equity: ₹1,00,000", str(target))
    assert "₹" in target.read_text(encoding="utf-8")


# ------------------------------------------------------------------ runs
def test_the_latest_run_is_found(journal):
    journal.resolve_live_run(strategy="momentum", now=NOW, config={})
    assert cli._latest_run_id(journal) is not None


def test_an_empty_journal_has_no_latest_run(journal):
    assert cli._latest_run_id(journal) is None


def test_reporting_on_an_empty_journal_is_a_sentence_not_a_traceback(
        tmp_path, capsys):
    args = parse("--journal", str(tmp_path / "j.db"), "report")
    assert cli.cmd_report(args) == 1
    assert "No runs in the journal yet." in capsys.readouterr().err


def test_reporting_on_a_run_that_does_not_exist_says_which_one(tmp_path, capsys):
    with Journal(str(tmp_path / "j.db")) as j:
        j.resolve_live_run(strategy="momentum", now=NOW, config={})
        j.commit()
    args = parse("--journal", str(tmp_path / "j.db"), "report", "--run", "999")
    assert cli.cmd_report(args) == 1
    assert "Run 999 not found." in capsys.readouterr().err


def test_calibrating_an_empty_journal_is_a_sentence_too(tmp_path, capsys):
    args = parse("--journal", str(tmp_path / "j.db"), "calibrate")
    assert cli.cmd_calibrate(args) == 1
    assert "No runs" in capsys.readouterr().err


def test_calibrating_a_run_with_no_decisions_does_not_download_anything(
        tmp_path, capsys):
    """Fetching a dataset to resolve zero decisions would spend a network call
    and a cache slot to produce an empty table."""
    with Journal(str(tmp_path / "j.db")) as j:
        run_id = j.resolve_live_run(strategy="momentum", now=NOW, config={})
        j.commit()
    args = parse("--journal", str(tmp_path / "j.db"), "calibrate", "--run", str(run_id))
    assert cli.cmd_calibrate(args) == 1
    assert "no decisions" in capsys.readouterr().err


def test_a_report_renders_for_a_journalled_run(tmp_path, capsys):
    with Journal(str(tmp_path / "j.db")) as j:
        j.resolve_live_run(strategy="momentum", now=NOW, config={})
        j.commit()
    assert cli.cmd_report(
        parse("--journal", str(tmp_path / "j.db"), "--market", "in", "report")) == 0
    out = capsys.readouterr().out
    assert "# Run 1: momentum" in out
    assert "STT and stamp duty" in out          # the caveats follow the market


# ---------------------------------------------------------------- backtest
def test_a_synthetic_backtest_runs_end_to_end(tmp_path, capsys):
    """The whole point of --synthetic: prove the engine runs with no key, no
    network and no vendor."""
    args = parse("--journal", str(tmp_path / "j.db"), "--market", "in",
                 "backtest", "--synthetic", "--days", "8", "--strategy", "momentum",
                 "--no-baseline", "--symbols", "RELIANCE,TCS")
    assert cli.cmd_backtest(args) == 0
    out = capsys.readouterr().out
    assert "# Backtest: momentum (IN intraday)" in out
    assert "Confidence calibration" in out


def test_a_synthetic_backtest_is_labelled_as_not_being_a_market(tmp_path, caplog):
    """A random walk is evidence that the code runs, and nothing else. Someone
    will quote the Sharpe from it unless the log says so."""
    args = parse("--journal", str(tmp_path / "j.db"), "--market", "in",
                 "backtest", "--synthetic", "--days", "8", "--strategy", "momentum",
                 "--no-baseline")
    with caplog.at_level("WARNING"):
        cli.cmd_backtest(args)
    assert "not a market" in caplog.text


def test_the_benchmark_is_added_to_the_dataset_even_if_not_requested(tmp_path,
                                                                    capsys):
    """Without it every report would say 'no benchmark recorded', which is the
    one number that decides whether the strategy was worth running."""
    args = parse("--journal", str(tmp_path / "j.db"), "--market", "in",
                 "backtest", "--synthetic", "--days", "8", "--strategy", "momentum",
                 "--no-baseline", "--symbols", "RELIANCE")
    cli.cmd_backtest(args)
    assert "buy-and-hold NIFTYBEES" in capsys.readouterr().out


def test_the_report_can_be_written_to_a_file(tmp_path):
    out = tmp_path / "report.md"
    args = parse("--journal", str(tmp_path / "j.db"), "--market", "in",
                 "backtest", "--synthetic", "--days", "8", "--strategy", "momentum",
                 "--no-baseline", "--out", str(out))
    assert cli.cmd_backtest(args) == 0
    assert "# Backtest" in out.read_text(encoding="utf-8")


def test_symbols_are_deduplicated_and_upper_cased(tmp_path):
    config = AppConfig(market="in")
    args = parse("backtest", "--synthetic", "--days", "4",
                 "--symbols", "reliance, RELIANCE ,tcs")
    market = cli._resolve_market(args, config)
    assert market.symbols.count("RELIANCE") == 1
    assert "TCS" in market.symbols


def test_no_symbols_means_the_whole_universe():
    config = AppConfig(market="in")
    args = parse("backtest", "--synthetic", "--days", "4")
    market = cli._resolve_market(args, config)
    assert set(config.universe) <= set(market.symbols)


# ------------------------------------------------------------------ doctor
def test_doctor_reports_the_configured_market(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_data_source_ok", lambda config: (True, "ok"))
    monkeypatch.setattr(cli, "_affordable_ok", lambda config: (True, "ok"))
    cli.cmd_doctor(parse("--journal", str(tmp_path / "j.db"), "--market", "in",
                         "doctor"))
    out = capsys.readouterr().out
    assert "NSE" in out or "India" in out
    assert "INR" in out


def test_doctor_does_not_demand_alpaca_keys_on_an_nse_run(tmp_path, monkeypatch,
                                                          capsys):
    """Reporting a failure that does not exist trains people to ignore it."""
    monkeypatch.setattr(cli, "_data_source_ok", lambda config: (True, "ok"))
    monkeypatch.setattr(cli, "_affordable_ok", lambda config: (True, "ok"))
    monkeypatch.setenv("STRATEGY", "momentum")
    code = cli.cmd_doctor(parse("--journal", str(tmp_path / "j.db"),
                                "--market", "in", "doctor"))
    assert "ALPACA_API_KEY" not in capsys.readouterr().out
    assert code == 0


def test_doctor_does_demand_alpaca_keys_on_a_us_run(tmp_path, capsys):
    code = cli.cmd_doctor(parse("--journal", str(tmp_path / "j.db"),
                                "--market", "us", "doctor"))
    out = capsys.readouterr().out
    assert "FAIL  ALPACA_API_KEY set" in out
    assert code == 1


def test_a_missing_anthropic_key_does_not_block_the_momentum_strategy(
        tmp_path, monkeypatch):
    """The control group costs nothing and needs no key; failing here would make
    the free path look broken."""
    monkeypatch.setattr(cli, "_data_source_ok", lambda config: (True, "ok"))
    monkeypatch.setattr(cli, "_affordable_ok", lambda config: (True, "ok"))
    monkeypatch.setenv("STRATEGY", "momentum")
    assert cli.cmd_doctor(parse("--journal", str(tmp_path / "j.db"),
                                "--market", "in", "doctor")) == 0


def test_a_missing_anthropic_key_does_block_the_claude_strategy(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(cli, "_data_source_ok", lambda config: (True, "ok"))
    monkeypatch.setattr(cli, "_affordable_ok", lambda config: (True, "ok"))
    monkeypatch.setenv("STRATEGY", "claude")
    assert cli.cmd_doctor(parse("--journal", str(tmp_path / "j.db"),
                                "--market", "in", "doctor")) == 1


def test_an_unreachable_feed_fails_the_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_data_source_ok",
                        lambda config: (False, "unreachable: timeout"))
    monkeypatch.setattr(cli, "_affordable_ok", lambda config: (True, "ok"))
    assert cli.cmd_doctor(parse("--journal", str(tmp_path / "j.db"),
                                "--market", "in", "doctor")) == 1
    assert "FAIL  market data" in capsys.readouterr().out


def test_the_us_data_check_is_skipped_without_credentials():
    ok, detail = cli._data_source_ok(AppConfig(market="us"))
    assert ok is False and "credentials" in detail


def test_a_feed_error_is_reported_rather_than_raised(monkeypatch):
    class Boom:
        def latest_prices(self, symbols, now):
            raise MarketDataError("vendor is down")

    monkeypatch.setattr("claude_trader.app.build_market_data", lambda config: Boom())
    ok, detail = cli._data_source_ok(AppConfig(market="in"))
    assert ok is False and "vendor is down" in detail


def test_a_feed_that_returns_no_price_is_not_a_pass(monkeypatch):
    """Reachable is not the same as usable, and a benchmark with no price makes
    every report's verdict 'no benchmark recorded'."""
    monkeypatch.setattr("claude_trader.app.build_market_data",
                        lambda config: type("E", (), {
                            "latest_prices": lambda self, s, n: {}})())
    ok, detail = cli._data_source_ok(AppConfig(market="in"))
    assert ok is False and "no price" in detail


def test_a_working_feed_reports_the_price_it_read(monkeypatch):
    monkeypatch.setattr("claude_trader.app.build_market_data",
                        lambda config: type("E", (), {
                            "latest_prices": lambda self, s, n: {"NIFTYBEES": 285.5}})())
    ok, detail = cli._data_source_ok(AppConfig(market="in"))
    assert ok is True and "NIFTYBEES" in detail


def test_a_writable_journal_passes_and_an_impossible_path_does_not(tmp_path):
    """The journal file IS the account on the paper broker, so a path that
    cannot be opened has to fail here rather than at the first order."""
    assert cli._journal_ok(str(tmp_path / "missing" / "dir" / "j.db")) is True
    assert cli._journal_ok(str(tmp_path)) is False


def test_a_real_timezone_resolves_and_an_invented_one_does_not():
    assert cli._tz_ok("Asia/Kolkata") is True
    assert cli._tz_ok("Mars/Olympus_Mons") is False


# -------------------------------------------------------------------- main
def test_main_returns_the_handler_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "run_live_cycle", lambda config: report())
    assert cli.main(["--journal", str(tmp_path / "j.db"), "trade"]) == 0


def test_a_trader_error_is_a_logged_message_not_a_traceback(monkeypatch, caplog):
    """A GitHub Actions run that ends in a stack trace tells the reader nothing
    about whether it was a missing key or a dead vendor."""
    def boom(config):
        raise TraderError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(cli, "run_live_cycle", boom)
    with caplog.at_level("ERROR"):
        assert cli.main(["trade"]) == 2
    assert "ANTHROPIC_API_KEY is not set" in caplog.text


def test_an_interrupt_exits_cleanly(monkeypatch):
    def boom(config):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_live_cycle", boom)
    assert cli.main(["trade"]) == 130


def test_an_unexpected_error_is_not_swallowed(monkeypatch):
    """Only errors this package raises deliberately are turned into exit codes;
    a bug must still surface as a bug."""
    def boom(config):
        raise ZeroDivisionError

    monkeypatch.setattr(cli, "run_live_cycle", boom)
    with pytest.raises(ZeroDivisionError):
        cli.main(["trade"])


# ------------------------------------------------------------------ dashboard
def test_the_dashboard_writes_one_self_contained_file(tmp_path, capsys):
    with Journal(str(tmp_path / "j.db")) as j:
        j.resolve_live_run(strategy="momentum", now=NOW, config={})
        j.commit()
    out = tmp_path / "page.html"
    args = parse("--journal", str(tmp_path / "j.db"), "dashboard",
                 "--out", str(out))

    assert cli.cmd_dashboard(args) == 0
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "https://" not in html.split("<footer>")[0]
    assert str(out.resolve()) in capsys.readouterr().out


def test_the_dashboard_on_an_empty_journal_says_what_to_run_first(
        tmp_path, capsys):
    args = parse("--journal", str(tmp_path / "j.db"), "dashboard",
                 "--out", str(tmp_path / "page.html"))
    assert cli.cmd_dashboard(args) == 1
    assert "No runs in the journal yet" in capsys.readouterr().err


def test_the_dashboard_on_a_missing_run_does_not_write_a_blank_page(
        tmp_path, capsys):
    """A page that renders for a run id that does not exist reads as 'this run
    did nothing', which is a claim about the account rather than about the id."""
    with Journal(str(tmp_path / "j.db")) as j:
        j.resolve_live_run(strategy="momentum", now=NOW, config={})
        j.commit()
    out = tmp_path / "page.html"
    args = parse("--journal", str(tmp_path / "j.db"), "dashboard",
                 "--run", "999", "--out", str(out))

    assert cli.cmd_dashboard(args) == 1
    assert not out.exists()
    assert "999" in capsys.readouterr().err


def test_the_dashboard_only_opens_a_browser_when_asked(tmp_path, monkeypatch):
    """Opening a window is a side effect. A scheduled job that renders a page
    every fifteen minutes must not also spawn a browser every fifteen minutes."""
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))
    with Journal(str(tmp_path / "j.db")) as j:
        j.resolve_live_run(strategy="momentum", now=NOW, config={})
        j.commit()

    out = tmp_path / "page.html"
    base = ["--journal", str(tmp_path / "j.db"), "dashboard", "--out", str(out)]
    assert cli.cmd_dashboard(parse(*base)) == 0
    assert opened == []

    assert cli.cmd_dashboard(parse(*base, "--open")) == 0
    assert len(opened) == 1


def test_the_buy_and_hold_control_is_not_the_default_run(tmp_path):
    """A backtest writes the strategy run first and its control second. Latest
    by id would resolve to the control and report an empty decision log for a
    run that made hundreds of decisions."""
    with Journal(str(tmp_path / "j.db")) as j:
        j.start_run("backtest", "momentum", NOW, {})
        j.start_run("backtest", "buy_and_hold", NOW, {})
        j.commit()
        assert j.query("SELECT strategy FROM runs WHERE id = ?",
                       (cli._latest_run_id(j),))[0]["strategy"] == "momentum"


# ------------------------------------------------------------ universe reach
class _Priced:
    """A market data source that knows prices and nothing else."""

    def __init__(self, prices):
        self._prices = prices

    def latest_prices(self, symbols, as_of):
        return {s: self._prices[s] for s in symbols if s in self._prices}


def _reach(monkeypatch, prices, **cfg):
    monkeypatch.setattr(cli, "build_market_data", lambda c: _Priced(prices),
                        raising=False)
    import claude_trader.app as app
    monkeypatch.setattr(app, "build_market_data", lambda c: _Priced(prices))
    return cli._affordable_ok(AppConfig(market="in", **cfg))


def test_reach_names_the_stocks_a_small_book_can_never_buy(monkeypatch):
    """The failure this catches is invisible otherwise: an unaffordable name is
    never picked, never logged, and quietly shrinks the universe under test."""
    ok, detail = _reach(
        monkeypatch,
        {"WIPRO": 180.0, "MARUTI": 13_565.0, "TCS": 2_302.0},
        starting_cash=2_000.0,
        universe=("WIPRO", "MARUTI", "TCS"),
        risk=RiskConfig(max_position_pct=0.40, max_notional_per_trade=800.0,
                        min_trade_notional=100.0),
    )
    assert not ok
    assert "MARUTI" in detail and "TCS" in detail
    assert "WIPRO" not in detail


def test_reach_passes_when_the_whole_universe_is_affordable(monkeypatch):
    ok, detail = _reach(
        monkeypatch,
        {"WIPRO": 180.0, "ITC": 269.0},
        starting_cash=2_000.0,
        universe=("WIPRO", "ITC"),
        risk=RiskConfig(max_position_pct=0.40, max_notional_per_trade=800.0,
                        min_trade_notional=100.0),
    )
    assert ok
    assert "all 2" in detail


def test_reach_is_a_warning_not_a_failure(monkeypatch, tmp_path, capsys):
    """A reduced universe is still a valid experiment, so it must not be an
    exit code -- only something the operator is told about."""
    monkeypatch.setattr(cli, "_data_source_ok", lambda config: (True, "ok"))
    monkeypatch.setattr(cli, "_affordable_ok",
                        lambda config: (False, "3/30 out of reach"))
    monkeypatch.setenv("STRATEGY", "momentum")
    code = cli.cmd_doctor(parse("--journal", str(tmp_path / "j.db"),
                                "--market", "in", "doctor"))
    assert code == 0
    assert "WARN  universe reach" in capsys.readouterr().out


def test_reach_is_not_asked_on_a_fractional_market(monkeypatch):
    """Fractional shares make the whole question moot -- any budget reaches any
    price -- and asking would cost a pointless round trip per symbol."""
    ok, detail = cli._affordable_ok(AppConfig(market="us", starting_cash=10.0))
    assert ok and "fractional" in detail
