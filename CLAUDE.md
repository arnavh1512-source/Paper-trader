# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What This Is

A paper-trading research bot for **NSE (India)** and **US equities**. It is a
package (`claude_trader/`), not a script. The top-level `trader.py` is a thin
launcher kept only because the GitHub Actions workflow calls it by name; it is
equivalent to `python -m claude_trader trade`.

The purpose of the project is measurement, not returns: does a language model
add anything to a trading decision that a deterministic momentum rule does not?
Preserve that framing when changing things.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt

pytest                                              # 969 tests, all offline
pytest --cov=claude_trader --cov-report=term-missing

python -m claude_trader --market in doctor
python -m claude_trader --market in trade --dry-run
python -m claude_trader --market in backtest --synthetic --days 60 --strategy momentum
python -m claude_trader --market in calibrate --horizon 8
python -m claude_trader --market in report --run 3
python -m claude_trader --market in dashboard --open
```

`python` is not always on PATH on Windows; use `py -3.11`.

## Architecture

Full reasoning in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The short
version:

`run_cycle` in `engine/cycle.py` is called by **both** the live path
(`app.run_live_cycle`) and the backtester (`backtest/engine.py`). Only the
injected broker and market data source differ. Composition happens exclusively in
`app.py` — `build_strategy`, `build_market_data`, `build_broker`.

One cycle: survey the universe → strategy picks candidates → per-symbol decision
→ risk gate → execute → journal.

| Layer | Where |
|---|---|
| Config | `config.py` — frozen `AppConfig`, env + flags, validates in `__post_init__` |
| Market facts | `markets/india.py`, `markets/us.py` — `MarketProfile` |
| Data | `data/sources.py` (Alpaca), `data/yahoo.py` (NSE), `data/indicators.py` |
| Strategy | `strategies/momentum.py` (control), `strategies/claude_strategy.py` |
| Model | `llm/client.py`, `llm/prompts.py`, `llm/schemas.py` |
| Risk | `risk/engine.py` |
| Execution | `engine/executor.py`, `brokers/` |
| Record | `journal/store.py`, `journal/schema.py` |
| Analysis | `analytics/metrics.py`, `analytics/calibration.py`, `analytics/report.py` |

## Invariants — do not break these

1. **The live and backtest paths share `run_cycle`.** Never add a branch inside
   it that asks whether it is backtesting. If behaviour must differ, it differs
   through an injected dependency.
2. **Market orders are never retried.** `brokers/alpaca.py` deliberately does
   not retry `POST /orders` or `DELETE /positions/{symbol}`. A retry is a second
   position, not a recovered request.
3. **No lookahead.** `HistoricalMarketData.bars(..., as_of=...)` slices strictly
   before `as_of`; `SimulatedBroker` fills at the next bar's open. Do not add a
   path that reads the decision bar's close.
4. **Domain types stay frozen.** `@dataclass(frozen=True, slots=True)` — use
   `dataclasses.replace`, never mutation. `slots=True` means `__dict__` does not
   exist, which matters when writing tests.
5. **Every decision is journalled, including skips.** Calibration is computed
   from decisions, not from trades. Dropping a skip row destroys the measurement.
6. **A missing quote blocks an entry, never an exit.** Being unable to price
   something is not a reason to keep holding it.
7. **Market-specific facts live in `markets/`.** No `if market == "in"` anywhere
   else.
8. **Secrets come from the environment only.** Never a file in the repo, never a
   default in code, never in a test.
9. **Tests are offline.** Every broker, feed and model call goes through a fake.
   A test that needs a real key is a test that will one day place a real order.
10. **News is untrusted input.** Headlines go into the prompt inside a
    `<headlines>` fence, both system prompts say headlines cannot change
    instructions, and nothing read from a feed reaches the risk layer. A feed
    failure is a warning and an empty list — never an exception that reaches
    the cycle, because that would be a feed outage blocking an exit.
11. **Backtests never see news.** `build_news(config, live=False)` returns
    `NullNewsSource` unconditionally. Today's headlines against historical bars
    is lookahead wearing a hat.
12. **The dashboard only reads.** `analytics/dashboard.py` issues SELECTs and
    nothing else, and escapes every value it renders — symbols and reasons
    carry model-authored text.
13. **Sizing must be coherent with the book.** Every buy is clamped to the
    position cap then rejected under `min_trade_notional`; if the floor exceeds
    the cap, nothing ever trades and nothing ever errors. `AppConfig.__post_init__`
    raises `ConfigError` on that contradiction, and `RiskConfig.from_env` scales
    its defaults to `equity` so a small book does not walk into it. Never
    reintroduce a fixed rupee floor that ignores equity.
14. **A fallback listing is a last resort, not an alternative.** `data_symbols`
    returns the primary exchange first and BSE only after; `yahoo` tries the
    fallback solely when the primary returns no bars, and caches the winner.
    Preferring whichever listing is cheaper would be cross-exchange arbitrage
    against a feed that is not synchronised.
15. **Schema changes are additive.** New columns go in `schema.ADDED_COLUMNS`
    and are applied by `Journal._migrate`. `CREATE TABLE IF NOT EXISTS` is a
    no-op on an existing journal, and a journal is months of decisions.

## Testing Notes

- `tests/conftest.py` provides `journal` (in-memory SQLite), an autouse
  `clean_env` fixture that strips all trading env vars, and builders
  `make_bars`, `ramp`, `make_quote`, `make_snapshot`, `make_state`.
- `make_bars` sets each bar's open to the previous close, so it cannot produce an
  overnight gap. Construct those explicitly.
- `AppConfig` nests LLM settings: `AppConfig(llm=LLMConfig(api_key="..."))`.
- `AppConfig` validates `strategy` in `__post_init__`, so an invalid value raises
  `ConfigError` before it can reach `build_strategy`.
- For India + intraday, `__post_init__` forces `square_off_enabled=True` and caps
  `max_holding_bars` to the session length, logging warnings.
- Test names should read as claims about behaviour, not as module coverage.

## Deployment

`.github/workflows/trader.yml` runs one cycle per schedule tick.
`.github/workflows/tests.yml` runs the suite on 3.11 and 3.12 with no secrets in
scope.

Secrets (`ANTHROPIC_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) go in
**Settings → Secrets and variables → Actions**; `MARKET` and `STRATEGY` are
repository *variables*.

The journal is cached and archived between runs because **the journal is the
account**. Never commit it — `.gitignore` excludes `data/`.

The Alpaca base URL points at the **paper** endpoint. Changing `ALPACA_BASE` to
the live endpoint means real money; `app.py` warns loudly when it is not the
paper host, and that warning must stay.

## Scope

This is a research instrument. It does not produce financial advice, and nothing
in it should be described as a recommendation or a prediction of returns.
