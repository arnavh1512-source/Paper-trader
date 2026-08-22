# What Was Built

A rebuild of `claude-trader` from a single-file script into a measurable,
two-market paper-trading research bot.

**Paper trading only. Nothing in this document or in the program is financial
advice, and no result from it predicts anything about real money.**

---

## 1. The starting point

The original was one file, roughly 300 lines. It asked Claude "what should I buy
from these 28 stocks", took the answer, and bought it if the model reported
confidence ≥ 7. It had:

- **No memory between runs.** Every 15 minutes it woke up with amnesia.
- **No record of decisions**, only whatever Alpaca happened to store.
- **No risk layer.** The model's opinion went straight to an order.
- **No backtest**, so no way to evaluate a change before running it with money.
- **An uncalibrated confidence gate.** `>= 7` was a number someone picked, and
  there was no mechanism — even in principle — to learn whether 7 differed
  from 4.
- **Prompts with a structural buy-bias.** "Which stocks should you buy?" is a
  question that presupposes buying.

It was not a bad *implementation* of a trading bot. It was a trading bot that
could not tell you anything about itself.

The rebuild is organised around a different goal: **not "make money" but "be
measurable."**

---

## 2. The invariant everything hangs off

> **The live cycle and the backtest run the same function.**

`engine/cycle.py::run_cycle(deps, now)` is called by both paths. The only
difference is what gets injected into `deps` — which broker, which data source,
which cost model.

```
                       +----------------------+
 app.run_live_cycle    |                      |   backtest.engine
   AlpacaMarketData    |      run_cycle       |   HistoricalMarketData
   AlpacaBroker /      |  survey -> decide -> |   SimulatedBroker
   PaperBroker         |  gate -> execute ->  |
                       |  journal             |
                       +----------------------+
```

This matters because the usual way a backtest lies is that it is a *different
program* from the live bot. There is no `if backtesting:` branch anywhere in the
decision path. If behaviour must differ, it differs through an injected
dependency — which means the difference is visible in exactly one place.

---

## 3. One cycle, step by step

```
1. broker.is_market_open(now)      -> closed? record it and stop
2. market.snapshots(universe)      -> ONE batched call for all symbols
3. strategy.pick(...)              -> 3-5 candidates, or abstain
4. for each candidate + every open position:
       market.bars() / market.quote()
       strategy.decide(...)        -> buy/sell/hold + confidence
       risk.evaluate(...)          -> RiskVerdict(ok, reason)
       executor.execute(...)       -> OrderRequest -> OrderResult
5. journal.record(...)             -> decisions, orders, skips, reasons
6. CycleReport -> summarise -> one log line
```

Step 4 runs for open positions too, not just new candidates — the old bot could
buy something and then never look at it again.

Steps 4 and 5 happen **whether or not an order results.** That is the single
most important change; section 7 explains why.

---

## 4. The modules

| Module | What it owns |
|---|---|
| `config.py` | One frozen `AppConfig` from env + flags. Validates in `__post_init__` and raises `ConfigError` rather than starting with a nonsensical setup. ~45 tunables. |
| `markets/` | `MarketProfile` per market — session hours, timezone, tick size, lot rules, benchmark, currency, cost model. **The only place a market-specific fact is allowed to live.** |
| `models.py` | Frozen domain types: `Bar`, `Quote`, `Snapshot`, `Position`, `Account`, `OrderRequest`, `OrderResult`, `Decision`, `Picks`, `PortfolioState`. |
| `data/` | `AlpacaMarketData`, `YahooMarketData`, plus `indicators.py` (ATR, RSI, SMA/EMA, volume profile, gap, range position). |
| `strategies/` | Two-method `Strategy` protocol. `MomentumStrategy` (deterministic control) and `ClaudeStrategy` (the subject). |
| `llm/` | Anthropic client with retries, response caching, and **schema validation** of every reply. |
| `risk/` | The gate. Sizing, caps, breakers, staleness, spread, cost-to-edge. |
| `engine/` | `run_cycle` plus the executor that turns an approved decision into an order. |
| `brokers/` | `AlpacaBroker` (live paper), `PaperBroker` (journal-backed book), `SimulatedBroker` (backtest fills). |
| `backtest/` | Dataset fetch/cache and the replay engine. |
| `analytics/` | Metrics, calibration, markdown reports. |
| `journal/` | SQLite schema and store. The system of record. |

45 source files, roughly 2,850 statements.

---

## 5. India, as a real market rather than a flag

`markets/india.py` is a full profile, not a symbol-list swap.

| | India (`--market in`) | US (`--market us`) |
|---|---|---|
| Session | 09:15–15:30 IST (25 bars) | 09:30–16:00 ET (26 bars) |
| Benchmark | `NIFTYBEES` | `SPY` |
| Data | Yahoo — **no key needed** | Alpaca |
| Broker | internal paper book (NSE has no free sandbox) | Alpaca paper API |
| Shares | whole shares, ₹0.05 tick | fractional, $0.01 tick |
| Starting cash | ₹100,000 | $10,000 |
| Max per trade | ₹10,000 | $100 |

The benchmark is `NIFTYBEES` and not the NIFTY 50 index deliberately — you
cannot buy an index. Comparing against something untradeable flatters the
strategy.

**Costs are modelled properly**, which for NSE is not a rounding error:
brokerage, STT at the *intraday* rate, stamp duty, exchange turnover fee, SEBI
fee, and GST on top. On ₹10,000 tickets this is frequently the entire difference
between a positive and a negative strategy. The US path models commission-free
trading with spread and slippage instead.

`--segment intraday` squares off before the close and pays the lower STT rate.
Choosing it **forces** `square_off_enabled=True` and caps `max_holding_bars` to
the session length. An intraday configuration that *could* hold overnight is a
configuration that will one day hold overnight — and then pay delivery STT on a
position nobody planned to have.

---

## 6. The risk layer

The strategy proposes; `risk/engine.py` disposes. It never sees the model's
reasoning, so it cannot be talked round by a confident tone.

- position count cap, sector concentration cap, correlation cap
- per-trade notional cap, max position % of equity, minimum cash reserve
- minimum trade notional (below which costs eat the trade)
- ATR-based sizing from `RISK_PER_TRADE_PCT` — position size derived from
  volatility, not a fixed amount
- hard stop, ATR stop, ATR target, trailing stop
- quote staleness and spread checks
- **cost-to-edge ratio** — rejects a trade whose modelled expected move does not
  clear its own transaction costs
- daily loss limit, and a max-drawdown breaker that halts new entries

Two behaviours worth calling out:

**A missing quote blocks an entry but never an exit.** Being unable to price
something is a reason not to buy it; it is not a reason to keep holding it.
(This was one of the bugs found in the final audit — a `None` quote was skipping
both the freshness *and* the spread gate entirely.)

**Modelled spreads do not reject.** On the Yahoo feed there is no order book, so
the spread is our own estimate. Rejecting on it would only ever mean rejecting
our own guess, so the check is skipped when the quote is flagged `modelled`.

---

## 7. Execution safety

- **Market orders are never retried.** `http.py` retries reads freely;
  `brokers/alpaca.py` explicitly does not retry `POST /orders` or
  `DELETE /positions/{symbol}`. A retried market order is not a recovered
  request — it is a second position. The old bot retried everything. This is the
  most expensive bug the adapter can have, and there is a test named after it.
- **Deterministic client order ids** derived from run and symbol, so a
  timeout-then-rerun is rejected by the broker instead of doubling up.
- **Exits go through `DELETE /positions/{symbol}`**, not a notional sell —
  Alpaca rejects notional sells against fractional positions, which is how
  model-driven exits used to fail silently.
- **Fills are read back from the broker**, never assumed from the request.

---

## 8. Measurement — the actual point

**Every decision is journalled, including the skipped ones and the reason.**

A journal of *trades* tells you about the decisions that passed the gate. A
journal of *decisions* tells you whether the gate is any good. The second is the
only thing calibration can be computed from, and the old bot had neither.

`calibrate` then resolves each past decision against what the price actually did
N bars later, and buckets outcomes by the confidence the strategy claimed. Real
output from the offline smoke run:

```
| Confidence          | Decisions | Hit rate | Avg fwd return | Benchmark |    Edge |
| 0-4 (no conviction) |         3 |    +0.0% |        -0.431% |   -1.079% | +0.648% |
| 5-6 (below gate)    |       933 |    +0.0% |        +0.078% |   +0.235% | -0.157% |
| 7 (at gate)         |        31 |   +45.2% |        -0.071% |   +0.656% | -0.727% |
| 8                   |        90 |   +50.0% |        +0.070% |   -0.044% | +0.114% |

Rank correlation: -0.01 | Buckets ordered correctly: no

Verdict: No relationship between confidence and outcome. The gate is
         filtering noise, not selecting edge.
```

That is the system reporting that its own gate is decoration — on random-walk
data, where it should, because there is no edge to find. That verdict line is
the feature.

**A free control group.** `--strategy momentum` runs the identical cycle, gate,
broker and reporting with a deterministic rule and zero API cost. Without it,
"was the model worth it?" is unanswerable. Reports compare against it *and*
against buy-and-hold.

**Reports that flag their own weakness.** Annualised figures drawn from short
samples get a `*` and a footnote, because six weeks extrapolated to a year is a
number that misleads silently. Every report carries market-appropriate caveats
and the words "not financial advice".

---

## 9. The CLI

```bash
python -m claude_trader --market in doctor           # config, keys, tz data, feed reachability
python -m claude_trader --market in trade --dry-run  # one cycle, decide + journal, no orders
python -m claude_trader --market in backtest --synthetic --days 60 --strategy momentum
python -m claude_trader --market in calibrate --horizon 8
python -m claude_trader --market in report --run 3
```

The backtest line needs **no API keys and no network at all** — offline
random-walk data, deterministic by seed. A change can be evaluated before it
ever touches a feed.

---

## 10. Tests

**879 tests, 98% coverage, entirely offline.**

Every broker, feed and model call goes through a fake. CI runs on Python 3.11
and 3.12 with **no secrets in scope** — a test that needs a real key is a test
that will one day place a real order.

The suite is organised by *the thing being defended*, not by module. Test names
read as claims:

- `test_a_market_order_is_never_retried`
- `test_a_missing_quote_stops_the_entry_but_not_the_cycle`
- `test_a_quantity_is_rounded_down_before_it_is_sent`
- `test_the_fill_is_read_back_from_the_broker_not_assumed`
- `test_a_short_leg_is_ignored_by_a_long_only_bot`

Per-module: `config.py`, `models.py`, `journal/store.py`, `alpaca.py`,
`yahoo.py`, `costs.py`, `llm/client.py` and `executor.py` at 100%; `cycle.py`
96%, `risk/engine.py` 96%, `cli.py` 90%.

---

## 11. Operations

- `.github/workflows/trader.yml` — one cycle per schedule tick. The NSE cron is
  active by default, with the US block commented directly beneath it. Windows
  are padded because GitHub's scheduler fires late routinely; the bot checks the
  calendar itself and no-ops outside the session.
- `.github/workflows/tests.yml` — the suite on 3.11 and 3.12,
  `--cov-fail-under=90`, plus an offline backtest smoke test.
- The journal is cached **and** archived as an artifact between runs, because
  **the journal is the account**: positions, cash, drawdown-breaker state and
  the model response cache all live in it. Losing the file resets the balance
  and silently clears a halt.
- `.gitignore` excludes `data/` — the account is never committed.
- The Alpaca base URL points at the **paper** endpoint, and `app.py` warns
  loudly if it does not.

---

## 12. The final audit

The full bug-check protocol was run: collect everything first, then fix in one
pass.

**1 critical**
- `pydantic` was missing from `requirements.txt` despite `llm/schemas.py`
  importing it. Every GitHub Actions run would have died with
  `ModuleNotFoundError`.

**2 high**
- `close_position` constructed an `OrderRequest` with neither qty nor notional,
  raising `ValueError` *inside the exit path*.
- `README.md` still documented the deleted single-file bot.

**5 medium**
- A `None` quote skipped both the freshness and spread gates.
- No `.gitignore` — the journal would have been committed.
- No pytest configuration and no CI test job.
- Stale `CLAUDE.md`.
- Empty `docs/`.

**2 low**
- An over-long line in `alpaca.py`.
- `strategies/base.py` at 0% coverage.

All twelve fixed. Suite green at 879 passed, 98% coverage, 0 blockers.

---

## 13. What is not claimed

- **Backtests are optimistic.** Fills are modelled at the open of the bar
  following the decision, plus slippage. Real fills are worse, especially in
  smaller NSE names.
- **Yahoo intraday data is delayed and occasionally wrong.** Good enough for
  research, not good enough for anything else.
- **An LLM is not a forecaster.** Given twenty bars of OHLC it will produce
  fluent reasoning for any direction you like. Calibration exists precisely
  because that fluency is not evidence.
- **A short run proves nothing**, which is why reports mark it.
- **This is not financial advice.** Nothing the program outputs is a
  recommendation, and no claim is made that any of it is profitable.

What it can now do that it could not before: report honestly whether the model
beats a deterministic rule, after real costs, with a record that can be audited.
