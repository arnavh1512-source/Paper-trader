# Architecture

This document explains *why* the code is shaped the way it is. The shape is
almost entirely a reaction to the failure modes of the thing it replaced: a
single 300-line script that asked Claude what to buy, bought it, kept no record,
and had no way of ever finding out whether it was any good.

---

## The one invariant

> **The live cycle and the backtest run the same function.**

`claude_trader.engine.cycle.run_cycle(deps, now)` is called by the live path and
by the backtester. The only things that differ are the objects injected into
`deps`: which broker, which market data source, which cost model.

Everything else in the design follows from wanting that sentence to stay true.
If a backtest can take a path the live cycle cannot, then a backtest result is a
statement about the backtester, not about the strategy.

```
                       +----------------------+
   app.run_live_cycle  |                      |  backtest.engine
   AlpacaMarketData    |      run_cycle       |  HistoricalMarketData
   AlpacaBroker /      |  (survey, decide,    |  SimulatedBroker
   PaperBroker         |   gate, execute,     |
                       |   journal)           |
                       +----------------------+
```

---

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | One frozen `AppConfig` built from env + flags. Validates, and raises `ConfigError` rather than starting with a nonsensical setup. |
| `markets/` | `MarketProfile` per market: session, timezone, tick size, lot rules, benchmark, currency, cost model. The only place a market-specific fact is allowed to live. |
| `models.py` | Frozen domain types: `Bar`, `Quote`, `Snapshot`, `Position`, `Account`, `OrderRequest`, `OrderResult`, `Decision`. |
| `data/` | `MarketDataSource` implementations (Alpaca, Yahoo), RSS headlines, and indicators. |
| `strategies/` | `Strategy` protocol; `MomentumStrategy` (the control) and `ClaudeStrategy` (the subject). |
| `llm/` | Anthropic client with retries, response caching, and **schema validation** of every reply. |
| `risk/` | The gate. Sizing, caps, breakers, staleness, spread, cost-to-edge. |
| `engine/` | `run_cycle` and the executor that turns an approved decision into an order. |
| `brokers/` | `Broker` protocol; Alpaca (live paper), `PaperBroker` (journal-backed book), `SimulatedBroker` (backtest). |
| `backtest/` | Dataset fetch/cache and the replay engine. |
| `analytics/` | Metrics, calibration, markdown reports, and the HTML dashboard. |
| `journal/` | SQLite schema and store. The system of record. |

---

## Design decisions, and what each one is defending against

### Protocol-based dependency injection

`MarketDataSource`, `Broker`, `Strategy`, `CostModel` and `PriceSource` are all
`@runtime_checkable` `Protocol`s. There is no inheritance hierarchy and no
framework.

*Defends against:* a test that needs a network connection. Every test in the
suite runs offline against a fake, which is why the whole suite finishes in
seconds and why no test can ever accidentally place an order.

### Frozen dataclasses everywhere

Domain types are `@dataclass(frozen=True, slots=True)`.

*Defends against:* a risk gate that mutates the decision it is evaluating, or an
executor that adjusts a quantity in place and leaves the journal describing an
order that was never sent. Changes go through `dataclasses.replace`, which
produces a new object with an obvious provenance.

### Anti-lookahead by construction, not by discipline

Two mechanisms, both structural:

- `HistoricalMarketData.bars(symbol, limit, as_of)` slices strictly *before*
  `as_of`. A strategy physically cannot see the bar it is deciding on.
- `SimulatedBroker` fills at the **open of the next bar**, never at the close of
  the decision bar.

*Defends against:* the single most common way a backtest lies. This is not a
convention someone has to remember; it is the only data the objects can return.

### Market orders are never retried

`http.py` retries reads. `brokers/alpaca.py` explicitly does not retry
`POST /orders` or `DELETE /positions/{symbol}`.

*Defends against:* the most expensive bug this adapter can have. A retried
market order is not a recovered request — it is a second position. The original
bot retried everything.

### Deterministic client order ids

Every order carries a `client_order_id` derived from the run and the symbol.

*Defends against:* a duplicate submission surviving a timeout-then-manual-rerun.
The broker rejects the second one.

### The journal is written even when nothing trades

Skipped decisions, the reason for each skip, abstentions, and the raw model
response are all persisted.

*Defends against:* survivorship bias in your own record. A journal of trades
tells you about the decisions that passed the gate. A journal of *decisions*
tells you whether the gate is any good — which is the only thing calibration can
be computed from.

### Calibration as a first-class command

`calibrate` resolves each past decision against the price N bars later and
buckets outcomes by claimed confidence.

*Defends against:* a confidence threshold that is decoration. The predecessor
gated on `confidence >= 7` and had no mechanism, even in principle, for learning
whether 7 differed from 4.

### A free control group

`--strategy momentum` runs the same cycle, the same risk gate, the same broker
and the same reporting, with a deterministic rule instead of a model, and costs
nothing to run.

*Defends against:* the question "was the model worth it?" being unanswerable.
Reports compare against it and against buy-and-hold.

### Explicit cost models

`costs.py` prices NSE brokerage, STT (with the lower intraday rate), stamp duty,
exchange and SEBI turnover fees, and GST — and the US spread/slippage model
separately.

*Defends against:* a strategy that is profitable before costs and negative
after. On small intraday tickets this is the difference, not a detail.

### Risk gate the strategy cannot argue with

The strategy proposes; `risk/engine.py` disposes. It has no access to the
model's reasoning and cannot be persuaded by a confident tone.

Notably: **a missing quote blocks an entry but never blocks an exit.** Being
unable to price something is a reason not to buy it and not a reason to keep
holding it.

### Reports that flag their own weakness

Annualised figures computed from short samples are marked `*` with a footnote.
Every report carries market-appropriate caveats and the words "not financial
advice".

*Defends against:* a six-week run being read as an annual return.

---

## Data flow of one cycle

```
AppConfig.from_env()
   |- MarketProfile          session, tick, costs, benchmark, currency
   |- build_market_data      Alpaca | Yahoo
   |- build_broker           Alpaca | PaperBroker (journal-backed)
   |- build_strategy         Claude | Momentum
        |
        \- run_cycle(deps, now)
             1. broker.is_market_open(now)     -> closed? record and stop
             2. market.snapshots(universe)     -> one batched call
             3. strategy.pick(...)             -> candidates, or abstain
             4. for each candidate + holding:
                  market.bars / market.quote
                  strategy.decide(...)         -> Decision(action, confidence)
                  risk.evaluate(...)           -> RiskVerdict(ok, reason)
                  executor.execute(...)        -> OrderRequest -> OrderResult
             5. journal.record(...)            -> decisions, orders, skips, run
             6. CycleReport                    -> app.summarise -> log line
```

Steps 4 and 5 happen for every symbol regardless of whether an order results.
That is the whole point.

---

## The journal

The journal is the system of record and, for the paper broker, the account
itself: positions, cash, the drawdown-breaker state and the model response cache
all live in it. Losing the file resets the balance and silently clears a halt,
which is why the scheduled workflow both caches and archives it.

Runs carry their configuration as JSON, risk limits included. Two runs with
different limits are different experiments, and storing the limits on the run
row is what makes them distinguishable months later.

---

## Testing

969 tests, ~98% statement coverage, all offline.

The suite is organised by the thing being defended rather than by module — test
names read as claims about behaviour ("a market order is never retried",
"a missing quote stops the entry but not the cycle"). A test whose name does not
state a consequence is usually a test that will not be missed when it breaks.

Fixtures live in `tests/conftest.py`: `journal` (in-memory SQLite), an autouse
`clean_env` that strips every trading environment variable so a developer's own
keys cannot leak into a run, and builders `make_bars`, `ramp`, `make_quote`,
`make_snapshot`, `make_state`.

CI runs the suite on Python 3.11 and 3.12 with **no secrets in scope**. A test
that needs a real key is a test that will one day place a real order.
