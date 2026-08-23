# claude-trader

A paper-trading research bot for **NSE (India)** and **US equities**, with the
same decision path running live and in backtest.

It is not a money machine and it is not advice. It is an instrument for asking
one question honestly: *does a language model add anything to a trading decision
that a simple momentum rule does not?* Every part of the design exists to make
that question answerable rather than to make the equity curve look good.

> **Paper trading only.** The default broker for the US is Alpaca's paper
> endpoint; NSE has no free sandbox, so it uses an internal paper book stored in
> the journal. Nothing here is financial advice, and no result from it predicts
> anything about real money.

---

## What it does in one cycle

```
survey the universe -> pick candidates -> decide per symbol -> risk gate -> execute -> journal
```

1. **Survey** — one batched snapshot call for the whole universe.
2. **Pick** — the strategy proposes a handful of symbols to look at closely.
   It may also *abstain*, which is a valid and frequently correct answer.
3. **Decide** — for each candidate (and every open position) the strategy sees
   bars, indicators, the current quote, its own position and unrealised P&L, and
   returns buy / sell / hold with a confidence score.
4. **Risk** — a gate that the strategy cannot talk its way past: position caps,
   sector caps, per-trade notional, cash reserve, quote staleness, spread,
   cost-to-edge ratio, daily loss limit and a drawdown breaker.
5. **Execute** — orders carry a deterministic client order id, and market orders
   are **never retried**. A retried market order is a second position.
6. **Journal** — every decision, order, fill, skip and reason is written to
   SQLite *whether or not it traded*. Decisions that were skipped are the most
   valuable rows in the database.

The backtester calls the exact same `run_cycle`. Only the broker and the data
source are swapped. There is no separate "backtest strategy" that can quietly
drift from the live one.

---

## Quick start

```bash
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env      # then edit it: your capital, your limits
```

`.env` is read at startup and never overrides a variable that is already set,
so a GitHub Actions secret always wins over a stale file in the checkout.

Run an offline backtest with no keys and no network at all:

```bash
python -m claude_trader --market in backtest --synthetic --days 60 --strategy momentum
```

Check your configuration and connectivity:

```bash
python -m claude_trader --market in doctor
```

Run one live paper cycle without sending anything:

```bash
python -m claude_trader --market in trade --dry-run
```

Then look at what it did:

```bash
python -m claude_trader --market in dashboard --open
```

---

## Commands

| Command | What it is for |
|---|---|
| `trade` | Run one live paper-trading cycle. This is what the scheduler calls. |
| `backtest` | Replay the decision path over history (`--synthetic` for offline). |
| `calibrate` | Resolve past decisions against what actually happened and score the confidence gate. |
| `report` | Render a markdown report for a journalled run, with a buy-and-hold benchmark. |
| `dashboard` | Write a single self-contained HTML page: positions, round trips, the order log, and every decision including the ones the risk gate blocked. |
| `doctor` | Check config, credentials, timezone data and feed reachability. |

Global flags: `--market {in,us}`, `--segment {intraday,delivery}`,
`--journal PATH`, `-v`.

### Calibration is the point

```bash
python -m claude_trader --market in calibrate --horizon 8
```

This buckets every decision by the confidence the strategy claimed and reports
what actually happened afterwards. If "confidence 9" is not measurably better
than "confidence 6", the confidence number is noise and the gate that reads it
is decoration. The original version of this bot had a hard-coded
`confidence >= 7` threshold and no way to know whether it meant anything.

---

## The two markets

| | India (`--market in`) | US (`--market us`) |
|---|---|---|
| Exchange | NSE | NYSE / NASDAQ |
| Session | 09:15–15:30 IST | 09:30–16:00 ET |
| Benchmark | `NIFTYBEES` | `SPY` |
| Data feed | Yahoo (no key needed) | Alpaca |
| Broker | internal paper book | Alpaca paper API |
| Shares | whole shares, ₹0.05 tick | fractional, $0.01 tick |
| Starting cash | ₹100,000 | $10,000 |
| Max per trade | ₹10,000 | $100 |
| Costs modelled | brokerage, STT, stamp duty, exchange + SEBI fees, GST | commission-free, spread + slippage |

`--segment intraday` (NSE only) squares off before the close and pays the lower
intraday STT rate. Choosing it forces square-off on and caps the maximum holding
period at the session — an intraday configuration that could hold overnight is a
configuration that will one day hold overnight.

---

## Configuration

Everything is environment variables; flags override them. The ones that matter
most:

| Variable | Default | Meaning |
|---|---|---|
| `MARKET` | `in` | `in` or `us` |
| `TRADE_SEGMENT` | `intraday` | NSE only: `intraday` or `delivery` |
| `STRATEGY` | `claude` | `claude` or `momentum` (the free control group) |
| `CLAUDE_MODEL` | `claude-sonnet-5` | Which Claude decides; `claude-opus-5` for deeper reasoning |
| `DRY_RUN` | `0` | Decide and journal, send no orders |
| `JOURNAL_PATH` | `data/journal.sqlite3` | The account lives here |
| `MAX_POSITIONS` | 5 | Concurrent holdings |
| `MAX_NOTIONAL_PER_TRADE` | market default | Hard cap per order |
| `MIN_CONFIDENCE` | 7 | The gate calibration exists to test |
| `STARTING_CASH` | market default | The paper account's opening balance |
| `NEWS_ENABLED` | `false` | Show recent headlines to the model on live cycles |
| `NEWS_MAX_HEADLINES` | 5 | Headlines per symbol |
| `NEWS_MAX_AGE_HOURS` | 24 | How stale a headline may be |
| `MAX_DRAWDOWN_PCT` | — | Breaker: halts new entries |
| `DAILY_LOSS_LIMIT_PCT` | — | Breaker: halts for the day |
| `RISK_PER_TRADE_PCT` | — | ATR-based position sizing |

Secrets — `ANTHROPIC_API_KEY`, and `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` for
the US path — come from the environment or GitHub Actions secrets. They are
never read from a file in the repo. The India path with `--strategy momentum`
needs **no keys at all**.

Run `python -m claude_trader doctor` to see which ones are missing and whether
that is fatal for your configuration.

---

## Seeing what it did

`dashboard` writes one HTML file with no scripts, no CDN, and no external
requests — open it from disk, or commit it, or mail it to yourself.

```bash
python -m claude_trader --market in dashboard --out data/dashboard.html --open
```

It shows open positions with their stops and targets, every closed round trip,
the raw order log, and — the section that matters — every decision the strategy
made, including the ones the risk layer refused, with the reason it gave. Holds
are counted rather than listed. No broker screen will show you the trades that
never happened; this is the only place they exist.

---

## News

Off by default. With `NEWS_ENABLED=true`, live cycles fetch recent headlines
from public RSS (Google News for individual companies, plus a couple of
market-wide feeds) and show them to the model as clearly-delimited untrusted
text.

Three things are deliberate:

- **Headlines are data, not instructions.** They arrive inside a `<headlines>`
  fence, and both system prompts state that nothing in a headline can change
  the model's instructions or output format. Nothing read from a feed ever
  reaches the risk layer — news can make the model *want* to trade, and it
  still has to get past a gate that never reads it.
- **Failure is silent and harmless.** A feed that is down, slow, or malformed
  produces no headlines and a logged warning. It never blocks a cycle, and it
  never blocks an exit.
- **Backtests never see news.** The feeds return today's headlines. Pricing a
  2024 bar against a 2026 headline is not a backtest, it is a machine for
  producing encouraging numbers.

The headlines a decision saw are stored with it, so the dashboard can show you
what the model was reading when it made the call.

---

## Running on GitHub Actions

`.github/workflows/trader.yml` runs one cycle per schedule tick. Set the three
secrets under **Settings → Secrets and variables → Actions**, and set `MARKET`
and `STRATEGY` as repository *variables*.

The journal is cached between runs and uploaded as an artifact, because **the
journal is the account**: positions, cash, the drawdown-breaker state and the
model response cache all live in it. Losing it resets the balance and silently
clears a halt.

The cron block for NSE is active by default; the US block is commented out
directly beneath it. GitHub's scheduler fires late routinely, so the windows are
padded — the bot checks the calendar itself and no-ops outside the session.

---

## Development

```bash
pytest
pytest --cov=claude_trader --cov-report=term-missing
```

875 tests, ~98% coverage. `.github/workflows/tests.yml` runs them on 3.11 and
3.12 with **no secrets in scope** — every test that touches a broker or the
model goes through a fake, by design.

Architecture, invariants and the reasoning behind each of them:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Honest limitations

- **Backtests are optimistic.** Fills are modelled at the next bar's open with a
  spread and slippage estimate. Real fills are worse, especially in the small
  NSE names.
- **Yahoo intraday data is delayed and occasionally wrong.** It is good enough
  for research and not good enough for anything else.
- **An LLM is not a forecaster.** Given twenty bars of OHLC it will produce
  fluent reasoning for any direction you like. Calibration is included precisely
  because that fluency is not evidence.
- **A short run proves nothing.** Annualised figures from a few weeks of samples
  are marked with `*` in reports for exactly this reason.
- **Not financial advice.** No output of this program is a recommendation.
