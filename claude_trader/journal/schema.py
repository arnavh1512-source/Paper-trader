"""Journal schema.

One file, one source of truth for what the bot saw, decided, and did. Without
this table set there is no way to answer "is this better than buying SPY", so
it is written on every cycle including cycles where nothing is traded.
"""

SCHEMA_VERSION = 2

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,              -- live | backtest
    strategy    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    notes       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS cycles (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ts             TEXT NOT NULL,
    market_open    INTEGER NOT NULL,
    equity         REAL NOT NULL,
    cash           REAL NOT NULL,
    position_count INTEGER NOT NULL,
    strategy_note  TEXT NOT NULL DEFAULT '',
    market_mood    TEXT NOT NULL DEFAULT 'neutral',
    picks_json     TEXT NOT NULL DEFAULT '[]',
    halted         INTEGER NOT NULL DEFAULT 0,
    halt_reason    TEXT NOT NULL DEFAULT '',
    abstained      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cycles_run ON cycles(run_id, ts);

-- Every decision is recorded, including holds and rejections. Holds are the
-- control group; without them confidence calibration is meaningless.
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    cycle_id        INTEGER NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
    ts              TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    action          TEXT NOT NULL,
    confidence      INTEGER NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    dollars         REAL,
    source          TEXT NOT NULL DEFAULT 'model',
    price           REAL NOT NULL DEFAULT 0,
    indicators_json TEXT NOT NULL DEFAULT '{}',
    prompt_sha      TEXT NOT NULL DEFAULT '',
    risk_approved   INTEGER NOT NULL DEFAULT 0,
    risk_reason     TEXT NOT NULL DEFAULT '',
    executed        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol, ts);

CREATE TABLE IF NOT EXISTS orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    cycle_id     INTEGER NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
    decision_id  INTEGER REFERENCES decisions(id) ON DELETE SET NULL,
    ts           TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    qty          REAL NOT NULL,
    price        REAL NOT NULL,
    notional     REAL NOT NULL,
    broker_id    TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT '',
    simulated    INTEGER NOT NULL DEFAULT 0,
    intent       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_orders_run ON orders(run_id, ts);

-- Bot-managed stops. Alpaca cannot bracket a notional fractional fill, so the
-- exit levels live here and are enforced at the top of each cycle.
CREATE TABLE IF NOT EXISTS position_risk (
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    symbol       TEXT NOT NULL,
    entry_price  REAL NOT NULL,
    entry_time   TEXT NOT NULL,
    stop_price   REAL NOT NULL,
    target_price REAL NOT NULL,
    high_water   REAL NOT NULL,
    atr_at_entry REAL NOT NULL DEFAULT 0,
    bars_held    INTEGER NOT NULL DEFAULT 0,
    is_open      INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (run_id, symbol)
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ts              TEXT NOT NULL,
    equity          REAL NOT NULL,
    cash            REAL NOT NULL,
    exposure        REAL NOT NULL DEFAULT 0,
    benchmark_price REAL,
    UNIQUE (run_id, ts)
);

-- Backfilled after the fact by the calibration job: what actually happened
-- after each decision, at several horizons.
CREATE TABLE IF NOT EXISTS outcomes (
    decision_id    INTEGER NOT NULL REFERENCES decisions(id) ON DELETE CASCADE,
    horizon_bars   INTEGER NOT NULL,
    entry_price    REAL NOT NULL,
    exit_price     REAL NOT NULL,
    forward_return REAL NOT NULL,
    benchmark_return REAL,
    resolved_at    TEXT NOT NULL,
    PRIMARY KEY (decision_id, horizon_bars)
);

-- Paper book for markets with no broker sandbox to lean on (NSE). Alpaca keeps
-- its own paper state server-side; here the journal *is* the account, which is
-- why the file has to survive between GitHub Actions runs.
CREATE TABLE IF NOT EXISTS paper_account (
    account          TEXT PRIMARY KEY,
    currency         TEXT NOT NULL DEFAULT 'INR',
    cash             REAL NOT NULL,
    starting_cash    REAL NOT NULL,
    day              TEXT NOT NULL DEFAULT '',
    day_start_equity REAL NOT NULL DEFAULT 0,
    charges_paid     REAL NOT NULL DEFAULT 0,
    realized_pnl     REAL NOT NULL DEFAULT 0,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_holdings (
    account   TEXT NOT NULL REFERENCES paper_account(account) ON DELETE CASCADE,
    symbol    TEXT NOT NULL,
    qty       REAL NOT NULL,
    avg_price REAL NOT NULL,
    opened_at TEXT NOT NULL,
    PRIMARY KEY (account, symbol)
);

-- Caches model responses so a backtest can be re-run without paying twice.
CREATE TABLE IF NOT EXISTS llm_cache (
    key        TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    response   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
);
"""
