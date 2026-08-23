"""SQLite-backed trade journal.

Everything the bot sees and does lands here. This is the only component whose
absence makes the rest meaningless, so it is written first in every cycle and
failures to write are loud.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..models import (
    Decision,
    OrderResult,
    Picks,
    PortfolioState,
    PositionRisk,
    RiskVerdict,
)
from .schema import ADDED_COLUMNS, SCHEMA_SQL, SCHEMA_VERSION


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


class Journal:
    """Append-only record of cycles, decisions, orders and equity."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._migrate()
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def _migrate(self) -> None:
        """Bring an older journal up to the current column set.

        A journal is months of decisions; recreating it to add a column would
        throw away the only record of what the bot actually did.
        """
        for table, column, decl in ADDED_COLUMNS:
            existing = {row["name"] for row in
                        self._conn.execute(f"PRAGMA table_info({table})")}
            if existing and column not in existing:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # ------------------------------------------------------------------ infra
    def commit(self) -> None:
        """The connection is in autocommit mode, so this only matters inside an
        explicit transaction. Callers say what they mean rather than relying on
        the isolation level staying as it is."""
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, params))

    # ------------------------------------------------------------------- runs
    def start_run(
        self,
        kind: str,
        strategy: str,
        started_at: datetime,
        config: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs(kind, strategy, started_at, config_json, notes)"
            " VALUES (?, ?, ?, ?, ?)",
            (kind, strategy, _iso(started_at), json.dumps(config or {}, default=str), notes),
        )
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, finished_at: datetime) -> None:
        self._conn.execute(
            "UPDATE runs SET finished_at = ? WHERE id = ?", (_iso(finished_at), run_id)
        )

    def latest_run(self, kind: str | None = None) -> sqlite3.Row | None:
        sql = "SELECT * FROM runs"
        params: tuple[Any, ...] = ()
        if kind:
            sql += " WHERE kind = ?"
            params = (kind,)
        sql += " ORDER BY id DESC LIMIT 1"
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def resolve_live_run(self, strategy: str, now: datetime, config: Mapping[str, Any]) -> int:
        """Live trading is a single continuous run across many GitHub Actions
        invocations, so reuse the open live run instead of starting a new one
        every 15 minutes."""
        rows = self.query(
            "SELECT id FROM runs WHERE kind = 'live' AND strategy = ?"
            " ORDER BY id DESC LIMIT 1",
            (strategy,),
        )
        if rows:
            return int(rows[0]["id"])
        return self.start_run("live", strategy, now, config)

    # ----------------------------------------------------------------- cycles
    def record_cycle(
        self,
        run_id: int,
        ts: datetime,
        state: PortfolioState,
        picks: Picks | None,
        market_open: bool,
        halted: bool = False,
        halt_reason: str = "",
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO cycles(run_id, ts, market_open, equity, cash, position_count,"
            " strategy_note, market_mood, picks_json, halted, halt_reason, abstained)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                _iso(ts),
                int(market_open),
                state.account.equity,
                state.account.cash,
                state.position_count,
                picks.strategy if picks else "",
                picks.market_mood if picks else "neutral",
                json.dumps(list(picks.symbols) if picks else []),
                int(halted),
                halt_reason,
                int(bool(picks.abstain)) if picks else 0,
            ),
        )
        return int(cur.lastrowid)

    # -------------------------------------------------------------- decisions
    def record_decision(
        self,
        run_id: int,
        cycle_id: int,
        ts: datetime,
        decision: Decision,
        price: float,
        indicators: Mapping[str, Any] | None = None,
        verdict: RiskVerdict | None = None,
        prompt_sha: str = "",
        news: Sequence[str] = (),
        executed: bool = False,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO decisions(run_id, cycle_id, ts, symbol, action, confidence,"
            " reason, dollars, source, price, indicators_json, prompt_sha,"
            " news_json,"
            " risk_approved, risk_reason, executed)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                cycle_id,
                _iso(ts),
                decision.symbol,
                decision.action.value,
                int(decision.confidence),
                decision.reason,
                decision.notional,
                decision.source,
                price,
                json.dumps(dict(indicators or {}), default=str),
                prompt_sha,
                json.dumps(list(news)),
                int(bool(verdict and verdict.approved)),
                verdict.reason if verdict else "",
                int(executed),
            ),
        )
        return int(cur.lastrowid)

    def mark_decision_executed(self, decision_id: int) -> None:
        self._conn.execute(
            "UPDATE decisions SET executed = 1 WHERE id = ?", (decision_id,)
        )

    # ----------------------------------------------------------------- orders
    def record_order(
        self,
        run_id: int,
        cycle_id: int,
        result: OrderResult,
        decision_id: int | None = None,
        intent: str = "",
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO orders(run_id, cycle_id, decision_id, ts, symbol, side, qty,"
            " price, notional, broker_id, status, simulated, intent)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                cycle_id,
                decision_id,
                _iso(result.submitted_at),
                result.symbol,
                result.side.value,
                result.qty,
                result.price,
                result.notional,
                result.order_id,
                result.status,
                int(result.simulated),
                intent,
            ),
        )
        return int(cur.lastrowid)

    def trades_today(self, run_id: int, day: datetime) -> int:
        prefix = _iso(day)[:10]
        rows = self.query(
            "SELECT COUNT(*) AS n FROM orders WHERE run_id = ? AND substr(ts, 1, 10) = ?",
            (run_id, prefix),
        )
        return int(rows[0]["n"]) if rows else 0

    # --------------------------------------------------------- position risks
    def upsert_position_risk(self, run_id: int, risk: PositionRisk) -> None:
        self._conn.execute(
            "INSERT INTO position_risk(run_id, symbol, entry_price, entry_time,"
            " stop_price, target_price, high_water, atr_at_entry, bars_held, is_open)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)"
            " ON CONFLICT(run_id, symbol) DO UPDATE SET"
            " entry_price=excluded.entry_price, entry_time=excluded.entry_time,"
            " stop_price=excluded.stop_price, target_price=excluded.target_price,"
            " high_water=excluded.high_water, atr_at_entry=excluded.atr_at_entry,"
            " bars_held=excluded.bars_held, is_open=1",
            (
                run_id,
                risk.symbol,
                risk.entry_price,
                _iso(risk.entry_time),
                risk.stop_price,
                risk.target_price,
                risk.high_water,
                risk.atr_at_entry,
                risk.bars_held,
            ),
        )

    def close_position_risk(self, run_id: int, symbol: str) -> None:
        self._conn.execute(
            "UPDATE position_risk SET is_open = 0 WHERE run_id = ? AND symbol = ?",
            (run_id, symbol),
        )

    def open_position_risks(self, run_id: int) -> tuple[PositionRisk, ...]:
        rows = self.query(
            "SELECT * FROM position_risk WHERE run_id = ? AND is_open = 1", (run_id,)
        )
        return tuple(
            PositionRisk(
                symbol=r["symbol"],
                entry_price=r["entry_price"],
                entry_time=_parse(r["entry_time"]),
                stop_price=r["stop_price"],
                target_price=r["target_price"],
                high_water=r["high_water"],
                atr_at_entry=r["atr_at_entry"],
                bars_held=r["bars_held"],
            )
            for r in rows
        )

    # ----------------------------------------------------------- equity curve
    def record_equity(
        self,
        run_id: int,
        ts: datetime,
        equity: float,
        cash: float,
        exposure: float = 0.0,
        benchmark_price: float | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO equity_curve(run_id, ts, equity, cash, exposure, benchmark_price)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(run_id, ts) DO UPDATE SET equity=excluded.equity,"
            " cash=excluded.cash, exposure=excluded.exposure,"
            " benchmark_price=excluded.benchmark_price",
            (run_id, _iso(ts), equity, cash, exposure, benchmark_price),
        )

    def equity_curve(self, run_id: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT ts, equity, cash, exposure, benchmark_price FROM equity_curve"
            " WHERE run_id = ? ORDER BY ts",
            (run_id,),
        )

    # --------------------------------------------------------------- outcomes
    def record_outcome(
        self,
        decision_id: int,
        horizon_bars: int,
        entry_price: float,
        exit_price: float,
        forward_return: float,
        benchmark_return: float | None,
        resolved_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO outcomes(decision_id, horizon_bars, entry_price,"
            " exit_price, forward_return, benchmark_return, resolved_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                decision_id,
                horizon_bars,
                entry_price,
                exit_price,
                forward_return,
                benchmark_return,
                _iso(resolved_at),
            ),
        )

    def unresolved_decisions(self, run_id: int, horizon_bars: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT d.* FROM decisions d"
            " LEFT JOIN outcomes o ON o.decision_id = d.id AND o.horizon_bars = ?"
            " WHERE d.run_id = ? AND o.decision_id IS NULL"
            " ORDER BY d.ts",
            (horizon_bars, run_id),
        )

    # -------------------------------------------------------------- llm cache
    def cache_get(self, key: str) -> str | None:
        rows = self.query("SELECT response FROM llm_cache WHERE key = ?", (key,))
        if not rows:
            return None
        self._conn.execute("UPDATE llm_cache SET hits = hits + 1 WHERE key = ?", (key,))
        return str(rows[0]["response"])

    def cache_put(self, key: str, model: str, response: str, now: datetime) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache(key, model, response, created_at, hits)"
            " VALUES (?, ?, ?, ?, COALESCE((SELECT hits FROM llm_cache WHERE key = ?), 0))",
            (key, model, response, _iso(now), key),
        )
