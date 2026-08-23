"""Journal-backed paper broker.

Alpaca hands you a paper account with server-side state; NSE has no equivalent
that does not require a KYC'd login and, at some brokers, a monthly fee. So for
Indian trading the journal *is* the account: cash and holdings live in SQLite,
and every fill is priced off live market data with the real statutory charges
deducted.

Two consequences worth being blunt about:

* **The journal file must persist between runs.** On GitHub Actions that means
  committing it back or restoring it from cache. Lose the file and the book
  resets to opening cash, which would silently erase a drawdown.
* **Fills are optimistic by exactly one modelled spread.** There is no queue, no
  partial fill and no impact beyond the configured slippage. Treat the results
  as an upper bound on what the same decisions would have earned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol

from ..costs import Charges, CostModel
from ..errors import BrokerError
from ..journal.store import Journal
from ..markets import MarketProfile
from ..models import Account, OrderRequest, OrderResult, Position, Side

log = logging.getLogger(__name__)


class PriceSource(Protocol):
    """Just the slice of ``MarketDataSource`` a broker needs."""

    def latest_prices(
        self, symbols: object, as_of: datetime
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class SlippageModel:
    """Market orders do not fill at the mid. On NIFTY large caps 5bps is a fair
    working assumption; on anything thinner it is charity."""

    bps: float = 5.0

    def apply(self, side: Side, reference: float) -> float:
        drift = reference * (self.bps / 10_000.0)
        return reference + drift if side is Side.BUY else reference - drift


class PaperBroker:
    """A broker whose entire state is rows in the journal."""

    def __init__(
        self,
        journal: Journal,
        market: PriceSource,
        profile: MarketProfile,
        costs: CostModel,
        account: str = "default",
        starting_cash: float | None = None,
        slippage: SlippageModel | None = None,
        clock: datetime | None = None,
    ) -> None:
        self._journal = journal
        self._market = market
        self._profile = profile
        self._costs = costs
        self._account = f"{account}:{profile.key}"
        self._slippage = slippage or SlippageModel()
        self._now = clock
        self._order_seq = 0
        self._ensure_account(
            starting_cash if starting_cash is not None else profile.starting_cash
        )

    # ------------------------------------------------------------- lifecycle
    def _ensure_account(self, starting_cash: float) -> None:
        rows = self._journal.query(
            "SELECT starting_cash FROM paper_account WHERE account = ?",
            (self._account,),
        )
        if rows:
            # The account already exists, and it is not rewritten here. Opening
            # balance is the denominator of every return figure in the journal;
            # moving it under months of recorded trades would silently restate
            # all of them.
            #
            # But a config that disagrees with the book is worth saying out
            # loud, because the symptom otherwise is nothing at all: the
            # operator raises STARTING_CASH, the bot keeps trading the old
            # balance, and the only evidence is a number on a dashboard that
            # never changes.
            stored = float(rows[0]["starting_cash"])
            if abs(stored - float(starting_cash)) > 0.005:
                log.warning(
                    "STARTING_CASH is %s but paper account %s was opened with "
                    "%s and keeps it. Delete the journal to start a new book "
                    "at the configured balance -- that discards its history.",
                    self._profile.money(starting_cash),
                    self._account,
                    self._profile.money(stored),
                )
            return
        self._journal.query(
            """INSERT INTO paper_account
               (account, currency, cash, starting_cash, day, day_start_equity,
                charges_paid, realized_pnl, updated_at)
               VALUES (?, ?, ?, ?, '', ?, 0, 0, ?)""",
            (
                self._account,
                self._profile.currency,
                float(starting_cash),
                float(starting_cash),
                float(starting_cash),
                self.now.isoformat(),
            ),
        )
        self._journal.commit()
        log.info(
            "Opened paper account %s with %s",
            self._account,
            self._profile.money(starting_cash),
        )

    def set_clock(self, now: datetime) -> None:
        self._now = now

    @property
    def now(self) -> datetime:
        if self._now is None:
            raise BrokerError("PaperBroker.set_clock must be called before trading")
        return self._now

    # ----------------------------------------------------------------- state
    def _row(self) -> Mapping[str, object]:
        rows = self._journal.query(
            "SELECT * FROM paper_account WHERE account = ?", (self._account,)
        )
        if not rows:
            raise BrokerError(f"paper account {self._account} vanished mid-run")
        return rows[0]

    def _holdings(self) -> dict[str, tuple[float, float]]:
        return {
            str(r["symbol"]): (float(r["qty"]), float(r["avg_price"]))
            for r in self._journal.query(
                "SELECT symbol, qty, avg_price FROM paper_holdings "
                "WHERE account = ? AND qty > 0 ORDER BY symbol",
                (self._account,),
            )
        }

    def _prices(self, symbols: tuple[str, ...]) -> Mapping[str, float]:
        if not symbols:
            return {}
        try:
            return self._market.latest_prices(symbols, self.now)
        except Exception as exc:  # data outage must not corrupt the book
            log.warning("mark-to-market unavailable (%s); using cost basis", exc)
            return {}

    def account(self) -> Account:
        row = self._row()
        holdings = self._holdings()
        prices = self._prices(tuple(holdings))
        cash = float(row["cash"])
        held = sum(
            qty * (prices.get(sym) or avg) for sym, (qty, avg) in holdings.items()
        )
        equity = cash + held

        today = self._profile.local(self.now).date().isoformat()
        if str(row["day"]) != today:
            # The day's starting equity anchors the daily loss circuit breaker,
            # so it is stamped once per session rather than recomputed.
            self._journal.query(
                "UPDATE paper_account SET day = ?, day_start_equity = ?, updated_at = ? "
                "WHERE account = ?",
                (today, equity, self.now.isoformat(), self._account),
            )
            self._journal.commit()
            day_start = equity
        else:
            day_start = float(row["day_start_equity"]) or equity

        return Account(
            equity=equity,
            cash=cash,
            buying_power=max(0.0, cash),
            last_equity=day_start,
        )

    def positions(self) -> tuple[Position, ...]:
        holdings = self._holdings()
        prices = self._prices(tuple(holdings))
        return tuple(
            Position(
                symbol=symbol,
                qty=qty,
                avg_entry_price=avg,
                current_price=prices.get(symbol) or avg,
            )
            for symbol, (qty, avg) in holdings.items()
        )

    def is_market_open(self, as_of: datetime) -> bool:
        """Delegates to the data source when it can answer, because a hardcoded
        holiday list goes stale every January."""
        probe = getattr(self._market, "is_trading_now", None)
        if callable(probe):
            return bool(probe(as_of))
        return self._profile.is_session_time(as_of)

    # ---------------------------------------------------------------- orders
    def _reference_price(self, symbol: str) -> float:
        prices = self._prices((symbol,))
        return float(prices.get(symbol) or 0.0)

    def submit(self, order: OrderRequest) -> OrderResult | None:
        reference = self._reference_price(order.symbol)
        if reference <= 0:
            log.warning("%s: no price, order dropped", order.symbol)
            return None

        price = self._profile.round_price(
            self._slippage.apply(order.side, reference)
        )
        if price <= 0:
            return None

        wanted = order.qty if order.qty is not None else (order.notional or 0.0) / price
        qty = self._profile.round_qty(wanted)
        if qty <= 0:
            log.info(
                "%s: %s rounds to zero shares at %s",
                order.symbol,
                self._profile.money(wanted * price),
                self._profile.money(price),
            )
            return None

        holdings = self._holdings()
        held_qty, held_avg = holdings.get(order.symbol, (0.0, 0.0))

        if order.side is Side.BUY:
            cash = float(self._row()["cash"])
            charges = self._costs.charges(Side.BUY, qty, price)
            while qty > 0 and qty * price + charges.total > cash + 1e-9:
                qty = self._profile.round_qty(qty - self._profile.lot_size)
                charges = self._costs.charges(Side.BUY, qty, price)
            if qty <= 0:
                log.info("%s: not enough cash for one lot", order.symbol)
                return None
            new_qty = held_qty + qty
            new_avg = ((held_qty * held_avg) + (qty * price)) / new_qty
            self._write_fill(
                symbol=order.symbol,
                qty=new_qty,
                avg=new_avg,
                cash_delta=-(qty * price + charges.total),
                charges=charges,
                realized=0.0,
            )
        else:
            if held_qty <= 0:
                log.warning("%s: nothing held to sell", order.symbol)
                return None
            qty = min(qty, held_qty)
            charges = self._costs.charges(Side.SELL, qty, price)
            realized = qty * (price - held_avg) - charges.total
            self._write_fill(
                symbol=order.symbol,
                qty=held_qty - qty,
                avg=held_avg,
                cash_delta=qty * price - charges.total,
                charges=charges,
                realized=realized,
            )

        self._order_seq += 1
        return OrderResult(
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            order_id=f"paper-{int(self.now.timestamp())}-{self._order_seq}",
            status="filled",
            submitted_at=self.now,
            simulated=True,
        )

    def _write_fill(
        self,
        symbol: str,
        qty: float,
        avg: float,
        cash_delta: float,
        charges: Charges,
        realized: float,
    ) -> None:
        """One transaction so a crash between the cash and holdings writes
        cannot leave the book claiming shares it never paid for."""
        with self._journal.transaction() as conn:
            if qty <= 1e-9:
                conn.execute(
                    "DELETE FROM paper_holdings WHERE account = ? AND symbol = ?",
                    (self._account, symbol),
                )
            else:
                conn.execute(
                    """INSERT INTO paper_holdings(account, symbol, qty, avg_price, opened_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(account, symbol) DO UPDATE SET
                           qty = excluded.qty, avg_price = excluded.avg_price""",
                    (self._account, symbol, qty, avg, self.now.isoformat()),
                )
            conn.execute(
                """UPDATE paper_account
                   SET cash = cash + ?, charges_paid = charges_paid + ?,
                       realized_pnl = realized_pnl + ?, updated_at = ?
                   WHERE account = ?""",
                (
                    cash_delta,
                    charges.total,
                    realized,
                    self.now.isoformat(),
                    self._account,
                ),
            )

    def close_position(self, symbol: str) -> OrderResult | None:
        held_qty, _ = self._holdings().get(symbol, (0.0, 0.0))
        if held_qty <= 0:
            return None
        return self.submit(
            OrderRequest(symbol=symbol, side=Side.SELL, qty=held_qty, intent="close")
        )

    # ------------------------------------------------------------- reporting
    def summary(self) -> dict[str, object]:
        row = self._row()
        account = self.account()
        return {
            "account": self._account,
            "currency": self._profile.currency,
            "equity": self._profile.money(account.equity),
            "cash": self._profile.money(account.cash),
            "starting cash": self._profile.money(float(row["starting_cash"])),
            "realized pnl": self._profile.money(float(row["realized_pnl"])),
            "charges paid": self._profile.money(float(row["charges_paid"])),
            "cost model": self._costs.name,
            "open positions": len(self._holdings()),
        }

    def reset(self, starting_cash: float | None = None) -> None:
        """Wipe the book. Separate from ``__init__`` so that starting over is
        always something someone asked for, never a side effect of a bad path."""
        cash = (
            starting_cash
            if starting_cash is not None
            else float(self._row()["starting_cash"])
        )
        with self._journal.transaction() as conn:
            conn.execute(
                "DELETE FROM paper_holdings WHERE account = ?", (self._account,)
            )
            conn.execute(
                """UPDATE paper_account
                   SET cash = ?, starting_cash = ?, day = '', day_start_equity = ?,
                       charges_paid = 0, realized_pnl = 0, updated_at = ?
                   WHERE account = ?""",
                (cash, cash, cash, self.now.isoformat(), self._account),
            )
        log.warning("Paper account %s reset to %s", self._account, self._profile.money(cash))
