"""Transaction cost models.

Costs are not a footnote in India. A Rs 10,000 delivery round trip pays roughly
Rs 40 -- about 0.40% -- most of it the flat Rs 15.34 DP charge on the sell side,
which does not shrink with ticket size. The same trade held intraday pays about
Rs 10.6, or 0.11%. On 15-minute bars with small tickets that difference decides
whether a strategy can clear its own friction, so it is modelled explicitly
rather than assumed away.

Rates below follow the SEBI/exchange schedule for NSE cash equities. They change
by circular; every number lives in a named constant so an update is a one-line
edit rather than an archaeology exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import Side

# --- NSE cash equity statutory rates --------------------------------------
STT_DELIVERY = 0.001           # 0.1% on both buy and sell
STT_INTRADAY_SELL = 0.00025    # 0.025% on the sell leg only
NSE_TXN_CHARGE = 0.0000297     # 0.00297% of turnover, both legs
SEBI_TURNOVER_FEE = 0.000001   # Rs 10 per crore
NSE_IPFT = 0.000001            # Rs 10 per crore
STAMP_DUTY_DELIVERY = 0.00015  # 0.015% on the buy leg only
STAMP_DUTY_INTRADAY = 0.00003  # 0.003% on the buy leg only
GST_RATE = 0.18                # on brokerage + exchange + SEBI fees
DP_CHARGE = 15.34              # flat, per scrip, on the delivery sell day

# --- US rates (sell side only, regulatory) --------------------------------
SEC_FEE = 0.0000278            # per dollar of sale proceeds
FINRA_TAF_PER_SHARE = 0.000166
FINRA_TAF_CAP = 8.30


@dataclass(frozen=True, slots=True)
class Charges:
    """Itemised so a report can show *where* the money went, not just how much."""

    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0
    depository: float = 0.0
    regulatory: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.brokerage
            + self.stt
            + self.exchange
            + self.sebi
            + self.stamp_duty
            + self.gst
            + self.depository
            + self.regulatory,
            2,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "brokerage": round(self.brokerage, 2),
            "stt": round(self.stt, 2),
            "exchange": round(self.exchange, 2),
            "sebi": round(self.sebi, 2),
            "stamp_duty": round(self.stamp_duty, 2),
            "gst": round(self.gst, 2),
            "depository": round(self.depository, 2),
            "regulatory": round(self.regulatory, 2),
            "total": self.total,
        }


@runtime_checkable
class CostModel(Protocol):
    def charges(self, side: Side, qty: float, price: float) -> Charges: ...

    def total(self, side: Side, qty: float, price: float) -> float: ...

    @property
    def name(self) -> str: ...


@dataclass(frozen=True, slots=True)
class IndianEquityCosts:
    """NSE cash segment.

    ``segment='intraday'`` assumes every position is squared off the same day.
    If the engine were allowed to hold overnight while this says intraday, the
    backtest would under-report costs, so AppConfig validates the square-off
    rule and this setting together.
    """

    segment: str = "intraday"
    brokerage_pct: float = 0.0003   # 0.03%, the common discount-broker rate
    brokerage_cap: float = 20.0     # per executed order
    dp_charge: float = DP_CHARGE

    def __post_init__(self) -> None:
        if self.segment not in {"intraday", "delivery"}:
            raise ValueError("segment must be 'intraday' or 'delivery'")

    @property
    def name(self) -> str:
        return f"NSE cash ({self.segment})"

    @property
    def is_intraday(self) -> bool:
        return self.segment == "intraday"

    def charges(self, side: Side, qty: float, price: float) -> Charges:
        turnover = max(0.0, qty * price)
        if turnover <= 0:
            return Charges()

        buying = side is Side.BUY

        if self.is_intraday:
            brokerage = min(turnover * self.brokerage_pct, self.brokerage_cap)
            stt = 0.0 if buying else turnover * STT_INTRADAY_SELL
            stamp = turnover * STAMP_DUTY_INTRADAY if buying else 0.0
            depository = 0.0
        else:
            # Delivery brokerage is zero at the discount brokers this targets.
            brokerage = 0.0
            stt = turnover * STT_DELIVERY
            stamp = turnover * STAMP_DUTY_DELIVERY if buying else 0.0
            # DP is charged per scrip on the sell leg; shown GST-inclusive.
            depository = 0.0 if buying else self.dp_charge * (1 + GST_RATE)

        exchange = turnover * NSE_TXN_CHARGE
        sebi = turnover * (SEBI_TURNOVER_FEE + NSE_IPFT)
        gst = (brokerage + exchange + sebi) * GST_RATE

        return Charges(
            brokerage=brokerage,
            stt=stt,
            exchange=exchange,
            sebi=sebi,
            stamp_duty=stamp,
            gst=gst,
            depository=depository,
        )

    def total(self, side: Side, qty: float, price: float) -> float:
        return self.charges(side, qty, price).total


@dataclass(frozen=True, slots=True)
class USEquityCosts:
    """Commission-free brokerage. The sell-side regulatory fees are real and
    small, and are included so 'zero commission' is not read as 'zero cost'."""

    commission_per_order: float = 0.0

    @property
    def name(self) -> str:
        return "US equities (commission-free)"

    def charges(self, side: Side, qty: float, price: float) -> Charges:
        turnover = max(0.0, qty * price)
        if turnover <= 0:
            return Charges()
        if side is Side.BUY:
            return Charges(brokerage=self.commission_per_order)
        regulatory = turnover * SEC_FEE + min(qty * FINRA_TAF_PER_SHARE, FINRA_TAF_CAP)
        return Charges(brokerage=self.commission_per_order, regulatory=regulatory)

    def total(self, side: Side, qty: float, price: float) -> float:
        return self.charges(side, qty, price).total


@dataclass(frozen=True, slots=True)
class NoCosts:
    """Only for isolating strategy signal from friction inside a test."""

    @property
    def name(self) -> str:
        return "none"

    def charges(self, side: Side, qty: float, price: float) -> Charges:
        return Charges()

    def total(self, side: Side, qty: float, price: float) -> float:
        return 0.0


def round_trip_cost_pct(model: CostModel, notional: float, price: float) -> float:
    """Cost of a full in-and-out as a fraction of notional.

    This is the number a strategy has to beat before it makes a rupee, which is
    why the report prints it next to the win rate.
    """
    if notional <= 0 or price <= 0:
        return 0.0
    qty = notional / price
    both = model.total(Side.BUY, qty, price) + model.total(Side.SELL, qty, price)
    return both / notional


def build_cost_model(market_key: str, segment: str = "intraday") -> CostModel:
    return IndianEquityCosts(segment=segment) if market_key == "in" else USEquityCosts()
