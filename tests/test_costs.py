"""Transaction costs.

The single biggest difference between trading NSE and trading US equities on a
small ticket. A US round trip costs about 0.003%; an NSE intraday round trip
costs about 0.106%, and a delivery round trip on a Rs 5,000 ticket costs 0.58%.
A strategy that ignores this is measuring a return that nobody could collect.
"""

from __future__ import annotations

import pytest

from claude_trader.costs import (
    GST_RATE,
    Charges,
    IndianEquityCosts,
    NoCosts,
    USEquityCosts,
    build_cost_model,
    round_trip_cost_pct,
)
from claude_trader.models import Side


def test_build_cost_model_dispatches_on_market():
    assert isinstance(build_cost_model("in"), IndianEquityCosts)
    assert build_cost_model("in", "delivery").segment == "delivery"
    assert isinstance(build_cost_model("us"), USEquityCosts)


def test_segment_must_be_a_real_segment():
    with pytest.raises(ValueError):
        IndianEquityCosts(segment="swing")


# ------------------------------------------------------------------- India
def test_intraday_buy_leg_pays_no_stt():
    """STT on intraday is a sell-side charge only. Billing it on both legs
    would roughly double the modelled cost of every trade."""
    charges = IndianEquityCosts("intraday").charges(Side.BUY, 10, 1_000.0)
    assert charges.stt == 0.0
    assert charges.brokerage == pytest.approx(3.0)      # 0.03% of 10,000
    assert charges.stamp_duty == pytest.approx(0.3)     # 0.003%, buy leg only
    assert charges.exchange == pytest.approx(0.297)
    assert charges.gst == pytest.approx(GST_RATE * (3.0 + 0.297 + 0.02))
    assert charges.depository == 0.0
    assert charges.total == pytest.approx(4.21, abs=0.01)


def test_intraday_sell_leg_pays_stt_and_no_stamp_duty():
    charges = IndianEquityCosts("intraday").charges(Side.SELL, 10, 1_000.0)
    assert charges.stt == pytest.approx(2.5)            # 0.025% of 10,000
    assert charges.stamp_duty == 0.0
    assert charges.total == pytest.approx(6.41, abs=0.01)


def test_brokerage_is_capped_per_order():
    """Rs 20 per order, so a large ticket is proportionally cheaper. Without
    the cap the model would over-charge every big trade."""
    model = IndianEquityCosts("intraday")
    small = model.charges(Side.BUY, 10, 1_000.0).brokerage
    large = model.charges(Side.BUY, 1_000, 1_000.0).brokerage
    assert small == pytest.approx(3.0)
    assert large == pytest.approx(20.0)


def test_delivery_costs_several_times_intraday():
    intraday = round_trip_cost_pct(IndianEquityCosts("intraday"), 20_000, 1_000)
    delivery = round_trip_cost_pct(IndianEquityCosts("delivery"), 20_000, 1_000)
    assert intraday == pytest.approx(0.00106, abs=0.0001)
    assert delivery > intraday * 2.5


def test_delivery_hurts_small_tickets_most():
    """The DP charge is flat per scrip, so it is a rounding error on a large
    ticket and a tax on a small one. This is the reason India defaults to
    intraday rather than delivery."""
    model = IndianEquityCosts("delivery")
    tiny = round_trip_cost_pct(model, 5_000, 1_000)
    big = round_trip_cost_pct(model, 50_000, 1_000)
    assert tiny == pytest.approx(0.0058, abs=0.0005)
    assert big < tiny / 2


def test_delivery_sell_leg_carries_the_dp_charge_inclusive_of_gst():
    charges = IndianEquityCosts("delivery").charges(Side.SELL, 10, 1_000.0)
    assert charges.depository == pytest.approx(15.34 * 1.18)
    assert IndianEquityCosts("delivery").charges(Side.BUY, 10, 1_000.0).depository == 0.0


def test_delivery_pays_stt_on_both_legs():
    model = IndianEquityCosts("delivery")
    assert model.charges(Side.BUY, 10, 1_000.0).stt == pytest.approx(10.0)
    assert model.charges(Side.SELL, 10, 1_000.0).stt == pytest.approx(10.0)


# ---------------------------------------------------------------------- US
def test_us_buys_are_free_and_sells_are_nearly_free():
    model = USEquityCosts()
    assert model.total(Side.BUY, 100, 100.0) == 0.0
    sell = model.charges(Side.SELL, 100, 100.0)
    assert sell.regulatory == pytest.approx(0.29, abs=0.01)


def test_us_round_trip_is_two_orders_of_magnitude_cheaper_than_nse():
    us = round_trip_cost_pct(USEquityCosts(), 10_000, 100)
    nse = round_trip_cost_pct(IndianEquityCosts("intraday"), 10_000, 1_000)
    assert us < 0.0001
    assert nse > us * 20


def test_finra_fee_is_capped():
    model = USEquityCosts()
    huge = model.charges(Side.SELL, 10_000_000, 1.0)
    assert huge.regulatory == pytest.approx(1.0 * 10_000_000 * 0.0000278 + 8.30)


# -------------------------------------------------------------------- edges
def test_zero_turnover_costs_nothing():
    for model in (IndianEquityCosts(), USEquityCosts(), NoCosts()):
        assert model.total(Side.BUY, 0, 1_000.0) == 0.0
        assert model.total(Side.BUY, 10, 0.0) == 0.0


def test_round_trip_pct_is_defined_at_zero():
    assert round_trip_cost_pct(IndianEquityCosts(), 0, 100) == 0.0
    assert round_trip_cost_pct(IndianEquityCosts(), 1_000, 0) == 0.0


def test_no_costs_is_free_and_says_so():
    assert NoCosts().name == "none"
    assert NoCosts().charges(Side.SELL, 10, 100.0).total == 0.0


def test_charges_total_sums_every_component():
    charges = Charges(
        brokerage=1, stt=2, exchange=3, sebi=4, stamp_duty=5, gst=6,
        depository=7, regulatory=8,
    )
    assert charges.total == 36
    assert charges.as_dict()["total"] == 36


def test_model_names_are_reportable():
    assert IndianEquityCosts("delivery").name == "NSE cash (delivery)"
    assert "commission-free" in USEquityCosts().name
