"""The deterministic risk layer.

This is the part of the bot that outranks the model. Every test here asserts
something the LLM is not allowed to talk its way past.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from claude_trader.config import RiskConfig
from claude_trader.costs import IndianEquityCosts, NoCosts, USEquityCosts
from claude_trader.markets import INDIA_MARKET, US_MARKET
from claude_trader.models import (
    Action,
    Decision,
    ExitReason,
    Position,
    PositionRisk,
)
from claude_trader.risk.engine import RiskEngine
from tests.conftest import IST, make_quote, make_snapshot, make_state

NOW = datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)   # 10:30 IST, mid-session


def engine(profile=INDIA_MARKET, costs=None, **risk_kwargs) -> RiskEngine:
    # The RiskConfig default cap is 100 -- a US-dollar figure. On NSE that buys
    # nothing, so every test here would fail on size rather than on the rule it
    # is actually exercising.
    risk_kwargs.setdefault("max_notional_per_trade", 10_000.0)
    return RiskEngine(
        RiskConfig(**risk_kwargs),
        profile=profile,
        costs=costs if costs is not None else IndianEquityCosts("intraday"),
    )


def buy(symbol="RELIANCE", confidence=8, notional=None) -> Decision:
    return Decision(symbol, Action.BUY, confidence, "test", notional=notional)


def held(symbol="RELIANCE", qty=10, price=1_000.0) -> Position:
    return Position(symbol, qty, price, price)


def risk_record(symbol="RELIANCE", entry=1_000.0, stop=980.0, target=1_030.0,
                atr=10.0, bars_held=0) -> PositionRisk:
    return PositionRisk(
        symbol=symbol,
        entry_price=entry,
        entry_time=NOW,
        stop_price=stop,
        target_price=target,
        high_water=entry,
        atr_at_entry=atr,
        bars_held=bars_held,
    )


# ============================================================ circuit breakers
def test_a_clean_book_may_open():
    state = make_state()
    assert engine().assess_portfolio(state, peak_equity=100_000, trades_today=0).may_open


def test_daily_loss_limit_halts_trading():
    state = make_state(cash=97_000.0, last_equity=100_000.0)
    verdict = engine().assess_portfolio(state, peak_equity=100_000, trades_today=0)
    assert verdict.halted and "daily loss limit" in verdict.reason


def test_drawdown_from_peak_halts_trading():
    state = make_state(cash=80_000.0)
    verdict = engine().assess_portfolio(state, peak_equity=100_000, trades_today=0)
    assert verdict.halted and "max drawdown" in verdict.reason


def test_drawdown_is_reported_in_the_local_currency():
    state = make_state(cash=80_000.0)
    verdict = engine().assess_portfolio(state, peak_equity=100_000, trades_today=0)
    assert "₹1,00,000.00" in verdict.reason


def test_trade_caps_halt_trading():
    state = make_state()
    assert engine().assess_portfolio(state, 100_000, trades_today=10).halted
    assert engine().assess_portfolio(
        state, 100_000, trades_today=0, trades_this_cycle=3
    ).halted


def test_a_full_book_stops_opening():
    positions = [held(s, 1, 1_000.0) for s in ("A", "B", "C", "D", "E")]
    state = make_state(cash=50_000.0, positions=positions)
    verdict = engine().assess_portfolio(state, 55_000, trades_today=0)
    assert verdict.halted and "max positions" in verdict.reason


# ================================================================== square-off
def test_square_off_fires_fifteen_minutes_before_the_bell():
    eng = engine(square_off_enabled=True)
    assert eng.square_off_due(datetime(2026, 3, 2, 15, 15, tzinfo=IST)) is True
    assert eng.square_off_due(datetime(2026, 3, 2, 15, 14, tzinfo=IST)) is False
    assert eng.square_off_due(datetime(2026, 3, 2, 10, 0, tzinfo=IST)) is False


def test_square_off_reads_utc_correctly():
    """09:45 UTC is 15:15 IST. A naive comparison against the wall clock would
    square off in the middle of the session."""
    eng = engine(square_off_enabled=True)
    assert eng.square_off_due(datetime(2026, 3, 2, 9, 45, tzinfo=timezone.utc)) is True
    assert eng.square_off_due(datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)) is False


def test_square_off_is_off_for_delivery_and_for_the_us():
    assert engine(square_off_enabled=False).square_off_due(
        datetime(2026, 3, 2, 15, 20, tzinfo=IST)
    ) is False
    assert engine(square_off_enabled=True).square_off_due(None) is False


def test_square_off_closes_every_position_at_once():
    eng = engine(square_off_enabled=True)
    state = make_state(
        cash=50_000.0, positions=[held("RELIANCE"), held("TCS", 5, 3_000.0)]
    )
    exits = eng.forced_exits(state, {}, now=datetime(2026, 3, 2, 15, 20, tzinfo=IST))
    assert {e.symbol for e in exits} == {"RELIANCE", "TCS"}
    assert all(e.reason is ExitReason.SQUARE_OFF for e in exits)
    assert "15m before close" in exits[0].detail


# ================================================================ forced exits
def test_stop_loss_fires_below_the_stop():
    eng = engine()
    state = make_state(positions=[held()], risks=[risk_record()])
    snaps = {"RELIANCE": make_snapshot("RELIANCE", price=975.0)}
    exits = eng.forced_exits(state, snaps, now=NOW)
    assert len(exits) == 1
    assert exits[0].reason is ExitReason.STOP_LOSS


def test_a_stop_above_the_entry_is_reported_as_a_trailing_stop():
    """The distinction matters for calibration: a trailing stop that fires is a
    trade that worked and gave some back, not a thesis that was wrong."""
    eng = engine()
    state = make_state(positions=[held()], risks=[risk_record(stop=1_020.0)])
    snaps = {"RELIANCE": make_snapshot("RELIANCE", price=1_015.0)}
    assert eng.forced_exits(state, snaps, now=NOW)[0].reason is ExitReason.TRAILING_STOP


def test_target_takes_profit():
    eng = engine()
    state = make_state(positions=[held()], risks=[risk_record()])
    snaps = {"RELIANCE": make_snapshot("RELIANCE", price=1_035.0)}
    assert eng.forced_exits(state, snaps, now=NOW)[0].reason is ExitReason.TAKE_PROFIT


def test_time_stop_closes_a_position_that_never_resolved():
    eng = engine(max_holding_bars=10)
    state = make_state(positions=[held()], risks=[risk_record(bars_held=10)])
    snaps = {"RELIANCE": make_snapshot("RELIANCE", price=1_000.0)}
    assert eng.forced_exits(state, snaps, now=NOW)[0].reason is ExitReason.TIME_STOP


def test_untracked_positions_get_the_blunt_percentage_backstop():
    """A position opened before this run has no stop record. Leaving it
    unguarded is how a bot discovers a 40% loser."""
    eng = engine(hard_stop_pct=0.08)
    state = make_state(positions=[Position("TCS", 10, 1_000.0, 900.0)])
    exits = eng.forced_exits(state, {}, now=NOW)
    assert exits[0].reason is ExitReason.STOP_LOSS
    assert "untracked" in exits[0].detail


def test_untracked_position_within_tolerance_is_left_alone():
    eng = engine(hard_stop_pct=0.08)
    state = make_state(positions=[Position("TCS", 10, 1_000.0, 980.0)])
    assert eng.forced_exits(state, {}, now=NOW) == ()


def test_a_healthy_position_is_not_exited():
    eng = engine()
    state = make_state(positions=[held()], risks=[risk_record()])
    snaps = {"RELIANCE": make_snapshot("RELIANCE", price=1_005.0)}
    assert eng.forced_exits(state, snaps, now=NOW) == ()


def test_a_position_with_no_price_is_skipped_rather_than_guessed_at():
    eng = engine()
    state = make_state(positions=[Position("TCS", 10, 1_000.0, 0.0)])
    assert eng.forced_exits(state, {}, now=NOW) == ()


# =================================================================== gatekeeping
def test_only_buys_are_sized():
    verdict = engine().approve_entry(
        Decision("RELIANCE", Action.HOLD, 9, ""), make_snapshot(), make_state()
    )
    assert not verdict.approved and verdict.reason == "not a buy"


def test_low_confidence_is_refused():
    verdict = engine(min_confidence=7).approve_entry(
        buy(confidence=6), make_snapshot(), make_state()
    )
    assert not verdict.approved and "below 7" in verdict.reason


def test_a_symbol_with_no_price_is_refused():
    snapshot = make_snapshot(price=0.0, atr=None)
    verdict = engine().approve_entry(buy(), snapshot, make_state())
    assert not verdict.approved and verdict.reason == "no usable price"


def test_a_stale_quote_is_refused():
    stale = make_quote("RELIANCE", 1_000.0, when=NOW - timedelta(minutes=30))
    snapshot = make_snapshot(quote=stale)
    verdict = engine(max_quote_age_seconds=900).approve_entry(
        buy(), snapshot, make_state(), now=NOW
    )
    assert not verdict.approved and "old" in verdict.reason


def test_a_wide_observed_spread_is_refused():
    wide = make_quote("RELIANCE", 1_000.0, when=NOW, spread_bps=200.0)
    verdict = engine(max_spread_bps=50.0).approve_entry(
        buy(), make_snapshot(quote=wide), make_state(), now=NOW
    )
    assert not verdict.approved and "spread" in verdict.reason


def test_a_modelled_spread_is_not_grounds_for_refusal():
    """Free NSE data has no order book, so the spread is our own assumption.
    Rejecting on it would only ever reject our own guess."""
    modelled = make_quote("RELIANCE", 1_000.0, when=NOW, spread_bps=200.0, modelled=True)
    verdict = engine(max_spread_bps=50.0).approve_entry(
        buy(), make_snapshot(quote=modelled), make_state(), now=NOW
    )
    assert verdict.approved


def test_a_full_book_refuses_a_new_name():
    positions = [held(s, 1, 1_000.0) for s in ("A", "B", "C", "D", "E")]
    state = make_state(cash=50_000.0, positions=positions)
    verdict = engine(max_positions=5).approve_entry(buy(), make_snapshot(), state)
    assert not verdict.approved and "max positions" in verdict.reason


def test_a_position_already_at_its_cap_is_not_added_to():
    state = make_state(cash=50_000.0, positions=[held("RELIANCE", 25, 1_000.0)])
    verdict = engine(max_position_pct=0.20).approve_entry(
        buy(), make_snapshot(), state
    )
    assert not verdict.approved and "size cap" in verdict.reason


def test_sector_concentration_is_refused():
    state = make_state(
        cash=80_000.0,
        positions=[held("HDFCBANK", 1, 1_000.0), held("ICICIBANK", 1, 1_000.0)],
    )
    verdict = engine(max_sector_positions=2).approve_entry(
        buy("SBIN"), make_snapshot("SBIN"), state
    )
    assert not verdict.approved and "sector banks" in verdict.reason


def test_a_different_sector_is_allowed():
    state = make_state(
        cash=80_000.0,
        positions=[held("HDFCBANK", 1, 1_000.0), held("ICICIBANK", 1, 1_000.0)],
    )
    verdict = engine(max_sector_positions=2).approve_entry(
        buy("TCS"), make_snapshot("TCS"), state
    )
    assert verdict.approved


def test_sector_value_limit_is_refused():
    state = make_state(
        cash=60_000.0, positions=[held("HDFCBANK", 40, 1_000.0)]
    )
    verdict = engine(max_sector_positions=5, max_sector_pct=0.40).approve_entry(
        buy("ICICIBANK"), make_snapshot("ICICIBANK"), state
    )
    assert not verdict.approved and "of equity" in verdict.reason


def test_a_correlated_name_is_refused():
    """Five names that move together are one position with extra steps."""
    returns = {
        "TCS": [0.01, -0.02, 0.03, 0.01, -0.01],
        "INFY": [0.011, -0.021, 0.031, 0.009, -0.011],
    }
    state = make_state(cash=90_000.0, positions=[held("TCS", 1, 1_000.0)])
    verdict = engine(max_correlation=0.85).approve_entry(
        buy("INFY"), make_snapshot("INFY"), state, returns=returns
    )
    assert not verdict.approved and "correlation" in verdict.reason


def test_missing_return_history_does_not_block_a_trade():
    state = make_state(cash=90_000.0, positions=[held("TCS", 1, 1_000.0)])
    verdict = engine().approve_entry(
        buy("INFY"), make_snapshot("INFY"), state, returns={"INFY": []}
    )
    assert verdict.approved


# ====================================================================== sizing
def test_size_is_the_smallest_binding_constraint():
    verdict = engine(max_notional_per_trade=10_000).approve_entry(
        buy(), make_snapshot(price=1_000.0, atr=10.0), make_state()
    )
    assert verdict.approved
    assert verdict.qty == 10
    assert verdict.notional == 10_000
    assert verdict.stop_price == 980.0
    assert verdict.target_price == 1_030.0
    assert "per-trade cap" in verdict.reason


def test_the_model_can_ask_for_less_but_not_for_more():
    smaller = engine(max_notional_per_trade=10_000).approve_entry(
        buy(notional=3_000), make_snapshot(price=1_000.0), make_state()
    )
    assert smaller.notional == 3_000 and "model suggestion" in smaller.reason
    greedy = engine(max_notional_per_trade=10_000).approve_entry(
        buy(notional=99_000), make_snapshot(price=1_000.0), make_state()
    )
    assert greedy.notional == 10_000


def test_nse_size_rounds_down_to_whole_shares():
    """Rs 10,000 against a Rs 1,500 share is a 6-share trade worth Rs 9,000 --
    and every downstream check has to be told that, not the Rs 10,000."""
    verdict = engine(max_notional_per_trade=10_000).approve_entry(
        buy(), make_snapshot(price=1_500.0, atr=15.0), make_state()
    )
    assert verdict.qty == 6
    assert verdict.notional == 9_000


def test_a_share_more_expensive_than_the_cap_is_untradable():
    verdict = engine(max_notional_per_trade=10_000).approve_entry(
        buy("MRF"), make_snapshot("MRF", price=120_000.0, atr=1_000.0), make_state()
    )
    assert not verdict.approved
    assert "does not cover one tradeable lot" in verdict.reason


def test_the_us_path_still_buys_fractional_shares():
    eng = RiskEngine(
        RiskConfig(max_notional_per_trade=100.0, min_trade_notional=1.0),
        profile=US_MARKET,
        costs=USEquityCosts(),
    )
    verdict = eng.approve_entry(
        buy("AAPL"), make_snapshot("AAPL", price=150.0, atr=2.0),
        make_state(cash=10_000.0),
    )
    assert verdict.approved
    assert 0 < verdict.qty < 1
    assert verdict.notional == pytest.approx(100.0, abs=0.01)


def test_volatility_budget_binds_on_a_jumpy_name():
    """Wide stop, small size. This is the whole point of ATR sizing."""
    verdict = engine(max_notional_per_trade=100_000, risk_per_trade_pct=0.01).approve_entry(
        buy(), make_snapshot(price=1_000.0, atr=100.0), make_state()
    )
    assert "volatility budget" in verdict.reason
    assert verdict.qty == 5   # (1,000 risk / 200 stop distance) * 1,000 price


def test_cash_reserve_is_never_deployed():
    state = make_state(cash=1_000.0)
    verdict = engine(
        max_notional_per_trade=100_000, min_cash_reserve_pct=0.05
    ).approve_entry(buy(), make_snapshot(price=1_000.0), state)
    assert not verdict.approved  # 950 deployable does not buy one Rs 1,000 share


def test_a_trade_below_the_minimum_ticket_is_refused():
    verdict = engine(
        max_notional_per_trade=400, min_trade_notional=500
    ).approve_entry(buy(), make_snapshot(price=100.0, atr=1.0), make_state())
    assert not verdict.approved and "below minimum trade" in verdict.reason


def test_without_atr_the_stop_falls_back_to_a_percentage():
    eng = engine(hard_stop_pct=0.08, max_notional_per_trade=10_000)
    snapshot = make_snapshot(price=1_000.0, atr=None)
    assert eng.stop_distance(snapshot, 1_000.0) == 80.0
    verdict = eng.approve_entry(buy(), snapshot, make_state())
    assert verdict.stop_price == 920.0


# ================================================================ cost hurdle
def test_costs_can_veto_an_otherwise_valid_trade():
    """A 0.3% target that pays 0.58% to get in and out is not a trade."""
    eng = engine(
        costs=IndianEquityCosts("delivery"),
        max_notional_per_trade=5_000,
        max_cost_ratio=0.35,
    )
    verdict = eng.approve_entry(
        buy(), make_snapshot(price=1_000.0, atr=1.0), make_state()
    )
    assert not verdict.approved
    assert "round-trip cost" in verdict.reason


def test_the_same_trade_is_fine_when_nothing_is_charged():
    """Proves the veto above comes from the cost model and not from sizing."""
    eng = engine(costs=NoCosts(), max_notional_per_trade=5_000, max_cost_ratio=0.35)
    verdict = eng.approve_entry(
        buy(), make_snapshot(price=1_000.0, atr=1.0), make_state()
    )
    assert verdict.approved


def test_a_target_big_enough_to_clear_costs_is_allowed():
    eng = engine(costs=IndianEquityCosts("intraday"), max_notional_per_trade=10_000)
    verdict = eng.approve_entry(
        buy(), make_snapshot(price=1_000.0, atr=10.0), make_state()
    )
    assert verdict.approved


# ============================================================= risk lifecycle
def test_initial_risk_records_the_levels_that_were_approved():
    eng = engine(max_notional_per_trade=10_000)
    snapshot = make_snapshot(price=1_000.0, atr=10.0)
    verdict = eng.approve_entry(buy(), snapshot, make_state())
    record = eng.initial_risk("RELIANCE", 1_000.0, snapshot, NOW, verdict)
    assert record.stop_price == verdict.stop_price
    assert record.target_price == verdict.target_price
    assert record.high_water == 1_000.0
    assert record.atr_at_entry == 10.0
    assert record.bars_held == 0


def test_trailing_uses_the_configured_multiple():
    eng = engine(trailing_stop_atr=2.5)
    trailed = eng.trail(risk_record(), price=1_100.0)
    assert trailed.stop_price == 1_075.0   # 1,100 - 2.5 * 10
    assert trailed.bars_held == 1
