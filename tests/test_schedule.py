"""The deployed cron has to survive GitHub's lateness, not cron's arithmetic.

Scheduled workflows do not fire on time. The first day of hourly running drifted
34, 47 and 54 minutes on consecutive ticks, which is enough to step a whole tick
past the close. The square-off tick is the one that cannot be missed: it is the
last of the day, nothing follows it to catch a failure, and skipping it carries
an intraday position overnight while the intraday cost model keeps billing
intraday rates -- so the P&L the experiment is measuring is wrong, quietly.

These tests read the real workflow file rather than a copy of its numbers. A
schedule that is correct in a fixture and wrong in the file that actually runs
is the failure this is here to prevent.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claude_trader.config import RiskConfig
from claude_trader.markets import get_market
from claude_trader.risk.engine import RiskEngine

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "trader.yml"

# Observed 34-54 minutes on day one. Designed for a full hour; asserted over the
# whole range rather than at the endpoints, because a schedule that works when
# late and fails when punctual is still broken.
MAX_DRIFT_MINUTES = 60

# A Monday NSE trades.
TRADING_DAY = datetime(2026, 8, 24, tzinfo=timezone.utc)


def _active_crons() -> list[str]:
    """Uncommented `- cron:` lines. Commented-out market blocks are inert and
    must not be asserted against -- the US block is deliberately parked."""
    text = WORKFLOW.read_text(encoding="utf-8")
    return re.findall(r"^\s*-\s*cron:\s*'([^']+)'", text, flags=re.MULTILINE)


def _square_off_minutes() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"SQUARE_OFF_MINUTES:.*?\|\|\s*'(\d+)'", text)
    assert match, "SQUARE_OFF_MINUTES is not pinned in the workflow"
    return int(match.group(1))


def _tick_times(cron: str) -> list[datetime]:
    minute, hours = cron.split()[0], cron.split()[1]
    minutes = [int(minute)] if minute.isdigit() else []
    assert minutes, f"unsupported minute field for this check: {cron}"
    start, _, end = hours.partition("-")
    span = range(int(start), int(end or start) + 1)
    return [TRADING_DAY.replace(hour=h, minute=m) for h in span for m in minutes]


@pytest.fixture(scope="module")
def engine() -> RiskEngine:
    profile = get_market("in")
    config = RiskConfig(
        square_off_enabled=True, square_off_minutes_before_close=_square_off_minutes()
    )
    return RiskEngine(config, profile=profile, costs=None)


@pytest.fixture(scope="module")
def ticks() -> list[datetime]:
    crons = _active_crons()
    assert crons, "no active cron in the workflow; the bot would never run"
    return sorted(t for cron in crons for t in _tick_times(cron))


def _drifted(tick: datetime) -> list[datetime]:
    return [tick + timedelta(minutes=d) for d in range(0, MAX_DRIFT_MINUTES + 1)]


def test_the_check_is_reading_the_real_workflow(ticks) -> None:
    # If parsing silently returns nothing, every assertion below passes vacuously.
    assert len(ticks) >= 4, f"suspiciously few ticks parsed: {ticks}"


def test_the_last_tick_squares_off_however_late_it_arrives(engine, ticks) -> None:
    """The one that matters. Nothing runs after it, so if any arrival time in
    the drift range fails to trigger square-off, some days end with an open
    intraday position and no mechanism left to close it."""
    profile = get_market("in")
    misses = [
        profile.local(t).strftime("%H:%M")
        for t in _drifted(ticks[-1])
        if not (engine.square_off_due(t) and profile.is_session_time(t))
    ]
    assert not misses, (
        "the final tick fails to square off when it arrives at these IST times: "
        f"{misses}. Widen SQUARE_OFF_MINUTES or move the tick earlier."
    )


def test_every_tick_lands_inside_the_session_even_at_full_drift(engine, ticks) -> None:
    profile = get_market("in")
    strays = [
        (profile.local(tick).strftime("%H:%M"), profile.local(t).strftime("%H:%M"))
        for tick in ticks
        for t in _drifted(tick)
        if not profile.is_session_time(t)
    ]
    assert not strays, f"ticks that drift out of the session: {sorted(set(strays))}"


def test_trading_ticks_precede_the_square_off_cutoff_when_punctual(engine, ticks) -> None:
    """Square-off blocks new entries. If every tick were past the cut-off the
    bot would never open a position -- silently, with a green workflow."""
    assert not engine.square_off_due(ticks[0]), "the first tick of the day already squares off"
    assert any(not engine.square_off_due(t) for t in ticks[:-1])
