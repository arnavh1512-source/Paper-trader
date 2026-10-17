"""The deployed schedule has to survive GitHub's scheduler, not cron's arithmetic.

Scheduled workflows do not fire on time, and sometimes do not fire at all: hourly
crons arrived 34-54 minutes late on day one and later hours late, with whole
ticks dropped. So cron no longer carries the cadence. It only starts the day's
job, and the job loops hourly on its own clock until the close.

The cycle that cannot be missed is the square-off: it is the last of the day,
nothing follows it to catch a failure, and skipping it carries an intraday
position overnight while the intraday cost model keeps billing intraday rates --
so the P&L the experiment is measuring is wrong, quietly.

These tests read the real workflow file, and run the real loop out of it under a
fake clock, rather than a copy of its numbers. A schedule that is correct in a
fixture and wrong in the file that actually runs is the failure this is here to
prevent.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from claude_trader.config import VALID_STRATEGIES, RiskConfig
from claude_trader.markets import get_market
from claude_trader.risk.engine import RiskEngine

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "trader.yml"
LOOP_STEP = "Trade the session"

# A Monday NSE trades.
TRADING_DAY = datetime(2026, 8, 24, tzinfo=timezone.utc)

# Hard ceiling on a GitHub-hosted job, and what setup (checkout, Python, pip,
# cache restore) plus the post-loop steps (dashboard, Pages, cache save,
# artifacts) are allowed to take around the loop step.
GITHUB_JOB_CEILING_MINUTES = 360
SETUP_AND_POST_MARGIN_MINUTES = 10
# Scheduled NSE arms, in order: the control first so nothing can delay it.
ARMS = ["momentum", "momentum_plus", "claude"]

# Steps that must run however the loop ends. Save journal is the one that
# matters most: the journal is the account.
ALWAYS_STEPS = (
    "Render dashboard",
    "Upload dashboard",
    "Stage dashboard for Pages",
    "Upload Pages artifact",
    "Save journal",
    "Upload journal artifact",
)

# First major of each action that runs on Node 24. Anything older is on the
# deprecated Node 20 runtime.
NODE24_MAJORS = {
    "actions/checkout": 5,
    "actions/setup-python": 6,
    "actions/cache/restore": 5,
    "actions/cache/save": 5,
    "actions/upload-artifact": 6,
    "actions/upload-pages-artifact": 5,
    "actions/deploy-pages": 5,
}


# --- reading the workflow ---------------------------------------------------


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _job() -> dict:
    return _workflow()["jobs"]["trade"]


def _step(name: str) -> dict:
    steps = [s for s in _job()["steps"] if s.get("name") == name]
    assert len(steps) == 1, f"expected exactly one step named {name!r}"
    return steps[0]


def _env_int(key: str) -> int:
    return int(_job()["env"][key])


def _crons() -> list[str]:
    # PyYAML reads the bare key `on` as boolean True (YAML 1.1).
    triggers = _workflow()[True]
    return [entry["cron"] for entry in triggers["schedule"]]


def _cron_time(cron: str) -> time:
    minute, hour, dom, month, dow = cron.split()
    assert minute.isdigit() and hour.isdigit(), f"one fixed time per trigger: {cron}"
    assert (dom, month, dow) == ("*", "*", "1-5"), f"weekdays only: {cron}"
    return time(int(hour), int(minute))


def _square_off_minutes() -> int:
    match = re.search(r"\|\|\s*'(\d+)'", _job()["env"]["SQUARE_OFF_MINUTES"])
    assert match, "SQUARE_OFF_MINUTES is not pinned in the workflow"
    return int(match.group(1))


def _at(t: time) -> datetime:
    return TRADING_DAY.replace(hour=t.hour, minute=t.minute)


def _close_utc() -> datetime:
    profile = get_market("in")
    local_day = profile.local(TRADING_DAY).date()
    return datetime.combine(local_day, profile.close_time, profile.tz).astimezone(timezone.utc)


@pytest.fixture(scope="module")
def engine() -> RiskEngine:
    config = RiskConfig(
        square_off_enabled=True, square_off_minutes_before_close=_square_off_minutes()
    )
    return RiskEngine(config, profile=get_market("in"), costs=None)


# --- running the real loop under a fake clock -------------------------------

_SHIMS = {
    # Only the forms the loop uses: epoch seconds, and HH:MM for log lines.
    "date": """
read -r c < "$FAKE_CLOCK"
case "$*" in
  *%s*) echo "$c" ;;
  *) printf '%02d:%02d\\n' $(( c % 86400 / 3600 )) $(( c % 3600 / 60 )) ;;
esac
""",
    "sleep": """
read -r c < "$FAKE_CLOCK"
echo $(( c + $1 )) > "$FAKE_CLOCK"
""",
    "timeout": """
echo "$1" >> "$FAKE_LOG.timeout"
shift
exec "$@"
""",
    # Records "epoch args", spends FAKE_CYCLE_SECONDS of fake time, and fails
    # for any arm listed in FAKE_FAIL.
    "python": """
read -r c < "$FAKE_CLOCK"
echo "$c $*" >> "$FAKE_LOG"
echo $(( c + FAKE_CYCLE_SECONDS )) > "$FAKE_CLOCK"
arm="${@: -1}"
case " ${FAKE_FAIL:-} " in *" $arm "*) exit 2 ;; esac
exit 0
""",
}


def _bash() -> str | None:
    found = shutil.which("bash")
    # On Windows, System32\\bash.exe is the WSL launcher, which cannot see
    # these shims. Git Bash can.
    if not found or "system32" in found.lower():
        return None
    return found


@dataclass(frozen=True, slots=True)
class LoopRun:
    start: datetime
    end: datetime
    calls: tuple[tuple[datetime, str], ...]  # (when, arm)
    timeouts: tuple[str, ...]
    returncode: int
    stdout: str

    def arms_at(self) -> list[tuple[datetime, list[str]]]:
        grouped: dict[datetime, list[str]] = {}
        iteration: datetime | None = None
        for when, arm in self.calls:
            if arm == "momentum" or iteration is None or arm in grouped[iteration]:
                iteration = when
                grouped.setdefault(iteration, [])
            grouped[iteration].append(arm)
        return list(grouped.items())


def _run_loop(
    tmp_path: Path,
    start: datetime,
    *,
    event: str = "schedule",
    market: str = "in",
    strategy: str = "momentum",
    api_key: str = "not-a-real-key",
    cycle_seconds: int = 60,
    fail: str = "",
) -> LoopRun:
    bash = _bash()
    if bash is None:
        pytest.skip("needs a POSIX bash to run the workflow's loop")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    for name, body in _SHIMS.items():
        shim = bin_dir / name
        shim.write_text("#!/bin/bash\n" + body.lstrip(), encoding="utf-8", newline="\n")
        shim.chmod(0o755)
    clock, log = tmp_path / "clock", tmp_path / "calls"
    clock.write_text(str(int(start.timestamp())), encoding="utf-8")

    job_env = _job()["env"]
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAKE_CLOCK": str(clock),
        "FAKE_LOG": str(log),
        "FAKE_CYCLE_SECONDS": str(cycle_seconds),
        "FAKE_FAIL": fail,
        "GITHUB_EVENT_NAME": event,
        "MARKET": market,
        "STRATEGY": strategy,
        "ANTHROPIC_API_KEY": api_key,
        **{
            key: str(job_env[key])
            for key in (
                "SESSION_CLOSE_UTC",
                "CYCLE_SLOT_MINUTE",
                "LOOP_BUDGET_MINUTES",
                "CYCLE_TIMEOUT_MINUTES",
            )
        },
    }
    script = _step(LOOP_STEP)["run"]
    # The flags GitHub uses for `shell: bash`; stricter than the default
    # `bash -e`, so a loop that survives this survives both.
    result = subprocess.run(
        [bash, "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    def epoch(raw: str) -> datetime:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)

    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    calls = []
    for line in lines:
        raw, *args = line.split()
        assert args[:4] == ["-m", "claude_trader", "trade", "--strategy"], line
        calls.append((epoch(raw), args[4]))
    timeout_log = Path(f"{log}.timeout")
    timeouts = tuple(timeout_log.read_text(encoding="utf-8").split()) if timeout_log.exists() else ()
    return LoopRun(
        start=start,
        end=epoch(clock.read_text(encoding="utf-8").strip()),
        calls=tuple(calls),
        timeouts=timeouts,
        returncode=result.returncode,
        stdout=result.stdout + result.stderr,
    )


def _arrivals() -> list[datetime]:
    """Every trigger as written, and the same triggers arriving late -- up to
    the close, which is as late as a trigger can arrive and still matter."""
    close = _close_utc()
    late = [timedelta(minutes=m) for m in (0, 2, 58, 133, 241)]
    times = {_at(_cron_time(c)) + d for c in _crons() for d in late}
    return sorted(t for t in times if t < close)


# --- the triggers -----------------------------------------------------------


def test_the_check_is_reading_the_real_workflow() -> None:
    # If parsing silently returns nothing, every assertion below passes vacuously.
    assert len(_crons()) >= 2, "one trigger is one dropped run away from a missed day"
    assert _step(LOOP_STEP)["run"].strip(), "the loop step has no script"


def test_triggers_fire_off_the_hour_and_never_before_the_open() -> None:
    profile = get_market("in")
    open_utc = datetime.combine(
        profile.local(TRADING_DAY).date(), profile.open_time, profile.tz
    ).astimezone(timezone.utc)
    for cron in _crons():
        at = _at(_cron_time(cron))
        assert at >= open_utc, f"{cron} fires before the NSE open"
        assert at < _close_utc(), f"{cron} fires after the NSE close"
        assert at.minute != 0, f"{cron} is on the hour, when GitHub drops the most runs"


def test_triggers_are_spread_so_each_can_cover_a_late_or_dropped_one() -> None:
    times = sorted(_at(_cron_time(c)) for c in _crons())
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(gap >= timedelta(hours=1) for gap in gaps), f"triggers bunched together: {times}"
    # The backstop has to land early enough to still trade, not just square off.
    assert times[-1] <= _close_utc() - timedelta(hours=2)


def test_session_close_constant_matches_the_market_profile() -> None:
    hh, mm = map(int, _job()["env"]["SESSION_CLOSE_UTC"].split(":"))
    assert _at(time(hh, mm)) == _close_utc()


# --- the loop's time budget -------------------------------------------------


def test_the_loop_fits_inside_the_job_and_the_job_inside_githubs_ceiling() -> None:
    worst_overrun = len(ARMS) * _env_int("CYCLE_TIMEOUT_MINUTES")  # every arm hung
    step_timeout = int(_step(LOOP_STEP)["timeout-minutes"])
    job_timeout = int(_job()["timeout-minutes"])
    assert _env_int("LOOP_BUDGET_MINUTES") + worst_overrun <= step_timeout
    assert step_timeout + SETUP_AND_POST_MARGIN_MINUTES <= job_timeout
    assert job_timeout <= GITHUB_JOB_CEILING_MINUTES


def test_the_earliest_trigger_has_budget_to_reach_the_square_off_slot(engine) -> None:
    """The budget is what stops a 03:47 job running into the 6h ceiling. It must
    not also stop it before the last slot, or the earliest trigger -- the one
    most likely to be the day's only run -- never squares off."""
    first = _at(min(_cron_time(c) for c in _crons()))
    budget_end = first + timedelta(minutes=_env_int("LOOP_BUDGET_MINUTES"))
    slot = _env_int("CYCLE_SLOT_MINUTE")
    last_slot = max(
        t
        for t in (_close_utc().replace(minute=slot) - timedelta(hours=h) for h in range(8))
        if t < _close_utc()
    )
    assert last_slot < budget_end, f"budget ends {budget_end:%H:%M}, last slot {last_slot:%H:%M}"
    assert engine.square_off_due(last_slot), f"the last slot {last_slot:%H:%M} UTC is early"


@pytest.mark.parametrize("start", _arrivals(), ids=lambda t: t.strftime("%H%M"))
def test_every_arrival_trades_to_the_close_and_squares_off(tmp_path, engine, start) -> None:
    profile = get_market("in")
    run = _run_loop(tmp_path, start)

    assert run.returncode == 0, run.stdout
    assert run.calls, "an in-session arrival traded nothing"
    assert all(when < _close_utc() for when, _ in run.calls), "a cycle started after the close"
    assert all(profile.is_session_time(when) for when, _ in run.calls)
    budget = timedelta(minutes=_env_int("LOOP_BUDGET_MINUTES"))
    # The budget gates iterations; the arms inside one run back to back, and
    # that overrun is what the step timeout is sized for.
    assert all(when < start + budget for when, _ in run.arms_at()), "an iteration started past the budget"
    assert run.end - start <= timedelta(minutes=int(_step(LOOP_STEP)["timeout-minutes"]))

    last_when, last_arms = run.arms_at()[-1]
    assert last_arms == ARMS
    assert engine.square_off_due(last_when), (
        f"arriving at {start:%H:%M} UTC, the day's last cycle is {last_when:%H:%M} UTC, "
        "before the square-off window"
    )
    assert all(engine.square_off_due(when) for when, arm in run.calls if when >= last_when)


def test_iterations_are_hourly_on_the_slot_after_the_first(tmp_path) -> None:
    start = _at(min(_cron_time(c) for c in _crons()))
    run = _run_loop(tmp_path, start)
    iterations = [when for when, _ in run.arms_at()]
    slot = _env_int("CYCLE_SLOT_MINUTE")
    assert iterations[0] == start, "the first cycle should run on arrival"
    assert all(t.minute == slot and t.second == 0 for t in iterations[1:])
    assert all(b - a == timedelta(hours=1) for a, b in zip(iterations[1:], iterations[2:]))


def test_the_first_cycle_of_the_day_can_still_open_positions(tmp_path, engine) -> None:
    """Square-off blocks new entries. If every cycle were past the cut-off the
    bot would never open a position -- silently, with a green workflow."""
    start = _at(min(_cron_time(c) for c in _crons()))
    run = _run_loop(tmp_path, start)
    assert not engine.square_off_due(run.calls[0][0])


def test_hung_cycles_still_end_the_loop_inside_its_step_timeout(tmp_path) -> None:
    start = _at(min(_cron_time(c) for c in _crons()))
    hung = _env_int("CYCLE_TIMEOUT_MINUTES") * 60
    run = _run_loop(tmp_path, start, cycle_seconds=hung)
    assert run.returncode == 0, run.stdout
    assert run.end - start <= timedelta(minutes=int(_step(LOOP_STEP)["timeout-minutes"]))
    assert set(run.timeouts) == {f"{_env_int('CYCLE_TIMEOUT_MINUTES')}m"}


def test_a_run_that_starts_after_the_close_trades_nothing(tmp_path) -> None:
    run = _run_loop(tmp_path, _close_utc() + timedelta(minutes=7))
    assert run.returncode == 0, run.stdout
    assert run.calls == ()


# --- the arms ---------------------------------------------------------------


def test_every_iteration_runs_the_control_first_then_every_other_arm(tmp_path) -> None:
    run = _run_loop(tmp_path, _at(time(5, 23)))
    iterations = run.arms_at()
    assert len(iterations) >= 4
    assert all(arms == ARMS for _, arms in iterations)


def test_the_scheduled_arms_are_exactly_the_valid_strategies() -> None:
    """A strategy added to config but not to the loop never builds a record;
    one in the loop but not in config fails every hour on ConfigError."""
    assert sorted(ARMS) == sorted(VALID_STRATEGIES)


def test_claude_is_skipped_without_a_key_and_the_keyless_arms_still_run(tmp_path) -> None:
    run = _run_loop(tmp_path, _at(time(5, 23)), api_key="")
    assert run.returncode == 0, "a missing key is a normal state, not a failure"
    keyless = [arm for arm in ARMS if arm != "claude"]
    assert all(arms == keyless for _, arms in run.arms_at())
    assert "ANTHROPIC_API_KEY not set; skipping the claude cycle" in run.stdout


def test_a_failing_claude_cycle_never_costs_another_arm_a_cycle(tmp_path) -> None:
    healthy = _run_loop(tmp_path / "ok", _at(time(5, 23)))
    broken = _run_loop(tmp_path / "bad", _at(time(5, 23)), fail="claude")
    for keyless in ("momentum", "momentum_plus"):
        kept = [when for when, arm in broken.calls if arm == keyless]
        assert kept == [when for when, arm in healthy.calls if arm == keyless]
        assert len(kept) >= 4, f"{keyless} stopped after a claude failure"
    assert broken.returncode != 0, "a failed cycle must still turn the run red"
    assert "::warning::claude cycle failed" in broken.stdout


def test_a_manual_run_is_one_cycle_of_the_chosen_strategy(tmp_path) -> None:
    run = _run_loop(tmp_path, _at(time(5, 23)), event="workflow_dispatch", strategy="claude")
    assert [arm for _, arm in run.calls] == ["claude"]
    assert run.returncode == 0, run.stdout


def test_scheduled_us_runs_keep_a_single_strategy(tmp_path) -> None:
    """Alpaca has one server-side paper account. Two arms there would trade the
    same book and journal each other's fills."""
    run = _run_loop(tmp_path, _at(time(5, 23)), market="us", strategy="momentum")
    assert run.calls and all(arm == "momentum" for _, arm in run.calls)


def test_the_strategy_env_stays_a_single_valid_value() -> None:
    """AppConfig validates STRATEGY before any --strategy override applies, so
    the env default has to be a real strategy even though the loop overrides it."""
    match = re.search(r"\|\|\s*'(\w+)'\s*}}", _job()["env"]["STRATEGY"])
    assert match and match.group(1) in VALID_STRATEGIES


# --- the steps around the loop ----------------------------------------------


@pytest.mark.parametrize("name", ALWAYS_STEPS)
def test_post_loop_steps_run_however_the_loop_ended(name) -> None:
    assert _step(name).get("if") == "always()"


def test_post_loop_steps_come_after_the_loop() -> None:
    names = [s.get("name") for s in _job()["steps"]]
    loop = names.index(LOOP_STEP)
    assert all(names.index(name) > loop for name in ALWAYS_STEPS)


def test_runs_queue_behind_each_other_instead_of_cancelling() -> None:
    """Two jobs writing one journal would fork the account; cancelling a running
    one would throw away its session."""
    assert _workflow()["concurrency"] == {"group": "paper-trader", "cancel-in-progress": False}


def test_the_dashboard_is_published_to_pages_only() -> None:
    assert "vercel" not in WORKFLOW.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("workflow", sorted(WORKFLOW.parent.glob("*.yml")), ids=lambda p: p.name)
def test_no_action_is_pinned_to_a_node20_major(workflow: Path) -> None:
    pins = re.findall(r"uses:\s*([\w./-]+)@v(\d+)", workflow.read_text(encoding="utf-8"))
    assert pins, f"no actions parsed from {workflow.name}"
    stale = [
        f"{action}@v{major}"
        for action, major in pins
        if int(major) < NODE24_MAJORS.get(action, 0)
    ]
    unknown = [action for action, _ in pins if action not in NODE24_MAJORS]
    assert not stale, f"still on the Node 20 runtime: {stale}"
    assert not unknown, f"add the Node 24 major for {unknown} to NODE24_MAJORS"
