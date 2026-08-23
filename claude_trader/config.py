"""Immutable configuration, assembled from the environment and validated once
at startup. Nothing downstream reads os.environ.

Config is market-aware. Which exchange the bot trades decides the currency, the
session clock, the universe, whether fractional shares exist, what a round trip
costs and therefore how big a position has to be before it is worth opening. All
of that comes from a ``MarketProfile`` rather than from scattered constants, so
adding a market means adding a profile -- not editing the decision path.
"""

from __future__ import annotations

import logging

import os
from pathlib import Path
from dataclasses import dataclass, field, fields, replace
from typing import Any

from .errors import ConfigError
from .markets import DEFAULT_MARKET, MARKETS, MarketProfile, get_market

_TRUTHY = {"1", "true", "yes", "on"}

VALID_STRATEGIES = ("claude", "momentum")
VALID_BROKERS = ("alpaca", "paper")
VALID_SEGMENTS = ("intraday", "delivery")


def load_dotenv(path: str = ".env") -> int:
    """Fill in missing environment variables from a local ``.env``.

    Existing values always win, so a GitHub Actions secret is never shadowed by
    a stale file someone left in the checkout. Returns how many names were set,
    which is the only thing worth logging: printing the values would print the
    keys.
    """
    file = Path(path)
    if not file.is_file():
        return 0
    loaded = 0
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if not name or name in os.environ:
            continue
        value = value.split(" #")[0].strip().strip("'\"")
        if value:
            os.environ[name] = value
            loaded += 1
    return loaded


def _env_str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUTHY


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw}") from exc


log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Every number the deterministic risk layer needs.

    These are hard limits enforced in code. The model cannot argue its way past
    any of them: it only ever proposes, risk disposes.
    """

    # --- sizing -------------------------------------------------------------
    max_notional_per_trade: float = 100.0
    max_position_pct: float = 0.20          # cap on any one name, % of equity
    risk_per_trade_pct: float = 0.01        # equity risked between entry and stop
    min_trade_notional: float = 1.00
    min_cash_reserve_pct: float = 0.05      # never deploy the last 5% of cash

    # --- exits --------------------------------------------------------------
    atr_stop_multiple: float = 2.0
    atr_target_multiple: float = 3.0
    trailing_stop_atr: float = 2.5
    max_holding_bars: int = 130             # ~2 sessions of 15m bars, then flat
    hard_stop_pct: float = 0.08             # backstop when ATR is unavailable
    square_off_enabled: bool = False
    square_off_minutes_before_close: int = 15

    # --- concentration ------------------------------------------------------
    max_positions: int = 5
    max_sector_positions: int = 2
    max_sector_pct: float = 0.40
    max_correlation: float = 0.85
    correlation_lookback: int = 60

    # --- circuit breakers ---------------------------------------------------
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_trades_per_cycle: int = 3
    max_trades_per_day: int = 10

    # --- gating -------------------------------------------------------------
    min_confidence: int = 7
    max_quote_age_seconds: int = 900
    max_spread_bps: float = 50.0
    max_cost_ratio: float = 0.35
    """Ceiling on round-trip transaction cost as a share of the expected move.

    An entry targeting a 1% gain that pays 0.6% to get in and out is not a trade,
    it is a donation. This is the most important gate on a market with material
    statutory charges, and it is why India defaults to intraday.
    """

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.max_notional_per_trade <= 0:
            problems.append("max_notional_per_trade must be > 0")
        if not 0 < self.max_position_pct <= 1:
            problems.append("max_position_pct must be in (0, 1]")
        if not 0 < self.risk_per_trade_pct <= 0.5:
            problems.append("risk_per_trade_pct must be in (0, 0.5]")
        if not 0 <= self.min_cash_reserve_pct < 1:
            problems.append("min_cash_reserve_pct must be in [0, 1)")
        if self.atr_stop_multiple <= 0:
            problems.append("atr_stop_multiple must be > 0")
        if self.atr_target_multiple <= self.atr_stop_multiple:
            problems.append("atr_target_multiple must exceed atr_stop_multiple")
        if self.max_positions < 1:
            problems.append("max_positions must be >= 1")
        if self.max_sector_positions < 1:
            problems.append("max_sector_positions must be >= 1")
        if not 0 < self.max_correlation <= 1:
            problems.append("max_correlation must be in (0, 1]")
        if not 1 <= self.min_confidence <= 10:
            problems.append("min_confidence must be in [1, 10]")
        if self.daily_loss_limit_pct <= 0 or self.max_drawdown_pct <= 0:
            problems.append("loss limits must be > 0")
        if not 0 < self.max_cost_ratio <= 1:
            problems.append("max_cost_ratio must be in (0, 1]")
        if self.square_off_minutes_before_close < 0:
            problems.append("square_off_minutes_before_close must be >= 0")
        if problems:
            raise ConfigError("Invalid RiskConfig: " + "; ".join(problems))

    @classmethod
    def from_env(
        cls,
        profile: MarketProfile | None = None,
        equity: float | None = None,
    ) -> "RiskConfig":
        """Defaults follow the market *and* the size of the book.

        A Rs 10,000 ticket and a 100 dollar ticket are not the same trade, and
        holding overnight on NSE costs several times what intraday costs -- so
        the starting numbers come from the profile instead of forcing every
        Indian user to discover and override eight variables.

        They also have to follow ``equity``, because the profile defaults are
        written for a one-lakh book and are actively incoherent on a small one.
        At Rs 2,000 a 20% position cap is Rs 400 while the Indian minimum ticket
        is Rs 500: every order is simultaneously too large and too small, and
        the bot trades exactly nothing, forever, without ever erroring. Scaling
        the ceiling *and* the floor to the book keeps the two ends apart.
        """
        profile = profile or get_market()
        indian = profile.key == "in"
        equity = equity if equity and equity > 0 else profile.starting_cash
        # Below this the profile's absolute rupee floors stop making sense.
        small_book = equity < profile.starting_cash / 4
        # One session of bars, less a little, keeps intraday genuinely intraday.
        default_hold = max(4, profile.bars_per_session - 2) if indian else 130
        return cls(
            max_notional_per_trade=_env_float(
                "MAX_NOTIONAL_PER_TRADE",
                min(profile.max_per_trade, equity * 0.40),
            ),
            # A small book cannot diversify and pretending otherwise just means
            # never reaching one whole share of anything.
            max_position_pct=_env_float(
                "MAX_POSITION_PCT", 0.40 if small_book else 0.20
            ),
            risk_per_trade_pct=_env_float("RISK_PER_TRADE_PCT", 0.01),
            min_trade_notional=_env_float(
                "MIN_TRADE_NOTIONAL",
                min(500.0, equity * 0.05) if indian else min(1.0, equity * 0.05),
            ),
            min_cash_reserve_pct=_env_float("MIN_CASH_RESERVE_PCT", 0.05),
            atr_stop_multiple=_env_float("ATR_STOP_MULTIPLE", 2.0),
            atr_target_multiple=_env_float("ATR_TARGET_MULTIPLE", 3.0),
            trailing_stop_atr=_env_float("TRAILING_STOP_ATR", 2.5),
            max_holding_bars=_env_int("MAX_HOLDING_BARS", default_hold),
            hard_stop_pct=_env_float("HARD_STOP_PCT", 0.08),
            square_off_enabled=_env_bool("SQUARE_OFF_ENABLED", indian),
            square_off_minutes_before_close=_env_int("SQUARE_OFF_MINUTES", 15),
            max_positions=_env_int("MAX_POSITIONS", 3 if small_book else 5),
            max_sector_positions=_env_int("MAX_SECTOR_POSITIONS", 2),
            max_sector_pct=_env_float("MAX_SECTOR_PCT", 0.40),
            max_correlation=_env_float("MAX_CORRELATION", 0.85),
            daily_loss_limit_pct=_env_float("DAILY_LOSS_LIMIT_PCT", 0.03),
            max_drawdown_pct=_env_float("MAX_DRAWDOWN_PCT", 0.15),
            max_trades_per_cycle=_env_int("MAX_TRADES_PER_CYCLE", 3),
            max_trades_per_day=_env_int("MAX_TRADES_PER_DAY", 10),
            min_confidence=_env_int("MIN_CONFIDENCE", 7),
            max_quote_age_seconds=_env_int("MAX_QUOTE_AGE_SECONDS", 900),
            max_spread_bps=_env_float("MAX_SPREAD_BPS", 50.0),
            max_cost_ratio=_env_float("MAX_COST_RATIO", 0.35),
        )

    def as_dict(self) -> dict[str, Any]:
        """Recorded verbatim with every run: a result is only reproducible if
        the limits that produced it were stored alongside it."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str = ""
    model: str = "claude-sonnet-5"
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_seconds: float = 45.0
    max_retries: int = 3
    cache_enabled: bool = True
    # Hard ceiling on billable calls in one process. 0 means no ceiling, which
    # is right for live trading -- an hourly cycle cannot run away. It exists
    # for backtests: replaying six months at 15-minute bars is ~10,000 calls,
    # and the first person to discover that is usually the invoice.
    max_api_calls: int = 0

    def __post_init__(self) -> None:
        if self.max_api_calls < 0:
            raise ConfigError("MAX_API_CALLS must be >= 0 (0 means no ceiling)")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=_env_str("ANTHROPIC_API_KEY"),
            model=_env_str("CLAUDE_MODEL", "claude-sonnet-5"),
            max_tokens=_env_int("CLAUDE_MAX_TOKENS", 1024),
            temperature=_env_float("CLAUDE_TEMPERATURE", 0.0),
            timeout_seconds=_env_float("CLAUDE_TIMEOUT", 45.0),
            max_retries=_env_int("CLAUDE_MAX_RETRIES", 3),
            cache_enabled=_env_bool("LLM_CACHE_ENABLED", True),
            max_api_calls=_env_int("MAX_API_CALLS", 0),
        )


@dataclass(frozen=True, slots=True)
class AppConfig:
    market: str = DEFAULT_MARKET
    broker: str = ""            # resolved from the market when left blank
    segment: str = "intraday"   # NSE cash: intraday | delivery
    alpaca_key: str = ""
    alpaca_secret: str = ""
    alpaca_base: str = "https://paper-api.alpaca.markets/v2"
    alpaca_data_base: str = "https://data.alpaca.markets/v2"
    feed: str = "iex"
    timeframe: str = ""         # defaults to the native timeframe of the market
    bar_lookback: int = 120
    universe: tuple[str, ...] = ()
    starting_cash: float = 0.0
    dry_run: bool = False
    verbose: bool = False
    journal_path: str = "data/journal.sqlite3"
    strategy: str = "claude"

    # --- news ---------------------------------------------------------------
    # Off by default. Headlines change what the model is shown, and a setting
    # that silently alters decisions is one that should be opted into.
    news_enabled: bool = False
    news_max_headlines: int = 5
    news_max_age_hours: float = 24.0

    risk: RiskConfig = field(default_factory=RiskConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def __post_init__(self) -> None:
        if self.market not in MARKETS:
            raise ConfigError(
                f"unknown MARKET {self.market!r}; expected one of "
                + ", ".join(sorted(MARKETS))
            )
        profile = MARKETS[self.market]
        # Fill market-derived blanks in place so every consumer sees a complete
        # config and nothing downstream has to know a default was implied.
        if not self.universe:
            object.__setattr__(self, "universe", profile.universe)
        if not self.timeframe:
            object.__setattr__(self, "timeframe", profile.timeframe)
        if self.starting_cash <= 0:
            object.__setattr__(self, "starting_cash", profile.starting_cash)
        if not self.broker:
            object.__setattr__(
                self, "broker", "alpaca" if self.market == "us" else "paper"
            )

        problems: list[str] = []
        # The sizing deadlock. Every buy is clamped down to the smallest of the
        # position cap, the per-trade ceiling and the deployable cash, and then
        # rejected if the result falls under the minimum ticket. When the floor
        # sits above that smallest ceiling, *every* order fails that last test
        # and the bot runs for months placing nothing while every component
        # reports itself healthy. Silence is the worst possible failure mode
        # here, so the contradiction is an error at startup rather than a
        # mystery in the journal.
        ceiling = min(
            self.starting_cash * self.risk.max_position_pct,
            self.risk.max_notional_per_trade,
            self.starting_cash * (1.0 - self.risk.min_cash_reserve_pct),
        )
        if ceiling < self.risk.min_trade_notional:
            sym = profile.currency_symbol
            problems.append(
                f"no order can ever be placed: the largest permitted buy is "
                f"{sym}{ceiling:,.2f} but MIN_TRADE_NOTIONAL is "
                f"{sym}{self.risk.min_trade_notional:,.2f}. Raise "
                f"MAX_POSITION_PCT / MAX_NOTIONAL_PER_TRADE / STARTING_CASH, "
                f"or lower MIN_TRADE_NOTIONAL"
            )
        if self.strategy not in VALID_STRATEGIES:
            problems.append(f"STRATEGY must be one of {', '.join(VALID_STRATEGIES)}")
        if self.broker not in VALID_BROKERS:
            problems.append(f"BROKER must be one of {', '.join(VALID_BROKERS)}")
        if self.segment not in VALID_SEGMENTS:
            problems.append(f"TRADE_SEGMENT must be one of {', '.join(VALID_SEGMENTS)}")
        if self.broker == "alpaca" and self.market != "us":
            problems.append("the Alpaca broker only serves the US market")
        if self.market == "in" and self.segment == "intraday":
            # Otherwise the cost model bills intraday rates on positions that
            # were actually carried overnight, which flatters every backtest.
            # These are corrected rather than rejected, because the combination
            # is not a preference the user could hold coherently -- but the
            # correction is logged, so it is never a silent change of intent.
            risk = self.risk
            if not risk.square_off_enabled:
                log.warning(
                    "segment 'intraday' forces SQUARE_OFF_ENABLED on; without it "
                    "the cost model under-charges positions held overnight"
                )
                risk = replace(risk, square_off_enabled=True)
            if risk.max_holding_bars >= profile.bars_per_session:
                capped = max(4, profile.bars_per_session - 2)
                log.warning(
                    "MAX_HOLDING_BARS %d exceeds one session (%d bars); capping to %d",
                    risk.max_holding_bars,
                    profile.bars_per_session,
                    capped,
                )
                risk = replace(risk, max_holding_bars=capped)
            if risk is not self.risk:
                object.__setattr__(self, "risk", risk)
        if problems:
            raise ConfigError("Invalid configuration: " + "; ".join(problems))

    # --------------------------------------------------------------- derived
    @property
    def profile(self) -> MarketProfile:
        return MARKETS[self.market]

    @property
    def currency(self) -> str:
        return self.profile.currency

    @property
    def benchmark(self) -> str:
        return self.profile.benchmark

    @property
    def periods_per_year(self) -> float:
        """Annualisation factor implied by this market's session length."""
        from .analytics.metrics import periods_per_year_for

        return periods_per_year_for(self.profile.bars_per_session)

    @property
    def is_paper(self) -> bool:
        """True when no order can reach a real exchange.

        The journal-backed paper broker is paper by construction; Alpaca is only
        paper while it is pointed at the paper endpoint.
        """
        return self.broker == "paper" or "paper-api" in self.alpaca_base

    def money(self, amount: float) -> str:
        return self.profile.money(amount)

    @classmethod
    def from_env(cls, **overrides: Any) -> "AppConfig":
        market = str(
            overrides.pop("market", "") or _env_str("MARKET", DEFAULT_MARKET)
        ).lower()
        segment = str(
            overrides.pop("segment", "") or _env_str("TRADE_SEGMENT", "intraday")
        ).lower()
        if market not in MARKETS:
            raise ConfigError(
                f"unknown MARKET {market!r}; expected one of "
                + ", ".join(sorted(MARKETS))
            )
        profile = MARKETS[market]
        symbols = tuple(
            dict.fromkeys(
                s.strip().upper() for s in _env_str("UNIVERSE").split(",") if s.strip()
            )
        )
        cfg = cls(
            market=market,
            broker=_env_str("BROKER").lower(),
            segment=segment,
            alpaca_key=_env_str("ALPACA_API_KEY"),
            alpaca_secret=_env_str("ALPACA_SECRET_KEY"),
            alpaca_base=_env_str(
                "ALPACA_BASE", "https://paper-api.alpaca.markets/v2"
            ).rstrip("/"),
            alpaca_data_base=_env_str(
                "ALPACA_DATA_BASE", "https://data.alpaca.markets/v2"
            ).rstrip("/"),
            feed=_env_str("ALPACA_FEED", "iex").lower(),
            timeframe=_env_str("BAR_TIMEFRAME", profile.timeframe),
            bar_lookback=_env_int("BAR_LOOKBACK", 120),
            universe=symbols or profile.universe,
            starting_cash=_env_float("STARTING_CASH", profile.starting_cash),
            dry_run=_env_bool("DRY_RUN", False),
            verbose=_env_bool("VERBOSE", False),
            journal_path=_env_str("JOURNAL_PATH", "data/journal.sqlite3"),
            strategy=_env_str("STRATEGY", "claude").lower(),
            news_enabled=_env_bool("NEWS_ENABLED", False),
            news_max_headlines=_env_int("NEWS_MAX_HEADLINES", 5),
            news_max_age_hours=_env_float("NEWS_MAX_AGE_HOURS", 24.0),
            risk=RiskConfig.from_env(
                profile, equity=_env_float("STARTING_CASH", profile.starting_cash)
            ),
            llm=LLMConfig.from_env(),
        )
        overrides = {k: v for k, v in overrides.items() if v is not None}
        return replace(cfg, **overrides) if overrides else cfg

    def require_live_credentials(self, need_llm: bool = True) -> None:
        """Raise if anything needed to place a paper order is absent.

        ``need_llm`` is False for jobs that only read market data (downloading a
        backtest dataset, for instance), which should not demand a model key.
        """
        missing: list[str] = []
        if self.broker == "alpaca":
            missing += [
                name
                for name, value in (
                    ("ALPACA_API_KEY", self.alpaca_key),
                    ("ALPACA_SECRET_KEY", self.alpaca_secret),
                )
                if not value
            ]
        if need_llm and self.strategy == "claude" and not self.llm.api_key:
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            raise ConfigError(
                "Missing required credentials: "
                + ", ".join(missing)
                + ". Set them as GitHub Actions secrets or environment variables."
            )
        if self.broker == "alpaca" and self.feed not in {"iex", "sip", "otc"}:
            raise ConfigError(f"Unknown ALPACA_FEED {self.feed}, expected iex or sip")
