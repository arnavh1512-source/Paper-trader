"""Configuration: defaults that follow the market, and limits that cannot be
configured into an incoherent state."""

from __future__ import annotations

import pytest

from claude_trader.config import AppConfig, LLMConfig, RiskConfig
from claude_trader.errors import ConfigError
from claude_trader.markets import INDIA_MARKET, US_MARKET


# ------------------------------------------------------------- market wiring
def test_india_is_the_default_market():
    assert AppConfig().market == "in"


def test_blanks_are_filled_from_the_profile():
    config = AppConfig(market="in")
    assert config.universe == INDIA_MARKET.universe
    assert config.timeframe == "15m"
    assert config.starting_cash == 100_000.0
    assert config.currency == "INR"
    assert config.benchmark == "NIFTYBEES"
    assert config.money(1_234_567.89) == "₹12,34,567.89"


def test_broker_defaults_to_the_one_that_serves_the_market():
    assert AppConfig(market="in").broker == "paper"
    assert AppConfig(market="us").broker == "alpaca"


def test_alpaca_cannot_be_pointed_at_nse():
    with pytest.raises(ConfigError, match="Alpaca broker only serves the US"):
        AppConfig(market="in", broker="alpaca")


def test_paper_broker_is_allowed_on_the_us_market():
    assert AppConfig(market="us", broker="paper").broker == "paper"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"market": "uk"}, "unknown MARKET"),
        ({"strategy": "vibes"}, "STRATEGY must be"),
        ({"broker": "zerodha"}, "BROKER must be"),
        ({"segment": "swing"}, "TRADE_SEGMENT must be"),
    ],
)
def test_invalid_configuration_is_rejected_at_construction(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        AppConfig(**kwargs)


def test_explicit_values_are_not_overwritten_by_the_profile():
    config = AppConfig(
        market="in", universe=("TCS",), timeframe="5m", starting_cash=50_000.0
    )
    assert config.universe == ("TCS",)
    assert config.timeframe == "5m"
    assert config.starting_cash == 50_000.0


# -------------------------------------------------------- intraday coherence
def test_intraday_forces_square_off():
    """Holding overnight while billing intraday rates would flatter every
    backtest, so the combination is corrected rather than trusted."""
    config = AppConfig(
        market="in", segment="intraday", risk=RiskConfig(square_off_enabled=False)
    )
    assert config.risk.square_off_enabled is True


def test_intraday_caps_the_holding_period_to_one_session():
    config = AppConfig(
        market="in", segment="intraday", risk=RiskConfig(max_holding_bars=130)
    )
    assert config.risk.max_holding_bars < INDIA_MARKET.bars_per_session
    assert config.risk.max_holding_bars == 23


def test_delivery_leaves_the_holding_period_alone():
    config = AppConfig(
        market="in", segment="delivery", risk=RiskConfig(max_holding_bars=130)
    )
    assert config.risk.max_holding_bars == 130
    assert config.risk.square_off_enabled is False


def test_us_is_not_subject_to_the_nse_segment_rules():
    config = AppConfig(
        market="us", broker="paper", risk=RiskConfig(max_holding_bars=130)
    )
    assert config.risk.max_holding_bars == 130
    assert config.risk.square_off_enabled is False


# ------------------------------------------------------------------- derived
def test_periods_per_year_counts_bars_not_wall_clock():
    """A 15-minute bar is not 1/96th of a day: markets are shut most of the
    time. Annualising on wall-clock time inflates the figure ~5x."""
    assert AppConfig(market="in").periods_per_year == 25 * 250
    assert AppConfig(market="us", broker="paper").periods_per_year == 26 * 250


def test_paper_detection():
    assert AppConfig(market="in").is_paper is True
    assert AppConfig(market="us").is_paper is True  # paper-api endpoint
    live = AppConfig(market="us", alpaca_base="https://api.alpaca.markets/v2")
    assert live.is_paper is False


# ------------------------------------------------------------------ from_env
def test_from_env_reads_the_environment(monkeypatch):
    monkeypatch.setenv("MARKET", "us")
    monkeypatch.setenv("BROKER", "paper")
    monkeypatch.setenv("STRATEGY", "momentum")
    monkeypatch.setenv("UNIVERSE", "aapl, msft ,aapl")
    monkeypatch.setenv("DRY_RUN", "yes")
    monkeypatch.setenv("VERBOSE", "1")
    config = AppConfig.from_env()
    assert config.market == "us"
    assert config.strategy == "momentum"
    assert config.universe == ("AAPL", "MSFT")  # upper-cased, de-duplicated
    assert config.dry_run is True
    assert config.verbose is True


def test_explicit_arguments_beat_the_environment(monkeypatch):
    monkeypatch.setenv("MARKET", "us")
    assert AppConfig.from_env(market="in").market == "in"


def test_none_overrides_are_ignored(monkeypatch):
    """The CLI passes None for flags the user did not give, which must leave
    the environment in charge rather than blanking it."""
    monkeypatch.setenv("STRATEGY", "momentum")
    config = AppConfig.from_env(strategy=None, journal_path=None)
    assert config.strategy == "momentum"
    assert config.journal_path == "data/journal.sqlite3"


def test_from_env_rejects_an_unknown_market(monkeypatch):
    monkeypatch.setenv("MARKET", "mars")
    with pytest.raises(ConfigError, match="unknown MARKET"):
        AppConfig.from_env()


def test_numeric_environment_variables_must_be_numeric(monkeypatch):
    monkeypatch.setenv("STARTING_CASH", "a lot")
    with pytest.raises(ConfigError, match="must be a number"):
        AppConfig.from_env()
    monkeypatch.delenv("STARTING_CASH")
    monkeypatch.setenv("MAX_POSITIONS", "several")
    with pytest.raises(ConfigError, match="must be an integer"):
        AppConfig.from_env()


# ------------------------------------------------------------------ risk cfg
def test_risk_defaults_follow_the_market():
    india = RiskConfig.from_env(INDIA_MARKET)
    us = RiskConfig.from_env(US_MARKET)
    assert india.max_notional_per_trade == 10_000.0   # rupees
    assert us.max_notional_per_trade == 100.0         # dollars
    assert india.min_trade_notional == 500.0
    assert us.min_trade_notional == 1.0
    assert india.square_off_enabled is True
    assert us.square_off_enabled is False
    assert india.max_holding_bars == 23                # one session, less a bit


def test_risk_environment_overrides_are_honoured(monkeypatch):
    monkeypatch.setenv("MIN_CONFIDENCE", "9")
    monkeypatch.setenv("MAX_COST_RATIO", "0.1")
    risk = RiskConfig.from_env(INDIA_MARKET)
    assert risk.min_confidence == 9
    assert risk.max_cost_ratio == 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_notional_per_trade": 0},
        {"max_position_pct": 1.5},
        {"risk_per_trade_pct": 0.9},
        {"min_cash_reserve_pct": 1.0},
        {"atr_stop_multiple": 0},
        {"atr_target_multiple": 1.0, "atr_stop_multiple": 2.0},
        {"max_positions": 0},
        {"max_sector_positions": 0},
        {"max_correlation": 0},
        {"min_confidence": 11},
        {"daily_loss_limit_pct": 0},
        {"max_cost_ratio": 0},
        {"square_off_minutes_before_close": -1},
    ],
)
def test_incoherent_risk_limits_are_rejected(kwargs):
    with pytest.raises(ConfigError, match="Invalid RiskConfig"):
        RiskConfig(**kwargs)


def test_risk_config_is_recorded_with_every_run():
    """A result is only reproducible if the limits that produced it were
    stored next to it."""
    data = RiskConfig().as_dict()
    assert data["min_confidence"] == 7
    assert "max_cost_ratio" in data


# ------------------------------------------------------------------ secrets
def test_missing_credentials_are_named(monkeypatch):
    config = AppConfig(market="us", strategy="claude")
    with pytest.raises(ConfigError) as exc:
        config.require_live_credentials()
    message = str(exc.value)
    assert "ALPACA_API_KEY" in message and "ANTHROPIC_API_KEY" in message


def test_data_only_jobs_do_not_need_a_model_key():
    config = AppConfig(market="in", strategy="claude")
    config.require_live_credentials(need_llm=False)  # paper broker, no keys needed


def test_indian_paper_trading_needs_no_alpaca_keys():
    config = AppConfig(market="in", strategy="momentum")
    config.require_live_credentials()


def test_unknown_alpaca_feed_is_rejected():
    config = AppConfig(
        market="us",
        strategy="momentum",
        alpaca_key="k",
        alpaca_secret="s",
        feed="delayed",
    )
    with pytest.raises(ConfigError, match="Unknown ALPACA_FEED"):
        config.require_live_credentials()


def test_llm_config_reads_its_own_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-test")
    llm = LLMConfig.from_env()
    assert llm.api_key == "sk-test"
    assert llm.model == "claude-test"
    assert llm.temperature == 0.0


# ------------------------------------------------------- sizing vs the book
def test_a_book_too_small_for_its_own_minimum_ticket_is_refused():
    """The deadlock this guard exists for.

    Every buy is clamped down to the position cap and then rejected if the
    result falls under the minimum ticket. When the floor sits above the cap,
    every order fails that test -- so the bot runs for months placing nothing
    while every component reports itself healthy. Silence is the worst failure
    mode available here, so it has to be an error at startup.
    """
    with pytest.raises(ConfigError, match="no order can ever be placed"):
        AppConfig(
            market="in",
            starting_cash=2_000.0,
            risk=RiskConfig(
                max_position_pct=0.20,     # -> Rs 400
                min_trade_notional=500.0,  # -> floor above the ceiling
            ),
        )


def test_the_per_trade_ceiling_can_deadlock_it_too():
    """The position cap is not the only ceiling; the tighter of the two binds."""
    with pytest.raises(ConfigError, match="no order can ever be placed"):
        AppConfig(
            market="in",
            starting_cash=100_000.0,
            risk=RiskConfig(
                max_position_pct=0.50,
                max_notional_per_trade=200.0,
                min_trade_notional=500.0,
            ),
        )


def test_a_coherent_small_book_is_accepted():
    config = AppConfig(
        market="in",
        starting_cash=2_000.0,
        risk=RiskConfig(
            max_position_pct=0.40,
            max_notional_per_trade=800.0,
            min_trade_notional=100.0,
        ),
    )
    assert config.starting_cash == 2_000.0


def test_sizing_defaults_scale_down_to_a_small_book(monkeypatch):
    """The profile defaults are written for one lakh and deadlock at Rs 2,000,
    so the defaults themselves have to follow the equity."""
    monkeypatch.setenv("MARKET", "in")
    monkeypatch.setenv("STARTING_CASH", "2000")
    risk = AppConfig.from_env().risk
    assert risk.min_trade_notional < 2_000 * risk.max_position_pct, "still deadlocked"
    assert risk.max_notional_per_trade <= 2_000
    assert risk.max_positions < 5, "a Rs 2,000 book cannot hold five names"


def test_the_one_lakh_defaults_are_left_alone(monkeypatch):
    """Scaling for small books must not quietly re-tune the normal case."""
    monkeypatch.setenv("MARKET", "in")
    risk = AppConfig.from_env().risk
    assert risk.max_position_pct == 0.20
    assert risk.max_notional_per_trade == 10_000.0
    assert risk.min_trade_notional == 500.0
    assert risk.max_positions == 5


def test_an_explicit_override_still_wins_over_the_scaled_default(monkeypatch):
    monkeypatch.setenv("MARKET", "in")
    monkeypatch.setenv("STARTING_CASH", "2000")
    monkeypatch.setenv("MAX_POSITION_PCT", "0.25")
    assert AppConfig.from_env().risk.max_position_pct == 0.25
