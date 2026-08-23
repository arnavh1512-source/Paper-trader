"""Loading a local .env.

A settings file only helps if it cannot quietly override a deployment secret,
and only helps if the values it does set never end up in a log line.
"""

from __future__ import annotations

import os

from claude_trader.config import load_dotenv


def test_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_dotenv() == 0


def test_values_are_read_from_the_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "# a comment\n\nSTARTING_CASH=250000\nMAX_POSITIONS=3\n", encoding="utf-8")

    assert load_dotenv() == 2
    assert os.environ["STARTING_CASH"] == "250000"
    assert os.environ["MAX_POSITIONS"] == "3"


def test_an_existing_variable_is_never_overwritten(tmp_path, monkeypatch):
    """A checked-out .env must not be able to shadow a GitHub Actions secret.
    The file is a convenience for a laptop, not a source of truth."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "the-real-one")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=stale\n", encoding="utf-8")

    assert load_dotenv() == 0
    assert os.environ["ANTHROPIC_API_KEY"] == "the-real-one"


def test_quotes_and_trailing_comments_are_stripped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        'MARKET="in"   # NSE\nSTRATEGY=momentum\n', encoding="utf-8")

    load_dotenv()
    assert os.environ["MARKET"] == "in"
    assert os.environ["STRATEGY"] == "momentum"


def test_a_blank_value_is_treated_as_unset(tmp_path, monkeypatch):
    """`.env.example` ships `ANTHROPIC_API_KEY=` as a placeholder. Setting it to
    an empty string would turn 'you forgot the key' into a 401 from the API."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")

    assert load_dotenv() == 0
    assert "ANTHROPIC_API_KEY" not in os.environ
