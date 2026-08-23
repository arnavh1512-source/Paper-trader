"""The autouse ``clean_env`` fixture is only as good as the list it strips.

Every variable the package reads has to appear in ``conftest.ENV_VARS``, or a
test that asserts a default silently starts asserting whatever is in the
developer's shell -- or in the ``.env`` sitting in the checkout. That failure
mode is quiet: the suite goes green on one machine and red on another, and the
red one is usually CI, hours later.

This happened. ``NEWS_ENABLED`` was added to ``config.py`` and not to
``ENV_VARS``, and the moment a real ``.env`` switched news on, a test asserting
"news is off by default" began reading that file. The guard below is cheaper
than remembering.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.conftest import ENV_VARS

PACKAGE = Path(__file__).resolve().parent.parent / "claude_trader"

# ``_env_str("NAME", ...)`` and friends, tolerating the line break black
# inserts after the opening paren when the default is long.
READS_ENV = re.compile(r"_env_[a-z]+\(\s*\"([A-Z][A-Z0-9_]*)\"")


def _declared_env_vars() -> set[str]:
    found: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        found.update(READS_ENV.findall(path.read_text(encoding="utf-8")))
    return found


def test_the_package_reads_env_vars_at_all() -> None:
    # If the regex stops matching, every other assertion here passes vacuously.
    declared = _declared_env_vars()
    assert len(declared) > 30, f"suspiciously few env vars found: {sorted(declared)}"
    assert "MARKET" in declared


def test_every_env_var_the_package_reads_is_stripped_between_tests() -> None:
    missing = sorted(_declared_env_vars() - set(ENV_VARS))
    assert not missing, (
        "these are read by claude_trader but not cleared by the clean_env "
        f"fixture, so a shell or .env value can leak into a test: {missing}"
    )


def test_env_vars_has_no_duplicates() -> None:
    duplicated = sorted({n for n in ENV_VARS if ENV_VARS.count(n) > 1})
    assert not duplicated, f"listed twice in ENV_VARS: {duplicated}"
