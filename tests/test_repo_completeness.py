"""A clone of this repository has to be able to import the package.

That sounds too obvious to test. It was not: ``.gitignore`` carried an
unanchored ``data/`` rule, meant for the journal directory at the repo root,
which also matched ``claude_trader/data/`` -- the market-data layer. Five source
files were never committed. Every local checkout still had them, so the suite was
green on the machine that wrote them and every clone was unimportable, including
CI and the scheduled trading run.

The failure mode is specifically invisible to normal testing: the files exist on
disk, imports resolve, coverage reports them. Only a fresh clone disagrees. So
the check has to ask git what it would actually hand to a clone, not the
filesystem.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(scope="module")
def tracked() -> set[Path]:
    try:
        out = _git("ls-files", "-z")
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a checkout
        pytest.skip("not a git checkout, or git is unavailable")
    return {Path(line) for line in out.split("\0") if line}


def _source_files() -> set[Path]:
    return {
        path.relative_to(REPO)
        for directory in ("claude_trader", "tests")
        for path in (REPO / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    }


def test_the_check_sees_a_real_repository(tracked: set[Path]) -> None:
    # Without this, an empty `tracked` set would make the assertions below
    # fail loudly rather than pass vacuously -- but an empty *source* set
    # would pass vacuously, so pin the one that can.
    assert len(_source_files()) > 40
    assert Path("claude_trader/config.py") in tracked


def test_every_source_file_would_reach_a_clone(tracked: set[Path]) -> None:
    missing = sorted(str(p).replace("\\", "/") for p in _source_files() - tracked)
    assert not missing, (
        "these exist on disk but git would not hand them to a clone, so the "
        "package is unimportable anywhere but here. Usually an unanchored "
        f".gitignore rule -- check `git check-ignore -v <path>`: {missing}"
    )
