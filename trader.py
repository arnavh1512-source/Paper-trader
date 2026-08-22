#!/usr/bin/env python3
"""Entry point for one scheduled trading cycle.

This file used to *be* the bot: a single script that asked Claude what to buy
and bought it. It is now a launcher, and everything it used to do lives in
``claude_trader/`` where each piece can be tested and measured on its own.

Kept as a top-level script because the GitHub Actions workflow calls it by
name, and because ``python trader.py`` is the shortest thing to type.

Equivalent:  python -m claude_trader trade
"""

from __future__ import annotations

import sys

from claude_trader.app import configure_logging, run_live_cycle, summarise
from claude_trader.config import AppConfig
from claude_trader.errors import TraderError


def main() -> int:
    config = AppConfig.from_env()
    configure_logging(verbose=config.verbose)
    try:
        report = run_live_cycle(config)
    except TraderError as exc:
        # A failed cycle is not a crash: the next one runs in fifteen minutes.
        # Exiting non-zero is how the scheduler learns something is wrong.
        print(f"cycle failed: {exc}", file=sys.stderr)
        return 2
    print(summarise(report, config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
