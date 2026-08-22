"""Tradable universe and the sector map used for concentration limits.

The sector tag is deliberately coarse. Its only job is to stop the bot from
holding five names that are really one bet.
"""

from __future__ import annotations

from types import MappingProxyType

# symbol -> coarse sector bucket
SECTORS: MappingProxyType = MappingProxyType(
    {
        # Mega-cap technology
        "AAPL": "tech",
        "MSFT": "tech",
        "NVDA": "semis",
        "AMD": "semis",
        "AMZN": "consumer_disc",
        "GOOGL": "tech",
        "META": "tech",
        "TSLA": "consumer_disc",
        "NFLX": "communication",
        "SHOP": "consumer_disc",
        "UBER": "consumer_disc",
        "RBLX": "communication",
        "PLTR": "tech",
        "COIN": "crypto_beta",
        # Financials
        "JPM": "financials",
        "BAC": "financials",
        "GS": "financials",
        "V": "financials",
        "MA": "financials",
        # Healthcare
        "JNJ": "healthcare",
        "UNH": "healthcare",
        "PFE": "healthcare",
        "ABBV": "healthcare",
        # Energy
        "XOM": "energy",
        "CVX": "energy",
        # Broad-market ETFs. Bucketed together on purpose: SPY/QQQ/DIA/IWM are
        # not diversification relative to one another, nor to mega-cap tech.
        "SPY": "broad_etf",
        "QQQ": "broad_etf",
        "DIA": "broad_etf",
        "IWM": "broad_etf",
        "GLD": "commodity",
    }
)

DEFAULT_UNIVERSE: tuple[str, ...] = tuple(SECTORS.keys())

BENCHMARK_SYMBOL = "SPY"


def sector_of(symbol: str) -> str:
    """Sector bucket for ``symbol``; unknown symbols get their own bucket so an
    unmapped ticker is never silently treated as diversifying."""
    return SECTORS.get(symbol.upper(), f"unknown:{symbol.upper()}")
