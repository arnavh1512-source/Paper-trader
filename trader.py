#!/usr/bin/env python3
"""
Claude Automated Paper Trader — GitHub Actions version
Runs once per execution. GitHub Actions handles the 15-minute schedule.
Credentials are read from environment variables (GitHub Secrets).
"""

import os
import json
import logging
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
# Credentials come from GitHub Secrets — never put real keys here
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYMBOLS               = ["AAPL", "TSLA"]   # stocks to trade
MAX_DOLLARS_PER_TRADE = 500                # max $ per trade
DRY_RUN               = False              # set True to test without placing orders

ALPACA_BASE = "https://paper-api.alpaca.markets/v2"
ET = ZoneInfo("America/New_York")

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("claude_trader")


# ─────────────────────────────────────────────
#  ALPACA HELPERS
# ─────────────────────────────────────────────
def alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }

def alpaca_get(path):
    r = requests.get(f"{ALPACA_BASE}{path}", headers=alpaca_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

def alpaca_post(path, body):
    r = requests.post(f"{ALPACA_BASE}{path}", headers=alpaca_headers(),
                      json=body, timeout=10)
    r.raise_for_status()
    return r.json()

def get_account():
    return alpaca_get("/account")

def get_positions():
    return alpaca_get("/positions")

def get_bars(symbol, limit=20):
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
        headers=alpaca_headers(),
        params={"timeframe": "15Min", "limit": limit, "feed": "iex"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("bars", [])

def get_latest_quote(symbol):
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest",
        headers=alpaca_headers(),
        params={"feed": "iex"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("quote", {})

def is_market_open():
    clock = alpaca_get("/clock")
    return clock.get("is_open", False)

def place_order(symbol, side, dollars):
    quote = get_latest_quote(symbol)
    ask = float(quote.get("ap", 0) or quote.get("bp", 0))
    if ask <= 0:
        log.warning(f"Could not get price for {symbol}, skipping.")
        return None
    body = {
        "symbol": symbol,
        "notional": str(round(dollars, 2)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    log.info(f"Placing {side.upper()} order: ${dollars} of {symbol} @ ~${ask:.2f}")
    return alpaca_post("/orders", body)


# ─────────────────────────────────────────────
#  CLAUDE DECISION ENGINE
# ─────────────────────────────────────────────
def build_market_snapshot(symbol, account, positions):
    bars  = get_bars(symbol, limit=20)
    quote = get_latest_quote(symbol)
    position = next((p for p in positions if p["symbol"] == symbol), None)

    closes = [b["c"] for b in bars] if bars else []
    sma5   = round(sum(closes[-5:]) / 5, 2)       if len(closes) >= 5  else "N/A"
    sma20  = round(sum(closes) / len(closes), 2)  if closes            else "N/A"
    price_change_pct = round((closes[-1] - closes[-5]) / closes[-5] * 100, 2) if len(closes) >= 5 else "N/A"

    bars_summary = "\n".join(
        f"  {b['t'][:16]}  O:{b['o']:.2f} H:{b['h']:.2f} L:{b['l']:.2f} C:{b['c']:.2f} V:{b['v']}"
        for b in bars[-8:]
    )

    pos_summary = "None"
    if position:
        pos_summary = (
            f"{position['qty']} shares | avg entry ${float(position['avg_entry_price']):.2f} "
            f"| market value ${float(position['market_value']):.2f} "
            f"| unrealized P&L ${float(position['unrealized_pl']):.2f} "
            f"({float(position['unrealized_plpc'])*100:.2f}%)"
        )

    return f"""
=== {symbol} Market Snapshot ===
Current Ask:    ${float(quote.get('ap', 0)):.2f}
Current Bid:    ${float(quote.get('bp', 0)):.2f}
5-bar SMA:      ${sma5}
20-bar SMA:     ${sma20}
Price change (last 5 bars): {price_change_pct}%

Recent 15-min bars (newest last):
{bars_summary}

Current position in {symbol}: {pos_summary}

Account:
  Portfolio value: ${float(account['portfolio_value']):.2f}
  Cash available:  ${float(account['cash']):.2f}
  Buying power:    ${float(account['buying_power']):.2f}
  Max per trade:   ${MAX_DOLLARS_PER_TRADE}
"""

def ask_claude(symbol, snapshot):
    system = """You are an expert quantitative trader managing a paper trading portfolio.
Analyze the market data and make a trading decision.

Respond ONLY with valid JSON — no explanation, no markdown:
{
  "action": "buy" | "sell" | "hold",
  "confidence": 1-10,
  "dollars": <number or null>,
  "reason": "<one sentence>"
}

Rules:
- Only trade if confidence >= 7
- "dollars" is how much to spend/sell (max is MAX_DOLLARS_PER_TRADE), null for hold
- Be conservative. Preserve capital.
- Never buy if insufficient cash. Never sell without a position.
"""
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 256,
            "system": system,
            "messages": [{"role": "user", "content": f"Analyze and decide for {symbol}:\n{snapshot}"}],
        },
        timeout=30,
    )
    r.raise_for_status()
    text = r.json()["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ─────────────────────────────────────────────
#  MAIN — runs once per GitHub Actions trigger
# ─────────────────────────────────────────────
def main():
    log.info(f"Claude Paper Trader starting — {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")

    if not ALPACA_API_KEY or not ANTHROPIC_API_KEY:
        log.error("Missing credentials! Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY as GitHub Secrets.")
        return

    if not is_market_open():
        log.info("Market is closed. Nothing to do.")
        return

    account   = get_account()
    positions = get_positions()
    log.info(f"Portfolio: ${float(account['portfolio_value']):.2f} | Cash: ${float(account['cash']):.2f}")

    for symbol in SYMBOLS:
        log.info(f"--- Analyzing {symbol} ---")
        try:
            snapshot = build_market_snapshot(symbol, account, positions)
            decision = ask_claude(symbol, snapshot)

            action     = decision.get("action", "hold").lower()
            confidence = decision.get("confidence", 0)
            dollars    = decision.get("dollars") or 0
            reason     = decision.get("reason", "")

            log.info(f"Claude: {action.upper()} | confidence: {confidence}/10 | ${dollars} | {reason}")

            if DRY_RUN:
                log.info("[DRY RUN] No order placed.")
                continue

            if action == "buy" and confidence >= 7 and dollars > 0:
                cash = float(account["cash"])
                trade_dollars = min(dollars, MAX_DOLLARS_PER_TRADE, cash * 0.95)
                if trade_dollars < 1:
                    log.warning("Insufficient cash, skipping.")
                    continue
                order = place_order(symbol, "buy", trade_dollars)
                if order:
                    log.info(f"BUY order placed: {order.get('id', '')[:12]}...")

            elif action == "sell" and confidence >= 7 and dollars > 0:
                position = next((p for p in positions if p["symbol"] == symbol), None)
                if not position:
                    log.info(f"No position in {symbol}, skipping sell.")
                    continue
                trade_dollars = min(dollars, MAX_DOLLARS_PER_TRADE, float(position["market_value"]))
                if trade_dollars < 1:
                    log.warning("Position too small, skipping.")
                    continue
                order = place_order(symbol, "sell", trade_dollars)
                if order:
                    log.info(f"SELL order placed: {order.get('id', '')[:12]}...")
            else:
                log.info(f"HOLD {symbol}")

        except Exception as e:
            log.error(f"Error processing {symbol}: {e}")

    log.info("Done.")

if __name__ == "__main__":
    main()
