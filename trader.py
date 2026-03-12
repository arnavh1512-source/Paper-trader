#!/usr/bin/env python3
"""
Claude Automated Paper Trader — Full Market Edition
Claude picks which stocks to trade AND decides the strategy each day.
Runs once per execution via GitHub Actions.
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
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MAX_DOLLARS_PER_TRADE = 100   # max $ per trade
MAX_POSITIONS         = 5     # max number of stocks to hold at once
DRY_RUN               = False # True = analyze only, no real orders

# Large universe of liquid stocks Claude can choose from
UNIVERSE = [
    # Big Tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    # Finance
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Energy
    "XOM", "CVX",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "GLD",
    # High momentum
    "AMD", "NFLX", "UBER", "SHOP", "COIN", "PLTR", "RBLX",
]

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

def is_market_open():
    return alpaca_get("/clock").get("is_open", False)

def get_bars(symbol, limit=20):
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
        headers=alpaca_headers(),
        params={"timeframe": "15Min", "limit": limit, "feed": "iex"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("bars", [])

def get_snapshots(symbols):
    """Get latest quote + bar for multiple symbols in one call."""
    r = requests.get(
        "https://data.alpaca.markets/v2/stocks/snapshots",
        headers=alpaca_headers(),
        params={"symbols": ",".join(symbols), "feed": "iex"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()

def get_latest_quote(symbol):
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest",
        headers=alpaca_headers(),
        params={"feed": "iex"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("quote", {})

def place_order(symbol, side, dollars):
    quote = get_latest_quote(symbol)
    ask = float(quote.get("ap", 0) or quote.get("bp", 0))
    if ask <= 0:
        log.warning(f"No price for {symbol}, skipping.")
        return None
    body = {
        "symbol": symbol,
        "notional": str(round(min(dollars, MAX_DOLLARS_PER_TRADE), 2)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    log.info(f"Placing {side.upper()} ${dollars} of {symbol} @ ~${ask:.2f}")
    return alpaca_post("/orders", body)


# ─────────────────────────────────────────────
#  CLAUDE API HELPER
# ─────────────────────────────────────────────
def ask_claude(system, prompt, max_tokens=512):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    r.raise_for_status()
    text = r.json()["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# ─────────────────────────────────────────────
#  STEP 1: Claude picks which stocks to focus on
# ─────────────────────────────────────────────
def claude_pick_stocks(account, positions, snapshots):
    current_holdings = [p["symbol"] for p in positions]
    cash = float(account["cash"])

    # Build a quick market overview from snapshots
    market_overview = []
    for sym, data in snapshots.items():
        try:
            daily = data.get("dailyBar", {})
            prev  = data.get("prevDailyBar", {})
            if not daily or not prev:
                continue
            close     = float(daily.get("c", 0))
            prev_close = float(prev.get("c", 1))
            change_pct = round((close - prev_close) / prev_close * 100, 2)
            volume    = int(daily.get("v", 0))
            market_overview.append(f"{sym}: ${close:.2f} ({'+' if change_pct>=0 else ''}{change_pct}%) vol:{volume:,}")
        except:
            continue

    system = """You are an expert stock picker. Given a market snapshot, pick the best 3-5 stocks to analyze for trading RIGHT NOW.
Consider: momentum, volume, volatility, and opportunity.

Respond ONLY with valid JSON — no explanation:
{
  "symbols": ["SYM1", "SYM2", "SYM3"],
  "strategy": "one sentence describing today's market strategy",
  "market_mood": "bullish" | "bearish" | "neutral"
}"""

    prompt = f"""Today is {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}.
Cash available: ${cash:.2f}
Currently holding: {current_holdings if current_holdings else 'nothing'}
Max positions allowed: {MAX_POSITIONS}

Market snapshot for {len(market_overview)} stocks:
{chr(10).join(market_overview)}

Pick the best 3-5 symbols to trade today from this universe: {', '.join(UNIVERSE)}"""

    response = ask_claude(system, prompt, max_tokens=256)
    return json.loads(response)


# ─────────────────────────────────────────────
#  STEP 2: Claude analyzes each stock and decides
# ─────────────────────────────────────────────
def build_snapshot(symbol, account, positions):
    bars = get_bars(symbol, limit=20)
    quote = get_latest_quote(symbol)
    position = next((p for p in positions if p["symbol"] == symbol), None)

    closes = [b["c"] for b in bars] if bars else []
    sma5   = round(sum(closes[-5:]) / 5, 2)      if len(closes) >= 5 else "N/A"
    sma20  = round(sum(closes) / len(closes), 2) if closes           else "N/A"
    change = round((closes[-1] - closes[-5]) / closes[-5] * 100, 2) if len(closes) >= 5 else "N/A"

    bars_text = "\n".join(
        f"  {b['t'][:16]}  O:{b['o']:.2f} H:{b['h']:.2f} L:{b['l']:.2f} C:{b['c']:.2f} V:{b['v']}"
        for b in bars[-8:]
    )

    pos_text = "None"
    if position:
        pos_text = (
            f"{position['qty']} shares | avg ${float(position['avg_entry_price']):.2f} "
            f"| value ${float(position['market_value']):.2f} "
            f"| P&L ${float(position['unrealized_pl']):.2f} ({float(position['unrealized_plpc'])*100:.2f}%)"
        )

    return f"""
=== {symbol} ===
Ask: ${float(quote.get('ap',0)):.2f} | Bid: ${float(quote.get('bp',0)):.2f}
5-bar SMA: ${sma5} | 20-bar SMA: ${sma20} | Change (5 bars): {change}%
Recent bars:
{bars_text}
Position: {pos_text}
Cash: ${float(account['cash']):.2f} | Max per trade: ${MAX_DOLLARS_PER_TRADE}
"""

def claude_trade_decision(symbol, snapshot, strategy):
    system = f"""You are an expert trader. Today's strategy: {strategy}
Analyze the data and decide what to do.

Respond ONLY with valid JSON:
{{
  "action": "buy" | "sell" | "hold",
  "confidence": 1-10,
  "dollars": <number or null>,
  "reason": "<one sentence>"
}}

Rules: Only trade if confidence >= 7. Be conservative. Never buy without cash. Never sell without a position."""

    response = ask_claude(system, f"Decide for {symbol}:\n{snapshot}", max_tokens=200)
    return json.loads(response)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    log.info(f"Claude Paper Trader (Full Market) — {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")

    if not ALPACA_API_KEY or not ANTHROPIC_API_KEY:
        log.error("Missing credentials in GitHub Secrets!")
        return

    if not is_market_open():
        log.info("Market is closed. Nothing to do.")
        return

    account   = get_account()
    positions = get_positions()
    log.info(f"Portfolio: ${float(account['portfolio_value']):.2f} | Cash: ${float(account['cash']):.2f} | Positions: {len(positions)}")

    # Step 1: Get market snapshots for the full universe
    log.info(f"Fetching market data for {len(UNIVERSE)} stocks...")
    try:
        snapshots = get_snapshots(UNIVERSE)
    except Exception as e:
        log.error(f"Failed to fetch snapshots: {e}")
        snapshots = {}

    # Step 2: Claude picks which stocks to trade today
    log.info("Asking Claude to pick today's stocks...")
    try:
        picks = claude_pick_stocks(account, positions, snapshots)
        symbols   = picks.get("symbols", ["AAPL", "TSLA"])
        strategy  = picks.get("strategy", "balanced momentum strategy")
        mood      = picks.get("market_mood", "neutral")
        log.info(f"Claude's picks: {symbols}")
        log.info(f"Strategy: {strategy}")
        log.info(f"Market mood: {mood}")
    except Exception as e:
        log.error(f"Stock picking failed: {e}, falling back to defaults")
        symbols  = ["AAPL", "TSLA", "NVDA"]
        strategy = "default momentum strategy"

    # Also include any symbols we currently hold (to manage exits)
    held = [p["symbol"] for p in positions if p["symbol"] not in symbols]
    if held:
        log.info(f"Also reviewing held positions: {held}")
        symbols = symbols + held

    # Step 3: Analyze each symbol and trade
    for symbol in symbols:
        log.info(f"--- {symbol} ---")
        try:
            snapshot = build_snapshot(symbol, account, positions)
            decision = claude_trade_decision(symbol, snapshot, strategy)

            action     = decision.get("action", "hold").lower()
            confidence = decision.get("confidence", 0)
            dollars    = decision.get("dollars") or 0
            reason     = decision.get("reason", "")

            log.info(f"Claude: {action.upper()} | confidence: {confidence}/10 | ${dollars} | {reason}")

            if DRY_RUN:
                log.info("[DRY RUN] No order placed.")
                continue

            if action == "buy" and confidence >= 7 and dollars > 0:
                if len(positions) >= MAX_POSITIONS:
                    log.info(f"Max positions ({MAX_POSITIONS}) reached, skipping buy.")
                    continue
                cash = float(account["cash"])
                trade_dollars = min(dollars, MAX_DOLLARS_PER_TRADE, cash * 0.95)
                if trade_dollars < 1:
                    log.warning("Not enough cash.")
                    continue
                order = place_order(symbol, "buy", trade_dollars)
                if order:
                    log.info(f"BUY order placed: {order.get('id','')[:12]}...")

            elif action == "sell" and confidence >= 7:
                position = next((p for p in positions if p["symbol"] == symbol), None)
                if not position:
                    log.info(f"No position in {symbol}, skipping sell.")
                    continue
                trade_dollars = min(dollars or MAX_DOLLARS_PER_TRADE, float(position["market_value"]))
                if trade_dollars < 1:
                    log.warning("Position too small.")
                    continue
                order = place_order(symbol, "sell", trade_dollars)
                if order:
                    log.info(f"SELL order placed: {order.get('id','')[:12]}...")
            else:
                log.info(f"HOLD {symbol}")

        except Exception as e:
            log.error(f"Error on {symbol}: {e}")

    log.info("Done.")

if __name__ == "__main__":
    main()
