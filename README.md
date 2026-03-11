# Claude Paper Trader

Automated paper trading bot powered by Claude AI + Alpaca. Runs free on GitHub Actions — no PC required.

## Setup (5 steps)

### 1. Create a GitHub account
Go to [github.com](https://github.com) and sign up for free.

### 2. Create a new repository
- Click the **+** icon top right → **New repository**
- Name it: `claude-trader`
- Set it to **Private**
- Click **Create repository**

### 3. Upload the files
- Click **uploading an existing file**
- Upload both `trader.py` and the `.github/workflows/trader.yml` file
- Click **Commit changes**

### 4. Add your secret keys
- Go to your repo → **Settings** → **Secrets and variables** → **Actions**
- Click **New repository secret** and add these 3 secrets:

| Name | Value |
|------|-------|
| `ALPACA_API_KEY` | Your Alpaca Key ID (e.g. PKXXXXXXXX) |
| `ALPACA_SECRET_KEY` | Your Alpaca Secret Key |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (sk-ant-...) |

### 5. Enable Actions
- Go to your repo → **Actions** tab
- Click **I understand my workflows, go ahead and enable them**

## That's it!
The bot will automatically run every 15 minutes on weekdays during US market hours (9:30am–4pm ET). You can watch it run under the **Actions** tab in your repo.

## Checking trades
- **GitHub Actions tab** → click any run to see Claude's decisions in the logs
- **[app.alpaca.markets](https://app.alpaca.markets)** → Orders / Positions to see actual trades
