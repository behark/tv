# Multi-Tier Trading Bot System

A sophisticated trading signal monitoring system featuring three distinct indicator tiers, each optimized for different trading approaches.

## Architecture

```
tv/
├── shared/              # Common modules used by all tiers
│   ├── base_bot.py      # Base bot class
│   ├── base_strategy.py # Base strategy class
│   ├── base_indicators.py # Common indicator calculations
│   ├── data_client.py   # Exchange data fetching
│   ├── notifier.py      # Telegram notifications
│   ├── signal_database.py # SQLite signal storage
│   ├── signal_tracker.py # TP/SL monitoring
│   ├── accuracy_filters.py # Signal quality filters
│   └── heartbeat.py     # Health monitoring
├── tier1_institutional/ # Tier 1 - Institutional flow
├── tier2_advanced/      # Tier 2 - Advanced technicals
├── tier3_smart_money/   # Tier 3 - Smart money concepts
├── trading-bot/         # Original bot (legacy)
└── launch_*.sh          # Launcher scripts
```

## Tier Overview

### Tier 1 - Institutional Trading
Focus: Institutional order flow and volume analysis

**Indicators:**
- VWAP with standard deviation bands
- Cumulative Volume Delta (CVD)
- Ichimoku Cloud (Tenkan, Kijun, Senkou spans)
- On-Balance Volume (OBV)
- Chaikin Money Flow (CMF)
- EMA Ribbon (8, 13, 21, 34, 55)
- RSI Divergence detection

**Best for:** Identifying institutional buying/selling pressure

### Tier 2 - Advanced Technical
Focus: Momentum and volatility analysis

**Indicators:**
- Stochastic RSI
- Fisher Transform
- FRAMA (Fractal Adaptive Moving Average)
- Waddah Attar Explosion
- Squeeze Momentum (Bollinger + Keltner)
- Half Trend
- Keltner Channels

**Best for:** Catching momentum shifts and squeeze breakouts

### Tier 3 - Smart Money Concepts
Focus: ICT-style institutional trading concepts

**Indicators:**
- Order Blocks (bullish/bearish)
- Fair Value Gaps (FVG)
- Liquidity Zones
- Market Structure (BOS/CHoCH)
- Premium/Discount Zones
- Swing highs/lows detection

**Best for:** Trading around institutional order flow and liquidity

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

Or create a `.env` file in the root directory.

### 3. Configure Trading Pairs
Edit the `config.yaml` in each tier directory to customize:
- Trading pairs
- Score thresholds
- Indicator weights
- Risk management settings

## Running the Bots

### Single Tier
```bash
# Tier 1 - Institutional
./launch_tier1.sh

# Tier 2 - Advanced Technical
./launch_tier2.sh

# Tier 3 - Smart Money
./launch_tier3.sh
```

### All Tiers (using tmux)
```bash
./launch_all.sh
```

This creates a tmux session with separate windows for each tier.

### Direct Python Execution
```bash
cd tier1_institutional && python bot.py
cd tier2_advanced && python bot.py
cd tier3_smart_money && python bot.py
```

## Features

### Signal Tracking
- SQLite database storage for all signals
- Automatic TP/SL hit detection
- P&L calculation and tracking
- Signal expiry management
- Performance statistics

### Telegram Notifications
- Real-time signal alerts
- TP/SL hit notifications
- Daily performance summaries
- Heartbeat status messages
- Error alerts

### Accuracy Filters
- Minimum confidence scoring
- Volume confirmation
- Volatility filtering
- Multi-timeframe alignment

### Risk Management
- Configurable score thresholds
- ATR-based stop losses
- Risk/reward ratio settings
- Signal cooldown periods

## Configuration

Each tier has its own `config.yaml` with:

```yaml
telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"

pairs:
  - BTC/USDT
  - ETH/USDT
  # ... more pairs

strategy:
  score_threshold: 4.0
  risk_reward_ratio: 1.5
  weights:
    indicator1: 2.0
    indicator2: 1.5
    # ... indicator weights
```

## Monitoring Same Pairs Across Tiers

All three tiers are configured to monitor the same default pairs:
- BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, DOGE/USDT
- ADA/USDT, AVAX/USDT, LINK/USDT, DOT/USDT, MATIC/USDT

This allows you to compare signal quality across different indicator approaches.

## Logs

Each tier creates its own log file:
- `tier1_institutional.log`
- `tier2_advanced.log`
- `tier3_smart_money.log`

And SQLite databases:
- `tier1_signals.db`
- `tier2_signals.db`
- `tier3_signals.db`

## License

MIT License
