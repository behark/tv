# Multi-Tier Trading Bot System

A sophisticated trading signal monitoring system featuring four distinct indicator tiers, each optimized for different trading approaches and win rate targets.

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
├── tier4_high_winrate/  # Tier 4 - High win rate (75-85%)
├── trading-bot/         # Original bot (legacy)
└── launch_*.sh          # Launcher scripts
```

## Tier Overview

| Tier | Focus | Expected Win Rate | R:R Ratio |
|------|-------|-------------------|-----------|
| Tier 1 | Institutional Flow | 55-65% | 2:1 |
| Tier 2 | Advanced Technical | 50-60% | 2.5:1 |
| Tier 3 | Smart Money (ICT) | 60-70% | 3:1 |
| **Tier 4** | **High Win Rate** | **75-85%** | **1.5:1** |

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

### Tier 4 - High Win Rate (NEW)
Focus: Maximum probability setups through multi-confluence

**Strategy:** Mean Reversion with Multiple Confirmations

**Entry Requirements:**
- RSI at extreme levels (< 30 or > 70)
- Price touching Bollinger Band
- Stochastic RSI confirmation
- Volume spike or climax
- Near support/resistance level
- RSI divergence (bonus)
- Prefer ranging market regime

**Indicators:**
- RSI with extreme detection
- Bollinger Bands (20, 2.0)
- Stochastic RSI
- Volume analysis (spike/climax detection)
- Support/Resistance levels
- Market regime detection (ADX-based)
- RSI divergence
- Multi-timeframe EMAs (21, 50, 200)

**Best for:** Traders who prioritize high win rate over R:R ratio

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

# Tier 4 - High Win Rate
./launch_tier4.sh
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
cd tier4_high_winrate && python bot.py
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

All four tiers are configured to monitor the same default pairs:
- BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, DOGE/USDT
- ADA/USDT, AVAX/USDT, LINK/USDT, DOT/USDT, MATIC/USDT

This allows you to compare signal quality across different indicator approaches.

## Logs

Each tier creates its own log file:
- `tier1_institutional.log`
- `tier2_advanced.log`
- `tier3_smart_money.log`
- `tier4_high_winrate.log`

And SQLite databases:
- `tier1_signals.db`
- `tier2_signals.db`
- `tier3_signals.db`
- `tier4_signals.db`

## Expected Performance Comparison

| Metric | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|--------|--------|--------|--------|--------|
| Win Rate | 55-65% | 50-60% | 60-70% | 75-85% |
| R:R Ratio | 2:1 | 2.5:1 | 3:1 | 1.5:1 |
| Signal Frequency | Medium | High | Low | Medium |
| Best Market | Trending | All | Trending | Ranging |

## License

MIT License
