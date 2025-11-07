# Confluence Trading Bot - Local Python Edition

A fully autonomous local Python trading bot that monitors cryptocurrency pairs on MEXC Futures using a multi-indicator confluence strategy. The bot sends real-time trading alerts to your Telegram with entry/exit prices, stop loss, take profit, and confidence scores.

## Features

- **Autonomous Monitoring**: Runs 24/7 on your local machine, continuously monitoring 8 crypto pairs on MEXC Futures
- **Multi-Indicator Confluence Strategy**: Combines EMA trend filter, RSI momentum, MACD direction, ADX strength, and volume confirmation
- **Real-Time Telegram Alerts**: Sends formatted alerts with entry price, stop loss, take profit, risk/reward ratio, and confidence score
- **No TradingView Required**: Fetches live data directly from MEXC exchange using ccxt library
- **Configurable Parameters**: All strategy parameters can be customized via config.yaml
- **Bar-Close Logic**: Only triggers alerts on completed 15-minute candles to avoid false signals
- **Duplicate Prevention**: Tracks last bar time per symbol to prevent duplicate alerts

## Strategy Overview

The bot uses the same confluence strategy as the TradingView Pine Script version:

### Indicators Used

1. **EMA (200)** - Trend filter: Long trades above EMA, short trades below EMA
2. **RSI (14)** - Momentum trigger: Identifies oversold/overbought conditions
3. **MACD (12, 26, 9)** - Direction confirmation: Confirms trend direction and momentum
4. **ADX (14)** - Trend strength filter: Only trades when trend is strong (ADX > 20)
5. **Volume** - Confirmation filter: Adds weight when volume is above average
6. **ATR (14)** - Risk management: Dynamic stop loss and take profit based on volatility

### Confluence Scoring System

Each indicator contributes points when its conditions are met:
- **Trend (EMA)**: 1.0 point
- **RSI**: 1.0 point
- **MACD**: 1.0 point
- **ADX**: 1.0 point
- **Volume**: 0.5 points

**Default Entry Threshold**: 2.5 points (requires 2-3 indicators to align)

### Risk Management

- **Stop Loss**: Entry ± (ATR × 1.5)
- **Take Profit**: Entry ± (ATR × 1.5 × 1.5) = 1.5:1 risk/reward ratio
- **ATR-based**: Adapts to market volatility automatically

## Requirements

- Python 3.10 or higher
- Linux (tested on Kubuntu, should work on any Linux distribution)
- Internet connection for exchange data and Telegram notifications
- Telegram bot token and chat ID

## Installation

### 1. Clone the Repository

```bash
cd /path/to/tv/trading-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Telegram Bot

If you haven't already created a Telegram bot:

1. Open Telegram and search for @BotFather
2. Send `/newbot` and follow the prompts
3. Copy the bot token
4. Send a message to your bot
5. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` to get your chat ID

### 4. Set Up Environment Variables

Copy the example .env file and add your credentials:

```bash
cp .env.example .env
nano .env
```

Edit `.env` and add your Telegram credentials:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 5. Configure Trading Pairs and Strategy

Edit `config.yaml` to customize:
- Trading pairs to monitor
- Strategy parameters (EMA length, RSI levels, etc.)
- Confluence weights and threshold
- Risk management settings (ATR multiplier, R:R ratio)

## Usage

### Start the Bot

```bash
python3 bot.py
```

Or if using a virtual environment:

```bash
source venv/bin/activate
python3 bot.py
```

### Stop the Bot

Press `Ctrl+C` to stop the bot gracefully.

### Run in Background

To run the bot in the background:

```bash
nohup python3 bot.py > bot_output.log 2>&1 &
```

To stop the background process:

```bash
pkill -f bot.py
```

### Auto-Start on Boot (Optional)

Create a systemd service to auto-start the bot on system boot:

1. Create a service file:

```bash
sudo nano /etc/systemd/system/trading-bot.service
```

2. Add the following content (adjust paths as needed):

```ini
[Unit]
Description=Confluence Trading Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/tv/trading-bot
ExecStart=/usr/bin/python3 /path/to/tv/trading-bot/bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

4. Check status:

```bash
sudo systemctl status trading-bot
```

## Configuration

### config.yaml

The main configuration file controls all bot behavior:

```yaml
# Exchange Settings
exchange:
  name: mexc
  type: futures
  testnet: false

# Trading Pairs
pairs:
  - LAB/USDT
  - MINA/USDT
  - APR/USDT
  - KITE/USDT
  - BLUAI/USDT
  - ON/USDT
  - RIVER/USDT
  - CLO/USDT

# Timeframe
timeframe: 15m

# Strategy Parameters
strategy:
  ema_length: 200
  rsi_length: 14
  rsi_oversold: 30
  rsi_overbought: 70
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  adx_length: 14
  adx_threshold: 20
  atr_length: 14
  atr_multiplier_sl: 1.5
  risk_reward_ratio: 1.5
  
  # Confluence Scoring
  weights:
    trend: 1.0
    rsi: 1.0
    macd: 1.0
    adx: 1.0
    volume: 0.5
  score_threshold: 2.5

# Bot Settings
bot:
  check_interval: 60  # seconds
  lookback_bars: 300  # historical bars to fetch
```

### Optimization Tips

**For More Signals (Higher Frequency):**
- Lower `score_threshold` to 2.0
- Reduce `adx_threshold` to 15
- Disable filters by setting `use_adx_filter: false`

**For Higher Accuracy (Lower Frequency):**
- Increase `score_threshold` to 3.0 or 3.5
- Increase `adx_threshold` to 25 or 30
- Increase `volume_multiplier` to 1.5

**For Different Risk/Reward:**
- Adjust `risk_reward_ratio` (1.5 is balanced, 2.0 for more profit potential, 1.0 for quicker exits)
- Adjust `atr_multiplier_sl` (1.5 is standard, 2.0 for wider stops, 1.0 for tighter stops)

## Telegram Alert Format

### Long Entry Alert

```
🟢 LONG SIGNAL - MINA/USDT

💰 Entry Price: 0.623400
🛑 Stop Loss: 0.614500
🎯 Take Profit: 0.641200
📊 Risk/Reward: 1:1.50

⭐ Confidence Score: 3.5/4.5 (78%)
📈 RSI: 35.2
💪 ADX: 28.4
⏰ Timeframe: 15m
🔔 Exchange: MEXC Futures
```

### Short Entry Alert

```
🔴 SHORT SIGNAL - MINA/USDT

💰 Entry Price: 0.623400
🛑 Stop Loss: 0.632300
🎯 Take Profit: 0.605600
📊 Risk/Reward: 1:1.50

⭐ Confidence Score: 3.2/4.5 (71%)
📈 RSI: 68.5
💪 ADX: 26.1
⏰ Timeframe: 15m
🔔 Exchange: MEXC Futures
```

## Monitored Pairs

The bot is configured to monitor these pairs on MEXC Futures:

- **LAB/USDT** - LabsGroup
- **MINA/USDT** - Mina Protocol
- **APR/USDT** - APR Coin
- **KITE/USDT** - Kite
- **BLUAI/USDT** - BluAI
- **ON/USDT** - On
- **RIVER/USDT** - River
- **CLO/USDT** - Callisto Network

All pairs are monitored on the **15-minute timeframe**.

## Expected Performance

Based on the confluence methodology and 15-minute crypto futures trading:

- **Expected Accuracy**: 45-60% (varies by market conditions)
- **Expected Profit Factor**: 1.5-2.5 with proper risk management
- **Signal Frequency**: 5-15 signals per week across all 8 pairs
- **Best Market Conditions**: Trending markets with clear directional moves

**Note**: These are estimates. Actual performance will vary based on market conditions, parameter settings, and execution.

## Logging

The bot logs all activity to `trading_bot.log` in the same directory. Log levels can be configured in `config.yaml`:

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about bot operation (default)
- **WARNING**: Warning messages
- **ERROR**: Error messages

View logs in real-time:

```bash
tail -f trading_bot.log
```

## Troubleshooting

### Bot Not Starting

**Error: "Telegram credentials not found"**
- Make sure `.env` file exists and contains `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`

**Error: "Failed to initialize exchange"**
- Check your internet connection
- Verify MEXC exchange is accessible from your location

### No Signals Being Generated

**Possible causes:**
- Score threshold is too high - lower it in `config.yaml`
- Market conditions don't meet confluence requirements
- Pairs are not available on MEXC - check startup message

### Duplicate Alerts

The bot has built-in duplicate prevention. If you're receiving duplicates:
- Restart the bot to reset the tracking
- Check that only one instance of the bot is running

### Pair Not Available

If a pair shows as unavailable in the startup message:
- Verify the pair symbol is correct (e.g., "MINA/USDT" not "MINAUSDT")
- Check if the pair is listed on MEXC Futures
- Some pairs may be spot-only or not available in your region

## Architecture

The bot is structured into modular components:

- **bot.py**: Main entry point and orchestration
- **data_client.py**: Exchange data fetching using ccxt
- **indicators.py**: Technical indicator calculations using pandas_ta
- **strategy.py**: Confluence strategy logic and signal generation
- **notifier.py**: Telegram notification formatting and sending
- **config.yaml**: Configuration file
- **.env**: Environment variables (credentials)

## Comparison with TradingView Version

| Feature | TradingView + Webhook | Local Python Bot |
|---------|----------------------|------------------|
| **Setup Complexity** | Medium (requires webhook deployment) | Low (runs locally) |
| **Dependencies** | TradingView subscription, public webhook | Python, internet connection |
| **Data Source** | TradingView | MEXC exchange directly |
| **Backtesting** | Yes (TradingView Strategy Tester) | No (alerts only) |
| **Customization** | Pine Script editing | Python code + YAML config |
| **Reliability** | Depends on TradingView + webhook uptime | Depends on local machine uptime |
| **Cost** | TradingView subscription + hosting | Free (local) |
| **Monitoring** | Multiple pairs via multiple alerts | All pairs in one bot |

## Security Notes

- **Never commit `.env` file** - It contains your Telegram credentials
- **Keep `.env` secure** - Anyone with your bot token can send messages as your bot
- **No API keys required** - The bot only reads public market data (no trading execution)
- **Local execution** - All data processing happens on your machine

## Limitations

- **No automatic trading** - The bot only sends alerts; you must execute trades manually
- **No backtesting** - Use the TradingView version for backtesting
- **Single timeframe** - Monitors only 15-minute timeframe (configurable but single at a time)
- **MEXC only** - Currently configured for MEXC Futures (can be adapted for other exchanges)
- **Requires uptime** - Bot must be running to monitor and send alerts

## Future Enhancements

Potential improvements (not currently implemented):

- Multi-timeframe analysis
- Multiple exchange support
- Web dashboard for monitoring
- Trade execution integration
- Performance tracking and statistics
- Email/SMS notifications
- Webhook integration for external systems

## Support

For issues, questions, or suggestions:
- Check the logs: `tail -f trading_bot.log`
- Review configuration: `config.yaml` and `.env`
- Verify pair availability on MEXC Futures
- Test Telegram connection: Send a test message to your bot

## License

Copyright (c) 2025, Confluence Trading Bot
This script may be freely distributed under the terms of GPL-3.0 license.

## Disclaimer

This bot is provided for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Always do your own research and never trade with money you cannot afford to lose. Past performance is not indicative of future results. The authors are not responsible for any financial losses incurred while using this bot.
