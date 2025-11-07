# Confluence Trading Strategy - Maximum Accuracy

A comprehensive multi-indicator confluence strategy designed for cryptocurrency trading on 15-minute timeframes. This strategy combines trend, momentum, and regime filters to generate high-probability trading signals with built-in risk management and detailed alert notifications.

## Strategy Overview

The Confluence Strategy uses a weighted scoring system that combines multiple technical indicators to identify high-probability trade setups. It only enters trades when multiple indicators align (confluence), which helps filter out false signals and improve accuracy.

## Key Features

- **Multi-Indicator Confluence System**: Combines EMA trend filter, RSI momentum, MACD direction, ADX strength, and volume confirmation
- **Comprehensive Alerts**: Detailed alert messages with entry price, stop loss, take profit, risk/reward ratio, and confluence score
- **Real-Time Performance Metrics**: On-chart display of accuracy percentage, profit factor, win rate, and more
- **ATR-Based Risk Management**: Dynamic stop loss and take profit levels based on market volatility
- **Fully Customizable**: All parameters can be adjusted to optimize for different markets and timeframes
- **Both Long and Short Trades**: Can trade both directions or be configured for long-only or short-only

## Indicators Used

### 1. EMA (Exponential Moving Average) - Trend Filter
- **Default**: 200-period EMA
- **Purpose**: Identifies the overall trend direction
- **Logic**: Long trades when price is above EMA, short trades when price is below EMA

### 2. RSI (Relative Strength Index) - Momentum Trigger
- **Default**: 14-period RSI with 30/70 levels
- **Purpose**: Identifies oversold/overbought conditions and momentum shifts
- **Logic**: Long signals on RSI crosses above oversold or RSI between 30-50, short signals on RSI crosses below overbought or RSI between 50-70

### 3. MACD (Moving Average Convergence Divergence) - Direction Confirmation
- **Default**: 12, 26, 9 settings
- **Purpose**: Confirms trend direction and momentum
- **Logic**: Long signals when MACD crosses above signal line or MACD is bullish, short signals when MACD crosses below signal line or MACD is bearish

### 4. ADX (Average Directional Index) - Trend Strength Filter
- **Default**: 14-period ADX with 20 threshold
- **Purpose**: Filters out weak trends and choppy markets
- **Logic**: Only takes trades when ADX is above threshold, indicating strong trend

### 5. Volume - Confirmation Filter
- **Default**: 1.2x average volume (20-period SMA)
- **Purpose**: Confirms that there's sufficient market participation
- **Logic**: Adds weight to signals when volume is above average

### 6. ATR (Average True Range) - Risk Management
- **Default**: 14-period ATR with 1.5x multiplier for stop loss
- **Purpose**: Sets dynamic stop loss and take profit based on market volatility
- **Logic**: Stop loss = Entry ± (ATR × 1.5), Take profit = Entry ± (ATR × 1.5 × 1.5 RR)

## Confluence Scoring System

The strategy uses a weighted scoring system where each indicator contributes points when its conditions are met:

- **Trend Filter**: 1.0 point (when price aligns with EMA)
- **RSI**: 1.0 point (when RSI shows momentum)
- **MACD**: 1.0 point (when MACD confirms direction)
- **ADX**: 1.0 point (when trend is strong)
- **Volume**: 0.5 points (when volume is above average)

**Default Entry Threshold**: 2.5 points

This means a trade requires at least 2-3 indicators to align before entry, reducing false signals and improving accuracy.

## Risk Management

- **Stop Loss**: ATR-based dynamic stop loss (default: 1.5 × ATR)
- **Take Profit**: Risk/reward ratio of 1.5:1 (default)
- **Position Sizing**: 100% of equity per trade (adjustable)
- **No Pyramiding**: Only one position at a time
- **Commission**: 0.1% per trade (typical crypto exchange fee)

## Alert System

The strategy provides two types of alerts:

### 1. Dynamic Alerts (alert() function)
These provide detailed information in real-time:

**Long Entry Alert Example:**
```
🟢 LONG SIGNAL - MINA/USDT
Entry Price: 0.6234
Stop Loss: 0.6145
Take Profit: 0.6412
Risk/Reward: 1:1.5
Confluence Score: 3.5/2.5
RSI: 35.2
ADX: 28.4
Timeframe: 15
```

**Exit Alert Example:**
```
✅ LONG EXIT - MINA/USDT
Exit Price: 0.6398
Entry Price: 0.6234
P&L: 2.63%
Timeframe: 15
```

### 2. Alert Conditions (alertcondition() fallback)
Simple alerts for users who prefer the traditional TradingView alert system.

## Performance Metrics Display

The strategy displays a real-time performance table in the top-right corner showing:

- **Accuracy**: Win rate percentage (color-coded: green ≥60%, orange ≥50%, red <50%)
- **Total Trades**: Number of completed trades
- **Winning Trades**: Number of profitable trades
- **Losing Trades**: Number of losing trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Win**: Average profit per winning trade
- **Average Loss**: Average loss per losing trade
- **Net Profit**: Total profit/loss percentage

## How to Use

### 1. Add to TradingView
1. Open TradingView and go to the Pine Editor
2. Create a new script
3. Copy and paste the entire `confluence_strategy.pine` code
4. Click "Add to Chart"

### 2. Configure for Your Trading Pairs
The strategy is optimized for these crypto pairs on 15-minute timeframe:
- LAB/USDT
- MINA/USDT
- APR/USDT
- KITE/USDT
- BLUAI/USDT
- ON/USDT
- RIVER/USDT
- CLO/USDT

### 3. Set Up Alerts
To receive alerts with detailed entry/exit information:

1. Click the "Alert" button (clock icon) in TradingView
2. Select "Confluence Strategy - Maximum Accuracy"
3. **Important**: In the "Condition" dropdown, select **"Any alert() function call"**
4. Configure your notification preferences (popup, email, webhook, etc.)
5. Click "Create"

For traditional alerts:
1. Select "Long Entry Alert" or "Short Entry Alert" from the condition dropdown
2. The message will be simpler but still functional

### 4. Optimize Settings (Optional)
You can adjust the following parameters to optimize for your specific needs:

**For More Trades (Higher Frequency):**
- Lower the "Entry Score Threshold" to 2.0 or 2.5
- Disable or reduce ADX threshold
- Reduce EMA length to 100 or 50

**For Higher Accuracy (Lower Frequency):**
- Increase the "Entry Score Threshold" to 3.0 or 3.5
- Increase ADX threshold to 25 or 30
- Increase volume multiplier to 1.5

**For Different Risk/Reward:**
- Adjust "Risk/Reward Ratio" (1.5 is balanced, 2.0 for more profit potential, 1.0 for quicker exits)
- Adjust "ATR Multiplier for Stop Loss" (1.5 is standard, 2.0 for wider stops, 1.0 for tighter stops)

## Expected Performance

Based on the confluence methodology and 15-minute crypto trading:

- **Expected Accuracy**: 45-60% (varies by market conditions)
- **Expected Profit Factor**: 1.5-2.5 (with proper risk management)
- **Trade Frequency**: 5-15 trades per week per pair (depends on threshold settings)
- **Best Market Conditions**: Trending markets with clear directional moves

**Note**: Past performance does not guarantee future results. Always backtest the strategy on your specific pairs and timeframes before live trading.

## Backtesting Recommendations

1. **Test on Historical Data**: Use at least 3-6 months of historical data
2. **Multiple Pairs**: Test on all your target pairs to see which perform best
3. **Different Timeframes**: While optimized for 15m, test on 5m and 30m as well
4. **Market Conditions**: Note performance during trending vs ranging markets
5. **Optimization**: Use TradingView's Strategy Tester to optimize parameters

## Tips for Maximum Profit

1. **Trade Multiple Pairs**: Monitor all 8 pairs to catch more opportunities
2. **Respect the Signals**: Don't override the strategy's decisions emotionally
3. **Adjust for Market Conditions**: In ranging markets, consider lowering position size or pausing
4. **Risk Management**: Never risk more than 1-2% of your capital per trade
5. **Review Performance**: Regularly check the performance metrics and adjust if needed
6. **Combine with Higher Timeframes**: Check 1H or 4H trend before taking 15m signals

## Customization Guide

### Trade Direction
- Enable/disable long or short trades independently
- Useful for directional bias or exchange limitations

### Trend Filter
- Toggle EMA filter on/off
- Adjust EMA length for faster/slower trend identification

### Confluence Weights
- Adjust individual indicator weights to emphasize certain signals
- Example: Increase MACD weight to 1.5 if you trust momentum more

### Score Threshold
- The most important parameter for trade frequency vs accuracy balance
- Start with default 2.5 and adjust based on backtest results

## Troubleshooting

**Problem**: Not getting any trades
- **Solution**: Lower the score threshold or disable some filters (ADX, Volume)

**Problem**: Too many losing trades
- **Solution**: Increase score threshold, enable all filters, increase ADX threshold

**Problem**: Alerts not showing detailed information
- **Solution**: Make sure you selected "Any alert() function call" when creating the alert

**Problem**: Stop loss too tight or too wide
- **Solution**: Adjust ATR multiplier (increase for wider stops, decrease for tighter)

## Version History

- **v1.0** (2025): Initial release with multi-indicator confluence system, comprehensive alerts, and performance metrics

## Support

For questions, issues, or suggestions, please open an issue on the GitHub repository.

## Disclaimer

This strategy is provided for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Always do your own research and never trade with money you cannot afford to lose. Past performance is not indicative of future results.

## License

Copyright (c) 2025, Confluence Trading Strategy
This script may be freely distributed under the terms of GPL-3.0 license.
