#!/usr/bin/env python3
"""
Advanced Confluence Trading Bot
Uses 10 superior indicators for enhanced signal quality

Indicators:
1. FRAMA - Fractal Adaptive Moving Average
2. Jurik RSX - Smoother RSI
3. SuperTrend - Trend direction
4. Chandelier Exit - ATR-based stops
5. Parabolic SAR - Acceleration trailing
6. IQR Bands - Interquartile Range Bands
7. Kirshenbaum Bands - Linear regression bands
8. MAD Bands - Mean Absolute Deviation Bands
9. Volatility Ratio - Squeeze/expansion detector
10. Klinger Volume Oscillator - Volume confirmation
"""

import os
import sys
import time
import logging
import yaml
from dotenv import load_dotenv
from typing import Dict, List

from data_client import DataClient
from indicators_advanced import AdvancedIndicatorCalculator
from strategy_advanced import AdvancedConfluenceStrategy
from notifier import TelegramNotifier

load_dotenv()

def setup_logging(config: dict):
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_file = config.get('logging', {}).get('file', 'trading_bot_advanced.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = 'config_advanced.yaml') -> dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['exchange', 'pairs', 'timeframe', 'strategy', 'bot']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        if 'logging' not in config:
            config['logging'] = {'level': 'INFO', 'file': 'trading_bot_advanced.log'}
        
        if 'enable_long' not in config:
            config['enable_long'] = True
        
        if 'enable_short' not in config:
            config['enable_short'] = True
        
        if 'strategy_mode' not in config:
            config['strategy_mode'] = 'trend_following'
        
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def main():
    print("=" * 80)
    print("🚀 Advanced Confluence Trading Bot - 10 Superior Indicators")
    print("=" * 80)
    
    config = load_config()
    logger = setup_logging(config)
    
    strategy_mode = config.get('strategy_mode', 'trend_following')
    logger.info(f"Starting Advanced Confluence Trading Bot in {strategy_mode.upper()} mode...")
    
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not telegram_token or not telegram_chat_id:
        logger.error("Telegram credentials not found in .env file")
        print("\n❌ Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env file")
        sys.exit(1)
    
    try:
        notifier = TelegramNotifier(telegram_token, telegram_chat_id)
        logger.info("Telegram notifier initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram notifier: {e}")
        sys.exit(1)
    
    try:
        data_client = DataClient(
            exchange_name=config['exchange']['name'],
            exchange_type=config['exchange']['type'],
            testnet=config['exchange'].get('testnet', False)
        )
        logger.info("Data client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize data client: {e}")
        notifier.send_error_message(f"Failed to initialize exchange: {e}")
        sys.exit(1)
    
    try:
        data_client.load_markets()
    except Exception as e:
        logger.error(f"Failed to load markets: {e}")
        notifier.send_error_message(f"Failed to load markets: {e}")
        sys.exit(1)
    
    pairs = config['pairs']
    available_pairs = data_client.get_available_pairs(pairs)
    
    active_pairs = [pair for pair, available in available_pairs.items() if available]
    
    if not active_pairs:
        logger.error("No available pairs to monitor")
        notifier.send_error_message("No available pairs found on exchange")
        sys.exit(1)
    
    logger.info(f"Monitoring {len(active_pairs)} pairs: {', '.join(active_pairs)}")
    
    startup_msg = f"🚀 <b>Advanced Trading Bot Started</b>\n\n"
    startup_msg += f"📊 <b>Strategy Mode:</b> {strategy_mode.replace('_', ' ').title()}\n"
    startup_msg += f"📈 <b>Monitoring:</b> {len(active_pairs)}/{len(pairs)} pairs on {config['exchange']['name'].upper()} ({config['timeframe']})\n\n"
    startup_msg += "<b>🎯 Indicators Used:</b>\n"
    startup_msg += "1️⃣ FRAMA - Adaptive trend baseline\n"
    startup_msg += "2️⃣ Jurik RSX - Smoother momentum\n"
    startup_msg += "3️⃣ SuperTrend - Direction trigger\n"
    startup_msg += "4️⃣ Chandelier Exit - Smart stops\n"
    startup_msg += "5️⃣ Parabolic SAR - Trailing\n"
    startup_msg += "6️⃣ IQR Bands - Mean reversion\n"
    startup_msg += "7️⃣ Kirshenbaum Bands - Breakouts\n"
    startup_msg += "8️⃣ MAD Bands - Alternative MR\n"
    startup_msg += "9️⃣ Volatility Ratio - Squeeze detector\n"
    startup_msg += "🔟 Klinger - Volume confirmation\n\n"
    startup_msg += f"<b>Available Pairs:</b>\n"
    for pair in active_pairs:
        startup_msg += f"✅ {pair}\n"
    
    unavailable = [pair for pair, available in available_pairs.items() if not available]
    if unavailable:
        startup_msg += f"\n<b>Unavailable Pairs:</b>\n"
        for pair in unavailable:
            startup_msg += f"❌ {pair}\n"
    
    startup_msg += f"\n✨ Ready to send high-quality signals!"
    notifier.send_message(startup_msg)
    
    indicator_calculator = AdvancedIndicatorCalculator(config)
    strategy = AdvancedConfluenceStrategy(config)
    
    timeframe = config['timeframe']
    check_interval = config['bot']['check_interval']
    lookback_bars = config['bot']['lookback_bars']
    
    logger.info(f"Bot configuration: mode={strategy_mode}, timeframe={timeframe}, "
               f"check_interval={check_interval}s, lookback_bars={lookback_bars}")
    
    print(f"\n✅ Bot started successfully!")
    print(f"📊 Strategy Mode: {strategy_mode.replace('_', ' ').title()}")
    print(f"📈 Monitoring {len(active_pairs)} pairs on {config['exchange']['name'].upper()} ({timeframe})")
    print(f"🔔 Alerts will be sent to your Telegram")
    print(f"⏱️  Checking every {check_interval} seconds")
    print(f"\nPress Ctrl+C to stop\n")
    print("=" * 80)
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            logger.info(f"Starting check iteration #{iteration}")
            
            for symbol in active_pairs:
                try:
                    logger.debug(f"Fetching data for {symbol}")
                    df = data_client.fetch_ohlcv(symbol, timeframe, lookback_bars)
                    
                    if df is None or len(df) < 200:
                        logger.warning(f"Insufficient data for {symbol}")
                        continue
                    
                    df_with_indicators = indicator_calculator.calculate_all(df)
                    
                    if df_with_indicators is None:
                        logger.warning(f"Failed to calculate indicators for {symbol}")
                        continue
                    
                    signal = strategy.check_signals(symbol, df_with_indicators)
                    
                    if signal:
                        logger.info(f"Signal detected: {signal['action']} {signal['symbol']} @ {signal['entry_price']:.6f}")
                        
                        action = signal['action']
                        emoji = "🟢" if action == 'LONG' else "🔴"
                        
                        message = f"{emoji} <b>{action} SIGNAL - {signal['symbol']}</b>\n\n"
                        message += f"💰 <b>Entry Price:</b> {signal['entry_price']:.6f}\n"
                        message += f"🛑 <b>Stop Loss:</b> {signal['stop_loss']:.6f}\n"
                        message += f"🎯 <b>Take Profit:</b> {signal['take_profit']:.6f}\n"
                        
                        risk = abs(signal['entry_price'] - signal['stop_loss'])
                        reward = abs(signal['take_profit'] - signal['entry_price'])
                        rr_ratio = reward / risk if risk > 0 else 0
                        message += f"📊 <b>Risk/Reward:</b> 1:{rr_ratio:.2f}\n\n"
                        
                        confidence_pct = (signal['score'] / signal['max_score']) * 100
                        message += f"⭐ <b>Confidence:</b> {signal['score']:.1f}/{signal['max_score']:.1f} ({confidence_pct:.0f}%)\n"
                        message += f"🎯 <b>Strategy:</b> {signal['strategy_mode'].replace('_', ' ').title()}\n"
                        message += f"📈 <b>RSX:</b> {signal['rsx']:.1f}\n"
                        message += f"💪 <b>ADX:</b> {signal['adx']:.1f}\n"
                        message += f"⏰ <b>Timeframe:</b> {timeframe}\n"
                        message += f"🔔 <b>Exchange:</b> {config['exchange']['name'].upper()}"
                        
                        success = notifier.send_message(message)
                        if success:
                            logger.info(f"Alert sent successfully for {signal['action']} {signal['symbol']}")
                        else:
                            logger.error(f"Failed to send alert for {signal['action']} {signal['symbol']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            logger.info(f"Completed check iteration #{iteration}. Sleeping for {check_interval}s...")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("\n\n🛑 Bot stopped by user")
        notifier.send_message("🛑 <b>Advanced Trading Bot Stopped</b>\n\nBot has been manually stopped.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        notifier.send_error_message(f"Bot crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
