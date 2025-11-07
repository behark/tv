#!/usr/bin/env python3
import os
import sys
import time
import logging
import yaml
from dotenv import load_dotenv
from typing import Dict, List

from data_client import DataClient
from indicators import IndicatorCalculator
from strategy import ConfluenceStrategy
from notifier import TelegramNotifier

load_dotenv()

def setup_logging(config: dict):
    log_level = getattr(logging, config['logging']['level'])
    log_file = config['logging']['file']
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("🤖 Confluence Trading Bot - MEXC Futures")
    print("=" * 60)
    
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting Confluence Trading Bot...")
    
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
            testnet=config['exchange']['testnet']
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
        notifier.send_error_message("No available pairs found on MEXC")
        sys.exit(1)
    
    logger.info(f"Monitoring {len(active_pairs)} pairs: {', '.join(active_pairs)}")
    
    notifier.send_startup_message(pairs, available_pairs)
    
    indicator_calculator = IndicatorCalculator(config)
    strategy = ConfluenceStrategy(config)
    
    timeframe = config['timeframe']
    check_interval = config['bot']['check_interval']
    lookback_bars = config['bot']['lookback_bars']
    
    logger.info(f"Bot configuration: timeframe={timeframe}, check_interval={check_interval}s, lookback_bars={lookback_bars}")
    
    print(f"\n✅ Bot started successfully!")
    print(f"📊 Monitoring {len(active_pairs)} pairs on MEXC Futures ({timeframe})")
    print(f"🔔 Alerts will be sent to your Telegram")
    print(f"⏱️  Checking every {check_interval} seconds")
    print(f"\nPress Ctrl+C to stop\n")
    print("=" * 60)
    
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
                        
                        success = notifier.send_signal_alert(signal)
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
        notifier.send_message("🛑 <b>Trading Bot Stopped</b>\n\nBot has been manually stopped.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        notifier.send_error_message(f"Bot crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
