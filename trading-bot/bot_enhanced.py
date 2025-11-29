#!/usr/bin/env python3
"""
Enhanced Confluence Trading Bot
Integrates all advanced features:
- Signal tracking with TP/SL monitoring
- Multi-timeframe confirmation
- Signal scoring and accuracy filters
- Heartbeat and daily summaries
- Graceful shutdown handling
"""

import os
import sys
import time
import signal
import logging
import yaml
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional

from data_client import DataClient
from indicators_advanced import AdvancedIndicatorCalculator
from strategy_advanced import AdvancedConfluenceStrategy
from notifier import TelegramNotifier
from signal_database import SignalDatabase
from signal_tracker import SignalTracker, SignalCooldownManager
from accuracy_filters import AccuracyFilterManager
from heartbeat import HeartbeatManager, DailySummaryManager, HealthMonitor

load_dotenv()

# Global shutdown flag
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global _shutdown_requested
    _shutdown_requested = True
    logging.info(f"Shutdown signal received ({signum})")


def setup_logging(config: dict):
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_file = config.get('logging', {}).get('file', 'trading_bot_enhanced.log')

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path: str = 'config_enhanced.yaml') -> dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ['exchange', 'pairs', 'timeframe', 'strategy', 'bot']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        # Set defaults
        defaults = {
            'logging': {'level': 'INFO', 'file': 'trading_bot_enhanced.log'},
            'enable_long': True,
            'enable_short': True,
            'strategy_mode': 'trend_following',
            'signal_tracking': {
                'enabled': True,
                'expiry_hours': 24,
                'cooldown_minutes': 60
            },
            'accuracy_filters': {
                'enable_mtf_filter': True,
                'enable_volume_filter': True,
                'enable_volatility_filter': True,
                'enable_signal_scoring': True,
                'min_signal_score': 6.0
            },
            'heartbeat': {
                'enabled': True,
                'interval_hours': 6
            },
            'daily_summary': {
                'enabled': True,
                'hour': 0,
                'minute': 0
            }
        }

        for key, value in defaults.items():
            if key not in config:
                config[key] = value

        return config

    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Falling back to config_advanced.yaml...")
        return load_config('config_advanced.yaml')
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def format_enhanced_signal_message(signal: Dict, signal_id: str, config: dict) -> str:
    """Format an enhanced signal alert message"""
    action = signal['action']
    symbol = signal['symbol']
    entry = signal['entry_price']
    sl = signal['stop_loss']
    tp = signal['take_profit']

    emoji = "🟢" if action == 'LONG' else "🔴"

    # Calculate R:R
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr_ratio = reward / risk if risk > 0 else 0

    # Get quality score if available
    quality_score = signal.get('quality_score', signal.get('score', 0))
    max_score = signal.get('max_score', 10)

    message = f"{emoji} <b>{action} SIGNAL - {symbol}</b>\n\n"
    message += f"🆔 <b>Signal ID:</b> #{signal_id}\n"
    message += f"📊 <b>Quality Score:</b> {quality_score:.1f}/10\n"

    # HTF confirmation if available
    if 'htf_trend' in signal:
        htf_emoji = "✅" if signal['htf_trend'] == action else "⚠️"
        message += f"{htf_emoji} <b>HTF Trend:</b> {signal['htf_trend']} ({signal.get('htf_confidence', 0)}%)\n"

    message += f"\n💰 <b>Entry Price:</b> {entry:.6f}\n"
    message += f"🛑 <b>Stop Loss:</b> {sl:.6f}\n"
    message += f"🎯 <b>Take Profit:</b> {tp:.6f}\n"
    message += f"📊 <b>Risk/Reward:</b> 1:{rr_ratio:.2f}\n"

    # Volume info if available
    if 'volume_ratio' in signal:
        vol_emoji = "🔥" if signal.get('volume_spike', False) else "📊"
        message += f"{vol_emoji} <b>Volume:</b> {signal['volume_ratio']:.1f}x avg\n"

    message += f"\n📈 <b>RSX:</b> {signal.get('rsx', 'N/A'):.1f}\n"
    message += f"💪 <b>ADX:</b> {signal.get('adx', 'N/A'):.1f}\n"
    message += f"🎯 <b>Strategy:</b> {signal.get('strategy_mode', 'unknown').replace('_', ' ').title()}\n"
    message += f"⏰ <b>Timeframe:</b> {config['timeframe']}\n"
    message += f"🔔 <b>Exchange:</b> {config['exchange']['name'].upper()}\n"

    message += f"\n<i>Tracking active - will alert on TP/SL hit</i>"

    return message


def main():
    global _shutdown_requested

    print("=" * 80)
    print("🚀 Enhanced Confluence Trading Bot")
    print("   Signal Tracking | Multi-Timeframe | Quality Scoring")
    print("=" * 80)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    config = load_config()
    logger = setup_logging(config)

    strategy_mode = config.get('strategy_mode', 'trend_following')
    logger.info(f"Starting Enhanced Trading Bot in {strategy_mode.upper()} mode...")

    # Initialize Telegram
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not telegram_token or not telegram_chat_id:
        logger.error("Telegram credentials not found in environment")
        print("\n❌ Error: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")
        sys.exit(1)

    try:
        notifier = TelegramNotifier(telegram_token, telegram_chat_id)
        logger.info("Telegram notifier initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram: {e}")
        sys.exit(1)

    # Initialize data client
    try:
        data_client = DataClient(
            exchange_name=config['exchange']['name'],
            exchange_type=config['exchange']['type'],
            testnet=config['exchange'].get('testnet', False),
            timeout=config.get('api_timeout', 30000)
        )
        data_client.load_markets()
        logger.info("Data client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize data client: {e}")
        notifier.send_error_message(f"Failed to initialize exchange: {e}")
        sys.exit(1)

    # Get available pairs
    pairs = config['pairs']
    available_pairs = data_client.get_available_pairs(pairs)
    active_pairs = [pair for pair, available in available_pairs.items() if available]

    if not active_pairs:
        logger.error("No available pairs to monitor")
        notifier.send_error_message("No available pairs found on exchange")
        sys.exit(1)

    logger.info(f"Monitoring {len(active_pairs)} pairs: {', '.join(active_pairs)}")

    # Initialize signal database and tracker
    signal_db = SignalDatabase(config.get('database_path', 'signals.db'))
    signal_tracker = SignalTracker(signal_db, config.get('signal_tracking', {}))
    cooldown_manager = SignalCooldownManager(
        signal_db,
        cooldown_minutes=config.get('signal_tracking', {}).get('cooldown_minutes', 60)
    )

    # Initialize accuracy filters
    accuracy_filters = AccuracyFilterManager(data_client, config.get('accuracy_filters', {}))

    # Initialize heartbeat and health monitoring
    heartbeat = HeartbeatManager(notifier, config.get('heartbeat', {}))
    daily_summary = DailySummaryManager(notifier, signal_db, config.get('daily_summary', {}))
    health_monitor = HealthMonitor(notifier, config)

    # Register signal tracker callbacks for TP/SL alerts
    def on_tp_hit(signal_data, **kwargs):
        message = signal_tracker.format_tp_hit_message(signal_data)
        notifier.send_message(message, priority="high")
        heartbeat.increment_signals()

    def on_sl_hit(signal_data, **kwargs):
        message = signal_tracker.format_sl_hit_message(signal_data)
        notifier.send_message(message, priority="high")

    def on_expired(signal_data, **kwargs):
        message = signal_tracker.format_expired_message(signal_data)
        notifier.send_message(message)

    signal_tracker.register_callback('on_tp_hit', on_tp_hit)
    signal_tracker.register_callback('on_sl_hit', on_sl_hit)
    signal_tracker.register_callback('on_expired', on_expired)

    # Initialize strategy components
    indicator_calculator = AdvancedIndicatorCalculator(config)
    strategy = AdvancedConfluenceStrategy(config)

    timeframe = config['timeframe']
    check_interval = config['bot']['check_interval']
    lookback_bars = config['bot']['lookback_bars']

    # Send startup message
    startup_msg = f"🚀 <b>Enhanced Trading Bot Started</b>\n\n"
    startup_msg += f"📊 <b>Strategy Mode:</b> {strategy_mode.replace('_', ' ').title()}\n"
    startup_msg += f"📈 <b>Monitoring:</b> {len(active_pairs)}/{len(pairs)} pairs\n"
    startup_msg += f"⏰ <b>Timeframe:</b> {timeframe}\n"
    startup_msg += f"🔔 <b>Exchange:</b> {config['exchange']['name'].upper()}\n\n"
    startup_msg += f"<b>Features Enabled:</b>\n"
    startup_msg += f"  ✅ Signal Tracking with TP/SL Alerts\n"
    startup_msg += f"  ✅ Multi-Timeframe Confirmation\n"
    startup_msg += f"  ✅ Signal Quality Scoring\n"
    startup_msg += f"  ✅ Volume & Volatility Filters\n"
    startup_msg += f"  ✅ Signal Cooldown Protection\n"
    startup_msg += f"  ✅ Daily Performance Summaries\n\n"
    startup_msg += f"<b>Available Pairs:</b>\n"
    for pair in active_pairs[:10]:  # Show first 10
        startup_msg += f"  ✅ {pair}\n"
    if len(active_pairs) > 10:
        startup_msg += f"  ... and {len(active_pairs) - 10} more\n"
    startup_msg += f"\n✨ Ready to send high-quality signals!"

    notifier.send_message(startup_msg)

    # Start background services
    heartbeat.start()
    daily_summary.start()

    print(f"\n✅ Bot started successfully!")
    print(f"📊 Strategy Mode: {strategy_mode.replace('_', ' ').title()}")
    print(f"📈 Monitoring {len(active_pairs)} pairs on {config['exchange']['name'].upper()} ({timeframe})")
    print(f"🔔 Alerts will be sent to Telegram")
    print(f"⏱️  Checking every {check_interval} seconds")
    print(f"\nPress Ctrl+C to stop gracefully\n")
    print("=" * 80)

    iteration = 0

    try:
        while not _shutdown_requested:
            iteration += 1
            heartbeat.increment_iteration()
            logger.info(f"Starting check iteration #{iteration}")

            # Collect current prices for signal tracking
            price_data = {}

            for symbol in active_pairs:
                if _shutdown_requested:
                    break

                try:
                    logger.debug(f"Fetching data for {symbol}")
                    df = data_client.fetch_ohlcv(symbol, timeframe, lookback_bars)

                    if df is None or len(df) < 200:
                        logger.warning(f"Insufficient data for {symbol}")
                        continue

                    # Store price data for signal tracking
                    last_bar = df.iloc[-1]
                    price_data[symbol] = {
                        'close': last_bar['close'],
                        'high': last_bar['high'],
                        'low': last_bar['low']
                    }

                    # Calculate indicators
                    df_with_indicators = indicator_calculator.calculate_all(df)

                    if df_with_indicators is None:
                        logger.warning(f"Failed to calculate indicators for {symbol}")
                        continue

                    # Check for new signals
                    signal = strategy.check_signals(symbol, df_with_indicators)

                    if signal:
                        # Check cooldown
                        if not cooldown_manager.can_signal(symbol, signal['action']):
                            remaining = cooldown_manager.get_remaining_cooldown(symbol, signal['action'])
                            logger.info(f"Signal for {symbol} blocked by cooldown ({remaining}m remaining)")
                            continue

                        # Get active signals for correlation filter
                        active_signal_symbols = [s['symbol'] for s in signal_db.get_active_signals()]

                        # Apply accuracy filters
                        passes, filter_results = accuracy_filters.filter_signal(
                            signal, df_with_indicators, timeframe, active_signal_symbols
                        )

                        if not passes:
                            logger.info(f"Signal rejected for {symbol}: {filter_results.get('rejection_reason')}")
                            continue

                        # Enhance signal with filter data
                        enhanced_signal = accuracy_filters.get_enhanced_signal(signal, filter_results)

                        # Add signal to tracker
                        signal_id = signal_tracker.add_signal(enhanced_signal)

                        if signal_id:
                            # Record cooldown
                            cooldown_manager.record_signal(symbol, signal['action'])

                            # Send alert
                            message = format_enhanced_signal_message(enhanced_signal, signal_id, config)
                            success = notifier.send_message(message)

                            if success:
                                logger.info(f"Alert sent: {signal['action']} {symbol} (ID: {signal_id})")
                                heartbeat.increment_signals()
                                health_monitor.record_success()
                            else:
                                logger.error(f"Failed to send alert for {symbol}")
                                health_monitor.record_failure("Failed to send Telegram alert")

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    health_monitor.record_failure(str(e))
                    heartbeat.increment_errors()
                    continue

            # Check active signals for TP/SL hits
            if price_data:
                closed_signals = signal_tracker.check_all_active_signals(price_data)
                if closed_signals:
                    logger.info(f"Closed {len(closed_signals)} signals this iteration")

            # Expire old signals
            signal_tracker.expire_old_signals()

            # Flush message queue if any
            notifier.flush_queue()

            # Health check
            health_monitor.record_success()

            logger.info(f"Completed iteration #{iteration}. Active signals: {signal_tracker.get_active_count()}")

            # Sleep with shutdown check
            for _ in range(check_interval):
                if _shutdown_requested:
                    break
                time.sleep(1)

    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        notifier.send_error_message(f"Bot crashed: {e}")
        raise

    finally:
        # Graceful shutdown
        logger.info("Initiating graceful shutdown...")

        # Stop background services
        heartbeat.stop()
        daily_summary.stop()

        # Get final stats
        stats = heartbeat.get_stats()
        perf = signal_tracker.get_performance_summary()

        shutdown_msg = "🛑 <b>Enhanced Trading Bot Stopped</b>\n\n"
        shutdown_msg += f"<b>Session Summary:</b>\n"
        shutdown_msg += f"  ⏱ Uptime: {stats['uptime_str']}\n"
        shutdown_msg += f"  🔄 Iterations: {stats['iterations']}\n"
        shutdown_msg += f"  📊 Signals Sent: {stats['signals']}\n"

        if perf['today']:
            shutdown_msg += f"\n<b>Today's Performance:</b>\n"
            shutdown_msg += f"  ✅ Wins: {perf['today'].get('wins', 0)}\n"
            shutdown_msg += f"  ❌ Losses: {perf['today'].get('losses', 0)}\n"
            shutdown_msg += f"  🎯 Win Rate: {perf['today'].get('win_rate', 0):.1f}%\n"

        shutdown_msg += f"\n<i>Shutdown completed gracefully</i>"

        notifier.send_message(shutdown_msg)
        logger.info("Bot stopped gracefully")
        print("\n\n🛑 Bot stopped gracefully")


if __name__ == "__main__":
    main()
