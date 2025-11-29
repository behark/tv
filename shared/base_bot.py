#!/usr/bin/env python3
"""
Base Bot Class
Provides common bot functionality for all tiers.
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Optional, List
from abc import ABC, abstractmethod

import yaml
import pandas as pd

from data_client import EnhancedExchangeClient
from notifier import TelegramNotifier
from signal_database import SignalDatabase
from signal_tracker import SignalTracker, SignalCooldownManager
from heartbeat import HeartbeatMonitor
from accuracy_filters import AccuracyFilterManager

logger = logging.getLogger(__name__)


class BaseBot(ABC):
    """
    Abstract base class for trading bots.
    Each tier inherits from this and provides tier-specific components.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.running = False
        self.loop_count = 0

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.exchange = self._init_exchange()
        self.notifier = self._init_notifier()
        self.database = self._init_database()
        self.tracker = self._init_tracker()
        self.cooldown = self._init_cooldown()
        self.heartbeat = self._init_heartbeat()
        self.accuracy_filters = self._init_accuracy_filters()

        # Initialize tier-specific components
        self.indicators = self._create_indicators()
        self.strategy = self._create_strategy()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(f"{self.strategy.strategy_name} bot initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file with environment variable substitution."""
        with open(config_path, 'r') as f:
            content = f.read()

        # Substitute environment variables
        content = os.path.expandvars(content)

        config = yaml.safe_load(content)
        self._validate_config(config)
        return config

    def _validate_config(self, config: dict):
        """Validate required configuration sections."""
        required = ['telegram', 'exchange', 'pairs', 'strategy']
        for section in required:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        if not config['telegram'].get('bot_token') or config['telegram']['bot_token'].startswith('$'):
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")
        if not config['telegram'].get('chat_id') or str(config['telegram']['chat_id']).startswith('$'):
            raise ValueError("TELEGRAM_CHAT_ID environment variable not set")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'bot.log')

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def _init_exchange(self) -> EnhancedExchangeClient:
        """Initialize exchange client."""
        exchange_config = self.config['exchange']
        return EnhancedExchangeClient(
            exchange_id=exchange_config.get('name', 'mexc'),
            config={
                'sandbox': exchange_config.get('sandbox', False),
                'timeout': exchange_config.get('timeout', 30000),
                'enableRateLimit': exchange_config.get('enableRateLimit', True)
            }
        )

    def _init_notifier(self) -> TelegramNotifier:
        """Initialize Telegram notifier."""
        telegram_config = self.config['telegram']
        return TelegramNotifier(
            bot_token=telegram_config['bot_token'],
            chat_id=str(telegram_config['chat_id'])
        )

    def _init_database(self) -> SignalDatabase:
        """Initialize signal database."""
        tracking_config = self.config.get('signal_tracking', {})
        db_path = tracking_config.get('database_path', 'signals.db')
        return SignalDatabase(db_path)

    def _init_tracker(self) -> SignalTracker:
        """Initialize signal tracker."""
        tracking_config = self.config.get('signal_tracking', {})
        return SignalTracker(
            database=self.database,
            config={'signal_expiry_hours': tracking_config.get('signal_expiry_hours', 24)}
        )

    def _init_cooldown(self) -> SignalCooldownManager:
        """Initialize cooldown manager."""
        tracking_config = self.config.get('signal_tracking', {})
        return SignalCooldownManager(
            database=self.database,
            cooldown_minutes=tracking_config.get('cooldown_minutes', 60)
        )

    def _init_heartbeat(self) -> Optional[HeartbeatMonitor]:
        """Initialize heartbeat monitor."""
        heartbeat_config = self.config.get('heartbeat', {})
        if not heartbeat_config.get('enabled', False):
            return None

        return HeartbeatMonitor(
            notifier=self.notifier,
            interval_seconds=heartbeat_config.get('interval_seconds', 300),
            include_stats=heartbeat_config.get('include_stats', True)
        )

    def _init_accuracy_filters(self) -> Optional[AccuracyFilterManager]:
        """Initialize accuracy filters."""
        filter_config = self.config.get('accuracy_filters', {})
        if not filter_config.get('enabled', False):
            return None

        return AccuracyFilterManager(filter_config)

    @abstractmethod
    def _create_indicators(self):
        """Create tier-specific indicator calculator. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _create_strategy(self):
        """Create tier-specific strategy. Must be implemented by subclass."""
        pass

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received...")
        self.running = False

    async def _check_available_pairs(self) -> Dict[str, bool]:
        """Check which pairs are available on the exchange."""
        pairs = self.config['pairs']
        available = {}

        await self.exchange.load_markets()

        for pair in pairs:
            symbol = self.exchange.format_symbol(pair)
            available[pair] = symbol in self.exchange.exchange.markets

        return available

    async def _fetch_and_process(self, symbol: str) -> Optional[Dict]:
        """Fetch data for a symbol and check for signals."""
        try:
            timeframe = self.config.get('timeframe', '15m')
            formatted_symbol = self.exchange.format_symbol(symbol)

            # Fetch OHLCV data
            df = await self.exchange.fetch_ohlcv(formatted_symbol, timeframe)

            if df is None or len(df) < 50:
                logger.debug(f"Insufficient data for {symbol}")
                return None

            # Calculate indicators
            df = self.indicators.calculate_all(df)

            if df is None or len(df) < 2:
                return None

            # Check active signals for TP/SL
            await self._check_active_signals(symbol, df)

            # Check for new signals
            signal = self.strategy.check_signals(symbol, df)

            if signal:
                # Apply accuracy filters
                if self.accuracy_filters:
                    passed, reason = self.accuracy_filters.filter_signal(signal, df)
                    if not passed:
                        logger.info(f"Signal filtered out: {reason}")
                        return None

                # Check cooldown
                if not self.cooldown.can_signal(symbol, signal['action']):
                    logger.debug(f"Signal in cooldown for {symbol} {signal['action']}")
                    return None

                return signal

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

        return None

    async def _check_active_signals(self, symbol: str, df: pd.DataFrame):
        """Check active signals for TP/SL hits."""
        active_signals = self.tracker.get_active_by_symbol(symbol)

        for signal in active_signals:
            status = self.tracker.check_signal_from_ohlcv(signal, df)

            if status == 'TP_HIT':
                updated = self.database.get_signal_by_id(signal['signal_id'])
                message = self.tracker.format_tp_hit_message(updated)
                self.notifier.send_message(message, priority="high")
                logger.info(f"TP HIT for {symbol}: {updated['pnl_percent']:.2f}%")

            elif status == 'SL_HIT':
                updated = self.database.get_signal_by_id(signal['signal_id'])
                message = self.tracker.format_sl_hit_message(updated)
                self.notifier.send_message(message, priority="high")
                logger.info(f"SL HIT for {symbol}: {updated['pnl_percent']:.2f}%")

    async def _process_signal(self, signal: Dict):
        """Process and send a new signal."""
        # Add to tracker
        signal_id = self.tracker.add_signal(signal)

        if signal_id:
            signal['signal_id'] = signal_id
            self.cooldown.record_signal(signal['symbol'], signal['action'])

            # Send notification
            success = self.notifier.send_signal_alert(signal)
            if success:
                logger.info(f"Signal sent: {signal['action']} {signal['symbol']} (ID: {signal_id})")
                if self.heartbeat:
                    self.heartbeat.increment_signals()
            else:
                logger.error(f"Failed to send signal notification")

    async def _main_loop(self):
        """Main monitoring loop."""
        pairs = self.config['pairs']
        interval = 60  # Check every 60 seconds

        while self.running:
            try:
                self.loop_count += 1

                for pair in pairs:
                    if not self.running:
                        break

                    signal = await self._fetch_and_process(pair)

                    if signal:
                        await self._process_signal(signal)

                    # Small delay between pairs
                    await asyncio.sleep(0.5)

                # Expire old signals
                if self.loop_count % 10 == 0:
                    self.tracker.expire_old_signals()

                # Heartbeat
                if self.heartbeat:
                    self.heartbeat.increment_iteration()

                # Wait for next iteration
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)

    async def run(self):
        """Run the bot."""
        self.running = True

        try:
            # Check available pairs
            available = await self._check_available_pairs()
            available_count = sum(1 for v in available.values() if v)

            logger.info(f"Starting {self.strategy.strategy_name} bot")
            logger.info(f"Monitoring {available_count}/{len(available)} pairs")

            # Send startup message
            self.notifier.send_startup_message(self.config['pairs'], available)

            # Start heartbeat
            if self.heartbeat:
                self.heartbeat.start()

            # Run main loop
            await self._main_loop()

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.notifier.send_error_message(str(e))

        finally:
            self.running = False
            if self.heartbeat:
                self.heartbeat.stop()
            logger.info("Bot stopped")

    def start(self):
        """Start the bot (synchronous entry point)."""
        asyncio.run(self.run())
