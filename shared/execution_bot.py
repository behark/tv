#!/usr/bin/env python3
"""
Execution Bot
Enhanced bot that actually executes trades on exchanges.

WARNING: This bot executes REAL trades with REAL money.
Always test thoroughly with sandbox mode first.
"""

import asyncio
import logging
import signal
import os
from datetime import datetime
from typing import Dict, Optional
from abc import abstractmethod

import yaml

from base_bot import BaseBot
from trade_executor import TradeExecutor, Position
from notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class ExecutionBot(BaseBot):
    """
    Enhanced bot with trade execution capabilities.

    Extends BaseBot to add:
    - Real order placement
    - Position management
    - Risk management
    - Execution notifications
    """

    def __init__(self, config_path: str, api_key: str, api_secret: str,
                 sandbox: bool = True):
        """
        Initialize execution bot.

        Args:
            config_path: Path to config.yaml
            api_key: Exchange API key
            api_secret: Exchange API secret
            sandbox: Use sandbox/testnet mode (RECOMMENDED)
        """
        # Load config first
        self.config = self._load_config_file(config_path)

        # Validate trading is enabled
        if not self.config.get('execution', {}).get('enabled', False):
            raise ValueError("Execution not enabled in config. Set execution.enabled = true")

        # Initialize executor
        exchange_id = self.config['exchange'].get('name', 'mexc')
        executor_config = {
            'max_position_size_percent': self.config.get('execution', {}).get('max_position_size_percent', 5.0),
            'max_open_positions': self.config.get('execution', {}).get('max_open_positions', 3),
            'risk_per_trade_percent': self.config.get('execution', {}).get('risk_per_trade_percent', 1.0),
            'use_stop_loss': self.config.get('execution', {}).get('use_stop_loss', True),
            'use_take_profit': self.config.get('execution', {}).get('use_take_profit', True),
        }

        self.executor = TradeExecutor(
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            config=executor_config,
            sandbox=sandbox
        )

        self.sandbox = sandbox

        # Call parent init
        super().__init__(config_path)

        # Execution state
        self.execution_enabled = True
        self.total_trades_executed = 0

        logger.info(f"ExecutionBot initialized (sandbox={sandbox})")

    def _load_config_file(self, config_path: str) -> dict:
        """Load configuration file."""
        with open(config_path, 'r') as f:
            content = f.read()
        content = os.path.expandvars(content)
        return yaml.safe_load(content)

    async def _process_signal(self, signal: Dict):
        """
        Process signal with trade execution.

        Override parent method to add execution.
        """
        # First, do normal signal processing (database, notifications)
        signal_id = self.tracker.add_signal(signal)

        if signal_id:
            signal['signal_id'] = signal_id
            self.cooldown.record_signal(signal['symbol'], signal['action'])

            # Send signal notification
            success = self.notifier.send_signal_alert(signal)
            if success:
                logger.info(f"Signal sent: {signal['action']} {signal['symbol']} (ID: {signal_id})")
                if self.heartbeat:
                    self.heartbeat.increment_signals()

            # Execute trade if enabled
            if self.execution_enabled:
                await self._execute_trade(signal)

    async def _execute_trade(self, signal: Dict):
        """Execute a trade based on the signal."""
        try:
            # Confirmation check
            if not self._should_execute(signal):
                logger.info(f"Skipping execution for {signal['symbol']} - conditions not met")
                return

            # Open position
            position = await self.executor.open_position(signal)

            if position:
                self.total_trades_executed += 1

                # Send execution notification
                self._send_execution_notification(position, signal)

                logger.info(f"Trade executed: {position.position_id}")
            else:
                logger.warning(f"Failed to execute trade for {signal['symbol']}")
                self.notifier.send_message(
                    f"⚠️ Failed to execute {signal['action']} on {signal['symbol']}",
                    priority="high"
                )

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            self.notifier.send_error_message(f"Execution error: {str(e)[:100]}")

    def _should_execute(self, signal: Dict) -> bool:
        """
        Check if we should execute this signal.

        Add any additional filters here.
        """
        # Check minimum score
        min_score = self.config.get('execution', {}).get('min_score_for_execution', 5.0)
        if signal.get('score', 0) < min_score:
            return False

        # Check if we have capacity
        open_positions = self.executor.get_open_positions()
        max_positions = self.config.get('execution', {}).get('max_open_positions', 3)
        if len(open_positions) >= max_positions:
            return False

        # Check if we already have a position on this symbol
        symbol_positions = [p for p in open_positions if p.symbol == signal['symbol']]
        if symbol_positions:
            return False

        return True

    def _send_execution_notification(self, position: Position, signal: Dict):
        """Send trade execution notification."""
        action_emoji = "🟢" if position.side == 'long' else "🔴"
        mode = "🧪 SANDBOX" if self.sandbox else "💰 LIVE"

        message = f"{action_emoji} <b>TRADE EXECUTED</b> {mode}\n\n"
        message += f"<b>Symbol:</b> {position.symbol}\n"
        message += f"<b>Side:</b> {position.side.upper()}\n"
        message += f"<b>Amount:</b> {position.amount:.6f}\n"
        message += f"<b>Entry:</b> {position.entry_price:.4f}\n"

        if position.stop_loss:
            message += f"<b>Stop Loss:</b> {position.stop_loss:.4f}\n"
        if position.take_profit:
            message += f"<b>Take Profit:</b> {position.take_profit:.4f}\n"

        message += f"\n<b>Position ID:</b> {position.position_id}\n"
        message += f"<b>Signal Score:</b> {signal.get('score', 0):.1f}\n"
        message += f"\n<i>Trade #{self.total_trades_executed}</i>"

        self.notifier.send_message(message, priority="high")

    async def _check_active_signals(self, symbol: str, df):
        """
        Check active signals for TP/SL hits.

        Override to also sync with exchange positions.
        """
        # Call parent method for signal tracking
        await super()._check_active_signals(symbol, df)

        # Sync executor positions with exchange
        await self.executor.sync_positions()

        # Check for closed positions and notify
        for position in self.executor.get_closed_positions():
            # Check if we already notified (simple check)
            if hasattr(position, '_notified'):
                continue

            self._send_close_notification(position)
            position._notified = True

    def _send_close_notification(self, position: Position):
        """Send position close notification."""
        pnl = position.pnl_percent or 0
        emoji = "✅" if pnl > 0 else "❌"

        message = f"{emoji} <b>POSITION CLOSED</b>\n\n"
        message += f"<b>Symbol:</b> {position.symbol}\n"
        message += f"<b>Side:</b> {position.side.upper()}\n"
        message += f"<b>Entry:</b> {position.entry_price:.4f}\n"
        message += f"<b>Exit:</b> {position.exit_price:.4f}\n"
        message += f"<b>P&L:</b> {pnl:+.2f}%\n"

        if position.pnl_absolute:
            message += f"<b>P&L $:</b> ${position.pnl_absolute:+.2f}\n"

        message += f"\n<b>Position ID:</b> {position.position_id}"

        priority = "high" if abs(pnl) > 2 else "normal"
        self.notifier.send_message(message, priority=priority)

    async def run(self):
        """Run the execution bot."""
        self.running = True

        try:
            # Check balance
            balance = await self.executor.get_balance('USDT')
            logger.info(f"Available balance: ${balance:.2f} USDT")

            if balance < 10:
                logger.warning("Low balance warning!")
                self.notifier.send_message("⚠️ Low balance warning: ${balance:.2f} USDT")

            # Check available pairs
            available = await self._check_available_pairs()
            available_count = sum(1 for v in available.values() if v)

            mode = "SANDBOX" if self.sandbox else "LIVE"
            logger.info(f"Starting ExecutionBot in {mode} mode")
            logger.info(f"Monitoring {available_count}/{len(available)} pairs")

            # Send startup message
            startup_msg = f"🤖 <b>Execution Bot Started</b>\n\n"
            startup_msg += f"Mode: {'🧪 Sandbox' if self.sandbox else '💰 LIVE'}\n"
            startup_msg += f"Balance: ${balance:.2f} USDT\n"
            startup_msg += f"Pairs: {available_count}\n"
            startup_msg += f"Max Positions: {self.config.get('execution', {}).get('max_open_positions', 3)}\n"
            startup_msg += f"Risk/Trade: {self.config.get('execution', {}).get('risk_per_trade_percent', 1.0)}%"

            self.notifier.send_message(startup_msg)

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

            # Close all positions if configured
            if self.config.get('execution', {}).get('close_on_shutdown', False):
                await self._close_all_positions()

            if self.heartbeat:
                self.heartbeat.stop()

            await self.executor.close()
            logger.info("ExecutionBot stopped")

    async def _close_all_positions(self):
        """Close all open positions on shutdown."""
        open_positions = self.executor.get_open_positions()

        if not open_positions:
            return

        logger.info(f"Closing {len(open_positions)} open positions...")

        for position in open_positions:
            try:
                await self.executor.close_position(position.position_id, 'shutdown')
            except Exception as e:
                logger.error(f"Error closing position {position.position_id}: {e}")

    def get_status(self) -> Dict:
        """Get current bot status."""
        open_positions = self.executor.get_open_positions()
        perf = self.executor.get_performance_summary()

        return {
            'running': self.running,
            'sandbox': self.sandbox,
            'execution_enabled': self.execution_enabled,
            'open_positions': len(open_positions),
            'total_trades': perf['total_trades'],
            'win_rate': perf['win_rate'],
            'total_pnl': perf['total_pnl_percent']
        }

    def pause_execution(self):
        """Pause trade execution (keep monitoring)."""
        self.execution_enabled = False
        logger.info("Execution paused")
        self.notifier.send_message("⏸️ Trade execution paused")

    def resume_execution(self):
        """Resume trade execution."""
        self.execution_enabled = True
        logger.info("Execution resumed")
        self.notifier.send_message("▶️ Trade execution resumed")
