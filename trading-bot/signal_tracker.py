#!/usr/bin/env python3
"""
Signal Tracker Module
Monitors active signals and detects when TP or SL is hit.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd

from signal_database import (
    SignalDatabase, STATUS_ACTIVE, STATUS_TP_HIT,
    STATUS_SL_HIT, STATUS_EXPIRED
)

logger = logging.getLogger(__name__)


class SignalTracker:
    """
    Tracks active trading signals and monitors for TP/SL hits.

    Features:
    - Real-time monitoring of active signals
    - TP/SL hit detection
    - Automatic signal expiry
    - Performance statistics
    - Callback support for notifications
    """

    def __init__(self, database: SignalDatabase, config: dict = None):
        self.db = database
        self.config = config or {}
        self._callbacks = {
            'on_tp_hit': [],
            'on_sl_hit': [],
            'on_expired': [],
            'on_signal_created': []
        }
        self._last_check = {}
        self._signal_expiry_hours = self.config.get('signal_expiry_hours', 24)

        logger.info("Signal tracker initialized")

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for signal events"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
            logger.debug(f"Registered callback for {event}")

    def _trigger_callbacks(self, event: str, signal: Dict, **kwargs):
        """Trigger all registered callbacks for an event"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(signal, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def add_signal(self, signal: Dict) -> Optional[str]:
        """
        Add a new signal for tracking.
        Returns the signal_id if successful.
        """
        signal_id = self.db.add_signal(signal)

        if signal_id:
            signal['signal_id'] = signal_id
            self._trigger_callbacks('on_signal_created', signal)
            logger.info(f"New signal tracking: {signal_id}")

        return signal_id

    def check_signal_status(self, signal: Dict, current_price: float,
                            high_price: float = None, low_price: float = None) -> Optional[str]:
        """
        Check if a signal has hit TP or SL.

        Args:
            signal: The signal dictionary from database
            current_price: Current close price
            high_price: Candle high (for more accurate TP/SL detection)
            low_price: Candle low (for more accurate TP/SL detection)

        Returns:
            Status string if signal closed, None if still active
        """
        if signal['status'] != STATUS_ACTIVE:
            return None

        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        action = signal['action']

        # Use high/low for more accurate detection, fallback to close price
        check_high = high_price if high_price is not None else current_price
        check_low = low_price if low_price is not None else current_price

        new_status = None
        exit_price = None

        if action == 'LONG':
            # LONG: TP hit if high >= take_profit, SL hit if low <= stop_loss
            if check_high >= take_profit:
                new_status = STATUS_TP_HIT
                exit_price = take_profit
            elif check_low <= stop_loss:
                new_status = STATUS_SL_HIT
                exit_price = stop_loss
        else:  # SHORT
            # SHORT: TP hit if low <= take_profit, SL hit if high >= stop_loss
            if check_low <= take_profit:
                new_status = STATUS_TP_HIT
                exit_price = take_profit
            elif check_high >= stop_loss:
                new_status = STATUS_SL_HIT
                exit_price = stop_loss

        if new_status:
            self.db.update_signal_status(signal['signal_id'], new_status, exit_price)

            # Get updated signal with P&L info
            updated_signal = self.db.get_signal_by_id(signal['signal_id'])

            if new_status == STATUS_TP_HIT:
                self._trigger_callbacks('on_tp_hit', updated_signal, exit_price=exit_price)
            else:
                self._trigger_callbacks('on_sl_hit', updated_signal, exit_price=exit_price)

            return new_status

        return None

    def check_all_active_signals(self, price_data: Dict[str, Dict]) -> List[Dict]:
        """
        Check all active signals against current price data.

        Args:
            price_data: Dict mapping symbol to price info
                        e.g., {'BTC/USDT': {'close': 67000, 'high': 67100, 'low': 66900}}

        Returns:
            List of signals that were closed (TP or SL hit)
        """
        closed_signals = []
        active_signals = self.db.get_active_signals()

        for signal in active_signals:
            symbol = signal['symbol']

            if symbol not in price_data:
                continue

            prices = price_data[symbol]
            status = self.check_signal_status(
                signal,
                current_price=prices.get('close'),
                high_price=prices.get('high'),
                low_price=prices.get('low')
            )

            if status:
                closed_signals.append({
                    'signal': signal,
                    'status': status,
                    'exit_price': prices.get('close')
                })

        return closed_signals

    def check_signal_from_ohlcv(self, signal: Dict, df: pd.DataFrame) -> Optional[str]:
        """
        Check if signal hit TP/SL using OHLCV dataframe.
        Checks the latest bar.
        """
        if df is None or len(df) < 1:
            return None

        last_bar = df.iloc[-1]
        return self.check_signal_status(
            signal,
            current_price=last_bar['close'],
            high_price=last_bar['high'],
            low_price=last_bar['low']
        )

    def expire_old_signals(self) -> List[Dict]:
        """Expire signals that have exceeded the expiry time"""
        expired = []
        active_signals = self.db.get_active_signals()
        expiry_time = datetime.now() - timedelta(hours=self._signal_expiry_hours)

        for signal in active_signals:
            created_at = datetime.fromisoformat(signal['created_at'])
            if created_at < expiry_time:
                self.db.update_signal_status(signal['signal_id'], STATUS_EXPIRED)
                updated_signal = self.db.get_signal_by_id(signal['signal_id'])
                self._trigger_callbacks('on_expired', updated_signal)
                expired.append(updated_signal)

        if expired:
            logger.info(f"Expired {len(expired)} old signals")

        return expired

    def get_active_count(self) -> int:
        """Get count of active signals"""
        return len(self.db.get_active_signals())

    def get_active_by_symbol(self, symbol: str) -> List[Dict]:
        """Get active signals for a symbol"""
        return self.db.get_active_signals_by_symbol(symbol)

    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        stats = self.db.get_overall_stats()
        today_stats = self.db.get_daily_stats()

        return {
            'overall': stats,
            'today': today_stats or {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl_percent': 0
            }
        }

    def format_tp_hit_message(self, signal: Dict) -> str:
        """Format TP hit notification message"""
        symbol = signal['symbol']
        action = signal['action']
        entry = signal['entry_price']
        exit_price = signal['exit_price']
        pnl = signal['pnl_percent']
        duration = signal['duration_minutes']
        signal_id = signal['signal_id']

        # Get today's stats
        today_stats = self.db.get_daily_stats()
        wins = today_stats['wins'] if today_stats else 1
        losses = today_stats['losses'] if today_stats else 0
        win_rate = today_stats['win_rate'] if today_stats else 100

        hours = duration // 60
        mins = duration % 60
        duration_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

        message = f"✅ <b>TARGET HIT - {symbol}</b>\n\n"
        message += f"🎯 Take Profit reached!\n\n"
        message += f"📊 <b>Signal ID:</b> #{signal_id}\n"
        message += f"📈 <b>Direction:</b> {action}\n"
        message += f"⏱ <b>Duration:</b> {duration_str}\n\n"
        message += f"💰 <b>Entry:</b> {entry:.6f}\n"
        message += f"🎯 <b>Exit:</b> {exit_price:.6f}\n"
        message += f"📈 <b>Profit:</b> +{pnl:.2f}%\n\n"
        message += f"📊 <b>Today's Stats:</b>\n"
        message += f"  Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%"

        return message

    def format_sl_hit_message(self, signal: Dict) -> str:
        """Format SL hit notification message"""
        symbol = signal['symbol']
        action = signal['action']
        entry = signal['entry_price']
        exit_price = signal['exit_price']
        pnl = signal['pnl_percent']
        duration = signal['duration_minutes']
        signal_id = signal['signal_id']

        # Get today's stats
        today_stats = self.db.get_daily_stats()
        wins = today_stats['wins'] if today_stats else 0
        losses = today_stats['losses'] if today_stats else 1
        win_rate = today_stats['win_rate'] if today_stats else 0

        hours = duration // 60
        mins = duration % 60
        duration_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

        message = f"❌ <b>STOPPED OUT - {symbol}</b>\n\n"
        message += f"🛑 Stop Loss triggered\n\n"
        message += f"📊 <b>Signal ID:</b> #{signal_id}\n"
        message += f"📈 <b>Direction:</b> {action}\n"
        message += f"⏱ <b>Duration:</b> {duration_str}\n\n"
        message += f"💰 <b>Entry:</b> {entry:.6f}\n"
        message += f"🛑 <b>Exit:</b> {exit_price:.6f}\n"
        message += f"📉 <b>Loss:</b> {pnl:.2f}%\n\n"
        message += f"📊 <b>Today's Stats:</b>\n"
        message += f"  Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%"

        return message

    def format_expired_message(self, signal: Dict) -> str:
        """Format signal expired notification message"""
        symbol = signal['symbol']
        action = signal['action']
        entry = signal['entry_price']
        signal_id = signal['signal_id']
        duration = signal.get('duration_minutes', 0)

        hours = duration // 60
        mins = duration % 60
        duration_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

        message = f"⏰ <b>SIGNAL EXPIRED - {symbol}</b>\n\n"
        message += f"Signal #{signal_id} has expired without hitting TP or SL.\n\n"
        message += f"📈 <b>Direction:</b> {action}\n"
        message += f"💰 <b>Entry:</b> {entry:.6f}\n"
        message += f"⏱ <b>Duration:</b> {duration_str}"

        return message


class SignalCooldownManager:
    """Manages cooldown periods for signals to prevent spam"""

    def __init__(self, database: SignalDatabase, cooldown_minutes: int = 60):
        self.db = database
        self.cooldown_minutes = cooldown_minutes
        self._last_signal_time = {}

    def can_signal(self, symbol: str, action: str) -> bool:
        """Check if a new signal is allowed for this symbol/action"""
        key = f"{symbol}_{action}"

        # Check in-memory cache first
        if key in self._last_signal_time:
            elapsed = (datetime.now() - self._last_signal_time[key]).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                logger.debug(f"Cooldown active for {key}: {elapsed:.1f} min elapsed of {self.cooldown_minutes}")
                return False

        # Check database for recent signals
        hours = self.cooldown_minutes / 60 + 1  # Add buffer
        recent = self.db.get_signals_for_symbol_in_timeframe(symbol, hours=hours)

        for signal in recent:
            if signal['action'] == action:
                created = datetime.fromisoformat(signal['created_at'])
                elapsed = (datetime.now() - created).total_seconds() / 60
                if elapsed < self.cooldown_minutes:
                    logger.debug(f"Recent signal found for {key}, cooldown active")
                    return False

        return True

    def record_signal(self, symbol: str, action: str):
        """Record that a signal was sent"""
        key = f"{symbol}_{action}"
        self._last_signal_time[key] = datetime.now()

    def get_remaining_cooldown(self, symbol: str, action: str) -> int:
        """Get remaining cooldown time in minutes"""
        key = f"{symbol}_{action}"

        if key in self._last_signal_time:
            elapsed = (datetime.now() - self._last_signal_time[key]).total_seconds() / 60
            remaining = self.cooldown_minutes - elapsed
            return max(0, int(remaining))

        return 0
