#!/usr/bin/env python3
"""
Heartbeat and Daily Summary Module
Provides periodic health checks and daily performance reports.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
import schedule

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """
    Manages periodic heartbeat messages and health monitoring.
    """

    def __init__(self, notifier, config: dict = None):
        self.notifier = notifier
        self.config = config or {}
        self._running = False
        self._thread = None
        self._last_heartbeat = None
        self._start_time = datetime.now()
        self._iteration_count = 0
        self._error_count = 0
        self._signals_count = 0

        # Heartbeat settings
        self.heartbeat_interval_hours = self.config.get('heartbeat_interval_hours', 6)
        self.enable_heartbeat = self.config.get('enable_heartbeat', True)

        logger.info(f"Heartbeat manager initialized (interval: {self.heartbeat_interval_hours}h)")

    def start(self):
        """Start the heartbeat background thread"""
        if not self.enable_heartbeat:
            logger.info("Heartbeat disabled in config")
            return

        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.info("Heartbeat thread started")

    def stop(self):
        """Stop the heartbeat thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Heartbeat thread stopped")

    def _heartbeat_loop(self):
        """Background loop for sending heartbeat messages"""
        while self._running:
            try:
                # Wait for the interval
                sleep_seconds = self.heartbeat_interval_hours * 3600
                time.sleep(sleep_seconds)

                if self._running:
                    self.send_heartbeat()

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(60)  # Wait a minute on error

    def send_heartbeat(self) -> bool:
        """Send a heartbeat status message"""
        try:
            uptime = datetime.now() - self._start_time
            uptime_str = self._format_uptime(uptime)

            message = "💓 <b>Bot Heartbeat</b>\n\n"
            message += f"✅ Status: Running\n"
            message += f"⏱ Uptime: {uptime_str}\n"
            message += f"🔄 Iterations: {self._iteration_count}\n"
            message += f"📊 Signals sent: {self._signals_count}\n"

            if self._error_count > 0:
                message += f"⚠️ Errors: {self._error_count}\n"

            # Add notifier stats
            notifier_stats = self.notifier.get_stats()
            message += f"\n📬 <b>Notification Stats:</b>\n"
            message += f"  Sent: {notifier_stats['success_count']}\n"
            message += f"  Failed: {notifier_stats['failed_count']}\n"
            message += f"  Queued: {notifier_stats['queue_size']}\n"

            message += f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            success = self.notifier.send_message(message)
            if success:
                self._last_heartbeat = datetime.now()
                logger.info("Heartbeat sent successfully")

            return success

        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            return False

    def _format_uptime(self, td: timedelta) -> str:
        """Format timedelta as readable string"""
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or not parts:
            parts.append(f"{minutes}m")

        return " ".join(parts)

    def increment_iteration(self):
        """Increment iteration counter"""
        self._iteration_count += 1

    def increment_signals(self, count: int = 1):
        """Increment signals counter"""
        self._signals_count += count

    def increment_errors(self, count: int = 1):
        """Increment error counter"""
        self._error_count += count

    def get_stats(self) -> Dict:
        """Get heartbeat statistics"""
        uptime = datetime.now() - self._start_time
        return {
            'uptime': uptime,
            'uptime_str': self._format_uptime(uptime),
            'iterations': self._iteration_count,
            'signals': self._signals_count,
            'errors': self._error_count,
            'last_heartbeat': self._last_heartbeat
        }


class DailySummaryManager:
    """
    Manages daily performance summary reports.
    """

    def __init__(self, notifier, signal_db, config: dict = None):
        self.notifier = notifier
        self.signal_db = signal_db
        self.config = config or {}
        self._running = False
        self._thread = None
        self._last_summary = None

        # Summary settings
        self.summary_hour = self.config.get('daily_summary_hour', 0)  # Midnight
        self.summary_minute = self.config.get('daily_summary_minute', 0)
        self.enable_summary = self.config.get('enable_daily_summary', True)

        logger.info(f"Daily summary manager initialized (time: {self.summary_hour:02d}:{self.summary_minute:02d})")

    def start(self):
        """Start the daily summary scheduler"""
        if not self.enable_summary:
            logger.info("Daily summary disabled in config")
            return

        self._running = True
        self._thread = threading.Thread(target=self._summary_loop, daemon=True)
        self._thread.start()
        logger.info("Daily summary thread started")

    def stop(self):
        """Stop the summary thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Daily summary thread stopped")

    def _summary_loop(self):
        """Background loop for sending daily summaries"""
        while self._running:
            try:
                now = datetime.now()
                target = now.replace(hour=self.summary_hour, minute=self.summary_minute, second=0, microsecond=0)

                # If target time has passed today, schedule for tomorrow
                if now >= target:
                    target += timedelta(days=1)

                # Calculate sleep time
                sleep_seconds = (target - now).total_seconds()

                # Sleep in chunks to allow for clean shutdown
                while sleep_seconds > 0 and self._running:
                    chunk = min(sleep_seconds, 60)
                    time.sleep(chunk)
                    sleep_seconds -= chunk

                if self._running:
                    self.send_daily_summary()

            except Exception as e:
                logger.error(f"Error in summary loop: {e}")
                time.sleep(60)

    def send_daily_summary(self, date: str = None) -> bool:
        """Send daily performance summary"""
        try:
            if date is None:
                # Get yesterday's date for the summary
                yesterday = datetime.now() - timedelta(days=1)
                date = yesterday.strftime('%Y-%m-%d')

            # Get stats from database
            daily_stats = self.signal_db.get_daily_stats(date)
            overall_stats = self.signal_db.get_overall_stats()

            message = f"📊 <b>Daily Trading Summary</b>\n"
            message += f"📅 {date}\n\n"

            if daily_stats:
                message += f"<b>Today's Performance:</b>\n"
                message += f"  📈 Total Signals: {daily_stats['total_signals']}\n"
                message += f"  ✅ Wins: {daily_stats['wins']}\n"
                message += f"  ❌ Losses: {daily_stats['losses']}\n"
                message += f"  ⏰ Expired: {daily_stats['expired']}\n"
                message += f"  🎯 Win Rate: {daily_stats['win_rate']:.1f}%\n"
                message += f"  💰 Total P&L: {daily_stats['total_pnl_percent']:+.2f}%\n"

                if daily_stats['best_trade_pnl'] > 0:
                    message += f"  🏆 Best Trade: +{daily_stats['best_trade_pnl']:.2f}%\n"
                if daily_stats['worst_trade_pnl'] < 0:
                    message += f"  📉 Worst Trade: {daily_stats['worst_trade_pnl']:.2f}%\n"

                message += f"  ⏱ Avg Duration: {daily_stats['avg_duration_minutes']:.0f}m\n"
            else:
                message += "<i>No trading activity today</i>\n"

            message += f"\n<b>Overall Statistics:</b>\n"
            message += f"  📊 Total Signals: {overall_stats['total_signals']}\n"
            message += f"  🔵 Active: {overall_stats['active']}\n"
            message += f"  🎯 Win Rate: {overall_stats['win_rate']:.1f}%\n"
            message += f"  💰 Total P&L: {overall_stats['total_pnl_percent']:+.2f}%\n"

            message += f"\n<i>Report generated at {datetime.now().strftime('%H:%M:%S')}</i>"

            success = self.notifier.send_message(message)
            if success:
                self._last_summary = datetime.now()
                logger.info(f"Daily summary sent for {date}")

            return success

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False

    def send_weekly_summary(self) -> bool:
        """Send weekly performance summary"""
        try:
            overall_stats = self.signal_db.get_overall_stats()

            # Get last 7 days of data
            weekly_signals = self.signal_db.get_recent_signals(hours=168)  # 7 days

            wins = sum(1 for s in weekly_signals if s['status'] == 'TP_HIT')
            losses = sum(1 for s in weekly_signals if s['status'] == 'SL_HIT')
            total_pnl = sum(s['pnl_percent'] or 0 for s in weekly_signals if s['status'] in ['TP_HIT', 'SL_HIT'])

            message = "📈 <b>Weekly Trading Summary</b>\n\n"
            message += f"<b>Last 7 Days:</b>\n"
            message += f"  📊 Total Signals: {len(weekly_signals)}\n"
            message += f"  ✅ Wins: {wins}\n"
            message += f"  ❌ Losses: {losses}\n"

            if wins + losses > 0:
                win_rate = (wins / (wins + losses)) * 100
                message += f"  🎯 Win Rate: {win_rate:.1f}%\n"

            message += f"  💰 Total P&L: {total_pnl:+.2f}%\n"

            message += f"\n<b>All-Time Stats:</b>\n"
            message += f"  📊 Total: {overall_stats['total_signals']}\n"
            message += f"  🎯 Win Rate: {overall_stats['win_rate']:.1f}%\n"
            message += f"  💰 Total P&L: {overall_stats['total_pnl_percent']:+.2f}%\n"

            return self.notifier.send_message(message)

        except Exception as e:
            logger.error(f"Error sending weekly summary: {e}")
            return False


# Alias for compatibility
HeartbeatMonitor = HeartbeatManager


class HealthMonitor:
    """
    Monitors bot health and sends alerts on issues.
    """

    def __init__(self, notifier, config: dict = None):
        self.notifier = notifier
        self.config = config or {}
        self._last_successful_check = None
        self._consecutive_failures = 0
        self._alert_threshold = self.config.get('alert_after_failures', 5)

    def record_success(self):
        """Record a successful operation"""
        self._last_successful_check = datetime.now()
        self._consecutive_failures = 0

    def record_failure(self, error: str = None):
        """Record a failed operation"""
        self._consecutive_failures += 1

        if self._consecutive_failures >= self._alert_threshold:
            self._send_health_alert(error)

    def _send_health_alert(self, error: str = None):
        """Send health alert"""
        message = "⚠️ <b>Bot Health Alert</b>\n\n"
        message += f"🔴 {self._consecutive_failures} consecutive failures detected!\n"

        if error:
            message += f"\nLast error: {error[:200]}\n"

        if self._last_successful_check:
            elapsed = datetime.now() - self._last_successful_check
            message += f"\nLast success: {elapsed.total_seconds() / 60:.0f} minutes ago"

        self.notifier.send_message(message, priority="high")

    def get_health_status(self) -> Dict:
        """Get current health status"""
        now = datetime.now()

        if self._last_successful_check:
            since_success = (now - self._last_successful_check).total_seconds()
            status = 'healthy' if since_success < 600 else 'degraded' if since_success < 1800 else 'unhealthy'
        else:
            status = 'unknown'

        return {
            'status': status,
            'consecutive_failures': self._consecutive_failures,
            'last_success': self._last_successful_check,
            'alert_threshold': self._alert_threshold
        }
