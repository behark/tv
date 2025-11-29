#!/usr/bin/env python3
"""
Signal Database Module
Handles SQLite storage for trading signals with full lifecycle tracking.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import json
import os

logger = logging.getLogger(__name__)

# Signal status constants
STATUS_ACTIVE = 'ACTIVE'
STATUS_TP_HIT = 'TP_HIT'
STATUS_SL_HIT = 'SL_HIT'
STATUS_EXPIRED = 'EXPIRED'
STATUS_CANCELLED = 'CANCELLED'


class SignalDatabase:
    """SQLite database for trading signal storage and tracking"""

    def __init__(self, db_path: str = 'signals.db'):
        self.db_path = db_path
        self._init_database()
        logger.info(f"Signal database initialized at {db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    score REAL,
                    max_score REAL,
                    strategy_mode TEXT,
                    timeframe TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP,
                    exit_price REAL,
                    pnl_percent REAL,
                    pnl_absolute REAL,
                    duration_minutes INTEGER,
                    indicators_json TEXT,
                    notes TEXT
                )
            ''')

            # Performance stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    expired INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    total_pnl_percent REAL DEFAULT 0,
                    avg_pnl_percent REAL DEFAULT 0,
                    best_trade_pnl REAL DEFAULT 0,
                    worst_trade_pnl REAL DEFAULT 0,
                    avg_duration_minutes REAL DEFAULT 0
                )
            ''')

            # Daily summaries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    summary_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at)')

            logger.info("Database schema initialized successfully")

    def generate_signal_id(self, symbol: str) -> str:
        """Generate unique signal ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        symbol_short = symbol.replace('/', '').replace('USDT', '')
        return f"{symbol_short}-{timestamp}"

    def add_signal(self, signal: Dict) -> Optional[str]:
        """Add a new signal to the database"""
        try:
            signal_id = self.generate_signal_id(signal['symbol'])

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO signals (
                        signal_id, symbol, action, entry_price, stop_loss, take_profit,
                        score, max_score, strategy_mode, timeframe, status, indicators_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_id,
                    signal['symbol'],
                    signal['action'],
                    signal['entry_price'],
                    signal['stop_loss'],
                    signal['take_profit'],
                    signal.get('score'),
                    signal.get('max_score'),
                    signal.get('strategy_mode', 'unknown'),
                    signal.get('timeframe', '15m'),
                    STATUS_ACTIVE,
                    json.dumps(signal.get('indicators', {}))
                ))

            logger.info(f"Signal added: {signal_id} - {signal['action']} {signal['symbol']}")
            return signal_id

        except Exception as e:
            logger.error(f"Error adding signal: {e}")
            return None

    def update_signal_status(self, signal_id: str, status: str, exit_price: float = None) -> bool:
        """Update signal status (TP_HIT, SL_HIT, EXPIRED, etc.)"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get original signal
                cursor.execute('SELECT * FROM signals WHERE signal_id = ?', (signal_id,))
                row = cursor.fetchone()

                if not row:
                    logger.warning(f"Signal not found: {signal_id}")
                    return False

                entry_price = row['entry_price']
                action = row['action']
                created_at = datetime.fromisoformat(row['created_at'])
                closed_at = datetime.now()
                duration_minutes = int((closed_at - created_at).total_seconds() / 60)

                # Calculate P&L
                pnl_percent = 0
                pnl_absolute = 0
                if exit_price:
                    if action == 'LONG':
                        pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                    else:  # SHORT
                        pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                    pnl_absolute = exit_price - entry_price if action == 'LONG' else entry_price - exit_price

                cursor.execute('''
                    UPDATE signals
                    SET status = ?, exit_price = ?, pnl_percent = ?, pnl_absolute = ?,
                        duration_minutes = ?, closed_at = ?, updated_at = ?
                    WHERE signal_id = ?
                ''', (
                    status, exit_price, pnl_percent, pnl_absolute,
                    duration_minutes, closed_at.isoformat(), closed_at.isoformat(),
                    signal_id
                ))

                # Update daily performance stats
                self._update_daily_stats(row['symbol'], status, pnl_percent, duration_minutes)

            logger.info(f"Signal {signal_id} updated: {status} (P&L: {pnl_percent:.2f}%)")
            return True

        except Exception as e:
            logger.error(f"Error updating signal: {e}")
            return False

    def _update_daily_stats(self, symbol: str, status: str, pnl_percent: float, duration_minutes: int):
        """Update daily performance statistics"""
        today = datetime.now().strftime('%Y-%m-%d')

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get or create today's stats
            cursor.execute('SELECT * FROM performance_stats WHERE date = ?', (today,))
            row = cursor.fetchone()

            if row:
                total_signals = row['total_signals'] + 1
                wins = row['wins'] + (1 if status == STATUS_TP_HIT else 0)
                losses = row['losses'] + (1 if status == STATUS_SL_HIT else 0)
                expired = row['expired'] + (1 if status == STATUS_EXPIRED else 0)
                total_pnl = row['total_pnl_percent'] + pnl_percent
                best = max(row['best_trade_pnl'], pnl_percent)
                worst = min(row['worst_trade_pnl'], pnl_percent)
                avg_duration = ((row['avg_duration_minutes'] * row['total_signals']) + duration_minutes) / total_signals
            else:
                total_signals = 1
                wins = 1 if status == STATUS_TP_HIT else 0
                losses = 1 if status == STATUS_SL_HIT else 0
                expired = 1 if status == STATUS_EXPIRED else 0
                total_pnl = pnl_percent
                best = pnl_percent
                worst = pnl_percent
                avg_duration = duration_minutes

            win_rate = (wins / total_signals * 100) if total_signals > 0 else 0
            avg_pnl = total_pnl / total_signals if total_signals > 0 else 0

            cursor.execute('''
                INSERT OR REPLACE INTO performance_stats (
                    date, total_signals, wins, losses, expired, win_rate,
                    total_pnl_percent, avg_pnl_percent, best_trade_pnl,
                    worst_trade_pnl, avg_duration_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today, total_signals, wins, losses, expired, win_rate,
                total_pnl, avg_pnl, best, worst, avg_duration
            ))

    def get_active_signals(self) -> List[Dict]:
        """Get all active signals"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM signals WHERE status = ? ORDER BY created_at DESC
            ''', (STATUS_ACTIVE,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_active_signals_by_symbol(self, symbol: str) -> List[Dict]:
        """Get active signals for a specific symbol"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM signals WHERE symbol = ? AND status = ?
                ORDER BY created_at DESC
            ''', (symbol, STATUS_ACTIVE))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_signal_by_id(self, signal_id: str) -> Optional[Dict]:
        """Get signal by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM signals WHERE signal_id = ?', (signal_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_recent_signals(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """Get recent signals within the specified hours"""
        since = datetime.now() - timedelta(hours=hours)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM signals
                WHERE created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (since.isoformat(), limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_daily_stats(self, date: str = None) -> Optional[Dict]:
        """Get performance stats for a specific date"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM performance_stats WHERE date = ?', (date,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_overall_stats(self) -> Dict:
        """Get overall performance statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total signals
            cursor.execute('SELECT COUNT(*) as count FROM signals')
            total = cursor.fetchone()['count']

            # By status
            cursor.execute('''
                SELECT status, COUNT(*) as count FROM signals GROUP BY status
            ''')
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

            # Aggregate P&L
            cursor.execute('''
                SELECT
                    AVG(pnl_percent) as avg_pnl,
                    SUM(pnl_percent) as total_pnl,
                    MAX(pnl_percent) as best_pnl,
                    MIN(pnl_percent) as worst_pnl,
                    AVG(duration_minutes) as avg_duration
                FROM signals WHERE status IN (?, ?)
            ''', (STATUS_TP_HIT, STATUS_SL_HIT))
            pnl_stats = cursor.fetchone()

            wins = status_counts.get(STATUS_TP_HIT, 0)
            losses = status_counts.get(STATUS_SL_HIT, 0)
            closed = wins + losses

            return {
                'total_signals': total,
                'active': status_counts.get(STATUS_ACTIVE, 0),
                'wins': wins,
                'losses': losses,
                'expired': status_counts.get(STATUS_EXPIRED, 0),
                'win_rate': (wins / closed * 100) if closed > 0 else 0,
                'avg_pnl_percent': pnl_stats['avg_pnl'] or 0,
                'total_pnl_percent': pnl_stats['total_pnl'] or 0,
                'best_trade_pnl': pnl_stats['best_pnl'] or 0,
                'worst_trade_pnl': pnl_stats['worst_pnl'] or 0,
                'avg_duration_minutes': pnl_stats['avg_duration'] or 0
            }

    def expire_old_signals(self, max_age_hours: int = 24) -> int:
        """Expire signals that are older than max_age_hours"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        expired_count = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT signal_id FROM signals
                WHERE status = ? AND created_at < ?
            ''', (STATUS_ACTIVE, cutoff.isoformat()))

            for row in cursor.fetchall():
                self.update_signal_status(row['signal_id'], STATUS_EXPIRED)
                expired_count += 1

        if expired_count > 0:
            logger.info(f"Expired {expired_count} old signals")

        return expired_count

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Remove signals older than specified days"""
        cutoff = datetime.now() - timedelta(days=days_to_keep)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM signals
                WHERE created_at < ? AND status != ?
            ''', (cutoff.isoformat(), STATUS_ACTIVE))
            deleted = cursor.rowcount

        logger.info(f"Cleaned up {deleted} old signals")
        return deleted

    def get_signals_for_symbol_in_timeframe(self, symbol: str, hours: int = 4) -> List[Dict]:
        """Get signals for a symbol within a timeframe (for cooldown checking)"""
        since = datetime.now() - timedelta(hours=hours)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM signals
                WHERE symbol = ? AND created_at >= ?
                ORDER BY created_at DESC
            ''', (symbol, since.isoformat()))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
