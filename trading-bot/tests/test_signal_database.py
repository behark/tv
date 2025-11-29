#!/usr/bin/env python3
"""
Unit tests for Signal Database module
"""

import unittest
import os
import sys
import tempfile
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_database import (
    SignalDatabase, STATUS_ACTIVE, STATUS_TP_HIT,
    STATUS_SL_HIT, STATUS_EXPIRED
)


class TestSignalDatabase(unittest.TestCase):
    """Test cases for SignalDatabase class"""

    def setUp(self):
        """Set up test database"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_file.close()
        self.db = SignalDatabase(self.temp_file.name)

    def tearDown(self):
        """Clean up test database"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_add_signal(self):
        """Test adding a signal to database"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0,
            'score': 7.5,
            'max_score': 10.0,
            'strategy_mode': 'trend_following',
            'timeframe': '15m'
        }

        signal_id = self.db.add_signal(signal)

        self.assertIsNotNone(signal_id)
        self.assertTrue(signal_id.startswith('BTC'))

    def test_get_signal_by_id(self):
        """Test retrieving a signal by ID"""
        signal = {
            'symbol': 'ETH/USDT',
            'action': 'SHORT',
            'entry_price': 3000.0,
            'stop_loss': 3100.0,
            'take_profit': 2800.0
        }

        signal_id = self.db.add_signal(signal)
        retrieved = self.db.get_signal_by_id(signal_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['symbol'], 'ETH/USDT')
        self.assertEqual(retrieved['action'], 'SHORT')
        self.assertEqual(retrieved['status'], STATUS_ACTIVE)

    def test_update_signal_status_tp_hit(self):
        """Test updating signal status to TP_HIT"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0
        }

        signal_id = self.db.add_signal(signal)
        success = self.db.update_signal_status(signal_id, STATUS_TP_HIT, 52000.0)

        self.assertTrue(success)

        updated = self.db.get_signal_by_id(signal_id)
        self.assertEqual(updated['status'], STATUS_TP_HIT)
        self.assertEqual(updated['exit_price'], 52000.0)
        self.assertGreater(updated['pnl_percent'], 0)

    def test_update_signal_status_sl_hit(self):
        """Test updating signal status to SL_HIT"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0
        }

        signal_id = self.db.add_signal(signal)
        success = self.db.update_signal_status(signal_id, STATUS_SL_HIT, 49000.0)

        self.assertTrue(success)

        updated = self.db.get_signal_by_id(signal_id)
        self.assertEqual(updated['status'], STATUS_SL_HIT)
        self.assertLess(updated['pnl_percent'], 0)

    def test_get_active_signals(self):
        """Test getting all active signals"""
        # Add multiple signals
        for i in range(3):
            signal = {
                'symbol': f'PAIR{i}/USDT',
                'action': 'LONG',
                'entry_price': 100.0,
                'stop_loss': 95.0,
                'take_profit': 110.0
            }
            self.db.add_signal(signal)

        active = self.db.get_active_signals()
        self.assertEqual(len(active), 3)

    def test_get_active_signals_by_symbol(self):
        """Test getting active signals for specific symbol"""
        # Add signals for different symbols
        self.db.add_signal({
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0
        })
        self.db.add_signal({
            'symbol': 'ETH/USDT',
            'action': 'SHORT',
            'entry_price': 3000.0,
            'stop_loss': 3100.0,
            'take_profit': 2800.0
        })

        btc_signals = self.db.get_active_signals_by_symbol('BTC/USDT')
        self.assertEqual(len(btc_signals), 1)
        self.assertEqual(btc_signals[0]['symbol'], 'BTC/USDT')

    def test_get_overall_stats(self):
        """Test getting overall statistics"""
        # Add and close some signals
        signal_id_1 = self.db.add_signal({
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0
        })
        self.db.update_signal_status(signal_id_1, STATUS_TP_HIT, 52000.0)

        signal_id_2 = self.db.add_signal({
            'symbol': 'ETH/USDT',
            'action': 'SHORT',
            'entry_price': 3000.0,
            'stop_loss': 3100.0,
            'take_profit': 2800.0
        })
        self.db.update_signal_status(signal_id_2, STATUS_SL_HIT, 3100.0)

        stats = self.db.get_overall_stats()

        self.assertEqual(stats['total_signals'], 2)
        self.assertEqual(stats['wins'], 1)
        self.assertEqual(stats['losses'], 1)
        self.assertEqual(stats['win_rate'], 50.0)

    def test_generate_signal_id(self):
        """Test signal ID generation"""
        signal_id = self.db.generate_signal_id('BTC/USDT')

        self.assertTrue(signal_id.startswith('BTC'))
        self.assertIn('-', signal_id)


class TestSignalDatabaseEdgeCases(unittest.TestCase):
    """Test edge cases for SignalDatabase"""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_file.close()
        self.db = SignalDatabase(self.temp_file.name)

    def tearDown(self):
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_update_nonexistent_signal(self):
        """Test updating a signal that doesn't exist"""
        success = self.db.update_signal_status('NONEXISTENT-123', STATUS_TP_HIT, 100.0)
        self.assertFalse(success)

    def test_short_pnl_calculation(self):
        """Test P&L calculation for short positions"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'SHORT',
            'entry_price': 50000.0,
            'stop_loss': 51000.0,
            'take_profit': 48000.0
        }

        signal_id = self.db.add_signal(signal)

        # TP hit for short (price went down)
        self.db.update_signal_status(signal_id, STATUS_TP_HIT, 48000.0)
        updated = self.db.get_signal_by_id(signal_id)

        # For SHORT: P&L = (entry - exit) / entry
        # (50000 - 48000) / 50000 = 4%
        self.assertGreater(updated['pnl_percent'], 0)


if __name__ == '__main__':
    unittest.main()
