#!/usr/bin/env python3
"""
Unit tests for Signal Tracker module
"""

import unittest
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_database import SignalDatabase, STATUS_ACTIVE, STATUS_TP_HIT, STATUS_SL_HIT
from signal_tracker import SignalTracker, SignalCooldownManager


class TestSignalTracker(unittest.TestCase):
    """Test cases for SignalTracker class"""

    def setUp(self):
        """Set up test database and tracker"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_file.close()
        self.db = SignalDatabase(self.temp_file.name)
        self.tracker = SignalTracker(self.db, {'signal_expiry_hours': 24})

    def tearDown(self):
        """Clean up"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_add_signal(self):
        """Test adding a signal through tracker"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0
        }

        signal_id = self.tracker.add_signal(signal)
        self.assertIsNotNone(signal_id)

    def test_check_long_tp_hit(self):
        """Test TP hit detection for LONG position"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0
        }

        signal_id = self.tracker.add_signal(signal)
        stored_signal = self.db.get_signal_by_id(signal_id)

        # Price hits TP
        status = self.tracker.check_signal_status(
            stored_signal,
            current_price=52500.0,
            high_price=52500.0,
            low_price=51000.0
        )

        self.assertEqual(status, STATUS_TP_HIT)

    def test_check_long_sl_hit(self):
        """Test SL hit detection for LONG position"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0
        }

        signal_id = self.tracker.add_signal(signal)
        stored_signal = self.db.get_signal_by_id(signal_id)

        # Price hits SL
        status = self.tracker.check_signal_status(
            stored_signal,
            current_price=48500.0,
            high_price=50500.0,
            low_price=48500.0
        )

        self.assertEqual(status, STATUS_SL_HIT)

    def test_check_short_tp_hit(self):
        """Test TP hit detection for SHORT position"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'SHORT',
            'entry_price': 50000.0,
            'stop_loss': 51000.0,
            'take_profit': 48000.0
        }

        signal_id = self.tracker.add_signal(signal)
        stored_signal = self.db.get_signal_by_id(signal_id)

        # Price hits TP (goes down for short)
        status = self.tracker.check_signal_status(
            stored_signal,
            current_price=47500.0,
            high_price=49000.0,
            low_price=47500.0
        )

        self.assertEqual(status, STATUS_TP_HIT)

    def test_check_short_sl_hit(self):
        """Test SL hit detection for SHORT position"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'SHORT',
            'entry_price': 50000.0,
            'stop_loss': 51000.0,
            'take_profit': 48000.0
        }

        signal_id = self.tracker.add_signal(signal)
        stored_signal = self.db.get_signal_by_id(signal_id)

        # Price hits SL (goes up for short)
        status = self.tracker.check_signal_status(
            stored_signal,
            current_price=51500.0,
            high_price=51500.0,
            low_price=49500.0
        )

        self.assertEqual(status, STATUS_SL_HIT)

    def test_signal_still_active(self):
        """Test signal remains active when neither TP nor SL hit"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0
        }

        signal_id = self.tracker.add_signal(signal)
        stored_signal = self.db.get_signal_by_id(signal_id)

        # Price is between SL and TP
        status = self.tracker.check_signal_status(
            stored_signal,
            current_price=50500.0,
            high_price=51000.0,
            low_price=50000.0
        )

        self.assertIsNone(status)

    def test_callback_registration(self):
        """Test callback registration and triggering"""
        callback_triggered = {'tp': False, 'sl': False}

        def on_tp(signal, **kwargs):
            callback_triggered['tp'] = True

        def on_sl(signal, **kwargs):
            callback_triggered['sl'] = True

        self.tracker.register_callback('on_tp_hit', on_tp)
        self.tracker.register_callback('on_sl_hit', on_sl)

        signal = {
            'symbol': 'BTC/USDT',
            'action': 'LONG',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0
        }

        signal_id = self.tracker.add_signal(signal)
        stored_signal = self.db.get_signal_by_id(signal_id)

        # Trigger TP
        self.tracker.check_signal_status(
            stored_signal,
            current_price=52500.0,
            high_price=52500.0,
            low_price=51000.0
        )

        self.assertTrue(callback_triggered['tp'])


class TestSignalCooldownManager(unittest.TestCase):
    """Test cases for SignalCooldownManager"""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_file.close()
        self.db = SignalDatabase(self.temp_file.name)
        self.cooldown = SignalCooldownManager(self.db, cooldown_minutes=60)

    def tearDown(self):
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_can_signal_initially(self):
        """Test that signals are allowed initially"""
        can_signal = self.cooldown.can_signal('BTC/USDT', 'LONG')
        self.assertTrue(can_signal)

    def test_cooldown_after_signal(self):
        """Test cooldown is active after signal"""
        self.cooldown.record_signal('BTC/USDT', 'LONG')

        can_signal = self.cooldown.can_signal('BTC/USDT', 'LONG')
        self.assertFalse(can_signal)

    def test_different_action_allowed(self):
        """Test different action is allowed during cooldown"""
        self.cooldown.record_signal('BTC/USDT', 'LONG')

        # SHORT should still be allowed
        can_signal = self.cooldown.can_signal('BTC/USDT', 'SHORT')
        self.assertTrue(can_signal)

    def test_different_symbol_allowed(self):
        """Test different symbol is allowed during cooldown"""
        self.cooldown.record_signal('BTC/USDT', 'LONG')

        # Different symbol should be allowed
        can_signal = self.cooldown.can_signal('ETH/USDT', 'LONG')
        self.assertTrue(can_signal)


if __name__ == '__main__':
    unittest.main()
