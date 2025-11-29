"""
Shared modules for multi-tier trading bots.
Provides common functionality for data fetching, notifications, signal tracking, and more.
"""

from .data_client import EnhancedExchangeClient
from .notifier import TelegramNotifier
from .signal_database import SignalDatabase, STATUS_ACTIVE, STATUS_TP_HIT, STATUS_SL_HIT, STATUS_EXPIRED
from .signal_tracker import SignalTracker, SignalCooldownManager
from .accuracy_filters import AccuracyFilterManager
from .heartbeat import HeartbeatMonitor
from .base_strategy import BaseStrategy
from .base_indicators import BaseIndicatorCalculator
from .base_bot import BaseBot

__all__ = [
    'EnhancedExchangeClient',
    'TelegramNotifier',
    'SignalDatabase',
    'STATUS_ACTIVE',
    'STATUS_TP_HIT',
    'STATUS_SL_HIT',
    'STATUS_EXPIRED',
    'SignalTracker',
    'SignalCooldownManager',
    'AccuracyFilterManager',
    'HeartbeatMonitor',
    'BaseStrategy',
    'BaseIndicatorCalculator',
    'BaseBot'
]

__version__ = '2.0.0'
