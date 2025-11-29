"""
Shared modules for multi-tier trading bots.
Provides common functionality for data fetching, notifications, signal tracking,
backtesting, and trade execution.
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
from .backtester import BacktestEngine, BacktestResult, BacktestReporter, Trade
from .trade_executor import TradeExecutor, Order, Position, OrderType, OrderSide, PositionStatus
from .execution_bot import ExecutionBot

__all__ = [
    # Data & Exchange
    'EnhancedExchangeClient',

    # Notifications
    'TelegramNotifier',

    # Signal Tracking
    'SignalDatabase',
    'STATUS_ACTIVE',
    'STATUS_TP_HIT',
    'STATUS_SL_HIT',
    'STATUS_EXPIRED',
    'SignalTracker',
    'SignalCooldownManager',

    # Filters & Monitoring
    'AccuracyFilterManager',
    'HeartbeatMonitor',

    # Base Classes
    'BaseStrategy',
    'BaseIndicatorCalculator',
    'BaseBot',

    # Backtesting
    'BacktestEngine',
    'BacktestResult',
    'BacktestReporter',
    'Trade',

    # Trade Execution
    'TradeExecutor',
    'Order',
    'Position',
    'OrderType',
    'OrderSide',
    'PositionStatus',
    'ExecutionBot'
]

__version__ = '3.0.0'
